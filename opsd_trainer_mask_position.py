import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from accelerate.utils import gather_object, is_peft_model
from torch import nn

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache

from opsd_trainer import OPSDTrainer


class MaskPositionOPSDTrainer(OPSDTrainer):
    """OPSD variant with mask-based position alignment.

    Rollout remains clean: the student generates from the original problem-only
    prompt.  During the loss forward pass, teacher and student receive identical
    token sequences containing the reference-solution privilege slot.  The
    teacher can attend to that slot; the student cannot.  Both forwards receive
    the same position_ids derived from a non-privilege mask, so ordinary tokens
    occupy the same positions for teacher and student.
    """

    def __init__(
        self,
        *args,
        position_alignment_debug=True,
        **kwargs,
    ):
        self.teacher_thinking = kwargs.get("teacher_thinking", True)
        super().__init__(*args, **kwargs)
        if self.reason_first:
            raise ValueError(
                "MaskPositionOPSDTrainer currently supports reason_first=False only. "
                "Use the original reason_first path or add a separate aligned reasoning variant."
            )

        self.position_alignment_debug = position_alignment_debug
        self._alignment_debug_printed = False
        self._privilege_slot_marker = "__OPSD_PRIVILEGE_SLOT__"

        print(f"\n{'='*80}")
        print("MASK-BASED POSITION ALIGNMENT ENABLED")
        print("Student rollout prompt remains clean; only loss prefixes are mask-position aligned.")
        print("Teacher sees the privilege slot; student masks it out.")
        print("Position ids are generated from the shared non-privilege mask.")
        print(f"{'='*80}\n")

    def _apply_teacher_chat_template(self, content):
        return self.processing_class.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.teacher_thinking,
        )

    def _build_privilege_slot_text_parts(self, problem):
        prefix = (
            f"Problem: {problem}\n\n"
            "Here is a reference solution to this problem:\n"
            "=== Reference Solution Begin ===\n"
        )
        suffix = (
            f"\n=== Reference Solution End ===\n"
            f"{self.data_collator.transition_prompt}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        return prefix, suffix

    def _tokenize_text(self, text):
        return self.processing_class(text, add_special_tokens=False).input_ids

    def _build_mask_position_prefix(self, problem, solution):
        content_prefix, content_suffix = self._build_privilege_slot_text_parts(problem)
        templated = self._apply_teacher_chat_template(
            content_prefix + self._privilege_slot_marker + content_suffix
        )
        if self._privilege_slot_marker not in templated:
            raise ValueError("Privilege slot marker was not preserved by the chat template.")

        templated_prefix, templated_suffix = templated.split(self._privilege_slot_marker, 1)

        shared_prefix_ids = self._tokenize_text(templated_prefix)
        privilege_slot_ids = self._tokenize_text(solution)
        shared_suffix_ids = self._tokenize_text(templated_suffix)

        prefix_ids = shared_prefix_ids + privilege_slot_ids + shared_suffix_ids
        non_privilege_mask = (
            [1] * len(shared_prefix_ids)
            + [0] * len(privilege_slot_ids)
            + [1] * len(shared_suffix_ids)
        )

        if len(prefix_ids) != len(non_privilege_mask):
            raise AssertionError("Internal mask-position alignment error: mask length mismatch.")

        return prefix_ids, non_privilege_mask, len(privilege_slot_ids), len(shared_prefix_ids)

    def _build_position_ids(self, position_source_mask):
        position_ids = position_source_mask.long().cumsum(dim=-1) - 1
        return position_ids.clamp_min(0)

    def _pad_prefixes(self, prefix_lists, non_privilege_masks, max_len, device):
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id for mask-position padding.")

        input_ids = torch.full(
            (len(prefix_lists), max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        teacher_attention_mask = torch.zeros(
            (len(prefix_lists), max_len),
            dtype=torch.long,
            device=device,
        )
        student_attention_mask = torch.zeros(
            (len(prefix_lists), max_len),
            dtype=torch.long,
            device=device,
        )
        position_source_mask = torch.zeros(
            (len(prefix_lists), max_len),
            dtype=torch.long,
            device=device,
        )

        for idx, (ids, non_privilege_mask) in enumerate(zip(prefix_lists, non_privilege_masks)):
            current_len = len(ids)
            if current_len == 0:
                continue
            input_ids[idx, :current_len] = torch.tensor(ids, dtype=torch.long, device=device)
            teacher_attention_mask[idx, :current_len] = 1
            non_privilege_tensor = torch.tensor(non_privilege_mask, dtype=torch.long, device=device)
            student_attention_mask[idx, :current_len] = non_privilege_tensor
            position_source_mask[idx, :current_len] = non_privilege_tensor

        return input_ids, teacher_attention_mask, student_attention_mask, position_source_mask

    def _build_mask_position_loss_inputs(self, inputs, generation_ids):
        device = self.accelerator.device
        prefix_lists = []
        non_privilege_masks = []
        prefix_lengths = []
        non_privilege_lengths = []
        slot_lengths = []
        slot_start_offsets = []

        for problem, solution in zip(inputs["problems"], inputs["solutions"]):
            prefix_ids, non_privilege_mask, slot_len, slot_start = self._build_mask_position_prefix(
                problem,
                solution,
            )
            prefix_lists.append(prefix_ids)
            non_privilege_masks.append(non_privilege_mask)
            prefix_lengths.append(len(prefix_ids))
            non_privilege_lengths.append(sum(non_privilege_mask))
            slot_lengths.append(slot_len)
            slot_start_offsets.append(slot_start)

        max_prefix_len = max(prefix_lengths)
        total_len = max_prefix_len + generation_ids.shape[1]
        max_length = getattr(self.args, "max_length", None)
        if max_length is not None and total_len > max_length:
            longest_idx = max(range(len(prefix_lengths)), key=lambda i: prefix_lengths[i])
            raise ValueError(
                "Mask-position aligned sequence exceeds max_length without truncation. "
                f"max_length={max_length}, batch_max_prefix_len={max_prefix_len}, "
                f"generation_len={generation_ids.shape[1]}, total_len={total_len}, "
                f"longest_example_index={longest_idx}, "
                f"longest_slot_len={slot_lengths[longest_idx]}, "
                f"longest_prefix_len={prefix_lengths[longest_idx]}. "
                "Increase --max_length or reduce --max_completion_length; this variant does not "
                "silently truncate privilege slots because that can change position alignment."
            )

        (
            prefix_ids,
            teacher_prefix_mask,
            student_prefix_mask,
            prefix_position_source_mask,
        ) = self._pad_prefixes(prefix_lists, non_privilege_masks, max_prefix_len, device)

        generation_attention_mask = torch.ones_like(generation_ids, device=device)
        if self.processing_class.pad_token_id is not None:
            generation_attention_mask[generation_ids == self.processing_class.pad_token_id] = 0

        input_ids = torch.cat([prefix_ids, generation_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prefix_mask, generation_attention_mask], dim=1)
        student_attention_mask = torch.cat([student_prefix_mask, generation_attention_mask], dim=1)
        position_source_mask = torch.cat([prefix_position_source_mask, generation_attention_mask], dim=1)
        position_ids = self._build_position_ids(position_source_mask)

        labels = input_ids.clone()
        labels[:, :max_prefix_len] = -100
        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        prefix_lengths_tensor = torch.tensor(prefix_lengths, dtype=torch.long, device=device)

        if self.position_alignment_debug:
            for idx, actual_prefix_len in enumerate(prefix_lengths):
                slot_start = slot_start_offsets[idx]
                slot_end = slot_start + slot_lengths[idx]
                student_slot_visible = int(student_attention_mask[idx, slot_start:slot_end].sum().item())
                teacher_slot_visible = int(teacher_attention_mask[idx, slot_start:slot_end].sum().item())
                if student_slot_visible != 0:
                    raise AssertionError("Student attention mask can see privilege slot tokens.")
                if teacher_slot_visible != slot_lengths[idx]:
                    raise AssertionError("Teacher attention mask cannot see all privilege slot tokens.")

                generation_start_position = int(position_ids[idx, max_prefix_len].item())
                expected_generation_start = non_privilege_lengths[idx]
                if generation_start_position != expected_generation_start:
                    raise AssertionError(
                        "Generation position id is not aligned with non-privilege prefix length: "
                        f"got {generation_start_position}, expected {expected_generation_start}"
                    )

            if not self._alignment_debug_printed and self.accelerator.is_main_process:
                print(f"\n{'='*80}")
                print("MASK-POSITION ALIGNMENT DEBUG")
                print(f"Batch max prefix length: {max_prefix_len}")
                print(f"Generation width: {generation_ids.shape[1]}")
                print(f"Slot lengths: {slot_lengths}")
                print(f"Prefix lengths including privilege: {prefix_lengths}")
                print(f"Non-privilege prefix lengths: {non_privilege_lengths}")
                print("Teacher can attend to privilege slots; student cannot.")
                print("Teacher/student share position_ids generated from non-privilege masks.")
                print(f"{'='*80}\n")
                self._alignment_debug_printed = True

        inputs["student_input_ids"] = input_ids
        inputs["student_attention_mask"] = student_attention_mask
        inputs["student_position_ids"] = position_ids
        inputs["student_prompt_length"] = max_prefix_len
        inputs["student_prompt_lengths_per_example"] = prefix_lengths_tensor
        inputs["teacher_input_ids"] = input_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask
        inputs["teacher_position_ids"] = position_ids
        inputs["teacher_prompt_length"] = max_prefix_len
        inputs["teacher_prompt_lengths_per_example"] = prefix_lengths_tensor
        inputs["labels"] = labels

        return {
            "max_prefix_len": max_prefix_len,
            "slot_lengths": slot_lengths,
            "prefix_lengths": prefix_lengths,
            "non_privilege_prefix_lengths": non_privilege_lengths,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
            position_ids=inputs["student_position_ids"],
        )

        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]

        if self.use_thinking_machines_loss:
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            student_log_probs_sampled = torch.gather(
                student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            del student_logits, student_log_probs
        else:
            student_logits_for_loss = student_logits
            del student_logits

        if return_outputs:
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student
        empty_cache()

        if self.use_ema_teacher:
            adapter_context = self._ema_teacher_context(model)
        elif self.fixed_teacher and is_peft_model(model):
            adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            adapter_context = nullcontext()

        with torch.no_grad(), adapter_context:
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                position_ids=inputs["teacher_position_ids"],
            )

            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]

            if self.use_thinking_machines_loss:
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                teacher_log_probs_sampled = torch.gather(
                    teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                del teacher_logits, teacher_log_probs
            else:
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

            del outputs_teacher
            empty_cache()

        if self.use_thinking_machines_loss:
            advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

            if shifted_labels is not None:
                mask = shifted_labels != -100
                advantage = advantage[mask]
                student_log_probs_sampled_masked = student_log_probs_sampled[mask]
            else:
                student_log_probs_sampled_masked = student_log_probs_sampled

            loss = -(advantage * student_log_probs_sampled_masked).mean()

            del (
                student_log_probs_sampled,
                teacher_log_probs_sampled,
                advantage,
                student_log_probs_sampled_masked,
            )
        else:
            loss = self.generalized_jsd_loss(
                student_logits=student_logits_for_loss,
                teacher_logits=teacher_logits_for_loss,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )
            del student_logits_for_loss, teacher_logits_for_loss

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return loss, minimal_output
        return loss

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | object],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        on_policy = True

        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs,
                self.generation_config,
                self.processing_class.pad_token_id,
            )
            generated_ids, _, _, prompt_texts, completion_texts = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model,
                    inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
                generated_ids, _, _ = result
                prompt_texts = self.processing_class.batch_decode(
                    inputs["student_prompts"],
                    skip_special_tokens=False,
                )
                student_prompt_len = inputs["student_prompt_length"]
                completion_ids = generated_ids[:, student_prompt_len:]
                completion_texts = self.processing_class.batch_decode(
                    completion_ids,
                    skip_special_tokens=False,
                )

        original_student_prompt_len = inputs["student_prompt_length"]
        generation_ids = generated_ids[:, original_student_prompt_len:]
        alignment_info = self._build_mask_position_loss_inputs(inputs, generation_ids)

        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {
                    "step": self.state.global_step,
                    "prompt": prompt,
                    "completion": completion,
                    "mask_position_alignment": True,
                    "max_prefix_len": alignment_info["max_prefix_len"],
                }
            )

        if random.random() < 0.01:
            print(f"\n{'='*80}")
            print(f"MASK-POSITION OPSD SAMPLE (Step {self.state.global_step}):")
            print(f"{'='*80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nClean rollout prompt:\n{prompt_texts[sample_idx]}")
            print(f"\nCompletion:\n{completion_texts[sample_idx]}")
            print(f"\nSlot length: {alignment_info['slot_lengths'][sample_idx]}")
            print(f"Prefix length: {alignment_info['prefix_lengths'][sample_idx]}")
            print(
                "Non-privilege prefix length: "
                f"{alignment_info['non_privilege_prefix_lengths'][sample_idx]}"
            )
            print(f"{'='*80}\n")

        loss = super(OPSDTrainer, self).training_step(model, inputs, num_items_in_batch)

        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv

        empty_cache()
        return loss
