import random

import torch
from accelerate.utils import gather_object
from torch import nn

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache

from opsd_trainer import OPSDTrainer


class PrivilegeSlotOPSDTrainer(OPSDTrainer):
    """OPSD variant with equal-length student/teacher privilege slots.

    Rollout stays unchanged: the student generates from the clean problem-only
    prompt.  Only the loss forward pass is rebuilt so teacher and student answer
    tokens start at identical positions.
    """

    def __init__(
        self,
        *args,
        privilege_placeholder_text="The privileged information is hidden.",
        position_alignment_debug=True,
        **kwargs,
    ):
        self.teacher_thinking = kwargs.get("teacher_thinking", True)
        super().__init__(*args, **kwargs)
        if self.reason_first:
            raise ValueError(
                "PrivilegeSlotOPSDTrainer currently supports reason_first=False only. "
                "Use the original reason_first path or add a separate aligned reasoning variant."
            )

        self.privilege_placeholder_text = privilege_placeholder_text
        self.position_alignment_debug = position_alignment_debug
        self._alignment_debug_printed = False
        self._privilege_slot_marker = "__OPSD_PRIVILEGE_SLOT__"

        placeholder_ids = self.processing_class(
            self.privilege_placeholder_text,
            add_special_tokens=False,
        ).input_ids
        if len(placeholder_ids) == 0:
            raise ValueError("privilege_placeholder_text must tokenize to at least one token.")
        self._placeholder_unit_ids = placeholder_ids

        print(f"\n{'='*80}")
        print("PRIVILEGE SLOT POSITION ALIGNMENT ENABLED")
        print("Student rollout prompt remains clean; only loss prefixes are aligned.")
        print(f"Placeholder text: {self.privilege_placeholder_text}")
        print(f"Placeholder unit tokens: {len(self._placeholder_unit_ids)}")
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

    def _make_placeholder_ids(self, target_len):
        if target_len == 0:
            return []

        repeats = (target_len + len(self._placeholder_unit_ids) - 1) // len(self._placeholder_unit_ids)
        return (self._placeholder_unit_ids * repeats)[:target_len]

    def _tokenize_text(self, text):
        return self.processing_class(text, add_special_tokens=False).input_ids

    def _build_aligned_prefix_pair(self, problem, solution):
        content_prefix, content_suffix = self._build_privilege_slot_text_parts(problem)
        templated = self._apply_teacher_chat_template(
            content_prefix + self._privilege_slot_marker + content_suffix
        )
        if self._privilege_slot_marker not in templated:
            raise ValueError("Privilege slot marker was not preserved by the chat template.")

        templated_prefix, templated_suffix = templated.split(self._privilege_slot_marker, 1)

        shared_prefix_ids = self._tokenize_text(templated_prefix)
        shared_suffix_ids = self._tokenize_text(templated_suffix)
        teacher_slot_ids = self._tokenize_text(solution)
        student_slot_ids = self._make_placeholder_ids(len(teacher_slot_ids))

        if len(student_slot_ids) != len(teacher_slot_ids):
            raise AssertionError(
                "Internal privilege slot alignment error: "
                f"student slot len {len(student_slot_ids)} != teacher slot len {len(teacher_slot_ids)}"
            )

        student_prefix_ids = shared_prefix_ids + student_slot_ids + shared_suffix_ids
        teacher_prefix_ids = shared_prefix_ids + teacher_slot_ids + shared_suffix_ids

        if len(student_prefix_ids) != len(teacher_prefix_ids):
            raise AssertionError(
                "Internal prompt alignment error: "
                f"student prefix len {len(student_prefix_ids)} != teacher prefix len {len(teacher_prefix_ids)}"
            )

        return student_prefix_ids, teacher_prefix_ids, len(teacher_slot_ids)

    def _pad_prefixes(self, prefix_lists, max_len, device):
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id for privilege-slot padding.")

        input_ids = torch.full(
            (len(prefix_lists), max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(prefix_lists), max_len),
            dtype=torch.long,
            device=device,
        )

        for idx, ids in enumerate(prefix_lists):
            current_len = len(ids)
            if current_len == 0:
                continue
            input_ids[idx, :current_len] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[idx, :current_len] = 1

        return input_ids, attention_mask

    def _build_aligned_loss_inputs(self, inputs, generation_ids):
        device = self.accelerator.device
        student_prefixes = []
        teacher_prefixes = []
        aligned_lengths = []
        slot_lengths = []

        for problem, solution in zip(inputs["problems"], inputs["solutions"]):
            student_prefix_ids, teacher_prefix_ids, slot_len = self._build_aligned_prefix_pair(
                problem,
                solution,
            )
            student_prefixes.append(student_prefix_ids)
            teacher_prefixes.append(teacher_prefix_ids)
            aligned_lengths.append(len(student_prefix_ids))
            slot_lengths.append(slot_len)

        max_prefix_len = max(aligned_lengths)
        total_len = max_prefix_len + generation_ids.shape[1]
        max_length = getattr(self.args, "max_length", None)
        if max_length is not None and total_len > max_length:
            longest_idx = max(range(len(aligned_lengths)), key=lambda i: aligned_lengths[i])
            raise ValueError(
                "Privilege-slot aligned sequence exceeds max_length without truncation. "
                f"max_length={max_length}, batch_max_prefix_len={max_prefix_len}, "
                f"generation_len={generation_ids.shape[1]}, total_len={total_len}, "
                f"longest_example_index={longest_idx}, "
                f"longest_slot_len={slot_lengths[longest_idx]}, "
                f"longest_prefix_len={aligned_lengths[longest_idx]}. "
                "Increase --max_length or reduce --max_completion_length; this variant does not "
                "silently truncate privilege slots because that can reintroduce position mismatch."
            )

        student_prefix_ids, student_prefix_mask = self._pad_prefixes(
            student_prefixes,
            max_prefix_len,
            device,
        )
        teacher_prefix_ids, teacher_prefix_mask = self._pad_prefixes(
            teacher_prefixes,
            max_prefix_len,
            device,
        )

        generation_attention_mask = torch.ones_like(generation_ids, device=device)
        if self.processing_class.pad_token_id is not None:
            generation_attention_mask[generation_ids == self.processing_class.pad_token_id] = 0

        student_input_ids = torch.cat([student_prefix_ids, generation_ids], dim=1)
        teacher_input_ids = torch.cat([teacher_prefix_ids, generation_ids], dim=1)
        student_attention_mask = torch.cat([student_prefix_mask, generation_attention_mask], dim=1)
        teacher_attention_mask = torch.cat([teacher_prefix_mask, generation_attention_mask], dim=1)

        labels = student_input_ids.clone()
        for idx, actual_prefix_len in enumerate(aligned_lengths):
            labels[idx, :actual_prefix_len] = -100
        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        aligned_lengths_tensor = torch.tensor(aligned_lengths, dtype=torch.long, device=device)

        if self.position_alignment_debug:
            for idx, actual_prefix_len in enumerate(aligned_lengths):
                student_positions_before_answer = int(student_attention_mask[idx, :max_prefix_len].sum().item())
                teacher_positions_before_answer = int(teacher_attention_mask[idx, :max_prefix_len].sum().item())
                if student_positions_before_answer != actual_prefix_len:
                    raise AssertionError("Student attention mask does not preserve aligned prefix length.")
                if teacher_positions_before_answer != actual_prefix_len:
                    raise AssertionError("Teacher attention mask does not preserve aligned prefix length.")

            if not self._alignment_debug_printed and self.accelerator.is_main_process:
                print(f"\n{'='*80}")
                print("PRIVILEGE SLOT ALIGNMENT DEBUG")
                print(f"Batch max aligned prefix length: {max_prefix_len}")
                print(f"Generation width: {generation_ids.shape[1]}")
                print(f"Slot lengths: {slot_lengths}")
                print(f"Aligned prefix lengths: {aligned_lengths}")
                print("Student and teacher prefix lengths match for every example.")
                print(f"{'='*80}\n")
                self._alignment_debug_printed = True

        inputs["student_input_ids"] = student_input_ids
        inputs["student_attention_mask"] = student_attention_mask
        inputs["student_prompt_length"] = max_prefix_len
        inputs["student_prompt_lengths_per_example"] = aligned_lengths_tensor
        inputs["teacher_input_ids"] = teacher_input_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask
        inputs["teacher_prompt_length"] = max_prefix_len
        inputs["teacher_prompt_lengths_per_example"] = aligned_lengths_tensor
        inputs["labels"] = labels

        return {
            "max_aligned_prefix_len": max_prefix_len,
            "slot_lengths": slot_lengths,
            "aligned_prefix_lengths": aligned_lengths,
        }

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
        alignment_info = self._build_aligned_loss_inputs(inputs, generation_ids)

        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {
                    "step": self.state.global_step,
                    "prompt": prompt,
                    "completion": completion,
                    "position_aligned_privilege_slot": True,
                    "max_aligned_prefix_len": alignment_info["max_aligned_prefix_len"],
                }
            )

        if random.random() < 0.01:
            print(f"\n{'='*80}")
            print(f"PRIVILEGE SLOT OPSD SAMPLE (Step {self.state.global_step}):")
            print(f"{'='*80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nClean rollout prompt:\n{prompt_texts[sample_idx]}")
            print(f"\nCompletion:\n{completion_texts[sample_idx]}")
            print(f"\nSlot length: {alignment_info['slot_lengths'][sample_idx]}")
            print(f"Aligned prefix length: {alignment_info['aligned_prefix_lengths'][sample_idx]}")
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
