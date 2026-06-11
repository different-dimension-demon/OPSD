import random

import torch
from accelerate.utils import gather_object
from torch import nn
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache

from opsd_trainer import OPSDTrainer
from opsd_trainer_mask_position import MaskPositionOPSDTrainer


class StudentAlignTeacherOPSDTrainer(MaskPositionOPSDTrainer):
    """OPSD variant where student positions are aligned to teacher positions.

    Rollout remains clean. During the loss forward pass, teacher and student
    receive identical token sequences containing the reference-solution
    privilege slot. The teacher can attend to that slot; the student cannot.
    Unlike mask-position alignment, the privilege slot still advances
    position_ids, so ordinary student tokens are shifted to the same positions
    as the teacher's natural privileged sequence.
    """

    def __init__(
        self,
        *args,
        position_alignment_debug=True,
        **kwargs,
    ):
        self.teacher_thinking = kwargs.get("teacher_thinking", True)
        OPSDTrainer.__init__(self, *args, **kwargs)
        if self.reason_first:
            raise ValueError(
                "StudentAlignTeacherOPSDTrainer currently supports reason_first=False only. "
                "Use the original reason_first path or add a separate aligned reasoning variant."
            )

        self.position_alignment_debug = position_alignment_debug
        self._alignment_debug_printed = False
        self._privilege_slot_marker = "__OPSD_PRIVILEGE_SLOT__"

        print(f"\n{'='*80}")
        print("STUDENT-ALIGNS-TO-TEACHER POSITION ALIGNMENT ENABLED")
        print("Student rollout prompt remains clean; only loss prefixes are aligned.")
        print("Teacher sees the privilege slot; student masks it out.")
        print("Privilege slots advance shared position_ids, matching teacher natural positions.")
        print(f"{'='*80}\n")

    def _pad_prefixes(self, prefix_lists, non_privilege_masks, max_len, device):
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id
        if pad_token_id is None:
            raise ValueError(
                "Tokenizer must define pad_token_id or eos_token_id for student-teacher alignment padding."
            )

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
            student_attention_mask[idx, :current_len] = torch.tensor(
                non_privilege_mask,
                dtype=torch.long,
                device=device,
            )
            position_source_mask[idx, :current_len] = 1

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
                "Student-teacher aligned sequence exceeds max_length without truncation. "
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
                expected_generation_start = actual_prefix_len
                if generation_start_position != expected_generation_start:
                    raise AssertionError(
                        "Generation position id is not aligned with teacher prefix length: "
                        f"got {generation_start_position}, expected {expected_generation_start}"
                    )

                if slot_lengths[idx] > 0:
                    slot_first_position = int(position_ids[idx, slot_start].item())
                    slot_last_position = int(position_ids[idx, slot_end - 1].item())
                    if slot_first_position != slot_start or slot_last_position != slot_end - 1:
                        raise AssertionError(
                            "Privilege slot did not advance natural position ids as expected: "
                            f"slot_start={slot_start}, got_first={slot_first_position}, "
                            f"got_last={slot_last_position}, slot_end={slot_end}"
                        )

            if not self._alignment_debug_printed and self.accelerator.is_main_process:
                print(f"\n{'='*80}")
                print("STUDENT-ALIGNS-TO-TEACHER POSITION DEBUG")
                print(f"Batch max prefix length: {max_prefix_len}")
                print(f"Generation width: {generation_ids.shape[1]}")
                print(f"Slot lengths: {slot_lengths}")
                print(f"Prefix lengths including privilege: {prefix_lengths}")
                print(f"Non-privilege prefix lengths: {non_privilege_lengths}")
                print("Teacher can attend to privilege slots; student cannot.")
                print("Teacher/student share position_ids; privilege slots advance positions.")
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
                    "student_align_teacher_alignment": True,
                    "max_prefix_len": alignment_info["max_prefix_len"],
                }
            )

        if random.random() < 0.01:
            print(f"\n{'='*80}")
            print(f"STUDENT-ALIGNS-TO-TEACHER OPSD SAMPLE (Step {self.state.global_step}):")
            print(f"{'='*80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nClean rollout prompt:\n{prompt_texts[sample_idx]}")
            print(f"\nCompletion:\n{completion_texts[sample_idx]}")
            print(f"\nSlot length: {alignment_info['slot_lengths'][sample_idx]}")
            print(f"Prefix length including privilege: {alignment_info['prefix_lengths'][sample_idx]}")
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
