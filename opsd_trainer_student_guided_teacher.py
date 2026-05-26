import random
from contextlib import nullcontext

import torch
from accelerate.utils import gather_object, is_peft_model
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from torch import nn

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache

from opsd_trainer import OPSDTrainer


class StudentGuidedTeacherOPSDTrainer(OPSDTrainer):
    """OPSD variant where the teacher first critiques the student's rollout."""

    def __init__(
        self,
        *args,
        teacher_guidance_max_new_tokens=512,
        teacher_guidance_temperature=0.7,
        teacher_guidance_top_p=0.95,
        teacher_guidance_top_k=20,
        teacher_guidance_mode="critique",
        teacher_guidance_thinking=True,
        teacher_scoring_thinking=True,
        include_student_answer_in_scoring=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_guidance_mode = teacher_guidance_mode
        self.teacher_guidance_thinking = teacher_guidance_thinking
        self.teacher_scoring_thinking = teacher_scoring_thinking
        self.include_student_answer_in_scoring = include_student_answer_in_scoring
        self.teacher_guidance_generation_config = GenerationConfig(
            max_new_tokens=teacher_guidance_max_new_tokens,
            temperature=teacher_guidance_temperature,
            top_p=teacher_guidance_top_p,
            top_k=teacher_guidance_top_k,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.teacher_guidance_generation_config.eos_token_id = self.model.generation_config.eos_token_id

    def _teacher_generation_context(self, model):
        if self.use_ema_teacher:
            return self._ema_teacher_context(model)

        unwrapped = self.accelerator.unwrap_model(model)
        if self.fixed_teacher and is_peft_model(unwrapped):
            return unwrapped.disable_adapter()
        return nullcontext()

    def _build_guidance_messages(self, problem, solution, student_answer):
        if self.teacher_guidance_mode == "hint":
            instruction = (
                "Compare the student's attempt with the reference solution. "
                "Give targeted hints and correction directions, but do not write a full solution."
            )
        elif self.teacher_guidance_mode == "corrected_reasoning":
            instruction = (
                "Compare the student's attempt with the reference solution. "
                "Identify the mistakes, then provide corrected reasoning that leads to the final answer."
            )
        else:
            instruction = (
                "Compare the student's attempt with the reference solution. "
                "Identify mistakes, missing reasoning steps, and key corrections. "
                "Provide concise guidance that would help the student solve this problem correctly. "
                "Do not merely copy the reference solution."
            )

        content = (
            f"Problem:\n{problem}\n\n"
            f"Reference solution:\n{solution}\n\n"
            f"Student answer:\n{student_answer}\n\n"
            f"{instruction}"
        )
        return [{"role": "user", "content": content}]

    def _build_scoring_messages(self, problem, solution, teacher_guidance, student_answer=None):
        student_block = ""
        if self.include_student_answer_in_scoring and student_answer is not None:
            student_block = f"\n\nStudent answer for context:\n{student_answer}"

        content = (
            f"Problem:\n{problem}\n\n"
            f"Reference solution:\n{solution}\n\n"
            f"Teacher guidance based on the student's attempt:\n{teacher_guidance}"
            f"{student_block}\n\n"
            "Now solve the problem correctly. Please reason step by step, and put your final answer within \\boxed{}."
        )
        return [{"role": "user", "content": content}]

    def _apply_chat_template(self, messages, enable_thinking=True):
        return self.processing_class.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def _generate_teacher_guidance(self, model, problems, solutions, student_answers):
        prompts = [
            self._apply_chat_template(
                self._build_guidance_messages(problem, solution, student_answer),
                enable_thinking=self.teacher_guidance_thinking,
            )
            for problem, solution, student_answer in zip(problems, solutions, student_answers)
        ]

        tokenized = self.processing_class(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max(1, self.args.max_length - self.teacher_guidance_generation_config.max_new_tokens),
            add_special_tokens=False,
        ).to(self.accelerator.device)

        original_use_cache = model.config.use_cache
        original_gen_use_cache = self.teacher_guidance_generation_config.use_cache
        model.config.use_cache = True
        self.teacher_guidance_generation_config.use_cache = True
        prompt_len = tokenized.input_ids.shape[1]

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                    generation_config=self.teacher_guidance_generation_config,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
        finally:
            model.config.use_cache = original_use_cache
            self.teacher_guidance_generation_config.use_cache = original_gen_use_cache

        completion_ids = outputs.sequences[:, prompt_len:]
        return self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

    def _tokenize_teacher_scoring_prompts(self, problems, solutions, teacher_guidance_texts, student_answers):
        prompts = [
            self._apply_chat_template(
                self._build_scoring_messages(
                    problem,
                    solution,
                    teacher_guidance,
                    student_answer=student_answer,
                ),
                enable_thinking=self.teacher_scoring_thinking,
            )
            for problem, solution, teacher_guidance, student_answer in zip(
                problems,
                solutions,
                teacher_guidance_texts,
                student_answers,
            )
        ]

        encoded_no_pad = self.processing_class(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.args.max_length,
            add_special_tokens=False,
        )
        prompt_lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_prompt_len = max(prompt_lengths)

        encoded = self.processing_class(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=False,
        ).to(self.accelerator.device)

        return encoded.input_ids, encoded.attention_mask, max_prompt_len

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
            generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model,
                    inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
                generated_ids, generated_attention_mask, _ = result
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

        problems = inputs["problems"]
        solutions = inputs["solutions"]

        with self._teacher_generation_context(model):
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                teacher_guidance_texts = self._generate_teacher_guidance(
                    unwrapped_model,
                    problems,
                    solutions,
                    completion_texts,
                )

        teacher_prompts, teacher_prompt_attention_mask, teacher_prompt_len = (
            self._tokenize_teacher_scoring_prompts(
                problems,
                solutions,
                teacher_guidance_texts,
                completion_texts,
            )
        )

        student_prompt_len = inputs["student_prompt_length"]
        generation_ids = generated_ids[:, student_prompt_len:]

        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask
        inputs["teacher_prompts"] = teacher_prompts
        inputs["teacher_prompt_attention_mask"] = teacher_prompt_attention_mask
        inputs["teacher_prompt_length"] = teacher_prompt_len

        teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0

        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask

        labels = generated_ids.clone()
        for i in range(labels.shape[0]):
            actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
            labels[i, :actual_prompt_len] = -100

        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        inputs["labels"] = labels

        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        for prompt, completion, guidance in zip(prompt_texts, completion_texts, teacher_guidance_texts):
            self._generation_outputs_buffer.append(
                {
                    "step": self.state.global_step,
                    "prompt": prompt,
                    "completion": completion,
                    "teacher_guidance": guidance,
                }
            )

        if random.random() < 0.01:
            print(f"\n{'='*80}")
            print(f"STUDENT-GUIDED TEACHER SAMPLE (Step {self.state.global_step}):")
            print(f"{'='*80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nPrompt:\n{prompt_texts[sample_idx]}")
            print(f"\nStudent completion:\n{completion_texts[sample_idx]}")
            print(f"\nTeacher guidance:\n{teacher_guidance_texts[sample_idx]}")
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
