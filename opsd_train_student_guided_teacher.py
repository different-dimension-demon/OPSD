import os
from dataclasses import dataclass, field

import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from data_collator_student_guided_teacher import StudentGuidedTeacherDataCollator
from opsd_train import CustomScriptArguments
from opsd_trainer_student_guided_teacher import StudentGuidedTeacherOPSDTrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class StudentGuidedTeacherScriptArguments(CustomScriptArguments):
    teacher_guidance_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum tokens for teacher critique/guidance generation."},
    )
    teacher_guidance_temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for teacher critique/guidance generation."},
    )
    teacher_guidance_top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p for teacher critique/guidance generation."},
    )
    teacher_guidance_top_k: int = field(
        default=20,
        metadata={"help": "Top-k for teacher critique/guidance generation."},
    )
    teacher_guidance_mode: str = field(
        default="critique",
        metadata={"help": "Teacher guidance style: critique, hint, or corrected_reasoning."},
    )
    include_student_answer_in_scoring: bool = field(
        default=False,
        metadata={
            "help": "If true, include the full student answer in the final teacher scoring prompt. "
            "Default false to avoid token-level future-token leakage."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((StudentGuidedTeacherScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_processes
    )

    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_config = (
            f"student_guided_opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    print(f"\n{'='*80}")
    print("STUDENT-GUIDED TEACHER OPSD RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Teacher guidance mode: {script_args.teacher_guidance_mode}")
    print(f"Include student answer in scoring: {script_args.include_student_answer_in_scoring}")
    print(f"{'='*80}\n")

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. "
            "The fixed teacher is implemented by disabling LoRA adapters."
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "student_guided_teacher": True,
                "teacher_guidance_max_new_tokens": script_args.teacher_guidance_max_new_tokens,
                "teacher_guidance_temperature": script_args.teacher_guidance_temperature,
                "teacher_guidance_top_p": script_args.teacher_guidance_top_p,
                "teacher_guidance_top_k": script_args.teacher_guidance_top_k,
                "teacher_guidance_mode": script_args.teacher_guidance_mode,
                "include_student_answer_in_scoring": script_args.include_student_answer_in_scoring,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
            },
        )

    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs
    training_args.presence_penalty = script_args.presence_penalty

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
    train_dataset = dataset["train"]

    data_collator = StudentGuidedTeacherDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        reason_first=False,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
    )

    trainer = StudentGuidedTeacherOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=False,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        teacher_guidance_max_new_tokens=script_args.teacher_guidance_max_new_tokens,
        teacher_guidance_temperature=script_args.teacher_guidance_temperature,
        teacher_guidance_top_p=script_args.teacher_guidance_top_p,
        teacher_guidance_top_k=script_args.teacher_guidance_top_k,
        teacher_guidance_mode=script_args.teacher_guidance_mode,
        teacher_guidance_thinking=script_args.teacher_thinking,
        teacher_scoring_thinking=script_args.teacher_thinking,
        include_student_answer_in_scoring=script_args.include_student_answer_in_scoring,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)
