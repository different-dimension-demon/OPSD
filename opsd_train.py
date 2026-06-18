import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from opsd_strategy_registry import (
    STRATEGY_SPECS,
    apply_strategy_defaults,
    strategy_trainer_kwargs,
    strategy_wandb_config,
)


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """OPSD script arguments shared by all OPSD strategies."""

    strategy: str = field(
        default="original",
        metadata={
            "help": (
                "OPSD training strategy. Valid values: "
                + ", ".join(sorted(STRATEGY_SPECS.keys()))
            )
        },
    )
    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of full-vocab JSD."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy as a fixed teacher by disabling LoRA adapters for teacher passes."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for output directory and WandB. If omitted, a strategy-aware name is generated."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty used during on-policy generation."},
    )
    reason_first: bool = field(
        default=False,
        metadata={"help": "Generate teacher rationalization before teacher scoring. Only supported by base OPSD."},
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict JSD loss to teacher top-k tokens. Strategy topk100 variants default this to 100."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={"help": "Clip per-token JSD loss. Set to 0 for no clipping."},
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={"help": "Use an EMA copy of student weights as teacher."},
    )
    ema_decay: float = field(
        default=0.999,
        metadata={"help": "EMA decay used when use_ema_teacher=True."},
    )
    student_thinking: bool = field(
        default=False,
        metadata={"help": "Enable Qwen3 thinking mode for student rollout."},
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={"help": "Enable Qwen3 thinking mode for teacher scoring/guidance."},
    )
    privilege_placeholder_text: str = field(
        default="The privileged information is hidden.",
        metadata={"help": "Placeholder text repeated for the privilege_slot strategy."},
    )
    position_alignment_debug: bool = field(
        default=True,
        metadata={"help": "Enable strict runtime checks for position-alignment strategies."},
    )
    teacher_guidance_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum tokens for student_guided_teacher guidance generation."},
    )
    teacher_guidance_temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for student_guided_teacher guidance generation."},
    )
    teacher_guidance_top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p for student_guided_teacher guidance generation."},
    )
    teacher_guidance_top_k: int = field(
        default=20,
        metadata={"help": "Top-k for student_guided_teacher guidance generation."},
    )
    teacher_guidance_mode: str = field(
        default="critique",
        metadata={"help": "Guidance mode: critique, hint, or corrected_reasoning."},
    )
    include_student_answer_in_scoring: bool = field(
        default=False,
        metadata={
            "help": "Include the full student answer in the final teacher scoring prompt for student_guided_teacher."
        },
    )


def _resolve_model_dtype(model_args):
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
            return dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        return model_args.torch_dtype
    if hasattr(model_args, "dtype") and model_args.dtype is not None:
        return model_args.dtype
    return torch.bfloat16


def _build_run_name(script_args, training_args, model_args, spec, effective_batch_size):
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
        return full_wandb_run_config

    model_name = model_args.model_name_or_path.split("/")[-1]
    parts = [
        f"opsd_{model_name}",
        spec.run_name_suffix,
        f"lr{lr_str}",
        f"bs{effective_batch_size}",
        f"tok{training_args.max_completion_length}",
    ]
    full_wandb_run_config = "_".join(part for part in parts if part)
    if script_args.fixed_teacher:
        full_wandb_run_config += "_fixteach"
    return full_wandb_run_config


def _strategy_supports_reason_first(strategy):
    return strategy in {"original", "topk100", "topk100_drop_negative", "topk100_aopd"}


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    spec = apply_strategy_defaults(script_args)

    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. "
            "The fixed teacher is implemented by disabling LoRA adapters."
        )
    if script_args.reason_first and not _strategy_supports_reason_first(script_args.strategy):
        raise ValueError(
            f"reason_first=True is not supported by strategy={script_args.strategy!r}. "
            "Use strategy=original/topk100 or disable reason_first."
        )

    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_processes
    )
    full_wandb_run_config = _build_run_name(
        script_args,
        training_args,
        model_args,
        spec,
        effective_batch_size,
    )

    print(f"\n{'='*80}")
    print("OPSD RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"Strategy: {script_args.strategy}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Top-k loss: {script_args.top_k_loss if script_args.top_k_loss > 0 else 'full vocabulary'}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb_config = {
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
            "use_tinker_loss": script_args.use_tinker_loss,
            "fixed_teacher": script_args.fixed_teacher,
            "use_ema_teacher": script_args.use_ema_teacher,
            "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
            "student_thinking": script_args.student_thinking,
            "teacher_thinking": script_args.teacher_thinking,
        }
        wandb_config.update(strategy_wandb_config(script_args))
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config=wandb_config,
        )

    model_dtype = _resolve_model_dtype(model_args)
    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

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

    reason_first = script_args.reason_first if _strategy_supports_reason_first(script_args.strategy) else False
    data_collator = spec.data_collator_cls(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        reason_first=reason_first,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        preserve_raw_fields=spec.preserve_raw_fields,
    )

    trainer_kwargs = {
        "model": model_args.model_name_or_path,
        "args": training_args,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "processing_class": tokenizer,
        "peft_config": get_peft_config(model_args),
        "use_thinking_machines_loss": script_args.use_tinker_loss,
        "fixed_teacher": script_args.fixed_teacher,
        "reason_first": reason_first,
        "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        "jsd_token_clip": script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        "use_ema_teacher": script_args.use_ema_teacher,
        "ema_decay": script_args.ema_decay,
        "student_thinking": script_args.student_thinking,
        "teacher_thinking": script_args.teacher_thinking,
    }
    trainer_kwargs.update(strategy_trainer_kwargs(script_args))
    trainer = spec.trainer_cls(**trainer_kwargs)

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
