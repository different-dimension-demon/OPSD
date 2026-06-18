from dataclasses import dataclass

from data_collator import SelfDistillationDataCollator
from opsd_trainer import OPSDTrainer
from opsd_trainer_position_alignment import PositionAlignmentOPSDTrainer
from opsd_trainer_privilege_slot import PrivilegeSlotOPSDTrainer
from opsd_trainer_student_guided_teacher import StudentGuidedTeacherOPSDTrainer
from opsd_trainer_topk_variants import (
    TopKAOPDNonPositiveOPSDTrainer,
    TopKDropNegativePositionOPSDTrainer,
)


@dataclass(frozen=True)
class OPSDStrategySpec:
    name: str
    trainer_cls: type
    data_collator_cls: type = SelfDistillationDataCollator
    preserve_raw_fields: bool = False
    run_name_suffix: str = ""
    default_top_k_loss: int | None = None


STRATEGY_SPECS = {
    "original": OPSDStrategySpec(
        name="original",
        trainer_cls=OPSDTrainer,
        run_name_suffix="",
    ),
    "topk100": OPSDStrategySpec(
        name="topk100",
        trainer_cls=OPSDTrainer,
        run_name_suffix="topk100",
        default_top_k_loss=100,
    ),
    "topk100_drop_negative": OPSDStrategySpec(
        name="topk100_drop_negative",
        trainer_cls=TopKDropNegativePositionOPSDTrainer,
        run_name_suffix="topk100_dropneg",
        default_top_k_loss=100,
    ),
    "topk100_aopd": OPSDStrategySpec(
        name="topk100_aopd",
        trainer_cls=TopKAOPDNonPositiveOPSDTrainer,
        run_name_suffix="topk100_aopd_nonpos",
        default_top_k_loss=100,
    ),
    "privilege_slot": OPSDStrategySpec(
        name="privilege_slot",
        trainer_cls=PrivilegeSlotOPSDTrainer,
        preserve_raw_fields=True,
        run_name_suffix="privilege_slot",
    ),
    "mask_position": OPSDStrategySpec(
        name="mask_position",
        trainer_cls=PositionAlignmentOPSDTrainer,
        preserve_raw_fields=True,
        run_name_suffix="mask_position",
    ),
    "student_align_teacher": OPSDStrategySpec(
        name="student_align_teacher",
        trainer_cls=PositionAlignmentOPSDTrainer,
        preserve_raw_fields=True,
        run_name_suffix="student_align_teacher",
    ),
    "student_guided_teacher": OPSDStrategySpec(
        name="student_guided_teacher",
        trainer_cls=StudentGuidedTeacherOPSDTrainer,
        preserve_raw_fields=True,
        run_name_suffix="student_guided_teacher",
    ),
}


def get_strategy_spec(strategy: str) -> OPSDStrategySpec:
    try:
        return STRATEGY_SPECS[strategy]
    except KeyError as exc:
        valid = ", ".join(sorted(STRATEGY_SPECS))
        raise ValueError(f"Unknown OPSD strategy {strategy!r}. Expected one of: {valid}") from exc


def apply_strategy_defaults(script_args):
    spec = get_strategy_spec(script_args.strategy)
    if spec.default_top_k_loss is not None and script_args.top_k_loss <= 0:
        script_args.top_k_loss = spec.default_top_k_loss
    return spec


def strategy_trainer_kwargs(script_args):
    strategy = script_args.strategy
    if strategy == "privilege_slot":
        return {
            "privilege_placeholder_text": script_args.privilege_placeholder_text,
            "position_alignment_debug": script_args.position_alignment_debug,
        }
    if strategy in {"mask_position", "student_align_teacher"}:
        return {
            "position_alignment_mode": strategy,
            "position_alignment_debug": script_args.position_alignment_debug,
        }
    if strategy == "student_guided_teacher":
        return {
            "teacher_guidance_max_new_tokens": script_args.teacher_guidance_max_new_tokens,
            "teacher_guidance_temperature": script_args.teacher_guidance_temperature,
            "teacher_guidance_top_p": script_args.teacher_guidance_top_p,
            "teacher_guidance_top_k": script_args.teacher_guidance_top_k,
            "teacher_guidance_mode": script_args.teacher_guidance_mode,
            "teacher_guidance_thinking": script_args.teacher_thinking,
            "teacher_scoring_thinking": script_args.teacher_thinking,
            "include_student_answer_in_scoring": script_args.include_student_answer_in_scoring,
        }
    return {}


def strategy_wandb_config(script_args):
    config = {
        "strategy": script_args.strategy,
        "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        "position_alignment_debug": script_args.position_alignment_debug,
        "privilege_placeholder_text": None,
        "teacher_guidance_mode": None,
    }
    if script_args.strategy == "privilege_slot":
        config["privilege_placeholder_text"] = script_args.privilege_placeholder_text
    if script_args.strategy == "student_guided_teacher":
        config.update(
            {
                "teacher_guidance_max_new_tokens": script_args.teacher_guidance_max_new_tokens,
                "teacher_guidance_temperature": script_args.teacher_guidance_temperature,
                "teacher_guidance_top_p": script_args.teacher_guidance_top_p,
                "teacher_guidance_top_k": script_args.teacher_guidance_top_k,
                "teacher_guidance_mode": script_args.teacher_guidance_mode,
                "include_student_answer_in_scoring": script_args.include_student_answer_in_scoring,
            }
        )
    return config
