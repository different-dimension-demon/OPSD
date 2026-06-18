#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/zju_wck/yzh/3_train/OPSD}"
cd "$PROJECT_DIR"

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/run_opsd.sh <1b|4b|8b> <strategy> [extra opsd_train.py args...]"
    echo "Strategies: original topk100 topk100_drop_negative topk100_aopd privilege_slot mask_position student_align_teacher student_guided_teacher"
    echo "Example: bash scripts/run_opsd.sh 4b student_align_teacher"
    echo "Dry run: DRY_RUN=1 bash scripts/run_opsd.sh 8b original"
    exit 1
fi

MODEL_SIZE="$1"
STRATEGY="$2"
shift 2

case "$MODEL_SIZE" in
    1b)
        MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}"
        RUN_PREFIX="${RUN_PREFIX:-qwen31b}"
        ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/configs/accelerate_2gpu.yaml}"
        PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
        GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
        PORT_BASE=12950
        ;;
    4b)
        MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
        RUN_PREFIX="${RUN_PREFIX:-qwen34b}"
        ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/configs/accelerate_2gpu_no_offload_auto.yaml}"
        PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
        GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
        PORT_BASE=12960
        ;;
    8b)
        MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"
        RUN_PREFIX="${RUN_PREFIX:-qwen38b}"
        ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/configs/accelerate_2gpu_no_offload_auto.yaml}"
        PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
        GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
        PORT_BASE=12970
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Expected one of: 1b 4b 8b"
        exit 1
        ;;
esac

case "$STRATEGY" in
    original)
        STRATEGY_INDEX=0
        RUN_SUFFIX="gen1024_fixteacher_temp11_forwardbeta0_clip005"
        DEFAULT_JSD_TOKEN_CLIP="0.05"
        ;;
    topk100)
        STRATEGY_INDEX=1
        RUN_SUFFIX="gen1024_fixteacher_temp11_forwardbeta0_clip005_topk100"
        DEFAULT_JSD_TOKEN_CLIP="0.05"
        ;;
    topk100_drop_negative)
        STRATEGY_INDEX=2
        RUN_SUFFIX="gen1024_fixteacher_temp11_topk100_dropneg"
        DEFAULT_JSD_TOKEN_CLIP="0.05"
        ;;
    topk100_aopd)
        STRATEGY_INDEX=3
        RUN_SUFFIX="gen1024_fixteacher_temp11_topk100_aopd_nonpos"
        DEFAULT_JSD_TOKEN_CLIP="0"
        export OPSD_AOPD_LOSS_TEMPERATURE="${OPSD_AOPD_LOSS_TEMPERATURE:-1.0}"
        ;;
    privilege_slot)
        STRATEGY_INDEX=4
        RUN_SUFFIX="gen1024_fixteacher_temp11_privilege_slot"
        DEFAULT_JSD_TOKEN_CLIP="0.05"
        ;;
    mask_position)
        STRATEGY_INDEX=5
        RUN_SUFFIX="gen1024_fixteacher_temp11_mask_position"
        DEFAULT_JSD_TOKEN_CLIP="0.05"
        ;;
    student_align_teacher)
        STRATEGY_INDEX=6
        RUN_SUFFIX="gen1024_fixteacher_temp11_student_align_teacher"
        DEFAULT_JSD_TOKEN_CLIP="0.05"
        ;;
    student_guided_teacher)
        STRATEGY_INDEX=7
        RUN_SUFFIX="gen1024_fixteacher_temp11_student_guided_teacher"
        DEFAULT_JSD_TOKEN_CLIP="0.05"
        ;;
    *)
        echo "Unknown strategy: $STRATEGY"
        echo "Expected one of: original topk100 topk100_drop_negative topk100_aopd privilege_slot mask_position student_align_teacher student_guided_teacher"
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((PORT_BASE + STRATEGY_INDEX))}"
RUN_CONFIG="${RUN_CONFIG:-${RUN_PREFIX}_${RUN_SUFFIX}}"
LOG_FILE="${LOG_FILE:-opsd_${MODEL_SIZE}_${STRATEGY}_train.log}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-$DEFAULT_JSD_TOKEN_CLIP}"
STUDENT_THINKING="${STUDENT_THINKING:-false}"
TEACHER_THINKING="${TEACHER_THINKING:-true}"

COMMAND=(
    accelerate launch
    --config_file "$ACCELERATE_CONFIG"
    --num_processes "$NUM_PROCESSES"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --main_process_port "$MAIN_PROCESS_PORT"
    opsd_train.py
    --strategy "$STRATEGY"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --learning_rate "${LEARNING_RATE:-5e-6}"
    --max_grad_norm "${MAX_GRAD_NORM:-0.1}"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_checkpointing
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --output_dir "${OUTPUT_DIR:-/zju_wck/yzh/3_train/OPSD/output/}"
    --run_config "$RUN_CONFIG"
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
    --max_steps "${MAX_STEPS:-100}"
    --max_completion_length "${MAX_COMPLETION_LENGTH:-1024}"
    --save_steps "${SAVE_STEPS:-20}"
    --logging_steps "${LOGGING_STEPS:-1}"
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
    --dtype "${DTYPE:-bfloat16}"
    --max_length "${MAX_LENGTH:-20000}"
    --beta "${BETA:-0}"
    --use_vllm
    --vllm_mode "${VLLM_MODE:-colocate}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --use_peft
    --lora_r "${LORA_R:-64}"
    --lora_alpha "${LORA_ALPHA:-128}"
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --temperature "${TEMPERATURE:-1.1}"
    --top_p "${TOP_P:-0.95}"
    --top_k "${TOP_K:-20}"
    --lmbda "${LMBDA:-1}"
    --fixed_teacher
    --student_thinking "$STUDENT_THINKING"
    --teacher_thinking "$TEACHER_THINKING"
    --jsd_token_clip "$JSD_TOKEN_CLIP"
    --wandb_project "${WANDB_PROJECT:-OPSD}"
)

case "$STRATEGY" in
    topk100|topk100_drop_negative|topk100_aopd)
        COMMAND+=(--top_k_loss "${TOP_K_LOSS:-100}")
        ;;
esac

if [ "$STRATEGY" = "privilege_slot" ]; then
    COMMAND+=(--privilege_placeholder_text "${PRIVILEGE_PLACEHOLDER_TEXT:-The privileged information is hidden.}")
fi

if [ "$STRATEGY" = "student_guided_teacher" ]; then
    COMMAND+=(
        --teacher_guidance_max_new_tokens "${TEACHER_GUIDANCE_MAX_NEW_TOKENS:-512}"
        --teacher_guidance_temperature "${TEACHER_GUIDANCE_TEMPERATURE:-0.7}"
        --teacher_guidance_top_p "${TEACHER_GUIDANCE_TOP_P:-0.95}"
        --teacher_guidance_top_k "${TEACHER_GUIDANCE_TOP_K:-20}"
        --teacher_guidance_mode "${TEACHER_GUIDANCE_MODE:-critique}"
    )
    if [ "${INCLUDE_STUDENT_ANSWER_IN_SCORING:-0}" = "1" ]; then
        COMMAND+=(--include_student_answer_in_scoring)
    fi
fi

if [ "$#" -gt 0 ]; then
    COMMAND+=("$@")
fi

echo "OPSD training launch"
echo "Model size: $MODEL_SIZE"
echo "Strategy: $STRATEGY"
echo "Run config: $RUN_CONFIG"
echo "Model: $MODEL_NAME_OR_PATH"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Num processes: $NUM_PROCESSES"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Per-device batch size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Main process port: $MAIN_PROCESS_PORT"
echo "Log: $PROJECT_DIR/$LOG_FILE"

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 &
echo "Started OPSD training in background."
