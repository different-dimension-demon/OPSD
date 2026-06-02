#!/bin/bash
set -e

cd /zju_wck/yzh/3_train/OPSD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"
RUN_CONFIG="${RUN_CONFIG:-qwen38b_gen1024_fixteacher_temp11_privilege_slot}"
LOG_FILE="${LOG_FILE:-opsd_8b_privilege_slot_train.log}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12964}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

nohup accelerate launch \
    --config_file scripts/configs/accelerate_2gpu_no_offload_auto.yaml \
    --num_processes "$NUM_PROCESSES" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    opsd_train_privilege_slot.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --output_dir /zju_wck/yzh/3_train/OPSD/output/ \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 1 \
    --max_steps 100 \
    --max_completion_length 1024 \
    --save_steps 20 \
    --logging_steps 1 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --student_thinking false \
    --teacher_thinking true \
    --jsd_token_clip 0.05 \
    --privilege_placeholder_text "The privileged information is hidden." \
    --position_alignment_debug true \
    --wandb_project OPSD \
    > "$LOG_FILE" 2>&1 &

echo "Started privilege-slot OPSD 8B training."
echo "Run config: $RUN_CONFIG"
echo "Model: $MODEL_NAME_OR_PATH"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Num processes: $NUM_PROCESSES"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Log: /zju_wck/yzh/3_train/OPSD/$LOG_FILE"
