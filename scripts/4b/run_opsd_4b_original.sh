#!/bin/bash
set -e

cd /zju_wck/yzh/3_train/OPSD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
RUN_CONFIG="${RUN_CONFIG:-qwen34b_gen1024_fixteacher_temp11_forwardbeta0_clip005}"
LOG_FILE="${LOG_FILE:-opsd_4b_original_train.log}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12961}"

nohup accelerate launch \
    --config_file scripts/configs/accelerate.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 1 \
    --main_process_port "$MAIN_PROCESS_PORT" \
    opsd_train.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --output_dir /zju_wck/yzh/3_train/OPSD/output/ \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
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
    --wandb_project OPSD \
    > "$LOG_FILE" 2>&1 &

echo "Started original OPSD 4B training."
echo "Run config: $RUN_CONFIG"
echo "Model: $MODEL_NAME_OR_PATH"
echo "Log: /zju_wck/yzh/3_train/OPSD/$LOG_FILE"
