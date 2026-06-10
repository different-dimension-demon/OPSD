#!/bin/bash
set -e

cd /zju_wck/yzh/3_train/OPSD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

nohup accelerate launch \
    --config_file scripts/configs/accelerate_2gpu.yaml \
    --num_processes 2 \
    --main_process_port "${MAIN_PROCESS_PORT:-12958}" \
    opsd_train_mask_position.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --output_dir /zju_wck/yzh/3_train/OPSD/output/ \
    --run_config qwen31b_gen1024_fixteacher_temp11_mask_position \
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
    --jsd_token_clip 0.05 \
    --student_thinking false \
    --teacher_thinking true \
    --position_alignment_debug true \
    --wandb_project OPSD \
    > opsd_1b_mask_position_train.log 2>&1 &

echo "Started mask-position OPSD 1B training."
echo "Log: /zju_wck/yzh/3_train/OPSD/opsd_1b_mask_position_train.log"
