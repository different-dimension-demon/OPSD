#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2,3
export TORCH_CUDA_ARCH_LIST="8.0"

nohup accelerate launch \
    --config_file scripts/configs/accelerate_2gpu.yaml \
    --num_processes 2 \
    --main_process_port 12953 \
    opsd_train_student_guided_teacher.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --output_dir /zju_wck/yzh/3_train/OPSD/output/ \
    --run_config qwen31b_gen1024_fixteacher_temp11_student_guided_teacher \
    --num_train_epochs 1 \
    --max_steps 100 \
    --max_completion_length 1024 \
    --teacher_guidance_max_new_tokens 512 \
    --teacher_guidance_temperature 0.7 \
    --teacher_guidance_top_p 0.95 \
    --teacher_guidance_top_k 20 \
    --teacher_guidance_mode critique \
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
    --wandb_project OPSD \
    > opsd_1b_student_guided_teacher_train.log 2>&1 &
