export CUDA_VISIBLE_DEVICES=2,3
export TORCH_CUDA_ARCH_LIST="8.0"

nohup accelerate launch \
    --config_file scripts/configs/accelerate_2gpu_ga16.yaml \
    --num_processes 2 \
    --main_process_port 19346 \
    grpo_train.py \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --output_dir /zju_wck/yzh/3_train/OPSD/output/ \
    --run_config qwen31b_grpo_gen1024_temp12 \
    --num_train_epochs 1 \
    --max_steps 100 \
    --num_iterations 2 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --logging_steps 1 \
    --save_steps 20 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --wandb_project OPSD \
    > grpo_1b_train.log 2>&1 &
