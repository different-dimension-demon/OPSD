export CUDA_VISIBLE_DEVICES=2,3
export TORCH_CUDA_ARCH_LIST="8.0"

nohup accelerate launch \
    --config_file scripts/configs/accelerate_2gpu.yaml \
    --num_processes 2 \
    --main_process_port 19347 \
    sft_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir /zju_wck/yzh/3_train/OPSD/output/qwen31b_sft \
    --num_train_epochs 1 \
    --max_steps 100 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 20000 \
    --logging_steps 1 \
    --save_steps 20 \
    > sft_1b_train.log 2>&1 &
