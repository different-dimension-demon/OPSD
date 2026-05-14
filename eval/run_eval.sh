#!/bin/bash

if [ -z "$OPSD_EVAL_BACKGROUND" ]; then
    export OPSD_EVAL_BACKGROUND=1
    nohup bash "$0" > eval_run.log 2>&1 &
    echo "Evaluation started in background. Log: $(pwd)/eval_run.log"
    exit 0
fi

BASE_MODEL="Qwen/Qwen3-1.7B"
EXP_DIR="/zju_wck/yzh/3_train/OPSD/output/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"
CUDA_DEVICES="2,3"
TENSOR_PARALLEL_SIZE=2

# evaluate base model performance
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
wait 

# after trained, evaluate the performance of the trained model. 
for step in 20 40 60 80 100; do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "aime24" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --checkpoint_dir "$EXP_DIR/checkpoint-$step"
done
