#!/bin/bash

PROJECT_DIR="/zju_wck/yzh/3_train/OPSD"
DEFAULT_EXP_DIR="$PROJECT_DIR/output/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"
EXP_DIR="${1:-$DEFAULT_EXP_DIR}"
RUN_NAME="${2:-$(basename "$EXP_DIR")}"
LOG_FILE="$PROJECT_DIR/eval_${RUN_NAME}.log"
EVAL_RESULTS_DIR="$EXP_DIR/eval_results"

if [ -z "$OPSD_EVAL_BACKGROUND" ]; then
    export OPSD_EVAL_BACKGROUND=1
    nohup bash "$0" "$EXP_DIR" "$RUN_NAME" > "$LOG_FILE" 2>&1 &
    echo "Evaluation started in background. Log: $LOG_FILE"
    exit 0
fi

BASE_MODEL="Qwen/Qwen3-1.7B"
CUDA_DEVICES="2,3"
TENSOR_PARALLEL_SIZE=2

echo "Evaluating experiment: $RUN_NAME"
echo "Experiment dir: $EXP_DIR"
echo "Log file: $LOG_FILE"
echo "Eval results dir: $EVAL_RESULTS_DIR"
mkdir -p "$EVAL_RESULTS_DIR"

# evaluate base model performance
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --output_file "$EVAL_RESULTS_DIR/base_aime24_thinking_temp1.0_valn12.json"
wait 

# after trained, evaluate the performance of the trained model. 
for step in 20 40 60 80 100; do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "aime24" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --checkpoint_dir "$EXP_DIR/checkpoint-$step" \
        --output_file "$EVAL_RESULTS_DIR/checkpoint-${step}_aime24_thinking_temp1.0_valn12.json"
done
