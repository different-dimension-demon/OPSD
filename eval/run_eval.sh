#!/bin/bash
set -e

PROJECT_DIR="/zju_wck/yzh/3_train/OPSD"

# Default one-click target. Override with:
#   bash run_eval.sh <EXP_DIR> <RUN_NAME>
DEFAULT_EXP_DIR="$PROJECT_DIR/output/qwen31b_sft"
EXP_DIR="${1:-$DEFAULT_EXP_DIR}"
RUN_NAME="${2:-sft_1b}"
LOG_FILE="$PROJECT_DIR/eval_${RUN_NAME}.log"
EVAL_RESULTS_DIR="$EXP_DIR/eval_results"

if [ -z "$OPSD_EVAL_BACKGROUND" ]; then
    export OPSD_EVAL_BACKGROUND=1
    nohup bash "$0" "$EXP_DIR" "$RUN_NAME" > "$LOG_FILE" 2>&1 &
    echo "Evaluation started in background."
    echo "Log: $LOG_FILE"
    echo "Eval results dir: $EVAL_RESULTS_DIR"
    exit 0
fi

# Fixed local/offline evaluation environment.
BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CUDA_DEVICES="0,1"
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9
MASTER_ADDR="127.0.0.1"
MASTER_PORT_BASE=29600

export HF_HOME="~/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Evaluating experiment: $RUN_NAME"
echo "Experiment dir: $EXP_DIR"
echo "Base model: $BASE_MODEL"
echo "CUDA devices: $CUDA_DEVICES"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT_BASE: $MASTER_PORT_BASE"
echo "Log file: $LOG_FILE"
echo "Eval results dir: $EVAL_RESULTS_DIR"

mkdir -p "$EVAL_RESULTS_DIR"

echo "Evaluating base model..."
MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT_BASE" \
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --output_file "$EVAL_RESULTS_DIR/base_aime24_thinking_temp1.0_valn12.json"

for step in 20 40 60 80 100; do
    echo "Evaluating checkpoint-$step..."
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=$((MASTER_PORT_BASE + step)) \
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "aime24" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --checkpoint_dir "$EXP_DIR/checkpoint-$step" \
        --output_file "$EVAL_RESULTS_DIR/checkpoint-${step}_aime24_thinking_temp1.0_valn12.json"
done

echo "Evaluation finished."
echo "Log file: $LOG_FILE"
echo "Eval results dir: $EVAL_RESULTS_DIR"
