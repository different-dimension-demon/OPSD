#!/bin/bash
set -e

PROJECT_DIR="${PROJECT_DIR:-/zju_wck/yzh/3_train/OPSD}"

if [ "$#" -lt 3 ]; then
    echo "Usage: bash $0 <BASE_MODEL> <EXP_DIR> <RUN_NAME>"
    exit 1
fi

BASE_MODEL="$1"
EXP_DIR="$2"
RUN_NAME="$3"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/eval_${RUN_NAME}.log}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-$EXP_DIR/eval_results}"

if [ -z "$OPSD_EVAL_BACKGROUND" ]; then
    export OPSD_EVAL_BACKGROUND=1
    nohup bash "$0" "$BASE_MODEL" "$EXP_DIR" "$RUN_NAME" > "$LOG_FILE" 2>&1 &
    echo "Evaluation started in background."
    echo "Log: $LOG_FILE"
    echo "Eval results dir: $EVAL_RESULTS_DIR"
    exit 0
fi

CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

find_free_port() {
    python - <<'PY'
import socket

HOST = "127.0.0.1"
PORT_BLOCK_SIZE = 8
MIN_PORT = 20000
MAX_PORT = 60000


def reserve_port_block(start_port):
    sockets = []
    try:
        for port in range(start_port, start_port + PORT_BLOCK_SIZE):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((HOST, port))
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as seed:
    seed.bind((HOST, 0))
    candidate = seed.getsockname()[1]

if MIN_PORT <= candidate <= MAX_PORT - PORT_BLOCK_SIZE and reserve_port_block(candidate):
    print(candidate)
else:
    for candidate in range(MIN_PORT, MAX_PORT - PORT_BLOCK_SIZE + 1):
        if reserve_port_block(candidate):
            print(candidate)
            break
    else:
        raise RuntimeError(f"Could not find {PORT_BLOCK_SIZE} consecutive free TCP ports")
PY
}

run_eval_if_missing() {
    local output_file="$1"
    shift

    if [ -s "$output_file" ]; then
        echo "Skipping existing result: $output_file"
        return 0
    fi

    local master_port
    master_port="$(find_free_port)"
    echo "MASTER_PORT: $master_port"
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$master_port" \
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$@"
}

cd "$PROJECT_DIR/eval"
mkdir -p "$EVAL_RESULTS_DIR"

echo "Evaluating experiment: $RUN_NAME"
echo "Experiment dir: $EXP_DIR"
echo "Base model: $BASE_MODEL"
echo "CUDA devices: $CUDA_DEVICES"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "Log file: $LOG_FILE"
echo "Eval results dir: $EVAL_RESULTS_DIR"

echo "Evaluating base model..."
BASE_OUTPUT_FILE="$EVAL_RESULTS_DIR/base_aime24_thinking_temp1.0_valn12.json"
run_eval_if_missing "$BASE_OUTPUT_FILE" python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --output_file "$BASE_OUTPUT_FILE"

for step in 20 40 60 80 100; do
    echo "Evaluating checkpoint-$step..."
    CHECKPOINT_OUTPUT_FILE="$EVAL_RESULTS_DIR/checkpoint-${step}_aime24_thinking_temp1.0_valn12.json"
    run_eval_if_missing "$CHECKPOINT_OUTPUT_FILE" python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "aime24" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --checkpoint_dir "$EXP_DIR/checkpoint-$step" \
        --output_file "$CHECKPOINT_OUTPUT_FILE"
done

echo "Evaluation finished."
echo "Log file: $LOG_FILE"
echo "Eval results dir: $EVAL_RESULTS_DIR"
