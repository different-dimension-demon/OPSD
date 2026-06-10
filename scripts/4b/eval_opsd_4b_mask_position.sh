#!/bin/bash
set -e

PROJECT_DIR="${PROJECT_DIR:-/zju_wck/yzh/3_train/OPSD}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
EXP_DIR="${EXP_DIR:-$PROJECT_DIR/output/qwen34b_gen1024_fixteacher_temp11_mask_position}"
RUN_NAME="${RUN_NAME:-opsd_4b_mask_position}"

export CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

bash "$PROJECT_DIR/eval/run_eval_experiment.sh" "$BASE_MODEL" "$EXP_DIR" "$RUN_NAME"
