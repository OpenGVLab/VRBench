#!/bin/bash
# GLM-4.1V-9B-Thinking VRBench 评测脚本 (8卡 64帧)
# 推理模型, MAX_TOKENS 调大以容纳 thinking 过程
# 依赖: transformers>=4.57 (原生 Glm4v), decord
export VLLM_LOGGING_LEVEL=ERROR
PYTHONWARNINGS="ignore"

PARTITION=${1:-""}
NUM_GPUS=${2:-8}
TOTAL_FRAMES=${3:-64}
VIDEO_ROOT=${4:-"/mnt/shared-storage-user/yujiashuo/VRBench"}
DATA_PATHS=(
  "data/VRBench_eval.jsonl"
)

MODEL="GLM-4.1V-9B-Thinking"
PROMPTS=("mcq")
MAX_TOKENS=8192

OUTPUT_DIR="outputs"
LOG_DIR="logs"

for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    if [ -z "$PARTITION" ]; then
      echo "[GLM-4.1V-9B-Thinking] $MODEL with $NUM_GPUS GPUs (all data)"
      LOG_FILE="$LOG_DIR/${MODEL}_${PROMPT}.log"
      PARTITION_ARG=""
    else
      echo "[GLM-4.1V-9B-Thinking] $MODEL with $NUM_GPUS GPUs (partition $PARTITION)"
      LOG_FILE="$LOG_DIR/${MODEL}_${PROMPT}_partition${PARTITION}.log"
      PARTITION_ARG="--partition $PARTITION"
    fi
    mkdir -p "$LOG_DIR"
    python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" \
      --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT" \
      --output_dir "$OUTPUT_DIR" --num_gpus "$NUM_GPUS" --max_tokens "$MAX_TOKENS" \
      $PARTITION_ARG > "$LOG_FILE" 2>&1
  done
done
