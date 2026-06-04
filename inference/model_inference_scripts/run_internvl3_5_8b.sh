#!/bin/bash
# InternVL3.5-8B VRBench 评测脚本
export VLLM_LOGGING_LEVEL=ERROR
PYTHONWARNINGS="ignore"

PARTITION=${1:-""}
NUM_GPUS=${2:-8}
TOTAL_FRAMES=${3:-64}
VIDEO_ROOT=${4:-"/mnt/shared-storage-user/yujiashuo/VRBench"}
DATA_PATHS=(
  "data/VRBench_eval.jsonl"
)

MODEL="InternVL3_5-8B"
PROMPTS=("mcq")
MAX_TOKENS=4096

OUTPUT_DIR="outputs"
LOG_DIR="logs"

for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    if [ -z "$PARTITION" ]; then
      echo "[InternVL3_5-8B] $MODEL with $NUM_GPUS GPUs (all data)"
      LOG_FILE="$LOG_DIR/${MODEL}_${PROMPT}.log"
      PARTITION_ARG=""
    else
      echo "[InternVL3_5-8B] $MODEL with $NUM_GPUS GPUs (partition $PARTITION)"
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
