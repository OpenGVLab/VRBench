#!/bin/bash
# Eagle2.5-8B VRBench 评测脚本 (8卡 64帧)
# NVIDIA Eagle 2.5, Qwen2.5-VL 风格 API, 自己预采样帧 (遵守数据 video_read_type)
# 依赖: transformers, flash-attn, decord, av
export VLLM_LOGGING_LEVEL=ERROR
PYTHONWARNINGS="ignore"

PARTITION=${1:-""}
NUM_GPUS=${2:-8}
TOTAL_FRAMES=${3:-64}
VIDEO_ROOT=${4:-"/mnt/shared-storage-user/yujiashuo/VRBench"}
DATA_PATHS=(
  "data/VRBench_eval.jsonl"
)

MODEL="Eagle2.5-8B"
PROMPTS=("mcq")
MAX_TOKENS=4096

OUTPUT_DIR="outputs"
LOG_DIR="logs"

for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    if [ -z "$PARTITION" ]; then
      echo "[Eagle2.5-8B] $MODEL with $NUM_GPUS GPUs (all data)"
      LOG_FILE="$LOG_DIR/${MODEL}_${PROMPT}.log"
      PARTITION_ARG=""
    else
      echo "[Eagle2.5-8B] $MODEL with $NUM_GPUS GPUs (partition $PARTITION)"
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
