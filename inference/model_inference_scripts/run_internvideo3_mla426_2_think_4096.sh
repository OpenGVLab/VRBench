#!/bin/bash
# InternVideo3 MLA426_2 think + 简洁 prompt, max_tokens=4096
export VLLM_LOGGING_LEVEL=ERROR
PYTHONWARNINGS="ignore"

PARTITION=${1:-""}
NUM_GPUS=${2:-8}
TOTAL_FRAMES=${3:-1024}
VIDEO_ROOT=${4:-"/mnt/shared-storage-user/yujiashuo/VRBench"}
DATA_PATHS=(
  "data/VRBench_eval.jsonl"
)

MODEL="InternVideo3_mla426_2_think"
PROMPTS=("mcq")
MAX_TOKENS=4096

OUTPUT_DIR="outputs"
LOG_DIR="logs"

# 清除旧的输出文件（含 gpu shard 和 merged）
echo "Cleaning old output files..."
rm -f outputs/InternVideo3_mla426_2_think_1024frame*.jsonl

for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    if [ -z "$PARTITION" ]; then
      echo "[InternVideo3_mla426_2_think 4096tok] $MODEL with $NUM_GPUS GPUs (all data)"
      LOG_FILE="$LOG_DIR/${MODEL}_${PROMPT}_4096tok.log"
      PARTITION_ARG=""
    else
      echo "[InternVideo3_mla426_2_think 4096tok] $MODEL with $NUM_GPUS GPUs (partition $PARTITION)"
      LOG_FILE="$LOG_DIR/${MODEL}_${PROMPT}_4096tok_partition${PARTITION}.log"
      PARTITION_ARG="--partition $PARTITION"
    fi
    mkdir -p "$LOG_DIR"
    python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" \
      --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT" \
      --output_dir "$OUTPUT_DIR" --num_gpus "$NUM_GPUS" --max_tokens "$MAX_TOKENS" \
      $PARTITION_ARG > "$LOG_FILE" 2>&1
  done
done
