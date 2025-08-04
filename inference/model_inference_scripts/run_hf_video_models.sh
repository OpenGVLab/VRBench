#!/bin/bash
export VLLM_LOGGING_LEVEL=ERROR
PYTHONWARNINGS="ignore"

TOTAL_FRAMES=${1:--1}
VIDEO_ROOT=${2:-"/path/to/video/root"}  # Default video root path
DATA_PATHS=(
  "data/VRBench_eval.jsonl"
)

# Keye series
KEYE_MODELS=(
  "Kwai-Keye/Keye-VL-8B-Preview"
)

# (Add more HF models here if needed)


PROMPTS=("mcq")

for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    for MODEL in "${KEYE_MODELS[@]}"; do
      echo "[KEYE] $MODEL"
        mkdir -p "$LOG_DIR"
        python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" \
          --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT" --n "$THREAD" > "$LOG_FILE" 2>&1 &
    done
  done
done

