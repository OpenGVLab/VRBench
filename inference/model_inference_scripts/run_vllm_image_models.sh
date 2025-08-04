#!/bin/bash

export VLLM_CONFIGURE_LOGGING=0
export VLLM_LOGGING_LEVEL=ERROR
export PYTHONWARNINGS="ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false

TOTAL_FRAMES=${1:--1}
VIDEO_ROOT=${2:-"/path/to/video/root"}  # Default video root path
DATA_PATHS=(
  "data/VRBench_eval.jsonl"
)

# InternVL series
INTERNVL_MODELS=(
  "OpenGVLab/InternVL2_5-8B"
)
# (Add more image models here if needed)

PROMPTS=("mcq")
for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    for MODEL in "${INTERNVL_MODELS[@]}"; do
      echo "[InternVL] $MODEL"
      python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT"
    done
  done
done
