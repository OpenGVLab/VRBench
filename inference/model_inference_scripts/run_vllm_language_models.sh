#!/bin/bash

export VLLM_CONFIGURE_LOGGING=0
export VLLM_LOGGING_LEVEL=ERROR
export PYTHONWARNINGS="ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false

TOTAL_FRAMES=${1:--1}
DATA_PATHS=(
  "data/VRBench_eval_debug.jsonl"
)

# Qwen series
QWEN_MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
)
# Llama series
LLAMA_MODELS=(
  "meta-llama/Llama-3.3-70B-Instruct"
)

PROMPTS=("mcq")
for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    for MODEL in "${QWEN_MODELS[@]}"; do
      echo "[Qwen] $MODEL"
      python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" --data_path "$DATA_PATH"
    done
    for MODEL in "${LLAMA_MODELS[@]}"; do
      echo "[Llama] $MODEL"
      python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" --data_path "$DATA_PATH"
    done
  done
done