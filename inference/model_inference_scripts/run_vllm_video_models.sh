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

# Qwen series
QWEN_MODELS=(
  "Qwen/Qwen2-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2-VL-2B-Instruct"
  "Qwen/Qwen2-VL-72B-Instruct-AWQ"
  "Qwen/Qwen2.5-VL-3B-Instruct"
)
# InternVL series
INTERNVL_MODELS=(
  "OpenGVLab/InternVL2-8B"
  "OpenGVLab/InternVL2_5-8B"
  "OpenGVLab/InternVL2_5-38B"
  "OpenGVLab/InternVL2_5-78B-AWQ"
)
# LLaVA series
LLAVA_MODELS=(
  "llava-hf/LLaVA-NeXT-Video-7B-hf"
  "llava-hf/LLaVA-NeXT-Video-34B-hf"
  "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
)

# Phi series
PHI_MODELS=(
  "microsoft/Phi-3.5-vision-instruct"
)

# Deepseek series
DEEPSEEK_MODELS=(
  "deepseek-ai/deepseek-vl2"
  "deepseek-ai/deepseek-vl2-tiny"
  "deepseek-ai/deepseek-vl2-small"
)

# Other models
OTHER_MODELS=(
  "mistral-community/pixtral-12b"
  "unsloth/Llama-3.2-11B-Vision-Instruct"
  "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit"
  "h2oai/h2ovl-mississippi-2b"
  "nvidia/NVLM-D-72B"
  "HuggingFaceM4/Idefics3-8B-Llama3"
  "rhymes-ai/Aria-Chat"
  "moonshotai/Kimi-VL-A3B-Thinking-2506"
  "MiMo-VL-7B-RL"
)

PROMPTS=("mcq")

for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    for MODEL in "${QWEN_MODELS[@]}"; do
      echo "[Qwen] $MODEL"
      python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT"
    done
    for MODEL in "${INTERNVL_MODELS[@]}"; do
      echo "[InternVL] $MODEL"
      python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT"
    done
    for MODEL in "${LLAVA_MODELS[@]}"; do
      echo "[LLaVA] $MODEL"
      python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT"
    done
    for MODEL in "${PHI_MODELS[@]}"; do
      echo "[Phi] $MODEL"
      python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT"
    done
    for MODEL in "${DEEPSEEK_MODELS[@]}"; do
      echo "[Deepseek] $MODEL"
      python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT"
    done
    for MODEL in "${OTHER_MODELS[@]}"; do
      echo "[Other] $MODEL"
      python3 main.py --model "$MODEL" --prompt "$PROMPT" --total_frames "$TOTAL_FRAMES" --data_path "$DATA_PATH" --video_root "$VIDEO_ROOT"
    done
  done
done