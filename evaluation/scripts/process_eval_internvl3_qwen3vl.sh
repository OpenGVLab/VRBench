#!/bin/bash
# 过程评分: InternVL3.5-8B / Qwen3-VL-8B / Qwen3-VL-4B
cd /mnt/shared-storage-user/yujiashuo/vrbench/evaluation

DATA_FILE="../inference/data/VRBench_eval.jsonl"
OUTPUT_DIR="eval_outputs/process_scores"
mkdir -p "$OUTPUT_DIR"

MODELS=(
  "InternVL3_5-8B_64frame"
  "Qwen3-VL-8B-Instruct_1024frame"
  "Qwen3-VL-4B-Instruct_1024frame"
)

for MODEL in "${MODELS[@]}"; do
  INFERENCE_FILE="../inference/outputs/${MODEL}.jsonl"
  OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}_process_eval.jsonl"

  echo "=== 评分: $MODEL ==="
  python3 run_process_eval.py \
    --inference_file "$INFERENCE_FILE" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --concurrency 20
done

echo "全部完成"
