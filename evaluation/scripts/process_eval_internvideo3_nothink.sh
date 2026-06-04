#!/bin/bash
# 过程评分: InternVideo3 mla426_2 nothink
cd /mnt/shared-storage-user/yujiashuo/vrbench/evaluation

DATA_FILE="../inference/data/VRBench_eval.jsonl"
OUTPUT_DIR="eval_outputs/process_scores"
mkdir -p "$OUTPUT_DIR"

echo "=== 评分: InternVideo3_mla426_2_nothink ==="
python3 run_process_eval.py \
  --inference_file "../inference/outputs/InternVideo3_mla426_2_nothink_1024frame.jsonl" \
  --data_file "$DATA_FILE" \
  --output_file "${OUTPUT_DIR}/InternVideo3_mla426_2_nothink_1024frame_process_eval.jsonl" \
  --concurrency 20
