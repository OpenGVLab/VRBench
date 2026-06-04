#!/bin/bash
# 过程评分: Molmo2-8B (非 thinking 模型)
cd /mnt/shared-storage-user/yujiashuo/vrbench/evaluation

DATA_FILE="../inference/data/VRBench_eval.jsonl"
OUTPUT_DIR="eval_outputs/process_scores"
mkdir -p "$OUTPUT_DIR"

MODEL="Molmo2-8B_64frame"
INFERENCE_FILE="../inference/outputs/${MODEL}.jsonl"
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}_process_eval.jsonl"

if [ ! -f "$INFERENCE_FILE" ]; then
  echo "缺推理文件: $INFERENCE_FILE"
  exit 1
fi

echo "=== 评分: $MODEL ==="
python3 run_process_eval.py \
  --inference_file "$INFERENCE_FILE" \
  --data_file "$DATA_FILE" \
  --output_file "$OUTPUT_FILE" \
  --concurrency 20

echo ""
echo "=== 汇总 ==="
python3 summarize_process_scores.py --eval_file "$OUTPUT_FILE"

echo ""
echo "完成"
