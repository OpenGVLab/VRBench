#!/bin/bash
# 过程评分: Keye-VL-1.5-8B / Eagle2.5-8B
# 两者都不是 thinking 模型，无 <think> 块
cd /mnt/shared-storage-user/yujiashuo/vrbench/evaluation

DATA_FILE="../inference/data/VRBench_eval.jsonl"
OUTPUT_DIR="eval_outputs/process_scores"
mkdir -p "$OUTPUT_DIR"

MODELS=(
  "Keye-VL-1_5-8B_64frame"
  "Eagle2.5-8B_64frame"
)

for MODEL in "${MODELS[@]}"; do
  INFERENCE_FILE="../inference/outputs/${MODEL}.jsonl"
  OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}_process_eval.jsonl"

  if [ ! -f "$INFERENCE_FILE" ]; then
    echo "[skip] 缺推理文件: $INFERENCE_FILE"
    continue
  fi

  echo "=== 评分: $MODEL ==="
  python3 run_process_eval.py \
    --inference_file "$INFERENCE_FILE" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --concurrency 20
done

echo ""
echo "=== 汇总 ==="
for MODEL in "${MODELS[@]}"; do
  OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}_process_eval.jsonl"
  if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    python3 summarize_process_scores.py --eval_file "$OUTPUT_FILE"
  fi
done

echo ""
echo "全部完成"
