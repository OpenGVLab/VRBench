#!/bin/bash
# 过程评分: MiniCPM-V-4.5 / GLM-4.1V-9B-Thinking
# GLM 是 thinking 模型，run_process_eval.py 内部会去掉 <think> 块
cd /mnt/shared-storage-user/yujiashuo/vrbench/evaluation

DATA_FILE="../inference/data/VRBench_eval.jsonl"
OUTPUT_DIR="eval_outputs/process_scores"
mkdir -p "$OUTPUT_DIR"

MODELS=(
  "MiniCPM-V-4_5_64frame"
  "GLM-4.1V-9B-Thinking_64frame"
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
