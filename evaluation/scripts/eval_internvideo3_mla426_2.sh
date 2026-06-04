#!/bin/bash
# InternVideo3_mla426_2 MCQ 准确率计算
# 无需 API，直接本地运行

cd /mnt/shared-storage-user/yujiashuo/vrbench/evaluation

MODEL_NAME="InternVideo3_mla426_2"
INFERENCE_FILE="../inference/outputs/InternVideo3_mla426_2_1024frame.jsonl"
GROUND_TRUTH_FILE="../inference/data/VRBench_eval.jsonl"
OUTPUT_FILE="./eval_outputs/${MODEL_NAME}_mcq_scores.json"

mkdir -p eval_outputs

python3 -c "
import json
import sys
sys.path.insert(0, '.')
from calculate_scores import extract_mcq_answer, load_ground_truth, load_inference_results
from collections import defaultdict

ground_truth = load_ground_truth('${GROUND_TRUTH_FILE}')
inference_results = load_inference_results('${INFERENCE_FILE}')

print(f'Ground truth: {len(ground_truth)} questions')
print(f'Inference results: {len(inference_results)} questions')

correct_count = 0
total_count = 0
category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
failed_extract = 0

for question_id, gt_data in ground_truth.items():
    if question_id in inference_results:
        total_count += 1
        predicted_answer = extract_mcq_answer(inference_results[question_id])
        correct_answer = gt_data['answer']
        reasoning_type = gt_data['reasoning_type']

        category_stats[reasoning_type]['total'] += 1

        if predicted_answer is None:
            failed_extract += 1
        elif predicted_answer == correct_answer:
            correct_count += 1
            category_stats[reasoning_type]['correct'] += 1

overall_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

print()
print('=' * 60)
print(f'${MODEL_NAME} VRBench MCQ Results')
print('=' * 60)
print(f'Overall Accuracy: {overall_accuracy:.2f}% ({correct_count}/{total_count})')
print(f'Failed to extract answer: {failed_extract}')
print()
print('By Reasoning Type:')
print('-' * 60)

results = {'model': '${MODEL_NAME}', 'overall_accuracy': round(overall_accuracy, 2), 'correct': correct_count, 'total': total_count, 'categories': {}}

for category in sorted(category_stats.keys()):
    stats = category_stats[category]
    acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f'  {category}: {acc:.2f}% ({stats[\"correct\"]}/{stats[\"total\"]})')
    results['categories'][category] = {'accuracy': round(acc, 2), 'correct': stats['correct'], 'total': stats['total']}

with open('${OUTPUT_FILE}', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\nResults saved to: ${OUTPUT_FILE}')
"
