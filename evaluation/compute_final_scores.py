#!/usr/bin/env python3
"""
汇总所有模型的 MCQ Accuracy + Process Score + Final Score
- MCQ Acc 来自 inference jsonl + ground truth
- Process Score 来自 process_scores/*_process_eval.jsonl 的 weighted 字段 (0-10)
- Final = (MCQ_Acc + PS*10) / 2
"""
import json
import os
import sys
from collections import defaultdict

EVAL_ROOT = "/mnt/shared-storage-user/yujiashuo/vrbench/evaluation"
INF_ROOT = "/mnt/shared-storage-user/yujiashuo/vrbench/inference"
sys.path.insert(0, EVAL_ROOT)
from calculate_scores import (
    load_ground_truth,
    load_inference_results,
    calculate_mcq_accuracy,
)

GT_FILE = f"{INF_ROOT}/data/VRBench_eval.jsonl"
PROC_DIR = f"{EVAL_ROOT}/eval_outputs/process_scores"

# (display_name, inference_filename, process_eval_filename)
MODELS = [
    ("InternVL3.5-8B (64f)",          "InternVL3_5-8B_64frame.jsonl",                       "InternVL3_5-8B_64frame_process_eval.jsonl"),
    ("Qwen3-VL-8B (1024f)",           "Qwen3-VL-8B-Instruct_1024frame.jsonl",                "Qwen3-VL-8B-Instruct_1024frame_process_eval.jsonl"),
    ("Qwen3-VL-4B (1024f)",           "Qwen3-VL-4B-Instruct_1024frame.jsonl",                "Qwen3-VL-4B-Instruct_1024frame_process_eval.jsonl"),
    ("InternVideo3-mla426 think",     "InternVideo3_mla426_2_think_1024frame.jsonl",         "InternVideo3_mla426_2_think_1024frame_process_eval.jsonl"),
    ("InternVideo3-mla426 nothink",   "InternVideo3_mla426_2_nothink_1024frame.jsonl",       "InternVideo3_mla426_2_nothink_1024frame_process_eval.jsonl"),
    ("MiniCPM-V-4.5 (64f)",           "MiniCPM-V-4_5_64frame.jsonl",                         "MiniCPM-V-4_5_64frame_process_eval.jsonl"),
    ("GLM-4.1V-9B-Thinking (64f)",    "GLM-4.1V-9B-Thinking_64frame.jsonl",                  "GLM-4.1V-9B-Thinking_64frame_process_eval.jsonl"),
    ("Keye-VL-1.5-8B (64f)",          "Keye-VL-1_5-8B_64frame.jsonl",                        "Keye-VL-1_5-8B_64frame_process_eval.jsonl"),
    ("Eagle2.5-8B (64f)",             "Eagle2.5-8B_64frame.jsonl",                           "Eagle2.5-8B_64frame_process_eval.jsonl"),
    ("Molmo2-8B (64f)",               "Molmo2-8B_64frame.jsonl",                             "Molmo2-8B_64frame_process_eval.jsonl"),
]


def load_process_scores(path):
    """returns: overall_avg (0-10), per_type {type: avg}, total_count"""
    type_scores = defaultdict(list)
    all_scores = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            ws = d['scores']['weighted']
            type_scores[d['reasoning_type']].append(ws)
            all_scores.append(ws)
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    per_type = {t: sum(v) / len(v) for t, v in type_scores.items()}
    return overall, per_type, len(all_scores)


def main():
    gt = load_ground_truth(GT_FILE)
    print(f"Ground truth: {len(gt)} questions across 960 videos\n")

    rows = []
    for name, inf_name, proc_name in MODELS:
        inf_path = f"{INF_ROOT}/outputs/{inf_name}"
        proc_path = f"{PROC_DIR}/{proc_name}"
        if not os.path.exists(inf_path):
            print(f"[skip] {name}: 缺 {inf_path}")
            continue
        if not os.path.exists(proc_path):
            print(f"[skip] {name}: 缺 {proc_path}")
            continue

        inf = load_inference_results(inf_path)
        n_videos = len(set(qid.rsplit('_', 1)[0] for qid in inf))
        mcq_acc, mcq_per_type, mcq_total, mcq_correct = calculate_mcq_accuracy(gt, inf)
        ps_overall, ps_per_type, ps_total = load_process_scores(proc_path)
        final = (mcq_acc + ps_overall * 10) / 2

        rows.append({
            "name": name,
            "videos": n_videos,
            "mcq_total": mcq_total,
            "mcq_correct": mcq_correct,
            "mcq_acc": mcq_acc,
            "ps_total": ps_total,
            "ps_overall": ps_overall,
            "final": final,
            "mcq_per_type": mcq_per_type,
            "ps_per_type": ps_per_type,
        })

    # ----- Overall table -----
    print("=" * 110)
    print(f"{'Model':<32}{'Vid':>5}{'MCQ_Q':>7}{'MCQ%':>8}{'PS_Q':>7}{'PS/10':>8}{'Final%':>9}")
    print("-" * 110)
    for r in rows:
        print(f"{r['name']:<32}{r['videos']:>5}{r['mcq_total']:>7}{r['mcq_acc']:>7.2f}%"
              f"{r['ps_total']:>7}{r['ps_overall']:>8.2f}{r['final']:>8.2f}%")
    print("=" * 110)

    # ----- Per-type breakdown -----
    all_types = sorted({t for r in rows for t in set(r['mcq_per_type']) | set(r['ps_per_type'])})
    print("\nPer reasoning_type breakdown (MCQ% / PS×10):\n")
    header = f"{'reasoning_type':<28}" + "".join(f"{r['name'][:18]:>20}" for r in rows)
    print(header)
    print("-" * len(header))
    for t in all_types:
        line = f"{t:<28}"
        for r in rows:
            mcq = r['mcq_per_type'].get(t, None)
            ps = r['ps_per_type'].get(t, None)
            mcq_s = f"{mcq:.1f}" if mcq is not None else " - "
            ps_s = f"{ps*10:.1f}" if ps is not None else " - "
            line += f"{mcq_s+'/'+ps_s:>20}"
        print(line)


if __name__ == "__main__":
    main()
