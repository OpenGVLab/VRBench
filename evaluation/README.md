# VRBench Evaluation Toolkit

This document is organised in **two independent parts**:

1. **DeepSeek-based rating** – call the DeepSeek API (or a local DeepSeek model) to rate previously generated answers.
2. **Score aggregation** – compute VRBench scores (overall and per-category) from the rating files.

> All commands assume execution from the project root.

---

## Part 1  Rating with DeepSeek

Scripts
* `evaluation/batch_evaluation_api_mp_time_sync.py`  (synchronous, multi-thread)
* `evaluation/batch_evaluation_api_mp_time_async.py` (asynchronous, `asyncio + aiohttp`)

These scripts **only rate answers that already exist** in an inference file – they do **not** generate answers themselves (answer generation happens in the `inference/` pipeline).

### Required inputs
* **source_file** – inference JSONL produced by `inference/` (must contain `mcq_result` or `openqa_result`).
* **summary_file** – video summaries JSONL (used by the rating prompt).

### Minimal example (sync)
```bash
python evaluation/batch_evaluation_api_mp_time_sync.py \
  --source_file  inference/outputs/my_model_inference.jsonl \
  --summary_file evaluation/data_for_eval/VRBench_summary.jsonl \
  --output_file  evaluation/eval_outputs/my_model_evaluation.jsonl \
  --model        deepseek           # DeepSeek is mandatory for rating \
  --api_key      $DEESEEK_API_KEY   # your DeepSeek key \
  --base_url     https://api.deepseek.com/v1
```

Key flags
| flag | required | description |
|------|----------|-------------|
| `--source_file`  | yes | inference JSONL to be rated |
| `--summary_file` | yes | video summaries (JSONL) |
| `--output_file`  | yes | write DeepSeek ratings here |
| `--model`        | yes | **must be `deepseek`** |
| `--api_key` / `--base_url` | yes | DeepSeek credentials |
| `--VLM` | optional | if the answers belong to a VLM task |
| `--separate` | optional | split system/user prompts (advanced)

Output JSONL (one line per QA item):
```jsonc
{
  "video_id": "...",
  "qa_id": "qa3",
  "rate": "5",          // DeepSeek rating (string 1-5)
  "reason": "...",      // optional rationale
  "eval_response": "..." // full model reply (optional)
}
```

The produced *evaluation file* is the `--evaluation_file` input for Part 2.

---

## Part 2  Score aggregation

### 2-1  Single model
```bash
python evaluation/calculate_scores.py \
  --ground_truth_file evaluation/data_for_eval/VRBench_eval.jsonl \
  --inference_file    evaluation/eval_outputs/my_model_inference.jsonl \
  --evaluation_file   evaluation/eval_outputs/my_model_evaluation.jsonl \
  --model_name        my_model \
  --output_file       evaluation/results/my_model_scores.json   # optional
```

| flag | required | description |
|------|----------|-------------|
| `--ground_truth_file` | yes | reference answers |
| `--inference_file`    | yes | answers produced by *inference* stage |
| `--evaluation_file`   | yes | DeepSeek rating file from Part 1 |
| `--model_name`        | yes | label in the table |
| `--output_file`       | no  | write JSON result (in addition to stdout) |

### 2-2  Batch mode
```bash
python evaluation/batch_calculate_scores.py \
  --data_dir          evaluation/eval_outputs \
  --ground_truth_file evaluation/data_for_eval/VRBench_eval.jsonl \
  --model_pattern     "*" \
  --output_dir        evaluation/batch_results \
  --output_file       batch_results.json
```

File naming inside `data_dir`
* `<model>_inference.jsonl`
* `<model>_evaluation.jsonl`

The script pairs files, runs the single-model scorer, and prints a consolidated table.

---

### Example table (truncated)
```
Model Name                         Overall    MCQ    OE    MCQ Cnt  OE Cnt
---------------------------------------------------------------------------
example_model                      0.6231     0.7662 4.7995  1403     1146
```

Field meaning
* **Overall** – mean of MCQ accuracy and OE average
* **MCQ** – multiple-choice accuracy
* **OE** – open-ended QA average

---
Questions or improvements? Open an issue or PR.
