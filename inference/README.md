# VRBench Inference Toolkit

This directory turns raw VRBench questions into **model answers** (inference stage).  Output JSONL files are later rated by the evaluation pipeline.

The toolkit contains three layers:
1. **`main.py`** – generic CLI entry point (handles data slicing, adaptive frame count, prompt selection).
2. **`model_inference/`** – model–specific wrappers (Video LLaVA, InternVL, Qwen-VL, etc.).
3. **`model_inference_scripts/`** – convenience shell launchers for batch runs.

---

## 1. Minimal command-line usage

```bash
python inference/main.py \
  --model "Qwen/Qwen2-VL-7B-Instruct" \
  --prompt "mcq" \
  --data_path "inference/data/VRBench_eval.jsonl" \
  --video_root "/absolute/path/to/VRBench" \
  --total_frames -1               # let the script decide
```

### Required flags
| flag | description |
|------|-------------|
| `--model` | HuggingFace model id or local name (must exist in `utils/constant.py::MODEL_PATH`) |
| `--data_path` | VRBench JSONL containing *relative* `video_path` fields |
| `--video_root` | absolute directory that prefixes every relative `video_path` |

Optional
* `--prompt`   – prompt template (default `single_round`, `mcq` most common)
* `--total_frames` – fixed frame count. Use `-1` to let the **adaptive frame solver** decide.
* `--output_dir`  – directory for `<model_name>_<suffix>.jsonl` outputs

---

## 2. Model path configuration (`utils/constant.py`)
The dictionary `MODEL_PATH` maps **model id → local weight path**.  Edit this file **once** before running inference, e.g.
```python
MODEL_PATH = {
    "Qwen/Qwen2-VL-7B-Instruct": "/data/models/qwen2-vl-7b",
    "llava-hf/LLaVA-NeXT-Video-7B-hf": "/data/models/llava-next-video-7b",
    # add your model here
}
```
If a requested model is missing you will get `model_path not found` and the run aborts.

---

## 3. Adaptive frame count logic
When `--total_frames` is **-1** the script selects a frame budget automatically:
1. **Precise mode** (default) – uses model tokenizer to estimate token cost.
2. **Approximate fallback** – heuristic based on video duration + model limits.
3. For **text-only LLMs** the frame count is forced to `-1` (video skipped).

You can override:
```bash
--total_frames 16                       # fixed
--use_precise_calculation false         # skip precise step
```

---

## 4. Integrating your own model
There are two supported back-ends:

| back-end | typical wrapper | notes |
|----------|-----------------|-------|
| **HuggingFace Transformers** | `model_inference/llava.py`, `internvideo.py`, etc. | Runs locally via `transformers` APIs. |
| **vLLM** | `model_inference/vllm_*` modules | Model served by the [vLLM](https://github.com/vllm-project/vllm) engine. |

### 4-1  Code changes checklist
1. **Create a wrapper** in `inference/model_inference/` – copy the closest example and adapt:
   * model / processor loading
   * `generate_response()` signature (see existing files)
2. **Register the local weight path** in `utils/constant.py::MODEL_PATH`.
3. **Add the model name** to the appropriate list in `model_inference_scripts/`.  
   * `run_hf_video_models.sh` for HF loaders
   * `run_vllm_video_models.sh` for vLLM loaders
4. **(Optional)** update `vllm_model_list.json` if you rely on the JSON switch logic in `main.py`.

### 4-2  Choosing a max frame count
| back-end | guidance |
|----------|----------|
| **HuggingFace** | GPU memory is the main constraint.  The *adaptive solver* (`--total_frames -1`) usually yields a safe number; override manually if needed. |
| **vLLM** | The engine streams tokens efficiently, allowing **larger frame counts**.  Start with `--total_frames 32` or `64` and increase while monitoring CUDA memory + throughput.  The adaptive solver still provides a baseline. |

> Tip: use the *debug dataset* below to verify your wrapper before running the full set.

---

## 5. Debug dataset
A two-video file for quick smoke testing:
```
inference/data/VRBench_debug.jsonl
```
Example:
```bash
python inference/main.py \
  --model "YourModel" \
  --prompt "mcq" \
  --data_path "inference/data/VRBench_debug.jsonl" \
  --video_root "/absolute/path/to/VRBench" \
  --output_dir "inference/debug_outputs"
```
The run completes in seconds and produces output JSONL files used later by the evaluator.

---

## 6. Batch shell scripts (`model_inference_scripts/`)
All scripts follow:
```
./<script>.sh [total_frames] [video_root]
```
| script | purpose |
|--------|---------|
| `run_hf_video_models.sh`       | HuggingFace video models |
| `run_vllm_video_models.sh`     | vLLM video models |
| `run_vllm_image_models.sh`     | vLLM image models |
| `run_vllm_language_models.sh`  | vLLM language (text) models – no `video_root` |

Modify the model arrays inside each script to include your custom entry.

---

## 7. Output file naming
* **VLM** – `<model_name>_<frames>frame.jsonl`  (e.g. `Qwen2.5-VL-7B_16frame.jsonl`)
* **LLM** – `<model_name>.jsonl`

Files are written to `--output_dir` and later consumed by the evaluation stage.

---
For questions please open an issue or PR.
