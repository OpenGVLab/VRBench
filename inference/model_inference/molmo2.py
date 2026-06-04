import os
import torch
import numpy as np
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from utils.prepare_input import prepare_qa_text_input
from utils.video_process import video_to_ndarrays, get_middle_frame_indices
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.utils import logging
from transformers.video_utils import VideoMetadata
from tqdm import tqdm
import jsonlines
import warnings


def _probe_video_meta(path: str, video_read_type: str):
    """Return (fps, total_frames). Cheap — opens the container/reader briefly, no full decode."""
    if video_read_type == 'decord':
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(path, ctx=cpu(0), num_threads=1)
            return float(vr.get_avg_fps()), len(vr)
        except Exception:
            video_read_type = 'av'
    if video_read_type == 'av':
        import av
        container = av.open(path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        total = stream.frames or 0
        if total == 0 and stream.duration and stream.time_base:
            total = int(float(stream.duration * stream.time_base) * fps)
        container.close()
        return fps, total or 1
    raise ValueError(f"unknown video_read_type: {video_read_type}")

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def generate_by_molmo2(model_name,
                       model_path,
                       queries,
                       prompt,
                       total_frames,
                       output_path: str,
                       temperature,
                       max_tokens,
                       device_id: int = 0):
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    device = f"cuda:{device_id}"

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, dtype="auto",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, trust_remote_code=True, dtype="auto", device_map={"": device},
    )
    model.eval()

    num_frames = total_frames if total_frames > 0 else 64
    result_key = f"{prompt['type']}_result"

    queries_pbar = tqdm(queries,
                        desc="Processing Videos",
                        unit="video",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    with jsonlines.open(output_path, 'a', flush=True) as f:
        for query in queries_pbar:
            queries_pbar.set_postfix_str(f"Video ID: {query['video_id']}")
            torch.cuda.empty_cache()

            output_dict = {"video_id": query['video_id'], result_key: {}}

            if prompt['type'] == 'mcq':
                qa_items = list(query['mcq'].items())

                # Decode video frames ourselves with VRBench's standard reader.
                # Molmo2's built-in decord reader chokes on .ts streams; passing a
                # numpy array makes load_video early-exit (see video_processing_molmo2.py:453).
                video_read_type = query.get('video_read_type', 'decord')
                try:
                    video_frames = video_to_ndarrays(
                        query['video_path'], num_frames=num_frames, video_read_type=video_read_type
                    )
                except Exception as e:
                    print(f"Warning: Failed to read video with {video_read_type}, fallback: {e}")
                    fallback_type = 'av' if video_read_type == 'decord' else 'decord'
                    video_frames = video_to_ndarrays(
                        query['video_path'], num_frames=num_frames, video_read_type=fallback_type
                    )

                video_np = np.asarray(video_frames)  # (T, H, W, 3) uint8

                # Build VideoMetadata so the processor can compute timestamps for the prompt.
                fps, total_frames_in_video = _probe_video_meta(query['video_path'], video_read_type)
                frames_indices = get_middle_frame_indices(total_frames_in_video, num_frames)
                video_metadata = VideoMetadata(
                    total_num_frames=total_frames_in_video,
                    fps=fps,
                    frames_indices=list(frames_indices),
                    duration=total_frames_in_video / fps if fps else None,
                )

                responses_pbar = tqdm(enumerate(qa_items),
                                      total=len(qa_items),
                                      desc="Generating Responses",
                                      unit="query",
                                      leave=False,
                                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

                for idx, (qa, qa_dict) in responses_pbar:
                    responses_pbar.set_postfix_str(f"QA ID: {qa}")

                    _, qa_text_prompt = prepare_qa_text_input(
                        video_summary=query['video_summary'],
                        qa_dict=qa_dict,
                        prompt=prompt
                    )

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                dict(type="text", text=qa_text_prompt),
                                dict(type="video", video=video_np),
                            ],
                        }
                    ]

                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        do_sample_frames=False,  # frames already sampled upstream
                        video_metadata=[video_metadata],
                    )
                    # token_type_ids triggers an `or_mask_function` path in modeling_molmo2.py:1537
                    # that requires torch>=2.6. Drop it here; image-block bidirectional masking
                    # degrades to plain causal — minor accuracy hit, but compatible with torch 2.4.
                    inputs.pop("token_type_ids", None)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    with torch.inference_mode():
                        generated_ids = model.generate(
                            **inputs,
                            do_sample=temperature > 0,
                            temperature=temperature if temperature > 0 else None,
                            max_new_tokens=max_tokens,
                        )
                    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
                    response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    output_dict[result_key][qa] = response

                    del inputs, generated_ids
                    torch.cuda.empty_cache()

                del video_np, video_frames

            f.write(output_dict)
            torch.cuda.empty_cache()


def generate_response(model_name: str,
                      model_path: str,
                      prompt: str,
                      queries: list,
                      total_frames: int,
                      output_path: str,
                      temperature: float = GENERATION_TEMPERATURE,
                      max_tokens: int = MAX_TOKENS,
                      device_id: int = 0):

    generate_by_molmo2(model_name,
                       model_path,
                       queries,
                       prompt=prompt,
                       total_frames=total_frames,
                       output_path=output_path,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       device_id=device_id)
