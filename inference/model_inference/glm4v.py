import os
import torch
import numpy as np
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from utils.prepare_input import prepare_qa_text_input
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from transformers.video_utils import VideoMetadata
from transformers.utils import logging
from decord import VideoReader, cpu
import av
from tqdm import tqdm
import jsonlines
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def _sample_with_decord(video_path, num_frames):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    vlen = len(vr)
    fps = float(vr.get_avg_fps())
    n = min(num_frames, vlen)
    indices = np.linspace(0, vlen - 1, n, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    return frames, vlen, fps, indices


def _sample_with_av(video_path, num_frames):
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    vlen = stream.frames
    if vlen <= 0:
        vlen = int(float(stream.duration * stream.time_base) * fps)
    n = min(num_frames, vlen)
    indices = np.linspace(0, vlen - 1, n, dtype=int).tolist()
    target = set(int(i) for i in indices)
    last = int(indices[-1])
    collected = {}
    for i, frame in enumerate(container.decode(stream)):
        if i in target:
            collected[i] = frame.reformat(format='rgb24').to_ndarray()
        if i >= last:
            break
    frames = np.stack([collected[int(i)] for i in indices])
    return frames, vlen, fps, indices


def sample_video_frames(video_path, num_frames, video_read_type='decord'):
    """Uniformly sample `num_frames` frames, returning frames array and VideoMetadata with real timestamps."""
    samplers = {'decord': _sample_with_decord, 'av': _sample_with_av}
    primary = video_read_type if video_read_type in samplers else 'decord'
    fallback = 'av' if primary == 'decord' else 'decord'
    try:
        frames, vlen, fps, indices = samplers[primary](video_path, num_frames)
    except Exception as e:
        print(f"{primary} failed for {video_path}: {e}; falling back to {fallback}")
        frames, vlen, fps, indices = samplers[fallback](video_path, num_frames)

    metadata = VideoMetadata(
        total_num_frames=int(vlen),
        fps=fps,
        duration=float(vlen / fps) if fps > 0 else None,
        frames_indices=indices,
    )
    return frames, metadata


def generate_by_glm4v(model_name,
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

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
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

                video_read_type = query.get('video_read_type', 'decord')
                frames, metadata = sample_video_frames(query['video_path'], num_frames, video_read_type)

                responses_pbar = tqdm(enumerate(qa_items),
                                      total=len(qa_items),
                                      desc="Generating Responses",
                                      unit="query",
                                      leave=False,
                                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

                for idx, (qa, qa_dict) in responses_pbar:
                    torch.cuda.empty_cache()
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
                                {"type": "video"},
                                {"type": "text", "text": qa_text_prompt},
                            ],
                        },
                    ]

                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    inputs = processor(
                        text=[text],
                        videos=[frames],
                        video_metadata=[metadata],
                        do_sample_frames=False,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(model.device)

                    gen_out = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                    )
                    output_text = processor.decode(
                        gen_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                    output_dict[result_key][qa] = output_text

                    del inputs, gen_out
                    torch.cuda.empty_cache()

                del frames

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

    generate_by_glm4v(model_name,
                      model_path,
                      queries,
                      prompt=prompt,
                      total_frames=total_frames,
                      output_path=output_path,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      device_id=device_id)
