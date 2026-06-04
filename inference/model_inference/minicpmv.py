import os
import torch
import numpy as np
from scipy.spatial import cKDTree
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from utils.prepare_input import prepare_qa_text_input
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging
from decord import VideoReader, cpu
import av
from PIL import Image
from tqdm import tqdm
import jsonlines
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

TIME_SCALE = 0.1
MAX_NUM_PACKING = 3


def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i + size] for i in range(0, len(arr), size)]


def _uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]


def _read_with_decord(video_path, num_frames):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    vlen = len(vr)
    fps = vr.get_avg_fps()
    n = min(num_frames, vlen)
    frame_idx = np.array(_uniform_sample(list(range(vlen)), n))
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_idx, fps, vlen / fps


def _read_with_av(video_path, num_frames):
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    vlen = stream.frames
    if vlen <= 0:
        vlen = int(float(stream.duration * stream.time_base) * fps)
    n = min(num_frames, vlen)
    frame_idx = np.array(_uniform_sample(list(range(vlen)), n))
    target = set(int(i) for i in frame_idx)
    last = int(frame_idx[-1])
    collected = {}
    for i, frame in enumerate(container.decode(stream)):
        if i in target:
            collected[i] = frame.reformat(format='rgb24').to_ndarray()
        if i >= last:
            break
    frames = np.stack([collected[int(i)] for i in frame_idx])
    return frames, frame_idx, fps, vlen / fps


def encode_video(video_path, num_frames, video_read_type='decord', force_packing=1):
    """Sample exactly `num_frames` frames uniformly and build temporal_ids."""
    readers = {'decord': _read_with_decord, 'av': _read_with_av}
    primary = video_read_type if video_read_type in readers else 'decord'
    fallback = 'av' if primary == 'decord' else 'decord'
    try:
        frames, frame_idx, fps, video_duration = readers[primary](video_path, num_frames)
    except Exception as e:
        print(f"{primary} failed for {video_path}: {e}; falling back to {fallback}")
        frames, frame_idx, fps, video_duration = readers[fallback](video_path, num_frames)

    packing_nums = min(force_packing, MAX_NUM_PACKING)

    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, TIME_SCALE)
    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)
    return frames, frame_ts_id_group


def generate_by_minicpmv(model_name,
                         model_path,
                         queries,
                         prompt,
                         total_frames,
                         output_path: str,
                         temperature,
                         max_tokens,
                         device_id: int = 0):
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation='sdpa',
    ).to(f"cuda:{device_id}").eval()

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

            video_read_type = query.get('video_read_type', 'decord')
            frames, frame_ts_id_group = encode_video(query['video_path'], num_frames, video_read_type)

            if prompt['type'] == 'mcq':
                qa_items = list(query['mcq'].items())

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

                    msgs = [{'role': 'user', 'content': frames + [qa_text_prompt]}]
                    response = model.chat(
                        msgs=msgs,
                        tokenizer=tokenizer,
                        use_image_id=False,
                        max_slice_nums=1,
                        temporal_ids=frame_ts_id_group,
                        sampling=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        max_new_tokens=max_tokens,
                    )
                    output_dict[result_key][qa] = response

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

    generate_by_minicpmv(model_name,
                         model_path,
                         queries,
                         prompt=prompt,
                         total_frames=total_frames,
                         output_path=output_path,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         device_id=device_id)
