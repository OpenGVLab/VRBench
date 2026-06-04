import os
import copy
import torch
import numpy as np
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from utils.prepare_input import prepare_qa_text_input
from transformers.utils import logging
from decord import VideoReader, cpu
import av
from tqdm import tqdm
import jsonlines
import warnings

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def _read_frames_decord(video_path, max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = vr.get_avg_fps()
    frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_idx, total_frame_num, avg_fps


def _read_frames_av(video_path, max_frames_num):
    container = av.open(video_path)
    stream = container.streams.video[0]
    avg_fps = float(stream.average_rate)
    total_frame_num = stream.frames
    if total_frame_num <= 0:
        total_frame_num = int(float(stream.duration * stream.time_base) * avg_fps)
    frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
    target = set(int(i) for i in frame_idx)
    last = int(frame_idx[-1])
    collected = {}
    for i, frame in enumerate(container.decode(stream)):
        if i in target:
            collected[i] = frame.reformat(format='rgb24').to_ndarray()
        if i >= last:
            break
    frames = np.stack([collected[int(i)] for i in frame_idx])
    return frames, frame_idx, total_frame_num, avg_fps


def load_video(video_path, max_frames_num, video_read_type='decord'):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0
    readers = {'decord': _read_frames_decord, 'av': _read_frames_av}
    primary = video_read_type if video_read_type in readers else 'decord'
    fallback = 'av' if primary == 'decord' else 'decord'
    try:
        frames, frame_idx, total_frame_num, avg_fps = readers[primary](video_path, max_frames_num)
    except Exception as e:
        print(f"{primary} failed for {video_path}: {e}; falling back to {fallback}")
        frames, frame_idx, total_frame_num, avg_fps = readers[fallback](video_path, max_frames_num)
    video_time = total_frame_num / avg_fps
    frame_time = ",".join([f"{i / avg_fps:.2f}s" for i in frame_idx])
    return frames, frame_time, video_time


def generate_by_llava_video(model_name,
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

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        torch_dtype="bfloat16",
        device_map={"": device},
    )
    model.eval()

    conv_template = "qwen_1_5"
    max_frames_num = total_frames if total_frames > 0 else 64
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
            frames, frame_time, video_time = load_video(
                query['video_path'], max_frames_num, video_read_type
            )
            video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(device).bfloat16()
            video = [video]
            time_instruction = (
                f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames "
                f"are uniformly sampled from it. These frames are located at {frame_time}."
                f"Please answer the following questions related to this video."
            )

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

                    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{qa_text_prompt}"
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()

                    input_ids = tokenizer_image_token(
                        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).to(device)

                    cont = model.generate(
                        input_ids,
                        images=video,
                        modalities=["video"],
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        max_new_tokens=max_tokens,
                    )
                    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
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

    generate_by_llava_video(model_name,
                            model_path,
                            queries,
                            prompt=prompt,
                            total_frames=total_frames,
                            output_path=output_path,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            device_id=device_id)
