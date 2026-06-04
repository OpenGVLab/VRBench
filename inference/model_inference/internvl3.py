import os
import torch
import numpy as np
import av
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from utils.prepare_input import prepare_qa_text_input
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging
from decord import VideoReader, cpu
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import jsonlines
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video(video_path, num_segments=32, input_size=448, max_num=1):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        seg_size = float(max_frame) / num_segments
        frame_indices = np.array([int(seg_size / 2 + np.round(seg_size * idx)) for idx in range(num_segments)])
        frame_indices = np.clip(frame_indices, 0, max_frame)

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
    except Exception as e:
        print(f"Warning: decord failed for {video_path}: {e}, trying av...")
        pixel_values_list, num_patches_list = [], []
        container = av.open(video_path)
        stream = container.streams.video[0]
        max_frame = stream.frames - 1
        if max_frame <= 0:
            max_frame = int(stream.duration * stream.time_base * float(stream.average_rate))
        seg_size = float(max_frame) / num_segments
        frame_indices = set(int(seg_size / 2 + np.round(seg_size * idx)) for idx in range(num_segments))
        for i, frame in enumerate(container.decode(stream)):
            if i > max(frame_indices):
                break
            if i in frame_indices:
                img = frame.to_image().convert("RGB")
                img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values = [transform(tile) for tile in img]
                pixel_values = torch.stack(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)
        container.close()

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def generate_by_internvl3(model_name,
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
    ).to(f"cuda:{device_id}").eval()

    generation_config = dict(
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        max_new_tokens=max_tokens,
        top_p=0.1 if temperature > 0 else None,
        num_beams=1,
    )

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

            num_segments = total_frames if total_frames > 0 else 64
            pixel_values, num_patches_list = load_video(
                query['video_path'],
                num_segments=num_segments,
                max_num=1,
            )
            pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

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

                    question = video_prefix + qa_text_prompt
                    response = model.chat(
                        tokenizer, pixel_values, question,
                        generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=False,
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

    generate_by_internvl3(model_name,
                          model_path,
                          queries,
                          prompt=prompt,
                          total_frames=total_frames,
                          output_path=output_path,
                          temperature=temperature,
                          max_tokens=max_tokens,
                          device_id=device_id)
