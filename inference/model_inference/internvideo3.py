import os
import torch
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from utils.prepare_input import prepare_qa_text_input
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.utils import logging
from qwen_vl_utils import process_vision_info
from utils.video_process import video_to_ndarrays
from PIL import Image
from tqdm import tqdm
import jsonlines
import warnings
import numpy as np

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def generate_by_internvideo3(model_name,
                            model_path,
                            queries,
                            prompt,
                            total_frames,
                            output_path: str,
                            temperature,
                            max_tokens,
                            device_id: int = 0):
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{device_id}",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained("/mnt/shared-storage-user/sfteval/sfteval_models/Qwen3-VL-8B-Instruct/", trust_remote_code=True)

    result_key = f"{prompt['type']}_result"

    actual_frames = total_frames if total_frames > 0 else 64
    min_pixels = actual_frames * 32 * 32 * 4
    max_pixels = actual_frames * 32 * 32 * 4

    is_nothink = "nothink" in model_name.lower()

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

                # Load video frames ONCE per video
                video_read_type = query.get('video_read_type', 'decord')
                try:
                    video_frames = video_to_ndarrays(
                        query['video_path'],
                        num_frames=total_frames if total_frames > 0 else 64,
                        video_read_type=video_read_type
                    )
                except Exception as e:
                    print(f"Warning: Failed to read video with {video_read_type}, trying fallback method: {e}")
                    fallback_type = 'av' if video_read_type == 'decord' else 'decord'
                    video_frames = video_to_ndarrays(
                        query['video_path'],
                        num_frames=total_frames if total_frames > 0 else 64,
                        video_read_type=fallback_type
                    )

                video_frames_pil = [Image.fromarray(frame).convert("RGB") for frame in video_frames]
                del video_frames

                # Process vision info ONCE per video (using a dummy message)
                dummy_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_frames_pil,
                                "min_pixels": min_pixels,
                                "max_pixels": max_pixels,
                            },
                            {"type": "text", "text": "dummy"},
                        ],
                    },
                ]
                _, cached_videos, cached_video_kwargs = process_vision_info(
                    dummy_messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True
                )
                cached_video_tensor = cached_videos[0][0]
                cached_video_kwargs = cached_videos[0][1]
                del dummy_messages

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
                                {
                                    "type": "video",
                                    "video": video_frames_pil,
                                    "min_pixels": min_pixels,
                                    "max_pixels": max_pixels,
                                },
                                {"type": "text", "text": qa_text_prompt},
                            ],
                        },
                    ]

                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        return_dict=False,
                        return_tensors="pt"
                    )

                    if is_nothink:
                        text = text + "<think>\n</think>\n\n"
                    else:
                        text = text + "<think>"

                    # Reuse cached video tensor
                    inputs = processor(
                        text=text,
                        images=None,
                        videos=cached_video_tensor,
                        do_resize=False,
                        frame_seqlen=total_frames if total_frames > 0 else 64,
                        return_tensors="pt",
                        **cached_video_kwargs
                    )

                    del text, messages

                    inputs = inputs.to(model.device)

                    gen_out = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        return_dict_in_generate=True,
                        use_cache=True,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                    )

                    generated_ids = gen_out.sequences
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    output_dict[result_key][qa] = output_text[0]

                    del inputs, gen_out, generated_ids, generated_ids_trimmed
                    torch.cuda.empty_cache()

                del video_frames_pil, cached_video_tensor, cached_video_kwargs

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

    generate_by_internvideo3(model_name,
                                model_path,
                                queries,
                                prompt=prompt,
                                total_frames=total_frames,
                                output_path=output_path,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                device_id=device_id)
