import os
import torch
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from utils.prepare_input import prepare_qa_text_input
from transformers import AutoModel, AutoProcessor
from transformers.utils import logging
from transformers import activations as _ta
if not hasattr(_ta, 'PytorchGELUTanh'):
    _ta.PytorchGELUTanh = _ta.GELUTanh
from keye_vl_utils import process_vision_info
from utils.video_process import video_to_ndarrays
from PIL import Image
from tqdm import tqdm
import jsonlines
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def generate_by_keye_vl_1_5(model_name,
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

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    result_key = f"{prompt['type']}_result"

    actual_frames = total_frames if total_frames > 0 else 64
    # Keye-VL-1.5 preprocessor: patch_size=14, merge_size=2 -> effective 28x28 per visual token
    # min/max_pixels are per-frame; target ~256 visual tokens/frame
    min_pixels = 32 * 28 * 28
    max_pixels = 256 * 28 * 28

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
                try:
                    video_frames = video_to_ndarrays(
                        query['video_path'], num_frames=actual_frames, video_read_type=video_read_type
                    )
                except Exception as e:
                    print(f"Warning: Failed to read video with {video_read_type}, fallback: {e}")
                    fallback_type = 'av' if video_read_type == 'decord' else 'decord'
                    video_frames = video_to_ndarrays(
                        query['video_path'], num_frames=actual_frames, video_read_type=fallback_type
                    )

                video_frames_pil = [Image.fromarray(frame).convert("RGB") for frame in video_frames]
                del video_frames

                dummy_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_frames_pil,
                             "min_pixels": min_pixels, "max_pixels": max_pixels},
                            {"type": "text", "text": "dummy"},
                        ],
                    },
                ]
                image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(dummy_messages)

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
                                {"type": "video", "video": video_frames_pil,
                                 "min_pixels": min_pixels, "max_pixels": max_pixels},
                                {"type": "text", "text": qa_text_prompt},
                            ],
                        },
                    ]

                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        **mm_processor_kwargs,
                    )
                    inputs = inputs.to(model.device)

                    gen_out = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_out)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    output_dict[result_key][qa] = output_text[0]

                    del inputs, gen_out, generated_ids_trimmed
                    torch.cuda.empty_cache()

                del video_frames_pil, video_inputs

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

    generate_by_keye_vl_1_5(model_name,
                            model_path,
                            queries,
                            prompt=prompt,
                            total_frames=total_frames,
                            output_path=output_path,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            device_id=device_id)
