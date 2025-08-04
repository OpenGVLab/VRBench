from transformers import AutoTokenizer
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

from argparse import Namespace
from typing import List
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModel
from vllm.assets.image import ImageAsset
from utils.video_process import video_to_ndarrays, video_to_ndarrays_fps, load_video_frames
from vllm.multimodal.utils import fetch_image
import os
import hashlib
import base64
import requests
from tqdm import tqdm
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input

def prepare_llava_next_video_inputs(model_name, 
                                    model_path,
                                    query, 
                                    prompt,
                                    total_frames: int=-1):

    def _create_input(prompt_text: str, video_data: list) -> dict:
        return {
            "prompt": f"USER: <video>\n{prompt_text} ASSISTANT:",
            "multi_modal_data": {"video": video_data}
        }
    
    def _get_video_data(video_path, total_frames, video_read_type):
        return video_to_ndarrays(path=video_path, num_frames=total_frames, video_read_type=video_read_type)

    if prompt['type'] == 'mcq':
        inputs, qa_id = [], []
        video_data = _get_video_data(query['video_path'], total_frames, query['video_read_type'])
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            inputs.append(_create_input(qa_text_prompt, video_data))
            qa_id.append(qa)
        return qa_id, inputs

def prepare_qwen2_inputs(model_name, 
                        model_path,
                        query,
                        prompt,
                        total_frames: int=-1):
    
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    
    def _create_input(messages, video_data) -> dict:
        return {
            "prompt": processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "multi_modal_data": {"video": video_data}
        }

    def _get_video_data(video_path, video_read_type):
        if total_frames == -1:
            return video_to_ndarrays(video_path, total_frames, video_read_type)
        return video_path, video_to_ndarrays(video_path, total_frames, video_read_type)

    if prompt['type'] == 'mcq':
        inputs, qa_id = [], []
        video_path, video_data = _get_video_data(query['video_path'], query['video_read_type'])
        
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": qa_text_prompt}
                ]
            }]
            inputs.append(_create_input(messages, video_data))
            qa_id.append(qa)
        return qa_id, inputs

def prepare_phi3v_inputs(model_name,
                        model_path, 
                        query, 
                        prompt,
                        total_frames: int=-1):
    def _create_input(prompt_text: str, vision_input: list) -> dict:
        """Helper function to create standardized input dictionary."""
        placeholders = "\n".join(f"<|image_{i}|>" for i, _ in enumerate(vision_input, start=1))
        text_input = f"<|user|>\n{placeholders}\n{prompt_text}<|end|>\n<|assistant|>\n"
        return {
            "prompt": text_input,
            "multi_modal_data": {"image": vision_input}
        }

    if prompt['type'] == 'mcq':
        inputs, qa_id = [], []
        # Prepare vision input (video frames) once per video
        vision_input_base64 = prepare_multi_image_input(model_name, query['video_path'], total_frames, video_read_type=query['video_read_type'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            inputs.append(_create_input(qa_text_prompt, vision_input))
            qa_id.append(qa)
        return qa_id, inputs

def prepare_deepseek_vl2_inputs(model_name, 
                                model_path,
                                query, 
                                prompt,
                                total_frames: int = -1):
    video_path, video_summary = query['video_path'], query['video_summary']
    
    def _create_input(prompt_text: str, vision_input: list) -> dict:
        """Helper function to create standardized input dictionary."""
        placeholder = "".join(f"image_{i}:<image>\n" for i, _ in enumerate(vision_input, start=1))
        text_input = f"<|User|>: {placeholder}{prompt_text}\n\n<|Assistant|>:"
        return {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            }
        }

    if prompt['type'] == 'mcq':
        inputs, qa_id = [], []
        vision_input_base64 = prepare_multi_image_input(model_name, video_path, total_frames, video_read_type=query['video_read_type'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=video_summary,
                qa_dict=qa_dict,
                prompt=prompt
            )
            inputs.append(_create_input(qa_text_prompt, vision_input))
            qa_id.append(qa)
        return qa_id, inputs

def prepare_aria_inputs(model_name, 
                        model_path,
                        query, 
                        prompt,
                        total_frames: int = -1):
    def _create_input(prompt_text: str, vision_input: list) -> dict:
        """Helper function to create standardized input dictionary."""
        placeholder = "<fim_prefix><|img|><fim_suffix>\n" * len(vision_input)
        return {
            "prompt": f"<|im_start|>user\n{placeholder}{prompt_text}<|im_end|>\n<|im_start|>assistant\n",
            "multi_modal_data": {"image": vision_input}
        }

    if prompt['type'] == 'mcq':
        inputs, qa_id = [], []
        vision_input_base64 = prepare_multi_image_input(model_name, query['video_path'], total_frames, video_read_type=query['video_read_type'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            inputs.append(_create_input(qa_text_prompt, vision_input))
            qa_id.append(qa)
        return qa_id, inputs

def prepare_general_vlm_inputs(model_name, 
                                model_path,
                                query, 
                                prompt,
                                total_frames: int = -1):
    def _create_input(prompt_text: str, vision_input: list) -> dict:
        """Helper function to create standardized input dictionary."""
        placeholders = "\n".join(f"Image-{i}: <image>\n" for i, _ in enumerate(vision_input, start=1))
        messages = [{'role': 'user', 'content': f"{placeholders}\n{prompt_text}"}]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
    if prompt['type'] == 'mcq':
        inputs, qa_id = [], []
        vision_input_base64 = prepare_multi_image_input(model_name, query['video_path'], total_frames, video_read_type=query['video_read_type'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            inputs.append(_create_input(qa_text_prompt, vision_input))
            qa_id.append(qa)
        return qa_id, inputs

def prepare_pixtral_inputs(model_name, 
                            model_path,
                            query, 
                            prompt,
                            inputs: list = [],
                            response: list = [],
                            total_frames: int = -1):
    def _create_input(prompt_text: str, vision_inputs: list) -> str:
        """Helper function to create standardized input string."""
        placeholders = "[IMG]" * len(vision_inputs)
        return f"<s>[INST]{prompt_text}\n{placeholders}[/INST]"

    if prompt['type'] == 'mcq':
        inputs = []
        qa_id = []
        vision_inputs = prepare_multi_image_input(model_name, query['video_path'], total_frames, video_read_type=query['video_read_type'])
        
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            inputs.append(_create_input(qa_text_prompt, vision_inputs))
            qa_id.append(qa)
        return qa_id, inputs

def prepare_mllama_inputs(model_name,
                          model_path, 
                          query, 
                          prompt,
                          total_frames: int = -1):   
    def _create_input(prompt_text: str, vision_inputs: list) -> dict:
        """Helper function to create standardized input dictionary."""
        placeholders = "<|image|>" * len(vision_inputs)
        messages = [{'role': 'user', 'content': f"{placeholders}\n{prompt_text}"}]
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return {
            "prompt": input_prompt,
            "multi_modal_data": {
                "image": vision_inputs
            },
        }

    if prompt['type'] == 'mcq':
        inputs, qa_id = [], []
        vision_input_base64 = prepare_multi_image_input(model_name, query['video_path'], total_frames, video_read_type=query['video_read_type'])
        vision_inputs = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            inputs.append(_create_input(qa_text_prompt, vision_inputs))
            qa_id.append(qa)
        return qa_id, inputs

def prepare_llava_onevision_inputs(model_name, 
                                model_path,
                                query, 
                                prompt,
                                total_frames: int = -1):
    
    def _create_input(prompt_text: str, video_data: list) -> dict:
        """Helper function to create standardized input dictionary."""
        return {
            "prompt": f"<|im_start|>user <video>\n{prompt_text}<|im_end|> <|im_start|>assistant\n",
            "multi_modal_data": {"video": video_data}
        }

    if prompt['type'] == 'mcq':
        inputs, qa_id = [], []
        if total_frames == -1:
            video_data = video_to_ndarrays_fps(path=query['video_path'], fps=1, max_frames=64)
        else:
            video_data = video_to_ndarrays(path=query['video_path'], num_frames=total_frames, video_read_type=query['video_read_type'])
        
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            inputs.append(_create_input(qa_text_prompt, video_data))
            qa_id.append(qa)
        return qa_id, inputs


def prepare_llava_next_video(model_name, 
                        model_path,
                        total_frames, 
                        temperature: float=1,
                        max_tokens: int=1024):
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                    max_tokens=max_tokens,
                                    stop_token_ids=stop_token_ids)
    if model_name == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        max_model_len = 8192
    elif model_name == "llava-hf/LLaVA-NeXT-Video-34B-hf":
        max_model_len = 4096
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    llm = LLM(model=model_path,
             max_model_len=max_model_len,
             limit_mm_per_prompt={"video": 1},
             tensor_parallel_size=min(torch.cuda.device_count(),4),
             )
    
    return llm, sampling_params

def prepare_qwen2(model_name, 
                model_path,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    
    llm = LLM(model=model_path,
              tensor_parallel_size=min(torch.cuda.device_count(),1),
              limit_mm_per_prompt={"video": 1})
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)
    
    return llm, sampling_params

def prepare_phi3v(model_name, 
                model_path,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": total_frames},
        tensor_parallel_size=min(torch.cuda.device_count(),4),
    )
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)

    return llm, sampling_params

def prepare_deepseek_vl2(model_name, 
                model_path,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    llm = LLM(model=model_path,
        max_model_len=4096,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": total_frames},
        tensor_parallel_size=min(torch.cuda.device_count(),4),
        
        )

    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)
    

    return llm, sampling_params

def prepare_aria(model_name, 
                model_path,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    inputs = []
    llm = LLM(model=model_path,
            tokenizer_mode="slow",
            trust_remote_code=True,
            limit_mm_per_prompt={"image": total_frames},
            tensor_parallel_size=min(torch.cuda.device_count(),4),
            )
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519,17]
    
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)

    return llm, sampling_params

def prepare_general_vlm(model_name, 
                    model_path,
                    total_frames, 
                    temperature: float=1,
                    max_tokens: int=1024):

    if "h2oai" in model_name:
        max_model_len=8192
    else:
        max_model_len=16384
    llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": total_frames},
            tensor_parallel_size=min(torch.cuda.device_count(),4),
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                trust_remote_code=True)
    if "h2oai" in model_name:
        stop_token_ids = [tokenizer.eos_token_id]
    else:
        stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     )

    return llm, sampling_params

def prepare_pixtral(model_name, 
                model_path,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    stop_token_ids = None
    llm = LLM(model=model_path, 
            max_model_len=8192,
            max_num_seqs=2,
            limit_mm_per_prompt={"image": total_frames}, 
            tensor_parallel_size=min(torch.cuda.device_count(),4),
            )
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)     
    return llm, sampling_params

def prepare_mllama(model_name, 
                model_path,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    if "11B" in model_name:
        llm = LLM(model=model_path,
            limit_mm_per_prompt={"image":total_frames},
            max_model_len=8192,
            max_num_seqs=2,
            enforce_eager=True,
            trust_remote_code=True,
            tensor_parallel_size=min(torch.cuda.device_count(),4),
        )
    else:
        llm = LLM(model=model_path,
            limit_mm_per_prompt={"image":total_frames},
            max_model_len=8192,
            max_num_seqs=2,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            enforce_eager=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
        )
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=[128001,128008,128009])
    return llm, sampling_params

def prepare_llava_onevision(model_name, 
                model_path,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    llm = LLM(model=model_path,
                max_model_len=32768,
                limit_mm_per_prompt={"video": 1},
                tensor_parallel_size=min(torch.cuda.device_count(),4),
                )  
    stop_token_ids = None  
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)    

    return llm, sampling_params


def prepare_kimivl(model_name, model_path, total_frames, temperature=1, max_tokens=1024):
    # Initialize vLLM model and processor
    llm = LLM(
        model_path,
        trust_remote_code=True,
        max_num_seqs=8,
        max_model_len=131072,
        limit_mm_per_prompt={"image": 256}
    )
    # Set sampling parameters
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    
    return llm, sampling_params

def prepare_kimivl_inputs(model_name, model_path, query, prompt, total_frames=-1):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    def _create_input(messages, video_data) -> dict:
        return {
            "prompt": processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt"),
            "multi_modal_data": {"image": video_data}
        }

    def _get_video_data(video_path, video_read_type, total_frames):
        return load_video_frames(video_path, video_read_type, total_frames)

    if prompt['type'] == 'mcq':
        inputs, qa_id = [], []
        video_data = _get_video_data(query['video_path'], query['video_read_type'], total_frames)
        
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            messages = [{"role": "user", "content": [{"type": "image", "image": ""} for _ in range(total_frames)] + [{"type": "text", "text":qa_text_prompt}]}]
            inputs.append(_create_input(messages, video_data))
            qa_id.append(qa)
        return qa_id, inputs

