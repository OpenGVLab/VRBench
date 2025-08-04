import os, torch
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
import json
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from decord import VideoReader, cpu
import av
from PIL import Image
import numpy as np
import decord
from decord import VideoReader, cpu
from torchvision import transforms
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
import hashlib
import requests
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import jsonlines
import base64
from utils.video_process import load_video_frames


def prepare_keyevl_input(prompt_text, frames):
    """Prepare input for Keye-VL model"""
    # Convert frames to base64
    frame_base64_list = []
    for frame in frames:
        # Convert PIL image to base64
        import io
        buffer = io.BytesIO()
        frame.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        frame_base64_list.append(img_base64)
    
    # Create placeholders for images (using similar format to Kimi-VL)
    placeholders = "\n".join(f"<|image_{i}|>" for i, _ in enumerate(frame_base64_list, start=1))
    text_input = f"<|user|>\n{placeholders}\n{prompt_text}<|end|>\n<|assistant|>\n"
    
    return text_input, frame_base64_list

def generate_by_keyevl_hf(model_name, 
                       model_path,
                       queries, 
                       prompt, 
                       total_frames, 
                       output_path: str,
                       temperature, 
                       max_tokens,
                       ):
    """Generate responses using Keye-VL model with HuggingFace"""
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Try with AutoProcessor if available
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
            model.eval()
            tokenizer = processor.tokenizer
        except Exception as e2:
            print(f"Failed to load with AutoProcessor: {e2}")
            raise e
    
    result_key = f"{prompt['type']}_result" 
    queries_pbar = tqdm(queries, 
                       desc="Processing Videos", 
                       unit="video",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    with jsonlines.open(output_path, 'a', flush=True) as f:
        for query in queries_pbar:
            queries_pbar.set_postfix_str(f"Video ID: {query['video_id']}")
            output_dict = {"video_id": query['video_id'], result_key: {}}
            # Load video frames
            frames = load_video_frames(query['video_path'], query['video_read_type'], total_frames)
            if prompt['type'] == 'mcq':
                for qa, qa_dict in query['mcq'].items():
                    _, qa_text_prompt = prepare_qa_text_input(
                        video_summary=query['video_summary'],
                        qa_dict=qa_dict,
                        prompt=prompt
                    )
                    # Prepare input for Keye-VL
                    text_input, frame_base64_list = prepare_keyevl_input(qa_text_prompt, frames)
                    # Tokenize input
                    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
                    # Generate response
                    with torch.no_grad():
                        try:
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
                            )
                            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                            output_dict[result_key][qa] = [response]
                        except Exception as e:
                            print(f"Generation failed: {e}")
                            output_dict[result_key][qa] = ["Generation failed"]
            
            f.write(output_dict)
            queries_pbar.update(1)


def generate_response(model_name: str, 
                     model_path: str,
                     prompt: str,
                     queries: list,
                     total_frames: int,
                     output_path: str,
                     temperature: float=GENERATION_TEMPERATURE,
                     max_tokens: int=MAX_TOKENS):
    """Main function to generate responses using Keye-VL model"""

    generate_by_keyevl_hf(
        model_name=model_name,
        model_path=model_path,
        queries=queries,
        prompt=prompt,
        total_frames=total_frames,
        output_path=output_path,
        temperature=temperature,
        max_tokens=max_tokens,
    ) 