import os, torch
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
from transformers import AutoTokenizer, AutoModel
from decord import VideoReader, cpu
import av
from PIL import Image
import numpy as np
import decord
from decord import VideoReader, cpu
from torchvision import transforms
decord.bridge.set_bridge("torch")
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
from utils.video_process import download_video
import hashlib
import requests
from tqdm import tqdm
import jsonlines

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=16, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames
    
def generate_by_internvideo2_5(model_name, 
                            model_path,
                            queries, 
                            prompt, 
                            total_frames, 
                            output_path: str,
                            temperature, 
                            max_tokens):
    # model setting
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda()
    tokenizer =  AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    result_key = f"{prompt['type']}_result"

    with jsonlines.open(output_path, 'a', flush=True) as f:
        for query in tqdm(queries):
            output_dict = {"video_id": query['video_id'], result_key: {}}
            # sample uniformly 8 frames from the video
            video_tensor = load_video(query['video_path'], num_segments=total_frames, return_msg=False)
            
            text_inputs, qa_id = [], []
            if prompt['type'] == 'mcq':
                for qa, qa_dict in query['one_step_mcq'].items():
                    _, qa_text_prompt = prepare_qa_text_input(
                        video_summary=query['video_summary'],
                        qa_dict=qa_dict,
                        prompt=prompt
                    )
                    text_inputs.append(qa_text_prompt)
                    qa_id.append(qa)
            text_inputs = [f"{q}" for q in text_inputs]
            
            video_tensor = video_tensor.to(model.device)
            inputs = [{
                    "prompt": text_input,
                    "multi_modal_data": {
                        "video": video_tensor
                    },
                } for text_input in text_inputs]
            
            for idx, input in enumerate(inputs):
                response = model.chat(tokenizer, '', input["prompt"], media_type='video', media_tensor=video_tensor, generation_config={'do_sample':False})
                output_dict[result_key][qa_id[idx]] = response
                del video_tensor
                torch.cuda.empty_cache()

def generate_response(model_name: str, 
                    model_path:str,
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    temperature: float=GENERATION_TEMPERATURE,
                    max_tokens: int=MAX_TOKENS):
    

    generate_by_internvideo2_5(model_name, 
                                model_path,
                                queries, 
                                prompt=prompt, 
                                total_frames=total_frames, 
                                output_path=output_path,
                                temperature=GENERATION_TEMPERATURE, 
                                max_tokens=MAX_TOKENS)
