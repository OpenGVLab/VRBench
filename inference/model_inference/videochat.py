import os, torch
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
import jsonlines
from transformers import AutoTokenizer, AutoModel
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import decord
from decord import VideoReader, cpu
from torchvision import transforms
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
from utils.video_process import download_video
import hashlib
import requests
from tqdm import tqdm


def get_videochat_flash(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    return model, tokenizer

def generate_by_videochat_flash(model_name, model_path, queries, prompt, total_frames, output_path, temperature, max_tokens):
    model, tokenizer = get_videochat_flash(model_path)
    max_num_frames = 512

    result_key = f"{prompt['type']}_result"
    
    with jsonlines.open(output_path, 'a', flush=True) as f:
        for query in tqdm(queries):
            output_dict = {"video_id": query['video_id'], result_key: {}}
            inputs, qa_id = [], []
            if prompt['type'] == 'mcq':
                for qa, qa_dict in query['mcq'].items():
                            _, qa_text_prompt1 = prepare_qa_text_input(
                                video_summary=query['video_summary'],
                                qa_dict=qa_dict,
                                prompt=prompt
                            )
                            inputs.append(qa_text_prompt1)
                            qa_id.append(qa)
                for idx, input in enumerate(inputs):
                    response = model.chat(
                        query['video_path'],
                        tokenizer,
                        input,
                        chat_history=None,
                        return_history=False,
                        max_num_frames=max_num_frames,
                        media_dict={'video_read_type': query['video_read_type']},
                        generation_config={
                            "max_new_tokens":max_tokens,
                            "temperature":temperature,
                            "do_sample":False,
                            "top_p":None,
                            "num_beams":1}
                        )
                    output_dict[result_key][qa_id[idx]] = response
                f.write(output_dict)
                
def generate_response(model_name: str, 
                    model_path:str,
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    temperature: float=GENERATION_TEMPERATURE,
                    max_tokens: int=MAX_TOKENS):
    generate_by_videochat_flash(model_name, 
                                model_path,
                                queries, 
                                prompt=prompt, 
                                total_frames=total_frames, 
                                output_path=output_path,
                                temperature=temperature, 
                                max_tokens=max_tokens)
