import os, sys
sys.path.append('VideoLLaMA2')
from videollama2 import model_init, mm_infer
# from videollama2.utils import disable_torch_init
# from utils.video_process import download_video
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
import hashlib
import requests
from tqdm import tqdm
import jsonlines

def get_videollama2(model_name):
    model, processor, tokenizer = model_init(model_name)
    return model, processor, tokenizer

def generate_response(model, processor, tokenizer, video_path, qa_text_prompt, modal='video'):
    # Handle single-turn Q&A
    input_data = {
        "prompt": qa_text_prompt,
        "multi_modal_data": {"video": video_path}
    }
    response = mm_infer(
        processor[modal](input_data['multi_modal_data']['video']),
        input_data['prompt'],
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        modal=modal
    )
    return response

def generate_by_videollama2(model_name, 
                                    model_path,
                                    queries, 
                                    prompt, 
                                    total_frames, 
                                    output_path: str,
                                    temperature, 
                                    max_tokens):
    model, processor, tokenizer = get_videollama2(model_name)
    result_key = f"{prompt['type']}_result"
    with jsonlines.open(output_path, 'a', flush=True) as f:
        for query in tqdm(queries):
            output_dict = {"video_id": query['video_id'], result_key: {}}
            inputs, qa_id = [], []
            if prompt['type'] == 'mcq':
                for qa, qa_dict in query['mcq'].items():
                    _, qa_text_prompt = prepare_qa_text_input(
                        video_summary=query['video_summary'],
                        qa_dict=qa_dict,
                        prompt=prompt
                    )
                    inputs.append(qa_text_prompt)
                    qa_id.append(qa)
                for idx, input in enumerate(inputs):        
                    response = generate_response(model, processor, tokenizer, query['video_path'], input)
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
    generate_by_videollama2(model_name, 
                                model_path,
                                queries, 
                                prompt=prompt, 
                                total_frames=total_frames, 
                                output_path=output_path,
                                temperature=temperature, 
                                max_tokens=max_tokens)
    