from utils.video_process import download_video
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
from utils.prepare_input import prepare_qa_text_input
from tqdm import tqdm
import jsonlines
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

def get_videollama3(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor

def create_input(qa_text_prompt, processor, video_path):
    text_input = f"{qa_text_prompt}"
            
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 180}},
                {"type": "text", "text": text_input},
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    return inputs
    
def generate_response(input, model, processor):
    output_ids = model.generate(
                **input,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                max_new_tokens=2048,
            )
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

def generate_by_videollama3(model_name, 
                            model_path,
                            queries, 
                            prompt, 
                            total_frames, 
                            output_path: str,
                            temperature, 
                            max_tokens):

    model, processor = get_videollama3(model_path)
    result_key = f"{prompt['type']}_result"

    with jsonlines.open(output_path, 'a', flush=True) as f:
        for idx, query in enumerate(tqdm(queries)):
            output_dict = {"video_id": query['video_id'], result_key: {}}
            inputs, qa_id = [], []
            if prompt['type'] == 'mcq':
                for qa, qa_dict in query['mcq'].items():
                    _, qa_text_prompt = prepare_qa_text_input(
                            model_name=model_name,
                            video_summary=query['video_summary'],
                            qa_dict=qa_dict,
                            round=1,
                            prompt=prompt
                        )
                    qa_id.append(qa)
                    inputs.append(create_input(qa_text_prompt, processor, query['video_path']))
            for idx, input in enumerate(inputs):
                response = generate_response(input, model, processor)
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
    generate_by_videollama3(model_name, 
                            model_path,
                            queries, 
                            prompt=prompt, 
                            total_frames=total_frames, 
                            output_path=output_path,
                            temperature=temperature, 
                            max_tokens=max_tokens)