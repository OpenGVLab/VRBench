from utils.vlm_prepare_input import *
import json
import jsonlines
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm


model_map = {
    "llava-hf/LLaVA-NeXT-Video-7B-hf": prepare_llava_next_video,
    "llava-hf/LLaVA-NeXT-Video-34B-hf": prepare_llava_next_video,
    "Qwen/Qwen2-VL-7B-Instruct": prepare_qwen2, 
    "Qwen/Qwen2-VL-2B-Instruct": prepare_qwen2,
    "Qwen/Qwen2-VL-72B-Instruct-AWQ": prepare_qwen2, 
    "microsoft/Phi-3.5-vision-instruct": prepare_phi3v, 
    "OpenGVLab/InternVL2-8B": prepare_general_vlm,
    "OpenGVLab/InternVL2_5-78B-AWQ": prepare_general_vlm,
    "OpenGVLab/InternVL2_5-8B": prepare_general_vlm,
    "OpenGVLab/InternVL2_5-38B":prepare_general_vlm,
    "mistral-community/pixtral-12b": prepare_pixtral,
    "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": prepare_llava_onevision,
    "unsloth/Llama-3.2-11B-Vision-Instruct": prepare_mllama,
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit": prepare_mllama,
    "h2oai/h2ovl-mississippi-2b": prepare_general_vlm,
    "nvidia/NVLM-D-72B": prepare_general_vlm,
    "HuggingFaceM4/Idefics3-8B-Llama3": prepare_general_vlm,
    "deepseek-ai/deepseek-vl2": prepare_deepseek_vl2,
    "deepseek-ai/deepseek-vl2-tiny": prepare_deepseek_vl2,
    "deepseek-ai/deepseek-vl2-small": prepare_deepseek_vl2,
    "rhymes-ai/Aria-Chat": prepare_aria,
    "Qwen/Qwen2.5-VL-3B-Instruct": prepare_qwen2,
    "Qwen/Qwen2.5-VL-7B-Instruct": prepare_qwen2,
    "moonshotai/Kimi-VL-A3B-Thinking-2506": prepare_kimivl,
    "MiMo-VL-7B-RL":prepare_qwen2,
}

model_input_map = {
    "llava-hf/LLaVA-NeXT-Video-7B-hf": prepare_llava_next_video_inputs,
    "llava-hf/LLaVA-NeXT-Video-34B-hf": prepare_llava_next_video_inputs,
    "Qwen/Qwen2-VL-7B-Instruct": prepare_qwen2_inputs, 
    "Qwen/Qwen2-VL-2B-Instruct": prepare_qwen2_inputs,
    "Qwen/Qwen2-VL-72B-Instruct-AWQ": prepare_qwen2_inputs, 
    "microsoft/Phi-3.5-vision-instruct": prepare_phi3v_inputs, 
    "OpenGVLab/InternVL2-8B": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-78B-AWQ": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-8B": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-38B":prepare_general_vlm_inputs,
    "mistral-community/pixtral-12b": prepare_pixtral_inputs,
    "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": prepare_llava_onevision_inputs,
    "unsloth/Llama-3.2-11B-Vision-Instruct": prepare_mllama_inputs,
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit": prepare_mllama_inputs,
    "h2oai/h2ovl-mississippi-2b": prepare_general_vlm_inputs,
    "nvidia/NVLM-D-72B": prepare_general_vlm_inputs,
    "HuggingFaceM4/Idefics3-8B-Llama3": prepare_general_vlm_inputs,
    "deepseek-ai/deepseek-vl2": prepare_deepseek_vl2_inputs,
    "deepseek-ai/deepseek-vl2-tiny": prepare_deepseek_vl2_inputs,
    "deepseek-ai/deepseek-vl2-small": prepare_deepseek_vl2_inputs,
    "rhymes-ai/Aria-Chat": prepare_aria_inputs,
    "Qwen/Qwen2.5-VL-3B-Instruct": prepare_qwen2_inputs,
    "Qwen/Qwen2.5-VL-7B-Instruct": prepare_qwen2_inputs,
    "moonshotai/Kimi-VL-A3B-Thinking-2506": prepare_kimivl_inputs,
    "MiMo-VL-7B-RL":prepare_qwen2_inputs,
}

def generate_response(model_name: str,   
                    model_path: str,             
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    temperature: float=GENERATION_TEMPERATURE,
                    max_tokens: int=MAX_TOKENS):
    if model_name not in model_map:
        raise ValueError(f"Model type {model_name} is not supported.")
    
    result_key = f"{prompt['type']}_result"

    llm, sampling_params = model_map[model_name](model_name, model_path, total_frames, temperature, max_tokens)
    
    queries_pbar = tqdm(queries, 
                       desc="Processing Videos", 
                       unit="video",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    with jsonlines.open(output_path, 'a', flush=True) as f:
        for query in queries_pbar:
            queries_pbar.set_postfix_str(f"Video ID: {query['video_id']}")
            
            output_dict = {"video_id": query['video_id'], result_key: {}}
            
            qa_id, inputs = model_input_map[model_name](model_name, model_path, query, prompt, total_frames=total_frames)
            
            responses_pbar = tqdm(enumerate(inputs), 
                            total=len(inputs),
                            desc="Generating Responses",
                            unit="query",
                            leave=False, 
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            
            for idx, input in responses_pbar:
                responses = llm.generate(input, sampling_params=sampling_params, use_tqdm=False)
                response = [response.outputs[0].text for response in responses]
                output_dict[result_key][qa_id[idx]] = response
                responses_pbar.set_postfix_str(f"QA ID: {qa_id[idx]}")
                        
            f.write(output_dict)
            queries_pbar.update(1)