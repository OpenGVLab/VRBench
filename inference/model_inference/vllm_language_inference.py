from utils.llm_prepare_input import *
import json
import jsonlines
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from vllm import LLM, SamplingParams
import torch

model_map = {
    "Qwen/Qwen2.5-7B-Instruct": prepare_qwen2,
    "meta-llama/Llama-3.3-70B-Instruct": prepare_llama3,
}

def generate_response(model_name: str,   
                    model_path: str,             
                    prompt: str,
                    queries: list,
                    output_path: str,
                    total_frames: int = -1,  # Not used for LLM models
                    temperature: float=GENERATION_TEMPERATURE,
                    max_tokens: int=MAX_TOKENS):
    
    result_key = f"{prompt['type']}_result"

    llm, sampling_params = model_map[model_name](model_name, model_path, temperature, max_tokens)

    with jsonlines.open(output_path, 'a', flush=True) as f:
        for query in queries:
            output_dict = {"video_id": query['video_id'], result_key: {}}
            qa_id, inputs = prepare_general_llm_inputs(model_name, model_path, query, prompt)
            for idx, input in enumerate(inputs):
                responses = llm.chat([input], sampling_params=sampling_params)
                response = [response.outputs[0].text for response in responses]
                output_dict[result_key][qa_id[idx]] = response
            f.write(output_dict)
