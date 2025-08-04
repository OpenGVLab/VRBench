from transformers import AutoTokenizer
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from vllm import LLM, SamplingParams
from utils.prepare_input import prepare_qa_text_input
from argparse import Namespace
from typing import List
import torch
from transformers import AutoProcessor, AutoTokenizer

from tqdm import tqdm
def prepare_general_llm_inputs(model_name, 
                            model_path,
                            query, 
                            prompt):
    
    def _create_input(qa_text_prompt):
        return [{"role": "user", "content": qa_text_prompt}]     
        
    inputs, qa_id = [], []
    if prompt['type'] == 'mcq':
        for qa, qa_dict in query['mcq'].items():
            _, qa_text_prompt = prepare_qa_text_input(
                video_summary=query['video_summary'],
                qa_dict=qa_dict,
                prompt=prompt
            )
            qa_id.append(qa)
            inputs.append(_create_input(qa_text_prompt))
        return qa_id, inputs

def prepare_qwen2(model_name, 
                model_path,
                temperature: float=1,
                max_tokens: int=1024):
    
    llm = LLM(model=model_path,
              tensor_parallel_size=min(torch.cuda.device_count(),4),
              trust_remote_code=True,
              )
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)
    return llm, sampling_params

def prepare_qwq(model_name, 
                model_path,
                temperature: float=1,
                max_tokens: int=1024):
    
    llm = LLM(model=model_path,
              tensor_parallel_size=min(torch.cuda.device_count(),2),
              trust_remote_code=True,
              disable_progress_bar=True,
              )
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)
    return llm, sampling_params

def prepare_internlm3(model_name, 
                model_path,
                temperature: float=1,
                max_tokens: int=1024):
    
    llm = LLM(model=model_path,
              tensor_parallel_size=min(torch.cuda.device_count(),4),
              trust_remote_code=True,
              disable_progress_bar=True,
              )
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)
    return llm, sampling_params

def prepare_llama3(model_name, 
                model_path,
                temperature: float=1,
                max_tokens: int=1024):
    
    llm = LLM(model=model_path,
              tensor_parallel_size=min(torch.cuda.device_count(),4),
              trust_remote_code=True,
              disable_progress_bar=True,
              )
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)
    return llm, sampling_params



