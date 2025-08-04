from utils.video_process import read_video, prepare_base64frames, prepare_base64_video
import requests
import os
import time
from tqdm import tqdm
import hashlib
import base64
import json

def dict_to_text(question, options):
    option_prompt = "\n".join(f"{k}: {v}" for k, v in options.items())
    return f"Question: {question}\nOptions:\n{option_prompt}"

def get_previous_reasoning(idx, previous_steps):
    """Generate reasoning text from previous steps."""
    if idx == 0:
        return ""
    return "\n".join(
        f"question: {step['question']}\nanswer: {step['options'][step['correct']]}"
        for step in previous_steps
    )


def prepare_qa_text_input(video_summary, qa_dict, prompt):
    if prompt["type"] == "mcq":
        qa_text_prompt = prompt["content"].substitute(
            multiple_choice_question=dict_to_text(qa_dict["question"], qa_dict["options"]),
            video_summary=video_summary
        )
        return {"type": "text", "text": qa_text_prompt}, qa_text_prompt
    else:
        raise ValueError(f"Invalid question type: {prompt['type']}")

def prepare_multi_image_input(model_name, video_path, total_frames, video_tmp_dir = "video_cache", video_read_type="decord"):
    base64frames = prepare_base64frames(model_name, video_path, total_frames, video_tmp_dir = video_tmp_dir, video_read_type=video_read_type)

    # for vllm models
    if model_name in json.load(open("model_inference/vllm_model_list.json"))['video']:
        return base64frames
    else:
        return [
            {
                "type": "image_url",
                'image_url': {
                    "url": f"data:image/jpeg;base64,{frame}",
                },
            } for frame in base64frames
        ]
