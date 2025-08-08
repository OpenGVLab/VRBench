from tqdm import tqdm
import json
import argparse
import os
import sys
from utils.constant import PROMPT, MODEL_PATH
from utils.model_config import calculate_optimal_frames, get_model_info
from utils.precise_token_calculation import calculate_optimal_frames_precise
from utils.video_process import get_duration
from transformers.utils import logging

logging.set_verbosity_error() 

def get_adaptive_frames(model_name: str, model_path: str, total_frames: int, data_path: str = None, use_precise_calculation: bool = True) -> int:
    """
    Get adaptive frame count based on model capabilities and content
    
    Args:
        model_name: Name of the model
        model_path: Path to the model
        total_frames: User-specified frames (if not default)
        data_path: Path to data file for content analysis
        use_precise_calculation: Whether to use precise token calculation
    
    Returns:
        Optimal number of frames
    """
    
    # If user manually specified frames, use them
    if total_frames != -1:
        print(f"Using user-specified frames: {total_frames}")
        return total_frames
    
    print(f"Calculating adaptive frames for model: {model_name}")
    
    # Use precise calculation if available
    if use_precise_calculation:
        try:
            optimal_frames = calculate_optimal_frames_precise(model_path)
            print(f"Precise calculation result: {optimal_frames} frames")
            return optimal_frames
        except Exception as e:
            print(f"Precise calculation failed: {e}")
            print("Falling back to approximate calculation...")
    
    # Fallback to approximate calculation
    try:
        optimal_frames = calculate_optimal_frames(model_name, model_path, data_path)
        print(f"Approximate calculation result: {optimal_frames} frames")
        return optimal_frames
    except Exception as e:
        print(f"Approximate calculation failed: {e}")
        print("Using default frames: 16")
        return 16

def main(
    model_name: str,
    model_path: str, 
    prompt: str, 
    queries: list, 
    total_frames: int, 
    output_path: str, 
    )-> None:
    vllm_model_list_path = os.path.join(os.path.dirname(__file__), "model_inference", "vllm_model_list.json")
    if not os.path.exists(vllm_model_list_path):
        vllm_model_list_path = os.path.join(os.path.dirname(__file__), "../inference/model_inference/vllm_model_list.json")
    
    if model_name in json.load(open(vllm_model_list_path))['video']:
        from model_inference.vllm_video_inference import generate_response
    elif "InternVideo2_5" in model_name:
        from model_inference.internvideo2_5 import generate_response
    elif "InternVideo2" in model_name:
        from model_inference.internvideo import generate_response
    elif "VideoLLaMA2" in model_name:
        from model_inference.videollama2 import generate_response
    elif "VideoChat" in model_name:
        from model_inference.videochat import generate_response
    elif "VideoLLaMA3" in model_name:
        from model_inference.videollama3 import generate_response
    elif "Keye-VL" in model_name:
        from model_inference.keyevl import generate_response
    elif model_name in json.load(open(vllm_model_list_path))['language']:
        from model_inference.vllm_language_inference import generate_response
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    generate_response(model_name=model_name,
                    model_path=model_path,
                    prompt=prompt,
                    queries=queries, 
                    total_frames=total_frames, 
                    output_path=output_path,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="single_round")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--video_root', type=str, required=True, help="Root directory for video files")
    parser.add_argument('--total_frames', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument("--api_base",type=str,default="")
    parser.add_argument("--use_precise_calculation", action="store_true", default=True, 
                       help="Use precise token calculation with actual tokenizer")
    parser.add_argument("--use_approximate_calculation", action="store_true", default=False,
                       help="Use approximate calculation (fallback)")

    args = parser.parse_args()
    
    model_name = args.model
    try:
        model_path = MODEL_PATH[model_name]
    except KeyError as e:
        print(f"{model_name} local path not found!")
        sys.exit(1)
    
    # Determine calculation method
    use_precise = args.use_precise_calculation and not args.use_approximate_calculation
    
    # Check if model is LLM (language model)
    vllm_model_list_path = os.path.join(os.path.dirname(__file__), "model_inference", "vllm_model_list.json")
    if not os.path.exists(vllm_model_list_path):
        vllm_model_list_path = os.path.join(os.path.dirname(__file__), "../inference/model_inference/vllm_model_list.json")
    is_llm = model_name in json.load(open(vllm_model_list_path))['language']
    
    if is_llm:
        # For LLM models, skip frame calculation and use text output
        total_frames = -1
        print(f"LLM model detected: {model_name}, skipping frame calculation")
    else:
        # Calculate adaptive frames for non-LLM models
        total_frames = get_adaptive_frames(
            model_name, 
            model_path, 
            args.total_frames, 
            args.data_path,
            use_precise_calculation=use_precise
        )

    try:
        prompt = PROMPT[args.prompt]
    except KeyError:
        print("Invalid prompt")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir
    
    output_name = model_name.split("/")[-1]
    if is_llm:
        # For LLM models, use simple text output name
        output_path = os.path.join(output_dir, f"{output_name}.jsonl")
    elif total_frames == -1:
        output_path = os.path.join(output_dir, f"{output_name}_1fps.jsonl")
    elif total_frames >= 1:
        output_path = os.path.join(output_dir, f"{output_name}_{total_frames}frame.jsonl")
    else:
        output_path = os.path.join(output_dir, f"{output_name}_text.jsonl")

    total_json_ls = [json.loads(line) for line in open(args.data_path, "r")]

    total_num = len(total_json_ls)

    exist_id = set()
    if os.path.exists(output_path):
        orig_output = [json.loads(line) for line in open(output_path, "r")]
        exist_id = set([long_vid['video_id'] for long_vid in orig_output])

    total_json_ls = [line for line in total_json_ls if line['video_id'] not in exist_id]
      
    print(f"{total_num} VIDEOS IN TOTAL, {len(exist_id)} VIDEOS EXISTING , {len(total_json_ls)} VIDEOS LEFT.")
    print(f"=========Running {model_name} with {total_frames} frames=========\n")

    # Convert relative video paths to absolute paths
    for query in total_json_ls:
        if 'video_path' in query:
            query['video_path'] = os.path.join(args.video_root, query['video_path'])
    
    main(
        model_name = model_name, 
        model_path = model_path,
        prompt = prompt, 
        queries = total_json_ls, 
        total_frames = total_frames, 
        output_path = output_path, 
        )
