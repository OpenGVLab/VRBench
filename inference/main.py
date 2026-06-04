from tqdm import tqdm
import json
import argparse
import os
import sys
import multiprocessing
from utils.constant import PROMPT, MODEL_PATH, MAX_TOKENS
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
    device_id: int = 0,
    max_tokens: int = MAX_TOKENS,
    )-> None:
    vllm_model_list_path = os.path.join(os.path.dirname(__file__), "model_inference", "vllm_model_list.json")
    if not os.path.exists(vllm_model_list_path):
        vllm_model_list_path = os.path.join(os.path.dirname(__file__), "../inference/model_inference/vllm_model_list.json")
    
    vllm_model_list = {'video': [], 'language': []}
    if os.path.exists(vllm_model_list_path):
        vllm_model_list = json.load(open(vllm_model_list_path))
    
    if model_name in vllm_model_list['video']:
        from model_inference.vllm_video_inference import generate_response
    elif "Qwen3-VL" in model_name or "qwen3-vl" in model_name.lower():
        from model_inference.qwen3vl import generate_response
    elif "InternVL3" in model_name:
        from model_inference.internvl3 import generate_response
    elif "InternVideo3" in model_name or "internvideo3" in model_name.lower():
        from model_inference.internvideo3 import generate_response
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
    elif "Keye-VL-1_5" in model_name or "Keye-VL-1.5" in model_name:
        from model_inference.keye_vl_1_5 import generate_response
    elif "Keye-VL" in model_name:
        from model_inference.keyevl import generate_response
    elif "GLM-4.1V" in model_name or "glm4v" in model_name.lower():
        from model_inference.glm4v import generate_response
    elif "MiniCPM-V" in model_name:
        from model_inference.minicpmv import generate_response
    elif "LLaVA-Video" in model_name or "llava-video" in model_name.lower():
        from model_inference.llava_video import generate_response
    elif "Molmo2" in model_name or "molmo2" in model_name.lower():
        from model_inference.molmo2 import generate_response
    elif "Eagle2.5" in model_name or "eagle2_5" in model_name.lower() or "eagle2.5" in model_name.lower():
        from model_inference.eagle2_5 import generate_response
    elif model_name in vllm_model_list['language']:
        from model_inference.vllm_language_inference import generate_response
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    generate_response(model_name=model_name,
                    model_path=model_path,
                    prompt=prompt,
                    queries=queries,
                    total_frames=total_frames,
                    output_path=output_path,
                    device_id=device_id,
                    max_tokens=max_tokens,
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
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use for parallel inference")
    parser.add_argument("--device_id", type=int, default=0,
                       help="GPU device ID (used when num_gpus=1 or for single GPU inference)")
    parser.add_argument("--partition", type=int, default=None,
                       help="Data partition index (0 for first half, 1 for second half). If None, process all data.")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                       help="Maximum number of tokens to generate")

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
    vllm_model_list = {'video': [], 'language': []}
    if os.path.exists(vllm_model_list_path):
        vllm_model_list = json.load(open(vllm_model_list_path))
    is_llm = model_name in vllm_model_list['language']
    
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
    partition_suffix = f"_partition{args.partition}" if args.partition is not None else ""
    if is_llm:
        # For LLM models, use simple text output name
        output_path = os.path.join(output_dir, f"{output_name}{partition_suffix}.jsonl")
    elif total_frames == -1:
        output_path = os.path.join(output_dir, f"{output_name}_1fps{partition_suffix}.jsonl")
    elif total_frames >= 1:
        output_path = os.path.join(output_dir, f"{output_name}_{total_frames}frame{partition_suffix}.jsonl")
    else:
        output_path = os.path.join(output_dir, f"{output_name}_text{partition_suffix}.jsonl")

    total_json_ls = [json.loads(line) for line in open(args.data_path, "r")]

    total_num = len(total_json_ls)

    exist_id = set()
    if os.path.exists(output_path):
        orig_output = [json.loads(line) for line in open(output_path, "r")]
        exist_id = set([long_vid['video_id'] for long_vid in orig_output])

    total_json_ls = [line for line in total_json_ls if line['video_id'] not in exist_id]
    
    if args.partition is not None:
        if args.partition not in [0, 1]:
            raise ValueError("Partition must be 0 or 1")
        partition_size = (len(total_json_ls) + 1) // 2
        if args.partition == 0:
            total_json_ls = total_json_ls[:partition_size]
            print(f"Using partition 0: first {len(total_json_ls)} videos")
        else:
            total_json_ls = total_json_ls[partition_size:]
            print(f"Using partition 1: last {len(total_json_ls)} videos")
      
    print(f"{total_num} VIDEOS IN TOTAL, {len(exist_id)} VIDEOS EXISTING , {len(total_json_ls)} VIDEOS LEFT.")
    print(f"=========Running {model_name} with {total_frames} frames=========\n")

    # Convert relative video paths to absolute paths
    for query in total_json_ls:
        if 'video_path' in query:
            video_path = query['video_path']
            if '/v001/' in video_path:
                video_path = video_path.replace('/v001/', '/')
            if video_path.startswith('VRBench/videos/'):
                video_path = video_path.replace('VRBench/videos/', 'videos/')
            query['video_path'] = os.path.join(args.video_root, video_path)
    
    if args.num_gpus > 1:
        num_gpus = min(args.num_gpus, 8)
        num_queries = len(total_json_ls)
        chunk_size = (num_queries + num_gpus - 1) // num_gpus
        
        def worker_process(gpu_id, queries_chunk, output_path_chunk):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            main(
                model_name=model_name,
                model_path=model_path,
                prompt=prompt,
                queries=queries_chunk,
                total_frames=total_frames,
                output_path=output_path_chunk,
                device_id=0,
                max_tokens=args.max_tokens,
            )
        
        processes = []
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * chunk_size
            end_idx = min((gpu_id + 1) * chunk_size, num_queries)
            if start_idx >= num_queries:
                break
            
            queries_chunk = total_json_ls[start_idx:end_idx]
            output_path_chunk = output_path.replace('.jsonl', f'_gpu{gpu_id}.jsonl')
            
            p = multiprocessing.Process(
                target=worker_process,
                args=(gpu_id, queries_chunk, output_path_chunk)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        output_files = [output_path.replace('.jsonl', f'_gpu{gpu_id}.jsonl') for gpu_id in range(num_gpus)]
        output_files = [f for f in output_files if os.path.exists(f)]
        
        if output_files:
            existing_lines = []
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    existing_lines = f.readlines()
            with open(output_path, 'w') as outfile:
                for line in existing_lines:
                    outfile.write(line)
                for fname in output_files:
                    with open(fname, 'r') as infile:
                        for line in infile:
                            outfile.write(line)
                    os.remove(fname)
            print(f"Merged results from {len(output_files)} GPU processes into {output_path}")
    else:
        main(
            model_name=model_name,
            model_path=model_path,
            prompt=prompt,
            queries=total_json_ls,
            total_frames=total_frames,
            output_path=output_path,
            device_id=args.device_id,
            max_tokens=args.max_tokens,
        )
