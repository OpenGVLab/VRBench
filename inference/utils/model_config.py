from transformers import AutoConfig, AutoTokenizer
import os
import json
from typing import Dict, Optional, Tuple

def get_model_max_length(model_path: str) -> int:
    """Get model's maximum context length from config"""
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Try different config fields
        if hasattr(config, 'max_position_embeddings'):
            return config.max_position_embeddings
        elif hasattr(config, 'model_max_length'):
            return config.model_max_length
        elif hasattr(config, 'max_sequence_length'):
            return config.max_sequence_length
        elif hasattr(config, 'context_length'):
            return config.context_length
        else:
            # Fallback to name-based inference
            return infer_max_length_from_name(model_path)
    except Exception as e:
        print(f"Warning: Failed to get config for {model_path}: {e}")
        return infer_max_length_from_name(model_path)

def infer_max_length_from_name(model_path: str) -> int:
    """Infer max length from model name when config is not available"""
    model_name = model_path.lower()
    
    # Based on existing configurations in the codebase
    if "llava-onevision" in model_name:
        return 32768
    elif "llava-next" in model_name:
        if "34b" in model_name:
            return 4096
        else:
            return 8192
    elif "deepseek" in model_name:
        return 4096
    elif "h2oai" in model_name:
        return 8192
    elif "pixtral" in model_name or "mllama" in model_name:
        return 8192
    elif "llava" in model_name:
        return 8192
    else:
        return 16384  # Default for most models

def get_tokens_per_frame(model_name: str) -> int:
    """Get estimated tokens consumed per frame for different models"""
    model_name_lower = model_name.lower()
    
    # Different models have different token consumption per frame
    if "llava-onevision" in model_name_lower:
        return 256  # Video models typically use fewer tokens per frame
    elif "qwen" in model_name_lower and "vl" in model_name_lower:
        return 512  # Qwen VL models
    elif "phi" in model_name_lower:
        return 256  # Phi models
    elif "deepseek" in model_name_lower:
        return 256  # DeepSeek models
    elif "internvl" in model_name_lower:
        return 512  # InternVL models
    elif "kimi" in model_name_lower:
        return 256  # Kimi models
    else:
        return 512  # Default for general VLM models

def estimate_text_tokens(video_summary: str, question: str, options: Dict[str, str], prompt_template: str) -> int:
    """Estimate token count for text content"""
    # Combine all text content
    full_text = f"{prompt_template}\n{video_summary}\n{question}\n"
    for key, value in options.items():
        full_text += f"{key}: {value}\n"
    
    # Rough estimation: 1 token ≈ 4 characters for English text
    estimated_tokens = len(full_text) // 4
    
    # Add some buffer for special tokens and formatting
    return estimated_tokens + 200

def calculate_optimal_frames(
    model_name: str, 
    model_path: str, 
    video_summary: str = "", 
    question: str = "", 
    options: Dict[str, str] = None,
    prompt_template: str = "",
    video_duration: float = None,
    reserve_tokens: int = 1024
) -> int:
    """
    Calculate optimal frame count based on model capacity and content
    
    Args:
        model_name: Name of the model
        model_path: Path to the model
        video_summary: Video summary text
        question: Question text
        options: Multiple choice options
        prompt_template: Prompt template
        video_duration: Video duration in seconds
        reserve_tokens: Tokens to reserve for output and special tokens
    
    Returns:
        Optimal number of frames
    """
    
    # Get model's maximum context length
    max_model_len = get_model_max_length(model_path)
    
    # Get tokens per frame for this model
    tokens_per_frame = get_tokens_per_frame(model_name)
    
    # Estimate text token consumption
    if options is None:
        options = {"A": "", "B": "", "C": "", "D": ""}
    text_tokens = estimate_text_tokens(video_summary, question, options, prompt_template)
    
    # Calculate available tokens for frames
    available_tokens = max_model_len - text_tokens - reserve_tokens
    
    # Calculate maximum possible frames
    max_possible_frames = max(1, available_tokens // tokens_per_frame)
    
    # Apply reasonable limits based on model type
    if "llava-onevision" in model_name.lower():
        max_frames = min(max_possible_frames, 64)
    elif "qwen" in model_name.lower() and "vl" in model_name.lower():
        max_frames = min(max_possible_frames, 32)
    elif "phi" in model_name.lower():
        max_frames = min(max_possible_frames, 16)
    elif "deepseek" in model_name.lower():
        max_frames = min(max_possible_frames, 8)
    else:
        max_frames = min(max_possible_frames, 32)
    
    # If video duration is available, consider it
    if video_duration:
        # Aim for 2 frames per second, but don't exceed model capacity
        duration_based_frames = min(int(video_duration * 2), max_frames)
        max_frames = max(4, duration_based_frames)
    
    # Ensure minimum frames
    max_frames = max(4, max_frames)
    
    return max_frames

def get_model_info(model_name: str, model_path: str) -> Dict[str, any]:
    """Get comprehensive model information for debugging"""
    max_length = get_model_max_length(model_path)
    tokens_per_frame = get_tokens_per_frame(model_name)
    
    return {
        "model_name": model_name,
        "max_context_length": max_length,
        "tokens_per_frame": tokens_per_frame,
        "estimated_max_frames": max_length // tokens_per_frame
    } 