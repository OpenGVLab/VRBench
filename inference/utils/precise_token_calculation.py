#!/usr/bin/env python3
"""
Precise token calculation using actual model tokenizers
"""

import os
import sys
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig
import torch

# Import the maximum content constants
from .max_content_constants import (
    MAX_VIDEO_SUMMARY, 
    MAX_QUESTION, 
    MAX_OPTIONS,
    MAX_VIDEO_SUMMARY_LENGTH,
    MAX_QUESTION_LENGTH,
    MAX_OPTIONS_LENGTH
)

def get_model_tokenizer(model_path: str) -> Optional[AutoTokenizer]:
    """Get the tokenizer for a specific model"""
    try:
        # Try to load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_fast=False  # Use slow tokenizer for better compatibility
        )
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer for {model_path}: {e}")
        return None

def calculate_text_tokens_with_tokenizer(
    tokenizer: AutoTokenizer,
    video_summary: str = None,
    question: str = None,
    options: Dict = None,
    prompt_template: str = None
) -> int:
    """
    Calculate exact token count using the model's tokenizer
    
    Args:
        tokenizer: The model's tokenizer
        video_summary: Video summary text (uses max if None)
        question: Question text (uses max if None)
        options: Options dict (uses max if None)
        prompt_template: Prompt template (uses default if None)
    
    Returns:
        Total token count
    """
    
    # Use maximum content if not provided
    if video_summary is None:
        video_summary = MAX_VIDEO_SUMMARY
    if question is None:
        question = MAX_QUESTION
    if options is None:
        options = MAX_OPTIONS
    if prompt_template is None:
        # Default prompt template
        prompt_template = """You are a helpful assistant. Please answer the following question based on the video summary provided.

Video Summary:
{video_summary}

Question: {question}

Options:
{options}

Please provide your answer and reasoning:"""

    # Construct the full text
    options_text = ""
    for key, value in options.items():
        options_text += f"{key}: {value}\n"
    
    full_text = prompt_template.format(
        video_summary=video_summary,
        question=question,
        options=options_text
    )
    
    # Tokenize the text
    try:
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        print(f"Warning: Tokenization failed: {e}")
        # Fallback to character-based estimation
        return len(full_text) // 4

def calculate_optimal_frames_precise(
    model_path: str,
    max_context_length: int = None,
    reserved_tokens: int = 1024,
    tokens_per_frame: int = None
) -> int:
    """
    Calculate optimal frame count using precise token calculation
    
    Args:
        model_path: Path to the model
        max_context_length: Maximum context length (auto-detected if None)
        reserved_tokens: Tokens reserved for output and special tokens
        tokens_per_frame: Tokens per frame (auto-detected if None)
    
    Returns:
        Optimal number of frames
    """
    
    print(f"Calculating optimal frames for: {model_path}")
    
    # Get model tokenizer
    tokenizer = get_model_tokenizer(model_path)
    if tokenizer is None:
        print("Warning: Could not load tokenizer, using fallback calculation")
        return 16  # Fallback to default
    
    # Get max context length if not provided
    if max_context_length is None:
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(config, 'max_position_embeddings'):
                max_context_length = config.max_position_embeddings
            elif hasattr(config, 'model_max_length'):
                max_context_length = config.model_max_length
            else:
                # Fallback based on model name
                if "7b" in model_path.lower() or "7b" in model_path.lower():
                    max_context_length = 32768
                elif "13b" in model_path.lower():
                    max_context_length = 32768
                else:
                    max_context_length = 8192
        except Exception as e:
            print(f"Warning: Could not detect max context length: {e}")
            max_context_length = 8192
    
    # Get tokens per frame if not provided
    if tokens_per_frame is None:
        tokens_per_frame = get_tokens_per_frame_by_model(model_path)
    
    # Calculate text tokens using actual tokenizer
    text_tokens = calculate_text_tokens_with_tokenizer(tokenizer)
    
    print(f"Max context length: {max_context_length}")
    print(f"Text tokens: {text_tokens}")
    print(f"Reserved tokens: {reserved_tokens}")
    print(f"Tokens per frame: {tokens_per_frame}")
    
    # Calculate available tokens for frames
    available_tokens = max_context_length - text_tokens - reserved_tokens
    
    if available_tokens <= 0:
        print(f"Warning: No tokens available for frames (available: {available_tokens})")
        return 4  # Minimum frames
    
    # Calculate optimal frames
    optimal_frames = available_tokens // tokens_per_frame
    
    # Apply reasonable limits
    optimal_frames = max(4, min(optimal_frames, 512))  # Between 4 and 512 frames
    
    print(f"Available tokens for frames: {available_tokens}")
    print(f"Optimal frames: {optimal_frames}")
    
    return optimal_frames

def get_tokens_per_frame_by_model(model_path: str) -> int:
    """Get tokens per frame based on model type"""
    
    model_name = model_path.lower()
    
    # Model-specific token consumption
    if "qwen" in model_name:
        if "vl" in model_name:
            return 512  # Qwen VL models
        else:
            return 256  # Regular Qwen models
    elif "llava" in model_name:
        return 256  # LLaVA models
    elif "internvl" in model_name or "internvideo" in model_name:
        return 256  # InternVL models
    elif "phi" in model_name:
        return 256  # Phi models
    elif "deepseek" in model_name:
        return 256  # DeepSeek models
    elif "kimi" in model_name:
        return 256  # Kimi models
    else:
        return 256  # Default

def test_precise_calculation():
    """Test the precise calculation with different models"""
    
    test_models = [
        "Qwen/Qwen2-VL-7B-Instruct",
        "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
        "microsoft/Phi-3.5-vision-instruct",
        "deepseek-ai/deepseek-vl2"
    ]
    
    print("=" * 80)
    print("PRECISE TOKEN CALCULATION TEST")
    print("=" * 80)
    
    for model in test_models:
        print(f"\nTesting model: {model}")
        print("-" * 50)
        
        try:
            optimal_frames = calculate_optimal_frames_precise(model)
            print(f"Result: {optimal_frames} frames")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_precise_calculation() 