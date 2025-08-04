#!/usr/bin/env python3
"""
VRBench batch scoring calculation script
Batch calculate MCQ accuracy and open-ended QA average scores for multiple models and generate comparison tables
"""

import os
import json
import argparse
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

def find_model_files(inference_dir, eval_dir):
    """
    Automatically find inference result files and corresponding evaluation result files
    """
    inference_files = {}
    eval_files = {}
    
    # Find inference result files
    if os.path.exists(inference_dir):
        for file in os.listdir(inference_dir):
            if file.endswith('.jsonl'):
                # Extract model base name (remove frame info)
                model_name = file.replace('.jsonl', '')
                # Remove frame info like _8frame, _25frame etc.
                import re
                model_base_name = re.sub(r'_\d+frame$', '', model_name)
                inference_files[model_base_name] = os.path.join(inference_dir, file)
    
    # Find evaluation result files
    if os.path.exists(eval_dir):
        for file in os.listdir(eval_dir):
            if file.endswith('_evaluation.jsonl'):
                model_name = file.replace('_evaluation.jsonl', '')
                eval_files[model_name] = os.path.join(eval_dir, file)
    
    # Find models with both inference and evaluation results
    common_models = set(inference_files.keys()) & set(eval_files.keys())
    
    model_files = {}
    for model in common_models:
        model_files[model] = {
            'inference': inference_files[model],
            'evaluation': eval_files[model]
        }
    
    return model_files

def calculate_single_model_score(model_name, inference_file, eval_file, ground_truth_file, output_dir):
    """
    Calculate scores for a single model
    """
    output_file = os.path.join(output_dir, f"{model_name}_scores.json")
    
    cmd = [
        sys.executable, "calculate_scores.py",
        "--ground_truth_file", ground_truth_file,
        "--inference_file", inference_file,
        "--evaluation_file", eval_file,
        "--output_file", output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            # Read generated result file
            with open(output_file, 'r', encoding='utf-8') as f:
                scores = json.load(f)
            return scores
        else:
            print(f"Failed to calculate scores for {model_name}: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        return None

def generate_comparison_table(all_scores, output_file):
    """
    Generate model comparison table
    """
    if not all_scores:
        print("No valid scoring data")
        return
    
    # Get all question categories
    all_categories = set()
    for scores in all_scores.values():
        all_categories.update(scores['category_scores'].keys())
    all_categories = sorted(all_categories)
    
    # Generate comparison table
    comparison = {
        'models': list(all_scores.keys()),
        'overall_comparison': {},
        'category_comparison': {}
    }
    
    # Overall comparison
    for model_name, scores in all_scores.items():
        comparison['overall_comparison'][model_name] = scores['overall_scores']
    
    # Category comparison
    for category in all_categories:
        comparison['category_comparison'][category] = {}
        for model_name, scores in all_scores.items():
            if category in scores['category_scores']:
                comparison['category_comparison'][category][model_name] = scores['category_scores'][category]
            else:
                comparison['category_comparison'][category][model_name] = {
                    'mcq_accuracy': 0.0,
                    'openqa_average': 0.0,
                    'combined_score': 0.0
                }
    
    # Save comparison results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    return comparison

def print_comparison_table(comparison):
    """
    Print comparison table
    """
    print("\n" + "="*80)
    print("VRBench Model Scoring Comparison")
    print("="*80)
    
    # Overall comparison
    print("\nOverall Score Comparison:")
    print("-"*80)
    print(f"{'Model Name':<30} {'MCQ Accuracy':<15} {'Open-ended QA Avg':<15} {'MCQ Stats':<15}")
    print("-"*80)
    
    for model_name in comparison['models']:
        overall = comparison['overall_comparison'][model_name]
        print(f"{model_name:<30} {overall['mcq_accuracy']:<15.2f}% {overall['openqa_average']:<15.2f}% {overall['mcq_stats']:<15}")
    
    # Category comparison
    print("\nCombined Scores by Question Category:")
    print("-"*80)
    header = f"{'Category':<25}"
    for model_name in comparison['models']:
        header += f"{model_name:<15}"
    print(header)
    print("-"*80)
    
    for category, category_scores in comparison['category_comparison'].items():
        row = f"{category:<25}"
        for model_name in comparison['models']:
            score = category_scores[model_name]['combined_score']
            row += f"{score:<15.2f}"
        print(row)
    
    print("-"*80)

def main():
    parser = argparse.ArgumentParser(description='Batch calculate VRBench scores')
    parser.add_argument('--ground_truth_file', type=str, 
                       default='../inference/data/VRBench_eval.jsonl',
                       help='Original data file path')
    parser.add_argument('--inference_dir', type=str, 
                       default='../inference/outputs',
                       help='Inference results file directory')
    parser.add_argument('--eval_dir', type=str, 
                       default='eval_outputs',
                       help='Evaluation results file directory')
    parser.add_argument('--output_dir', type=str, 
                       default='eval_outputs',
                       help='Output results directory')
    parser.add_argument('--models', type=str, nargs='*',
                       help='Specify model names to process (optional)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Finding model files...")
    model_files = find_model_files(args.inference_dir, args.eval_dir)
    
    if not model_files:
        print("No matching inference and evaluation result files found")
        print(f"Inference results directory: {args.inference_dir}")
        print(f"Evaluation results directory: {args.eval_dir}")
        return
    
    # Process only specified models if provided
    if args.models:
        model_files = {k: v for k, v in model_files.items() if k in args.models}
    
    print(f"Found {len(model_files)} models:")
    for model_name in model_files.keys():
        print(f"  - {model_name}")
    
    print("\nStarting score calculation...")
    all_scores = {}
    
    for model_name, files in model_files.items():
        print(f"\nProcessing model: {model_name}")
        scores = calculate_single_model_score(
            model_name, 
            files['inference'], 
            files['evaluation'], 
            args.ground_truth_file, 
            args.output_dir
        )
        
        if scores:
            all_scores[model_name] = scores
            print(f"  MCQ Accuracy: {scores['overall_scores']['mcq_accuracy']:.2f}%")
            print(f"  Open-ended QA Average: {scores['overall_scores']['openqa_average']:.2f}%")
    
    if all_scores:
        # Generate comparison table
        comparison_file = os.path.join(args.output_dir, "model_comparison.json")
        comparison = generate_comparison_table(all_scores, comparison_file)
        
        # Print comparison table
        print_comparison_table(comparison)
        
        print(f"\nDetailed comparison results saved to: {comparison_file}")
        print(f"Individual model detailed scoring results saved in: {args.output_dir}")
    else:
        print("No model scores calculated successfully")

if __name__ == "__main__":
    main() 