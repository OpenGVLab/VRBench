#!/usr/bin/env python3
"""
VRBench scoring calculation script
Calculate MCQ accuracy and open-ended QA average scores, including scores by question category
"""

import json
import argparse
from collections import defaultdict
import re

def extract_mcq_answer(response_text):
    """
    Extract MCQ choice from model response
    Enhanced version based on ds_v3.py approach with improved text processing
    """
    if not response_text:
        return None
    
    # Define punctuation processing (adapted from ds_v3.py)
    period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    comma_strip = re.compile(r"(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
    
    def process_punctuation(text):
        """Clean punctuation from text (adapted from ds_v3.py)"""
        output = text
        for p in punct:
            if (p + " " in text or " " + p in text) or (re.search(comma_strip, text) is not None):
                output = output.replace(p, "")
            else:
                output = output.replace(p, " ")
        output = period_strip.sub("", output, re.UNICODE)
        return output
    
    # 1. Find answer in \boxed{} format (highest priority)
    boxed_match = re.search(r'\\boxed\{([A-E])\}', response_text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()
    
    # 2. Find answer in <Answer> tags (second highest priority)
    answer_match = re.search(r'<Answer>\s*([A-E])', response_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # 3. Try to match option format "A. content" at the beginning of lines (from ds_v3.py)
    option_format_matches = re.findall(r'^([A-E])\.\s*(.+)$', response_text.strip(), re.IGNORECASE | re.MULTILINE)
    if option_format_matches:
        return option_format_matches[-1][0].upper()  # Return the last match
    
    # 4. Find choice after "Answer" keywords (explicit answer statements)
    answer_patterns = [
        r'Answer[:\s]+([A-E])(?:\s|$|\.|,)',         # Answer: A, Answer A
        r'answer is[:\s]+([A-E])(?:\s|$|\.|,)',      # answer is A, answer is: A
        r'correct answer[:\s]+([A-E])(?:\s|$|\.|,)',  # correct answer A
        r'final answer[:\s]+([A-E])(?:\s|$|\.|,)',   # final answer A
        r'final[:\s]+([A-E])(?:\s|$|\.|,)',          # final: A, final A
        r'therefore[:\s]+([A-E])(?:\s|$|\.|,)',      # therefore A
        r'conclusion[:\s]+([A-E])(?:\s|$|\.|,)',     # conclusion A
    ]
    
    # Collect all matches from answer patterns
    all_answer_matches = []
    for pattern in answer_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        all_answer_matches.extend(matches)
    
    if all_answer_matches:
        return all_answer_matches[-1].upper()  # Return the last explicit answer
    
    # 5. Find choices that appear as option labels
    option_patterns = [
        r'([A-E])\.',  # A., B., C., D., E.
        r'([A-E])\)',  # A), B), C), D), E)
        r'([A-E]):',   # A:, B:, C:, D:, E:
    ]
    
    for pattern in option_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()  # Return the last match
    
    # 6. Look for patterns like "D " (letter followed by space)
    space_pattern = r'(?:^|\n|\s)([A-E])\s+(?=[A-Z]|$|\n)'
    matches = re.findall(space_pattern, response_text, re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].upper()
    
    # 7. Text preprocessing and final extraction (adapted from ds_v3.py)
    processed_text = response_text.replace("\n", " ").replace("\t", " ").strip()
    processed_text = process_punctuation(processed_text)
    processed_text = processed_text.strip("'").strip('"').strip(")").strip("(").strip().lower()
    
    # Find single letter A-E in processed text
    processed_letters = re.findall(r'\b([A-E])\b', processed_text, re.IGNORECASE)
    if processed_letters:
        return processed_letters[-1].upper()
    
    # 8. Enhanced fallback: find standalone letters A-E at word boundaries in original text
    # Look for letters that are likely to be final answers (near end of text or after keywords)
    choices = re.findall(r'\b([A-E])\b', response_text)
    if choices:
        # Check if any choice appears near the end of the text or after keywords
        text_lower = response_text.lower()
        end_portion = text_lower[-50:]  # Last 50 characters
        
        # Look for letters in the end portion first
        end_choices = re.findall(r'\b([A-E])\b', end_portion, re.IGNORECASE)
        if end_choices:
            return end_choices[-1].upper()
        
        # Otherwise return the last choice found anywhere
        return choices[-1].upper()
    
    return None

def load_ground_truth(data_file):
    """
    Load ground truth answers from original data file
    """
    ground_truth = {}
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            video_id = data['video_id']
            if 'mcq' in data:
                for qa_id, qa_data in data['mcq'].items():
                    question_id = f"{video_id}_{qa_id}"
                    ground_truth[question_id] = {
                        'answer': qa_data['answer'],
                        'reasoning_type': qa_data['reasoning_type']
                    }
    return ground_truth

def load_inference_results(inference_file):
    """
    Load inference results
    """
    inference_results = {}
    with open(inference_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            video_id = data['video_id']
            if 'mcq_result' in data:
                for qa_id, responses in data['mcq_result'].items():
                    question_id = f"{video_id}_{qa_id}"
                    # Take first response if multiple exist
                    response = responses[0] if isinstance(responses, list) else responses
                    inference_results[question_id] = response
    return inference_results

def load_evaluation_results(eval_file):
    """
    Load evaluation results (open-ended QA scores)
    """
    evaluation_results = {}
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question_id = data['id'][0] if isinstance(data['id'], list) else data['id']
            rate = data.get('rate', '0')
            question_type = data['type'][0] if isinstance(data['type'], list) else data['type']
            
            try:
                score = float(rate)
            except (ValueError, TypeError):
                score = 0.0
            
            evaluation_results[question_id] = {
                'score': score,
                'type': question_type
            }
    
    return evaluation_results

def calculate_mcq_accuracy(ground_truth, inference_results):
    """
    Calculate MCQ accuracy
    """
    correct_count = 0
    total_count = 0
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for question_id, gt_data in ground_truth.items():
        if question_id in inference_results:
            total_count += 1
            predicted_answer = extract_mcq_answer(inference_results[question_id])
            correct_answer = gt_data['answer']
            reasoning_type = gt_data['reasoning_type']
            
            category_stats[reasoning_type]['total'] += 1
            
            if predicted_answer == correct_answer:
                correct_count += 1
                category_stats[reasoning_type]['correct'] += 1
    
    overall_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    category_accuracies = {}
    for category, stats in category_stats.items():
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        category_accuracies[category] = accuracy
    
    return overall_accuracy, category_accuracies, total_count, correct_count

def calculate_openqa_scores(evaluation_results):
    """
    Calculate open-ended QA average scores
    """
    total_score = 0
    total_count = 0
    category_scores = defaultdict(list)
    
    for question_id, eval_data in evaluation_results.items():
        score = eval_data['score']
        question_type = eval_data['type']
        
        total_score += score
        total_count += 1
        category_scores[question_type].append(score)
    
    overall_average = (total_score / total_count * 10) if total_count > 0 else 0  # Convert to percentage
    
    category_averages = {}
    for category, scores in category_scores.items():
        avg_score = (sum(scores) / len(scores) * 10) if scores else 0  # Convert to percentage
        category_averages[category] = avg_score
    
    return overall_average, category_averages, total_count

def calculate_combined_scores(mcq_accuracies, openqa_averages):
    """
    Calculate combined scores for each question category (average of MCQ accuracy and open-ended QA scores)
    """
    combined_scores = {}
    all_categories = set(mcq_accuracies.keys()) | set(openqa_averages.keys())
    
    for category in all_categories:
        mcq_score = mcq_accuracies.get(category, 0)
        openqa_score = openqa_averages.get(category, 0)
        
        # Use available score if category has only one type of question
        if mcq_score > 0 and openqa_score > 0:
            combined_score = (mcq_score + openqa_score) / 2
        elif mcq_score > 0:
            combined_score = mcq_score
        elif openqa_score > 0:
            combined_score = openqa_score
        else:
            combined_score = 0
            
        combined_scores[category] = combined_score
    
    return combined_scores

def main():
    parser = argparse.ArgumentParser(description='Calculate VRBench scores')
    parser.add_argument('--ground_truth_file', type=str, required=True,
                       help='Original data file path (containing ground truth answers)')
    parser.add_argument('--inference_file', type=str, required=True,
                       help='Inference results file path')
    parser.add_argument('--evaluation_file', type=str, required=True,
                       help='Evaluation results file path')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output results file path (optional)')
    
    args = parser.parse_args()
    
    print("Loading data...")
    
    # Load data
    ground_truth = load_ground_truth(args.ground_truth_file)
    inference_results = load_inference_results(args.inference_file)
    evaluation_results = load_evaluation_results(args.evaluation_file)
    
    print(f"Loaded {len(ground_truth)} ground truth answers")
    print(f"Loaded {len(inference_results)} inference results")
    print(f"Loaded {len(evaluation_results)} evaluation results")
    
    # Calculate MCQ accuracy
    mcq_accuracy, mcq_category_accuracies, mcq_total, mcq_correct = calculate_mcq_accuracy(
        ground_truth, inference_results)
    
    # Calculate open-ended QA average scores
    openqa_average, openqa_category_averages, openqa_total = calculate_openqa_scores(
        evaluation_results)
    
    # Calculate combined scores
    combined_scores = calculate_combined_scores(mcq_category_accuracies, openqa_category_averages)
    
    # Prepare results output
    results = {
        'overall_scores': {
            'mcq_accuracy': round(mcq_accuracy, 2),
            'openqa_average': round(openqa_average, 2),
            'mcq_stats': f"{mcq_correct}/{mcq_total}",
            'openqa_stats': f"{openqa_total} questions"
        },
        'category_scores': {}
    }
    
    # Output results
    print("\n" + "="*60)
    print("VRBench Scoring Results")
    print("="*60)
    print(f"MCQ Accuracy: {mcq_accuracy:.2f}% ({mcq_correct}/{mcq_total})")
    print(f"Open-ended QA Average: {openqa_average:.2f}% ({openqa_total} questions)")
    print("\nScores by Question Category:")
    print("-"*60)
    
    for category in sorted(combined_scores.keys()):
        mcq_score = mcq_category_accuracies.get(category, 0)
        openqa_score = openqa_category_averages.get(category, 0)
        combined_score = combined_scores[category]
        
        results['category_scores'][category] = {
            'mcq_accuracy': round(mcq_score, 2),
            'openqa_average': round(openqa_score, 2),
            'combined_score': round(combined_score, 2)
        }
        
        print(f"{category}:")
        print(f"  MCQ Accuracy: {mcq_score:.2f}%")
        print(f"  Open-ended QA Average: {openqa_score:.2f}%")
        print(f"  Combined Score: {combined_score:.2f}%")
        print()
    
    # Save results to file
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 