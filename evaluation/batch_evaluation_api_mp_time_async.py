import json
import re
import os
import argparse
from dataset import ReasoningEvaluationDatasetVLM, ReasoningEvaluationDatasetLLM
from model_api import AsyncBaseModel  # Other model classes can be handled similarly if needed
from torch.utils.data import DataLoader
import time
import jsonlines
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiofiles

# Proxy settings (uncomment and set if needed)
# proxy_address = 'http://user:password@ip:port/'
# os.environ["http_proxy"] = proxy_address
# os.environ["https_proxy"] = proxy_address
# os.environ["HTTP_PROXY"] = proxy_address
# os.environ["HTTPS_PROXY"] = proxy_address

def extract_numbers(text):
    """
    Extract all numbers from a string (supports int, float, negative)
    Args:
        text (str): input string
    Returns:
        list: list of number strings
    Example:
        >>> "Temperature -5.6℃, wind speed 3m/s")
        ['-5.6', '3']
        >>> "Price 12.34.56 error")
        ['12.34', '.56']
    """
    pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)'  # match int, float, negative
    return re.findall(pattern, text)


async def async_process_sample(sample, model, total_summary_dict, separate, semaphore):
    async with semaphore:
        """
        Process a single sample, return the processed result.
        Note: Since DataLoader is set to batch_size=1, each field in sample is a list of length 1.
        """

        id = sample['id'][0]
        question = sample['question'][0]
        response = sample['steps_and_answer'][0].strip('Assistant:')
        # print(response)
        # exit()
        answer = sample['answer'][0]
        procedure = sample['procedure'][0]
        
        question_type = sample['type'][0]

        # Call different evaluation interfaces based on question type and separate parameter
        if separate:
            # Use independent scoring interface, calculate total score based on four dimensions: 0.4, 0.4, 0.1, 0.1
            if question_type.strip() in ['Event Attribution', 'Multi-element Inference', 'Implicit Inference', 'Logical Linkage']:
                # Call interface based on UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE
                separate_eval_response = await model.evaluate_unique_answer_response(question, response, procedure, answer)
                # Extract scores for each dimension
                step_match = re.search(r"<step_matching>(.*?)</step_matching>", separate_eval_response)
                logical_match = re.search(r"<logical_consistency>(.*?)</logical_consistency>", separate_eval_response)
                factual_match = re.search(r"<factual_accuracy>(.*?)</factual_accuracy>", separate_eval_response)
                clarity_match = re.search(r"<process_clarity>(.*?)</process_clarity>", separate_eval_response)
                rationale_match = re.search(r"<rationale>(.*?)</rationale>", separate_eval_response)
                try:
                    step_score = float(step_match.group(1).strip())
                except Exception:
                    step_score = float(extract_numbers(step_match.group(1).strip())[0]) if step_match else 0.0
                try:
                    logical_score = float(logical_match.group(1).strip())
                except Exception:
                    logical_score = float(extract_numbers(logical_match.group(1).strip())[0]) if logical_match else 0.0
                try:
                    factual_score = float(factual_match.group(1).strip())
                except Exception:
                    factual_score = float(extract_numbers(factual_match.group(1).strip())[0]) if factual_match else 0.0
                try:
                    clarity_score = float(clarity_match.group(1).strip())  
                except Exception:
                    clarity_score = float(extract_numbers(clarity_match.group(1).strip())[0]) if clarity_match else 0.0
                # Calculate total score
                weighted_score = step_score * 0.4 + logical_score * 0.4 + factual_score * 0.1 + clarity_score * 0.1
                rate = str(weighted_score)
                reason = rationale_match.group(1).strip() if rationale_match else ""
                eval_response = separate_eval_response

            elif question_type.strip() in ['Hypothetical Reasoning', 'Event Prediction']:
                id_trim = id.rsplit('_', 1)[0]
                summary = total_summary_dict.get(id_trim, "")
                # Call interface based on NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE
                separate_eval_response = await model.evaluate_non_unique_answer_response(summary, question, response, procedure, answer)
                # Extract scores for each dimension
                relevance_match = re.search(r"<relevance>(.*?)</relevance>", separate_eval_response)
                logical_match = re.search(r"<logical_consistency>(.*?)</logical_consistency>", separate_eval_response)
                factual_match = re.search(r"<factual_accuracy>(.*?)</factual_accuracy>", separate_eval_response)
                clarity_match = re.search(r"<clarity>(.*?)</clarity>", separate_eval_response)
                rationale_match = re.search(r"<rationale>(.*?)</rationale>", separate_eval_response)
                try:
                    relevance_score = float(relevance_match.group(1).strip())
                except Exception:
                    relevance_score = float(extract_numbers(relevance_match.group(1).strip())[0]) if relevance_match else 0.0
                try:
                    logical_score = float(logical_match.group(1).strip())
                except Exception:
                    logical_score = float(extract_numbers(logical_match.group(1).strip())[0]) if logical_match else 0.0
                try:
                    factual_score = float(factual_match.group(1).strip())
                except Exception:
                    factual_score = float(extract_numbers(factual_match.group(1).strip())[0]) if factual_match else 0.0
                try:
                    clarity_score = float(clarity_match.group(1).strip())  
                except Exception:
                    clarity_score = float(extract_numbers(clarity_match.group(1).strip())[0]) if clarity_match else 0.0
                # Calculate total score (note: in non-unique types, the first dimension is relevance)
                weighted_score = relevance_score * 0.4 + logical_score * 0.4 + factual_score * 0.1 + clarity_score * 0.1
                rate = str(weighted_score)
                reason = rationale_match.group(1).strip() if rationale_match else ""
                eval_response = separate_eval_response

            elif question_type.strip() in ['Event Summarization']:
                # Do not process event summarization type
                eval_response = ""
                rate = ""
                reason = ""
            else:
                print(f'{id}, {question_type.strip()} did not match any category!')
                eval_response = ""
                rate = ""
                reason = ""
        else:
            # Use original interface: return overall score and reason, format as <rate>...</rate> and <reason>...</reason>
            if question_type.strip() in ['Event Attribution', 'Multi-element Inference', 'Implicit Inference', 'Logical Linkage']:
                eval_response = await model.evaluate_unique_answer_response(question, response, procedure, answer)
            elif question_type.strip() in ['Hypothetical Reasoning', 'Event Prediction']:
                id_trim = id.rsplit('_', 1)[0]
                summary = total_summary_dict.get(id_trim, "")
                eval_response = await model.evaluate_non_unique_answer_response(summary, question, response, procedure, answer)
            elif question_type.strip() in ['Event Summarization']:
                # Do not process event summarization type
                eval_response = ""
            else:
                print(f'{id}, {question_type.strip()} did not match any category!')
                eval_response = ""
            # Extract <rate> and <reason> tag content using regex
            pattern = r"<rate>(.*?)</rate>|<reason>(.*?)</reason>"
            matches = re.findall(pattern, eval_response)
            rate = ""
            reason = ""
            for match in matches:
                if match[0]:
                    rate = match[0]
                if match[1]:
                    reason = match[1]

        result = sample.copy() 
        result['rate'] = rate
        result['reason'] = reason
        result['eval_response'] = eval_response

        # time.sleep(3)
        
        return result

async def async_batch_evaluation(args):
    # Construct dataset and DataLoader (batch_size=1)
    if args.VLM:
        dataset = ReasoningEvaluationDatasetVLM(args.source_file, args.output_file, args.summary_file)
    else:
        dataset = ReasoningEvaluationDatasetLLM(args.source_file, args.output_file, args.summary_file)
    bs = 1
    n_worker = 16
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        num_workers=20,
        pin_memory=False,
        shuffle=False,
        persistent_workers=True if n_worker > 0 else False,
    )
    
    # Model initialization
    if args.model == "gpt-4o":
        api_key = "sk-so7zhdci3gxgggAyXM3UxevY0XsJGlzUzOXyMgdcQswplf7T"
        base_url = "https://api.claudeshop.top/v1/"
        model = AsyncBaseModel(api_key=api_key, base_url=base_url, separate=args.separate)
    else:
        model = AsyncBaseModel(model=args.model, api_key=args.api_key, base_url=args.base_url, separate=args.separate)
    

    # # Load summary file, build video_id -> video_summary dictionary
    # with open(args.summary_file, 'r') as file:
    #     total_vid_ls = [json.loads(line) for line in file]
    #     total_summary_dict = {item['video_id']: item['video_summary'] for item in total_vid_ls}
    
    async with aiofiles.open(args.summary_file, 'r') as f:
        total_vid_ls = [json.loads(line) async for line in f]
    total_summary_dict = {item['video_id']: item['video_summary'] for item in total_vid_ls}

    # Use thread pool to process each sample in parallel
    # futures = []
    # max_threads = 20  # Adjust thread count as needed
    semaphore = asyncio.Semaphore(20)
    tasks = []

    for batch in dataloader:
        task = asyncio.create_task(
            async_process_sample(batch, model, total_summary_dict, args.separate, semaphore)
        )
        tasks.append(task)
    
    results = []
    batch_size = 50
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch_tasks)
      
        # Asynchronously write to file
        async with aiofiles.open(args.output_file, 'a') as f:
            for result in batch_results:
                await f.write(json.dumps(result) + '\n')
      
        print(f"Processed {i + len(batch_results)} samples")


   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str, default='/mnt/petrelfs/wuyue/CODE/longvideoreasoning/multi-step-reasoning_llm_eval/inference_result/gpt4o_cot_result.jsonl')
    parser.add_argument('--summary_file', type=str, default='/mnt/petrelfs/wuyue/CODE/longvideoreasoning/LLM_Eval/data/processed_question_answer.jsonl')
    parser.add_argument('--model', type=str, default='deepseek', help='selected model for inference')
    parser.add_argument('--api_key', type=str, default='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoieXVqaWFzaHVvIiwiZXhwIjoxNzk4NzYxNjAwfQ.rkEcndq5pYnz_qVcBBk_pjua4cUV7vEm1rkUCtNjNjM')
    parser.add_argument('--base_url', type=str, default='https://180.163.156.42:21020/v1')
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--separate', action='store_true', default=False)
    parser.add_argument('--VLM', action='store_true', default=False)

    args = parser.parse_args()

    asyncio.run(async_batch_evaluation(args))