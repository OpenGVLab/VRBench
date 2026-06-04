#!/usr/bin/env python3
"""
VRBench 过程评分脚本
读取模型推理输出，去掉 think 内容，用 DeepSeek 对推理过程打分
"""

import json
import os
import re
import argparse
import asyncio
import aiofiles
from openai import AsyncOpenAI
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = """You are a reasoning process evaluation model.
Given the question, compare the model's reasoning process with the correct reference process and evaluate across four dimensions.
Provide separate scores (0-10) per dimension with specific evaluation standards.
Provide key and very brief explanation of the scores.

Evaluation Dimensions:
    1. **Step Matching** (0-10):
    - Evaluate alignment of reasoning steps with the reference reasoning process
    - Detect omissions of critical steps or redundant additions
    - Verify completeness of problem decomposition and sequence validity

    2. **Logical Consistency** (0-10):
    - Validate causal connections in the reasoning chain
    - Identify logical leaps or argument discontinuities
    - Assess congruence between assumptions and conclusions

    3. **Factual Accuracy** (0-10):
    - Verify verifiability of all factual claims
    - Detect conflicts with established truths
    - Evaluate frequency and impact of factual errors

    4. **Process Clarity** (0-10):
    - Analyze clarity and organization of step presentation
    - Check terminology accuracy and consistency
    - Assess effectiveness in explaining complex concepts

Scoring Standards:
    For each dimension:
    9-10: Exemplary performance with no flaws
    7-8: Non-critical deviations present
    5-6: Quality-impairing defects
    3-4: Serious validity-compromising errors
    0-2: Fundamental functionality failure

Output Format:
<step_matching>[0-10]</step_matching>
<logical_consistency>[0-10]</logical_consistency>
<factual_accuracy>[0-10]</factual_accuracy>
<process_clarity>[0-10]</process_clarity>
<rationale>
    [Per-dimension basis:
    - Highlight alignment strengths
    - Specify critical deficiencies]
</rationale>
"""

NON_UNIQUE_SYSTEM_PROMPT = """You are a reasoning process evaluation model.
Given the question, your task is to evaluate the model's reasoning process with the provided correct reasoning process and video summary across four distinct dimensions.
Provide separate scores (0-10) for each criterion based on its specific evaluation standards.
Provide key and very brief explanation of the scores.

Evaluation Dimensions:
    1. **Relevance** (0-10):
    - Assess whether the reasoning process closely relates to the provided correct reasoning process
    - Evaluate whether the reasoning process fully addresses the question requirements
    - Consider handling of edge cases and alternative approaches

    2. **Logical Consistency** (0-10):
    - Examine the coherence between reasoning steps
    - Verify absence of contradictions or fallacies
    - Assess structural integrity and progression validity

    3. **Factual Accuracy** (0-10):
    - Verify correctness of factual claims in the provided
    - Check consistency with provided reference correct reasoning process and video summary
    - Evaluate error frequency and severity

    4. **Clarity and Persuasiveness** (0-10):
    - Assess explanation clarity and organization
    - Evaluate effectiveness of supporting evidence
    - Consider presentation logic and accessibility

Dimension Scoring Standards:
    For each dimension:
    9-10: Exemplary performance with no notable issues
    7-8:  Minor imperfections with negligible impact
    5-6:  Moderate issues affecting quality
    3-4:  Serious deficiencies impairing validity
    0-2:  Fundamental failures in this dimension

Output Format:
<relevance>[0-10]</relevance>
<logical_consistency>[0-10]</logical_consistency>
<factual_accuracy>[0-10]</factual_accuracy>
<clarity>[0-10]</clarity>
<rationale>
    [For each dimension with simply explanation:
    - Key strengths identified
    - Specific weaknesses noted]
</rationale>
"""


def strip_think(text):
    """去掉 </think> 及之前的所有内容"""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def format_process(procedure):
    if isinstance(procedure, str):
        return procedure
    sorted_steps = sorted(procedure.items(), key=lambda x: int(x[0]))
    return '\n'.join([f"<Step {num}> {desc}" for num, desc in sorted_steps])


def parse_scores_unique(response):
    """解析 unique answer 类型的评分"""
    step = re.search(r"<step_matching>(.*?)</step_matching>", response)
    logical = re.search(r"<logical_consistency>(.*?)</logical_consistency>", response)
    factual = re.search(r"<factual_accuracy>(.*?)</factual_accuracy>", response)
    clarity = re.search(r"<process_clarity>(.*?)</process_clarity>", response)

    def safe_float(match):
        if not match:
            return 0.0
        try:
            return float(match.group(1).strip())
        except:
            nums = re.findall(r'[\d.]+', match.group(1))
            return float(nums[0]) if nums else 0.0

    s, l, f, c = safe_float(step), safe_float(logical), safe_float(factual), safe_float(clarity)
    weighted = s * 0.4 + l * 0.4 + f * 0.1 + c * 0.1
    return {"step_matching": s, "logical_consistency": l, "factual_accuracy": f, "process_clarity": c, "weighted": weighted}


def parse_scores_non_unique(response):
    """解析 non-unique answer 类型的评分"""
    relevance = re.search(r"<relevance>(.*?)</relevance>", response)
    logical = re.search(r"<logical_consistency>(.*?)</logical_consistency>", response)
    factual = re.search(r"<factual_accuracy>(.*?)</factual_accuracy>", response)
    clarity = re.search(r"<clarity>(.*?)</clarity>", response)

    def safe_float(match):
        if not match:
            return 0.0
        try:
            return float(match.group(1).strip())
        except:
            nums = re.findall(r'[\d.]+', match.group(1))
            return float(nums[0]) if nums else 0.0

    r, l, f, c = safe_float(relevance), safe_float(logical), safe_float(factual), safe_float(clarity)
    weighted = r * 0.4 + l * 0.4 + f * 0.1 + c * 0.1
    return {"relevance": r, "logical_consistency": l, "factual_accuracy": f, "clarity": c, "weighted": weighted}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def call_deepseek(client, model, system_prompt, user_prompt):
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=60.0,
    )
    return completion.choices[0].message.content


async def eval_one(client, model, qa_id, question, response, procedure, answer, reasoning_type, video_summary, semaphore):
    async with semaphore:
        unique_types = ['Event Attribution', 'Multi-element Inference', 'Implicit Inference', 'Logical Linkage']
        non_unique_types = ['Hypothetical Reasoning', 'Event Prediction']

        if reasoning_type in unique_types:
            user_prompt = f"""# Question
{question}
# Model's reasoning process and Answer
{response}
# Correct reasoning step and Answer
Reasoning Step:
{procedure}
Answer:
{answer}
Please provide your rating and brief reasons."""
            eval_response = await call_deepseek(client, model, SYSTEM_PROMPT, user_prompt)
            scores = parse_scores_unique(eval_response)

        elif reasoning_type in non_unique_types:
            summary = video_summary[:10000] if video_summary else ""
            user_prompt = f"""# Video Summary
{summary}
# Question
{question}
# Model's reasoning process and Answer
{response}
# Correct reasoning step and Answer
Answer:
{answer}
Reasoning Step:
{procedure}
Please provide your rating and brief reasons."""
            eval_response = await call_deepseek(client, model, NON_UNIQUE_SYSTEM_PROMPT, user_prompt)
            scores = parse_scores_non_unique(eval_response)

        else:
            # Event Summarization 等类型跳过
            return None

        return {
            "id": qa_id,
            "reasoning_type": reasoning_type,
            "scores": scores,
            "eval_response": eval_response,
        }


async def main():
    parser = argparse.ArgumentParser(description="VRBench 过程评分")
    parser.add_argument("--inference_file", type=str, required=True, help="模型推理输出文件")
    parser.add_argument("--data_file", type=str, required=True, help="VRBench_eval.jsonl (含 ground truth)")
    parser.add_argument("--output_file", type=str, required=True, help="评分输出文件")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash-beta")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    # 加载 ground truth
    gt_data = {}
    with open(args.data_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            gt_data[d['video_id']] = d

    # 加载推理结果
    inference = {}
    with open(args.inference_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            inference[d['video_id']] = d.get('mcq_result', {})

    # 加载已完成的评分（断点续传）
    exist_ids = set()
    if args.output_file and __import__('os').path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                d = json.loads(line)
                exist_ids.add(d['id'])

    # 构建评分任务
    tasks_data = []
    for video_id, mcq_result in inference.items():
        if video_id not in gt_data:
            continue
        gt = gt_data[video_id]
        video_summary = gt.get('video_summary', '')
        mcq_gt = gt.get('mcq', {})

        for qa_key, model_response in mcq_result.items():
            qa_id = f"{video_id}_{qa_key}"
            if qa_id in exist_ids:
                continue
            if qa_key not in mcq_gt:
                continue

            qa_info = mcq_gt[qa_key]
            response_clean = strip_think(model_response) if isinstance(model_response, str) else str(model_response)

            tasks_data.append({
                "qa_id": qa_id,
                "question": qa_info['question'],
                "response": response_clean,
                "procedure": format_process(qa_info['reasoning_process']),
                "answer": qa_info['original_answer'],
                "reasoning_type": qa_info['reasoning_type'],
                "video_summary": video_summary,
            })

    model_name = args.inference_file.split('/')[-1].replace('.jsonl', '')
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"总计 {len(tasks_data)} 个 QA 需要评分，已跳过 {len(exist_ids)} 个已完成")
    print(f"{'='*60}")

    if not tasks_data:
        print("无需评分，全部已完成")
        return

    # 异步评分
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        eval_one(client, args.model, td["qa_id"], td["question"], td["response"],
                 td["procedure"], td["answer"], td["reasoning_type"], td["video_summary"], semaphore)
        for td in tasks_data
    ]

    # 分批处理并写入，带进度条
    from tqdm import tqdm
    batch_size = 50
    completed = 0
    errors = 0
    pbar = tqdm(total=len(tasks_data), desc=f"[{model_name}] 过程评分", unit="qa",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    async with aiofiles.open(args.output_file, 'a') as f:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    errors += 1
                    pbar.set_postfix_str(f"errors={errors}")
                    continue
                if r is None:
                    pbar.update(1)
                    continue
                await f.write(json.dumps(r, ensure_ascii=False) + '\n')
                completed += 1
                pbar.update(1)
            pbar.set_postfix_str(f"done={completed}, errors={errors}")

    pbar.close()
    print(f"\n[{model_name}] 评分完成: {completed} 条写入 {args.output_file}，{errors} 条失败")


if __name__ == "__main__":
    asyncio.run(main())
