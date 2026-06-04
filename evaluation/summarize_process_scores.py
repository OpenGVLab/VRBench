#!/usr/bin/env python3
"""汇总过程评分结果"""

import json
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True)
    args = parser.parse_args()

    type_scores = defaultdict(list)
    total_scores = []

    with open(args.eval_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            score = d['scores']['weighted']
            reasoning_type = d['reasoning_type']
            type_scores[reasoning_type].append(score)
            total_scores.append(score)

    print(f"文件: {args.eval_file}")
    print(f"总计 {len(total_scores)} 个 QA")
    print(f"平均分: {sum(total_scores)/len(total_scores):.2f}/10")
    print()
    for t in sorted(type_scores.keys()):
        scores = type_scores[t]
        print(f"  {t}: {sum(scores)/len(scores):.2f} ({len(scores)} 题)")


if __name__ == "__main__":
    main()
