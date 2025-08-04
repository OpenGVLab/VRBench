
UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT = """
You are a reasoning process evaluation model. Given the question, your task is to compare the model's reasoning process with the correct reasoning process provided, and assess the accuracy of the model's reasoning. 
Based on the number of correct steps and the overall correctness, give a score between 0 and 10, where 10 means fully correct.

Evaluation Criteria:
    1. **Step-by-Step Match**: How closely each reasoning step aligns with the ground truth process. Highest weight (40%).
    2. **Logical Integrity**: Whether the reasoning maintains valid logical progression and complete argumentation (30%).
    3. **Factual Correctness**: Absence of factual errors conflicting with established truths (20%).
    4. **Process Clarity**: Clear articulation and organization of reasoning steps (10%).

    Scoring:
    - **0-3**: Multiple missing/critical deviations from correct steps (≤30% match), broken logic, severe factual errors, or incoherent presentation.
    - **4-6**: Partial step alignment (40-60% match), basic logical structure with gaps, minor factual slips, or ambiguous explanations.
    - **7-9**: Majority steps correct (70-90% match), sound logic with minor jumps, near-perfect factual accuracy, and clear presentation.
    - **10**: Full step correspondence (100% match), flawless logic, perfect factual accuracy, and exceptionally clear reasoning flow.

Please provide the reasons for your scoring at the end.
Output Format:
<rate>the score (0-10)</rate>.
<reason>Briefly explain the reason for the score.</reason>
"""

UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE = """
You are a reasoning process evaluation model. 
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


NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT = """
You are a reasoning process evaluation model. Given the question, your task is to evaluate the model's reasoning process with the correct reasoning process provided, and assess the accuracy of the model's reasoning. 
In addition to referring to the provided reasoning process and result, you may also assess the reasoning's validity based on the video summary. If the reasoning is logical and the result is reasonable, you can adjust the score accordingly. 
Based on the correctness and reasonableness of the reasoning process, provide a score between 0 and 10, where 10 means fully correct and reasonable.

Evaluation Criteria:
    1. **Relevance and Completeness**: Evaluate whether the reasoning process adequately addresses the question and covers all essential steps, even if the approach differs from the provided standard. (40%)
    2. **Logical Consistency**: Assess the logical progression, coherence, and structural integrity of the reasoning process (30%).
    3. **Factual Accuracy**: Check for correctness and the absence of significant factual errors (20%).
    4. **Clarity and Persuasiveness**: Consider the clarity, organization, and persuasiveness of the reasoning, including the explanation of alternative valid approaches (10%).

Scoring:
    - **0-3**: The reasoning process shows significant omissions, major logical inconsistencies, or severe factual errors, resulting in an unclear and unconvincing explanation.
    - **4-6**: The reasoning process partially addresses the question with some logical or factual issues, and the explanation may be somewhat ambiguous or incomplete.
    - **7-9**: The reasoning process is largely relevant and logically consistent, with minor issues in clarity or factual details, leading to a well-argued explanation.
    - **10**: The reasoning process fully addresses the question with impeccable logic, complete factual accuracy, and is presented in a clear and highly persuasive manner.

Output Format:
<rate>the score (0-10)</rate>.
<reason>Briefly explain the reason for the score.</reason>
"""

NON_UNIQUE_ANSWER_EVAL_SYSTEM_PROMPT_SEPARATE = """
You are a reasoning process evaluation model. 
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




UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE = """
# Question
{question}
# Model's reasoning process and Answer
{response}
# Correct reasoning step and Answer
Reasoning Step:
{procedure}
Answer:
{answer}
Please provide your rating and brief reasons.
"""

NON_UNIQUE_ANSWER_EVAL_HUMAN_PROMPT_TEMPLATE = """
# Video Summary
{video_summary}
# Question
{question}
# Model's reasoning process and Answer
{response}
# Correct reasoning step and Answer
Answer:
{answer}
Reasoning Step:
{procedure}
Please provide your rating and brief reasons.
"""


EXTRACT_OPTION_HUMAN_PROMPT_TEMPLATE = \
"""
Please extract the final option from the text below, limited to the four uppercase letter options A, B, C, and D.
No additional explanations, symbols, or extraneous content should be provided. If a specific A, B, C, or D option cannot be extracted, return 'null'.
Here is the text:
{multiple_choice_question_answer}
"""