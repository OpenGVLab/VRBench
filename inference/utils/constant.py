from string import Template

MAX_TOKENS = 1024
GENERATION_TEMPERATURE = 1.0
GENERATION_SEED = 215

MCQ_COT_PROMPT = Template("""
You are a helpful video understanding assistant that answers multi-choice questions through step-by-step reasoning based on the video and its summary.

# Instructions:
1. Break down the reasoning process into clear, specific events.
2. Conclude with the best option letter(A/B/C/D) at last.

# Output Format:
<Step 1> Description of event/observation 
<Step 2> Description of event/observation 
...  
<Answer> [Option letter]

# Multiple Choice Question
$multiple_choice_question

# Video Summary
$video_summary
""")


MCQ = {
    "type":"mcq",
    "content":MCQ_COT_PROMPT
}

PROMPT = {
    "mcq": MCQ,
}

MODEL_PATH = {
    "Qwen/Qwen2-VL-7B-Instruct":"/fs-computility/llm/shared/llmeval/models/opencompass_hf_hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/",
    "Qwen/Qwen2.5-VL-7B-Instruct":"/fs-computility/ai4sData/shared/models/Qwen2.5-VL-7B-Instruct/",
    "OpenGVLab/InternVL2_5-8B":"/fs-computility/ai4sData/shared/models/InternVL2_5-8B/",
    "internvideo:OpenGVLab/InternVideo2_5_Chat_8B":"/fs-computility/video/shared/hf_weight/InternVideo2_5_Chat_8B/",
    "Qwen/Qwen2.5-7B-Instruct":"/fs-computility/ai4sData/shared/models/Qwen2.5-7B-Instruct/",
    "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf":"/fs-computility/video/shared/hf_weight/llava-onevision-qwen2-7b-ov-hf",
    "meta-llama/Llama-3.3-70B-Instruct":"/fs-computility/ai4sData/shared/models/Llama-3.3-70B-Instruct/",
    "deepseek-ai/deepseek-vl2":"/fs-computility/ai4sData/shared/models/deepseek-vl2/",
    "microsoft/Phi-3.5-vision-instruct":"/fs-computility/ai4sData/shared/models/Phi-3.5-vision-instruct/",
    "moonshotai/Kimi-VL-A3B-Thinking-2506":"/fs-computility/video/shared/hf_weight/Kimi-VL-A3B-Thinking-2506/",
    "Kwai-Keye/Keye-VL-8B-Preview": "/fs-computility/video/shared/hf_weight/Keye-VL-8B-Preview",
    "MiMo-VL-7B-RL": "/fs-computility/video/shared/hf_weight/MiMo-VL-7B-RL/",
}