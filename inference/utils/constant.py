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


MCQ_THINK_PROMPT = Template("""You are a helpful video understanding assistant. Answer the following multiple-choice question based on the video and its summary. Output only the option letter (A/B/C/D).

# Multiple Choice Question
$multiple_choice_question

# Video Summary
$video_summary
""")

MCQ = {
    "type":"mcq",
    "content":MCQ_COT_PROMPT
}

MCQ_THINK = {
    "type":"mcq",
    "content":MCQ_THINK_PROMPT
}

PROMPT = {
    "mcq": MCQ,
    "mcq_think": MCQ_THINK,
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
    "InternVideo3": "/mnt/shared-storage-user/sfteval/xtuner_saved_model/internvl3.5/export_meta_testqwenmla_64k_896_107/20260108074743/hf-2605",
    "InternVideo3_mla426_2": "/mnt/shared-storage-user/sfteval/xtuner_saved_model/internvideo3/sft_mla_426_2/20260430173425/hf-5000_new",
    "InternVideo3_mla426_2_think": "/mnt/shared-storage-user/sfteval/xtuner_saved_model/internvideo3/sft_mla_426_2/20260430173425/hf-5000_new",
    "InternVideo3_mla426_2_nothink": "/mnt/shared-storage-user/sfteval/xtuner_saved_model/internvideo3/sft_mla_426_2/20260430173425/hf-5000_new",
    "Qwen3-VL-8B-Instruct": "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/cadac78306af287f801b75a5565ede58f323f472",
    "Qwen3-VL-4B-Instruct": "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/f2981ed65cf8ac8b860135c9115dff9dd7c7c80c",
    "InternVL3_5-8B": "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--OpenGVLab--InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79",
    "MiniCPM-V-4_5": "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--openbmb--MiniCPM-V-4_5/snapshots/343b490578adc7e03b687846c33cadd8a00cb80d",
    "LLaVA-Video-7B-Qwen2": "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--lmms-lab--LLaVA-Video-7B-Qwen2",
    "Molmo2-8B": "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--allenai--Molmo2-8B",
    "Keye-VL-1_5-8B": "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--Kwai-Keye-Keye-VL-1_5-8B",
    "GLM-4.1V-9B-Thinking": "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--zai-org-GLM-4.1V-9B-Thinking",
    "Eagle2.5-8B": "/mnt/shared-storage-user/yanziang/Eagle2.5-8B",
}