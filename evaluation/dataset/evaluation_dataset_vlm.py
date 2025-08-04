import json
import re
import os
from torch.utils.data import Dataset, DataLoader

class ReasoningEvaluationDatasetVLM(Dataset):
    def __init__(self, label_root, exits_root, summary_file=None):
        self.label_root = label_root
        self.exist_root = exits_root
        self.summary_file = summary_file
        
        # Load summary data if provided
        self.summary_data = {}
        if summary_file and os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.summary_data[data['video_id']] = data
        
        label_ls = []
        with open(self.label_root, 'r', encoding='utf-8') as f:
            label_ls = [json.loads(line) for line in f]

        self.qa_ls = []
        # num = 0
        for video in label_ls:
            # Check if this is inference output format (mcq_result) or evaluation format (mcq_answer)
            if 'mcq_result' in video:
                # Inference output format: {"video_id": "...", "mcq_result": {"qa2": ["..."], ...}}
                video_id = video['video_id']
                summary_video = self.summary_data.get(video_id, {})
                
                for qa_k, qa_result in video['mcq_result'].items():
                    qa_dict = {}
                    qa_dict['id'] = video_id + '_' + qa_k
                    
                    # Get question, answer, procedure, type from summary file
                    if video_id in self.summary_data and 'mcq' in self.summary_data[video_id]:
                        mcq_data = self.summary_data[video_id]['mcq']
                        if qa_k in mcq_data:
                            qa_v = mcq_data[qa_k]
                            qa_dict['question'] = qa_v['question']
                            qa_dict['answer'] = qa_v['original_answer']
                            try:
                                qa_dict['procedure'] = self.format_process(qa_v['reasoning_process'])
                                qa_dict['type'] = qa_v['reasoning_type']
                            except:
                                qa_dict['procedure'] = self.format_process(qa_v['reasoning_process'])
                                qa_dict['type'] = qa_v['reasoning_type']
                        else:
                            # Fallback if qa_k not found in summary
                            qa_dict['question'] = f"Question for {qa_k}"
                            qa_dict['answer'] = f"Answer for {qa_k}"
                            qa_dict['procedure'] = ""
                            qa_dict['type'] = "Unknown"
                    else:
                        # Fallback if video not found in summary
                        qa_dict['question'] = f"Question for {qa_k}"
                        qa_dict['answer'] = f"Answer for {qa_k}"
                        qa_dict['procedure'] = ""
                        qa_dict['type'] = "Unknown"
                    
                    qa_dict['steps_and_answer'] = qa_result[0] if qa_result else ""
                    self.qa_ls.append(qa_dict)
            else:
                # Original evaluation format: {"video_id": "...", "mcq": {...}, "mcq_answer": {...}}
                for qa_k, qa_v  in video['mcq'].items():
                    qa_dict = {}
                    qa_dict['id'] = video['video_id']+'_' + qa_k
                    qa_dict['question'] = qa_v['question']
                    qa_dict['answer'] = qa_v['original_answer']
                    try:
                        qa_dict['procedure'] = self.format_process(qa_v['reasoning_process'])
                        qa_dict['type'] = qa_v['reasoning_type']
                    except:
                        qa_dict['procedure'] = self.format_process(qa_v['reasoning_process'])
                        qa_dict['type'] = qa_v['reasoning_type']
                    try:
                        qa_dict['steps_and_answer'] = video['mcq_answer'][qa_k]['reasoning_steps_and_answer']
                        # print('2')
                    except:
                        # num += 1
                        print(video['video_id'])
                        continue
                    self.qa_ls.append(qa_dict)
        # print(num/len(label_ls))
        # exit()

        data = []
        if os.path.exists(self.exist_root):
            with open(self.exist_root, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

        exist_id_ls = [qa['id'][0] for qa in data]
        self.qa_ls = [qa for qa in self.qa_ls if qa['id'] not in exist_id_ls]

        print(f"remain qa num: {len(self.qa_ls)}, exist qa num: {len(data)}")
    
    def format_process(self, procedure):
        if isinstance(procedure, str):
            return procedure
        sorted_steps = sorted(procedure.items(), key=lambda x: int(x[0]))
        return '\n'.join([f"<Step {num}> {desc}" for num, desc in sorted_steps])
        

    def __len__(self):
        return len(self.qa_ls)

    def __getitem__(self, index):
        return self.qa_ls[index]
    

class ReasoningEvaluationDatasetLLM(Dataset):
    def __init__(self, label_root, exits_root, original_data_root):
        self.label_root = label_root
        self.exist_root = exits_root
        self.original_data_root = original_data_root#.rsplit('.', 1)[0] + 'json'
        label_ls = []
        o_data = {}
        with open(self.original_data_root, 'r', encoding='utf-8') as f_a:
            for line in f_a:
                entry = json.loads(line)
                video_id = entry['video_id']
                for k, v in entry['one_step_mcq'].items():
                    tmp_id = video_id+'_'+k
                    o_data[tmp_id] = {}
                    o_data[tmp_id]['id'] = tmp_id
                    o_data[tmp_id]['question'] = v['original_question']
                    o_data[tmp_id]['answer'] = v['original_answer']
                    o_data[tmp_id]['type'] = v['reasoning_type']
                    o_data[tmp_id]['procedure'] = self.format_process(v['reasoning_process'])

        with open(self.label_root, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            for k, v in data_dict['video_results'].items():
                label_ls.append(v)

        data = []
        if os.path.exists(self.exist_root):
            with open(self.exist_root, 'r', encoding='utf-8-sig') as f:
                data = [json.loads(line) for line in f]

        exist_id_ls = [qa['id'][0] for qa in data]

        self.qa_ls = []
        for video in label_ls:
            for qa_k, qa_v  in video['question_results'].items():
                qa_k_id, typ = qa_k.rsplit('_', 1)
                if qa_k_id in exist_id_ls:
                    continue
                
                if typ == 'one':
                    qa_dict = o_data[qa_k_id]
                    qa_dict['id'] = qa_k_id
                    qa_dict['steps_and_answer'] = qa_v['model_response']
                    if not qa_v['model_response']:
                        continue
                    self.qa_ls.append(qa_dict)

        print(f"remain qa num: {len(self.qa_ls)}, exist qa num: {len(data)}")
    
    def format_process(self, procedure):
        if isinstance(procedure, str):
            return procedure
        sorted_steps = sorted(procedure.items(), key=lambda x: int(x[0]))
        return '\n'.join([f"<Step {num}> {desc}" for num, desc in sorted_steps])
        

    def __len__(self):
        return len(self.qa_ls) 

    def __getitem__(self, index):
        return self.qa_ls[index] 
    

if __name__ == '__main__':
    # dataset = ReasoningEvaluationDatasetVLM('qwen2_vl_7b_result_frames_64_max_token_256_0305.jsonl', '')
    dataset = ReasoningEvaluationDatasetLLM('eva_qwen2.5_7_1.jsonl', '','VRBench_eval_0305.jsonl')

    qa_list = dataset.qa_ls

    output_path = 'output_test2.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for qa_item in qa_list:
            json_str = json.dumps(qa_item, ensure_ascii=False)
            f.write(json_str + '\n')
    
    print(f"Successfully wrote {len(qa_list)} items to {output_path}")

