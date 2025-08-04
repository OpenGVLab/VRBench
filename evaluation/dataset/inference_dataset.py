import json
import re
import os
from torch.utils.data import Dataset

class VideoMultiStepReasoningDataset(Dataset):
    def __init__(self, label_root, exist_root):
        self.label_root = label_root
        self.exist_root = exist_root
        self.label_ls = []
        self.exist_ls = []
        with open(self.label_root, 'r') as f:
            self.label_ls = [json.loads(line) for line in f]
        self.qa_ls = []
        for video in self.label_ls:
            for qa in video['question_answer_list']:
                qa['video_summary'] = video['video_summary']
                self.qa_ls.append(qa)
        data = []
        if os.path.exists(self.exist_root):
            with open(exist_root, 'r') as f:
                data = [json.loads(line) for line in f]
        self.exist_ls = [qa['id'] for qa in data]
        self.qa_ls = [qa for qa in self.qa_ls if qa['id'] not in self.exist_ls]
        print(f"load {len(self.qa_ls)} qa, exist {len(self.exist_ls)} qa, total {len(self.qa_ls) + len(self.exist_ls)} qa")

    def __len__(self):
        return len(self.qa_ls)

    def __getitem__(self, index):
        return (
            self.qa_ls[index]['id'],
            self.qa_ls[index]['time_range'],
            self.qa_ls[index]['question'],
            self.qa_ls[index]['answer'],
            self.qa_ls[index]['procedure'],
            self.qa_ls[index]['multiple_choice_question'],
            self.qa_ls[index]['type'],
            self.qa_ls[index]['video_summary']
        )