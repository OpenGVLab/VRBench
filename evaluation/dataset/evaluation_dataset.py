import json
import re
import os
from torch.utils.data import Dataset, DataLoader

class ReasoningEvaluationDataset(Dataset):
    def __init__(self, label_root, exits_root):
        self.label_root = label_root
        self.exist_root = exits_root
        self.label_ls = []
        with open(self.label_root, 'r') as f:
            self.label_ls = [json.loads(line) for line in f]

        self.exist_ls = []
        if os.path.exists(self.exist_root):
            with open(self.exist_root, 'r') as f:
                self.exist_ls = [json.loads(line) for line in f]

        exist_id_ls = [qa['id'][0] for qa in self.exist_ls]
        self.label_ls = [qa for qa in self.label_ls if qa['id'] not in exist_id_ls]

        print(f"remain qa num: {len(self.label_ls)}, exist qa num: {len(self.exist_ls)}")

        
    def __len__(self):
        return len(self.label_ls)

    def __getitem__(self, index):
        return self.label_ls[index]