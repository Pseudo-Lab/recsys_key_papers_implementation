import torch
import torch.nn as nn
import pandas as pd
import random

from torch.utils.data import Dataset

PAD = 0
MASK = 1

class Bert4RecDataset(Dataset):
    def __init__(self, data_csv, group_by_col, split_mode, train_history, valid_history, data_col) -> None:
        super().__init__()
        self.data_csv = data_csv
        self.group_by_col = group_by_col
        self.split_mode = split_mode
        self.train_history = train_history
        self.valid_history = valid_history
        self.data_col = data_col
        
        self.groups_df = self.data_csv.groupby(by=self.group_by_col)
        self.groups = list(self.groups_df.groups)
        
    def __getitem__(self, index):
        group = self.groups[index]
        group_df = self.groups_df.get_group(group)
        sequnece = self.get_sequence(group_df)
        
        trg_items = sequnece[self.data_col].tolist()
        
        if self.split_mode == "train":
            src_items = self.mask_seqeunce(trg_items)
        else:
            src_items = self.mask_last_elements_sequence(trg_items)
        
        trg_items = self.pad_sequence(trg_items)
        src_items = self.pad_sequence(src_items)
        
        trg_mask = [ 1 if t != PAD else 0 for t in trg_items]
        src_mask = [ 1 if t != PAD else 0 for t in src_items]
        
        src_items = torch.tensor(src_items, dtype=torch.long)
        trg_items = torch.tensor(trg_items, dtype=torch.long)
        src_mask = torch.tensor(src_mask, dtype=torch.long)
        trg_mask = torch.tensor(trg_mask, dtype=torch.long)
        
        return {
            "source": src_items,
            "target": trg_items,
            "source_mask": src_mask,
            "target_mask": trg_mask
        }
    
    def __len__(self):
        return len(self.groups)
    
    def get_sequence(self, group_df: pd.DataFrame):
        if self.split_mode == "train":
            _ = group_df.shape[0] - self.valid_history
            end_idx = random.randint(10, _ if _ >= 10 else 10)
        else:
            end_idx = group_df.shape[0]
        
        start_idx = max(0, end_idx - self.train_history)
        
        sequnece = group_df[start_idx:end_idx]
        
        return sequnece
    
    def mask_seqeunce(self, sequence: list, p: float = 0.8):
        return [ s if random.random() < p else MASK for s in sequence]
    
    def mask_last_elements_sequence(self, sequence):
        return sequence[:-self.valid_history] + self.mask_seqeunce(sequence[-self.valid_history:], 0.5)
    
    def pad_sequence(self, token: list):
        if len(token) < self.train_history:
            token = token + [PAD] * (self.train_history - len(token))
        return token
    
if __name__ == "__main__":
    
    df = pd.read_csv("../data/ml-1m/ml-1m.txt", sep=" ")
    df.columns = ["userId", "movieId"]
    
    dataset = Bert4RecDataset(df, "userId", "train", 120, 5, "movieId")
    print(dataset.__getitem__(1))