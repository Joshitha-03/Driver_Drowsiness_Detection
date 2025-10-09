import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, csv_files, seq_len=30, stride=5):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        dfs = [pd.read_csv(f) for f in csv_files]
        self.df = pd.concat(dfs, ignore_index=True)
        feat = self.df[['ear','mar','head_angle']].values.astype(np.float32)
        mins = feat.min(axis=0)
        maxs = feat.max(axis=0)
        self.df[['ear','mar','head_angle']] = (feat - mins)/(maxs - mins + 1e-6)
        self.seq_len = seq_len
        self.stride = stride
        self.samples = []
        self.prepare_samples()

    def prepare_samples(self):
        arr = self.df[['ear','mar','head_angle']].values
        labels = self.df['label'].values
        for start in range(0, len(arr)-self.seq_len+1, self.stride):
            seq = arr[start:start+self.seq_len]
            lab = int(round(labels[start:start+self.seq_len].mean()))
            self.samples.append((seq.astype('float32'), np.int64(lab)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq), torch.tensor(label)
