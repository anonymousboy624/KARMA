"""
etth2_dataset.py
----------------
PyTorch Dataset and DataLoaders for the ETT-h2 (hourly) forecasting data.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_DEFAULT_DIR = "/workspaces/KARMA-/data/generated/etth2"


class ETTh2Dataset(Dataset):
    def __init__(self, data_dir=_DEFAULT_DIR, split="train"):
        self.X = np.load(os.path.join(data_dir, f"X_{split}.npy")).astype(np.float32)
        self.y = np.load(os.path.join(data_dir, f"y_{split}.npy")).astype(np.float32)
        print(f"Loaded {split}: X={self.X.shape}  y={self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def get_dataloaders(data_dir=_DEFAULT_DIR, batch_size=32, num_workers=2):
    train = DataLoader(ETTh2Dataset(data_dir, "train"), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val   = DataLoader(ETTh2Dataset(data_dir, "val"),   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test  = DataLoader(ETTh2Dataset(data_dir, "test"),  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train, val, test
