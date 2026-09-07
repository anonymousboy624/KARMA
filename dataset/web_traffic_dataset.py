"""
web_traffic_dataset.py
----------------------
PyTorch Dataset class for the prepared web traffic time series data.

This class loads the preprocessed sequences and provides them for training deep learning models.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class WebTrafficDataset(Dataset):
    """
    Dataset for web traffic time series forecasting.

    Loads preprocessed sequences from numpy files and provides them as PyTorch tensors.

    Parameters
    ----------
    data_dir : str
        Directory containing the preprocessed data files
    split : str
        Which split to load: 'train', 'val', or 'test'
    """

    def __init__(
        self,
        data_dir="/workspaces/KARMA-/data/generated/web_traffic",
        split="train",
    ):
        self.data_dir = data_dir
        self.split = split

        # Load data
        self.X = np.load(os.path.join(data_dir, f"X_{split}.npy")).astype(np.float32)
        self.y = np.load(os.path.join(data_dir, f"y_{split}.npy")).astype(np.float32)

        print(f"Loaded {split} data: X shape {self.X.shape}, y shape {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # (seq_length, D)
        y = torch.from_numpy(self.y[idx])  # (pred_horizon, D)

        return x, y


def get_dataloaders(
    data_dir="/workspaces/KARMA-/data/generated/web_traffic",
    batch_size=32,
    num_workers=2,
):
    """
    Create PyTorch DataLoaders for train/val/test splits.

    Parameters
    ----------
    data_dir : str
        Directory containing the preprocessed data
    batch_size : int
        Batch size for DataLoaders
    num_workers : int
        Number of workers for DataLoaders

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        PyTorch DataLoaders for each split
    """

    train_dataset = WebTrafficDataset(data_dir, "train")
    val_dataset = WebTrafficDataset(data_dir, "val")
    test_dataset = WebTrafficDataset(data_dir, "test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = WebTrafficDataset()
    print(f"Dataset length: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample x shape: {x.shape}, y shape: {y.shape}")
