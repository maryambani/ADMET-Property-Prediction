from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class Tox21Dataset(Dataset):
    def __init__(self, data_dir: str | Path = "data/processed"):
        data_dir = Path(data_dir)
        X = np.load(data_dir / "X.npy")
        y = np.load(data_dir / "y.npy")

        self.X = torch.from_numpy(X)
        self.mask = torch.from_numpy(~np.isnan(y))
        self.y = torch.from_numpy(np.nan_to_num(y, nan=0.0))

        tasks_file = data_dir / "tasks.txt"
        if tasks_file.exists():
            self.tasks = tasks_file.read_text().strip().split("\n")
        else:
            self.tasks = [f"task_{i}" for i in range(self.y.shape[1])]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.mask[idx]
