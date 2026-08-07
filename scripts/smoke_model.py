import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from admet.data import load_config
from admet.dataset import Tox21Dataset
from admet.model import build_model

if __name__ == "__main__":
    config = load_config()
    ds = Tox21Dataset()
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)

    model = build_model(input_dim=ds.X.shape[1], n_tasks=ds.y.shape[1], config=config)
    x, y, mask = next(iter(loader))
    logits = model(x)

    print(f"dataset: {len(ds)} molecules, {ds.y.shape[1]} tasks")
    print(f"batch x: {tuple(x.shape)}")
    print(f"batch logits: {tuple(logits.shape)}")
    print(f"labeled in batch: {mask.sum().item()} / {mask.numel()}")
