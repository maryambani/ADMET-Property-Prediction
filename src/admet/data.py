import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from admet.featurize import smiles_to_fingerprint

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_tox21(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "tox21.csv.gz"
    if out.exists():
        return out

    print(f"downloading tox21 -> {out}")
    urllib.request.urlretrieve(TOX21_URL, out)
    return out


def load_tox21_df(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f)


def get_task_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("mol_id", "smiles")]


def featurize_tox21(df: pd.DataFrame, radius: int, n_bits: int):
    task_cols = get_task_columns(df)
    labels = df[task_cols].to_numpy(dtype=np.float32)

    fingerprints = []
    valid_idx = []

    for i, smiles in enumerate(df["smiles"]):
        try:
            fp = smiles_to_fingerprint(smiles, radius=radius, n_bits=n_bits)
        except ValueError:
            continue
        fingerprints.append(fp)
        valid_idx.append(i)

    X = np.stack(fingerprints)
    y = labels[valid_idx]

    return X, y, task_cols


def prepare_dataset(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    data_cfg = config["data"]

    raw_path = download_tox21(Path("data/raw"))
    df = load_tox21_df(raw_path)

    X, y, task_cols = featurize_tox21(
        df,
        radius=data_cfg["fingerprint_radius"],
        n_bits=data_cfg["fingerprint_bits"],
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    (out_dir / "tasks.txt").write_text("\n".join(task_cols))

    print(f"molecules: {len(df)} -> {X.shape[0]} after dropping bad SMILES")
    print(f"X shape: {X.shape}")  # (n_molecules, n_bits)
    print(f"y shape: {y.shape}")  # (n_molecules, n_tasks)
    print(f"tasks: {task_cols}")
