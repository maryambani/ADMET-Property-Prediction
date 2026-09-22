from pathlib import Path

import torch
from rdkit.Chem import Draw

from admet.featurize import smiles_to_fingerprint, smiles_to_mol
from admet.model import build_model


class Predictor:
    def __init__(self, checkpoint_path: str | Path = "checkpoints/best.pt"):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"no checkpoint at {checkpoint_path}, run scripts/train.py first"
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.hparams = ckpt["hparams"]
        self.tasks = ckpt["tasks"]
        self.val_auc = ckpt.get("val_auc")
        self.epoch = ckpt.get("epoch")

        self.model = build_model(
            input_dim=self.hparams["fingerprint_bits"],
            n_tasks=len(self.tasks),
            config={
                "model": {
                    "hidden_dims": self.hparams["hidden_dims"],
                    "dropout": self.hparams["dropout"],
                }
            },
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, smiles: str) -> dict[str, float]:
        fp = smiles_to_fingerprint(
            smiles,
            radius=self.hparams["fingerprint_radius"],
            n_bits=self.hparams["fingerprint_bits"],
        )
        x = torch.from_numpy(fp).unsqueeze(0)
        probs = torch.sigmoid(self.model(x)).squeeze(0).tolist()
        return dict(zip(self.tasks, probs))


def draw_molecule(smiles: str, size: tuple[int, int] = (350, 300)):
    mol = smiles_to_mol(smiles)
    return Draw.MolToImage(mol, size=size)
