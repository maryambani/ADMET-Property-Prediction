from pathlib import Path

import torch
from rdkit.Chem import Draw

from admet.featurize import fingerprint_and_bits, smiles_to_fingerprint, smiles_to_mol
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
        self.test_auc = ckpt.get("test_auc")
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

    def atom_contributions(self, smiles: str, task: str) -> dict[int, float]:
        # gradient of one assay logit w.r.t. the fingerprint, mapped back onto atoms
        task_idx = self.tasks.index(task)
        mol = smiles_to_mol(smiles)
        fp, bits = fingerprint_and_bits(
            mol,
            radius=self.hparams["fingerprint_radius"],
            n_bits=self.hparams["fingerprint_bits"],
        )
        x = torch.from_numpy(fp).unsqueeze(0)
        x.requires_grad_(True)
        logit = self.model(x)[0, task_idx]
        (grad,) = torch.autograd.grad(logit, x)
        contrib = (grad * x).squeeze(0).detach().numpy()

        scores: dict[int, float] = {}
        for bit, hits in bits.items():
            weight = float(contrib[bit])
            if weight <= 0:
                continue
            for center, _radius in hits:
                scores[center] = scores.get(center, 0.0) + weight
        return scores


def draw_molecule(smiles: str, size: tuple[int, int] = (350, 300), atom_scores: dict | None = None):
    mol = smiles_to_mol(smiles)
    if not atom_scores:
        return Draw.MolToImage(mol, size=size)

    top = max(atom_scores.values())
    # keep the stronger atoms. a flat tint on every atom is hard to read
    keep = {atom: score for atom, score in atom_scores.items() if score >= 0.35 * top}
    colors = {
        atom: (1.0, 1.0 - 0.75 * score / top, 1.0 - 0.75 * score / top)
        for atom, score in keep.items()
    }
    return Draw.MolToImage(
        mol,
        size=size,
        highlightAtoms=list(keep),
        highlightAtomColors=colors,
    )
