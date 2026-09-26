from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from admet.data import load_config
from admet.dataset import Tox21Dataset
from admet.metrics import compute_auc, masked_bce_with_logits
from admet.model import build_model


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_hparams(config: dict, epochs: int | None = None) -> dict:
    hparams = {
        "seed": config["seed"],
        "hidden_dims": config["model"]["hidden_dims"],
        "dropout": config["model"]["dropout"],
        "batch_size": config["training"]["batch_size"],
        "lr": config["training"]["lr"],
        "epochs": config["training"]["epochs"],
        "patience": config["training"]["patience"],
        "val_split": config["training"]["val_split"],
        "test_split": config["training"]["test_split"],
        "fingerprint_radius": config["data"]["fingerprint_radius"],
        "fingerprint_bits": config["data"]["fingerprint_bits"],
    }
    if epochs is not None:
        hparams["epochs"] = epochs
    return hparams


def make_loaders(dataset, val_split: float, test_split: float, batch_size: int, seed: int):
    n_val = int(len(dataset) * val_split)
    n_test = int(len(dataset) * test_split)
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    return loaders, sizes


class EarlyStopping:
    # stop when the metric hasn't improved for `patience` epochs
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value > self.best:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    ys, probs, masks = [], [], []

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = model(x)
        loss = masked_bce_with_logits(logits, y, mask)
        n = int(mask.sum().item())
        total_loss += loss.item() * n
        total_n += n

        ys.append(y.cpu().numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())
        masks.append(mask.cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    mask_np = np.concatenate(masks)
    aucs = compute_auc(y_true, y_prob, mask_np)
    mean_auc = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return total_loss / max(total_n, 1), mean_auc, aucs


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = masked_bce_with_logits(logits, y, mask)
        loss.backward()
        optimizer.step()

        n = int(mask.sum().item())
        total_loss += loss.item() * n
        total_n += n

    return total_loss / max(total_n, 1)


def train(
    config_path: str = "configs/default.yaml",
    epochs: int | None = None,
    use_wandb: bool | None = None,
):
    config = load_config(config_path)
    wandb_cfg = config.get("wandb", {})
    if use_wandb is None:
        use_wandb = wandb_cfg.get("enabled", False)

    hparams = build_hparams(config, epochs)

    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project=wandb_cfg.get("project", "admet-tox21"),
            config=hparams,
        )
        # during a sweep the agent overrides values, so read them back
        hparams = dict(run.config)

    set_seed(hparams["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = Tox21Dataset()
    loaders, sizes = make_loaders(
        dataset,
        val_split=hparams["val_split"],
        test_split=hparams["test_split"],
        batch_size=hparams["batch_size"],
        seed=hparams["seed"],
    )
    print(
        f"train: {sizes['train']}  val: {sizes['val']}  test: {sizes['test']}  "
        f"tasks: {dataset.y.shape[1]}"
    )

    model = build_model(
        input_dim=dataset.X.shape[1],
        n_tasks=dataset.y.shape[1],
        config={"model": {"hidden_dims": hparams["hidden_dims"], "dropout": hparams["dropout"]}},
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    n_epochs = hparams["epochs"]
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    best_path = out_dir / "best.pt"
    stopper = EarlyStopping(patience=hparams["patience"])
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
        val_loss, mean_auc, task_aucs = evaluate(model, loaders["val"], device)
        print(
            f"epoch {epoch:03d}/{n_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={mean_auc:.4f}"
        )

        if run is not None:
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": mean_auc,
            }
            for task_idx, auc in task_aucs.items():
                metrics[f"val_auc/{dataset.tasks[task_idx]}"] = auc
            run.log(metrics)

        if stopper.step(mean_auc):
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hparams": hparams,
                    "tasks": dataset.tasks,
                    "val_auc": mean_auc,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  saved best -> {best_path} (auc={mean_auc:.4f})")
        elif stopper.should_stop:
            print(f"early stopping: no val improvement for {stopper.patience} epochs")
            break

    # test set is evaluated once, with the weights that were best on val
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_auc, test_task_aucs = evaluate(model, loaders["test"], device)
    ckpt["test_auc"] = test_auc
    ckpt["test_task_aucs"] = {dataset.tasks[i]: a for i, a in test_task_aucs.items()}
    torch.save(ckpt, best_path)

    print(f"done. best epoch {best_epoch}  val auc {stopper.best:.4f}  test auc {test_auc:.4f}")

    if run is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_auc"] = stopper.best
        run.summary["test_auc"] = test_auc
        for task, auc in ckpt["test_task_aucs"].items():
            run.summary[f"test_auc/{task}"] = auc
        run.finish()

    return best_path, test_auc
