import numpy as np
from sklearn.metrics import roc_auc_score


def masked_bce_with_logits(logits, y, mask):
    import torch.nn.functional as F

    # per-element loss, then average only over labeled assays
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    mask = mask.float()
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray) -> dict:
    """Per-task ROC-AUC. Skips a task if it only has one class in the labeled rows."""
    n_tasks = y_true.shape[1]
    scores = {}
    for t in range(n_tasks):
        m = mask[:, t].astype(bool)
        if m.sum() == 0:
            continue
        yt = y_true[m, t]
        yp = y_prob[m, t]
        if len(np.unique(yt)) < 2:
            continue
        scores[t] = float(roc_auc_score(yt, yp))
    return scores
