import torch

from admet.metrics import masked_bce_with_logits
from admet.model import Tox21Model


def test_masked_loss_ignores_unlabeled():
    logits = torch.zeros(2, 3)
    y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])

    loss = masked_bce_with_logits(logits, y, mask)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_train_one_epoch_runs():
    from admet.train import train_one_epoch

    model = Tox21Model(input_dim=16, n_tasks=3, hidden_dims=[8], dropout=0.0)
    x = torch.randn(4, 16)
    y = torch.randint(0, 2, (4, 3)).float()
    mask = torch.ones(4, 3, dtype=torch.bool)
    loader = [((x, y, mask))]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))
    assert loss > 0
