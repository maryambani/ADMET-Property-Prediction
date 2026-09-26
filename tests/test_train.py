import torch
from torch.utils.data import TensorDataset

from admet.metrics import masked_bce_with_logits
from admet.model import Tox21Model
from admet.train import EarlyStopping, make_loaders


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2)
    assert stopper.step(0.70) is True
    assert stopper.step(0.75) is True
    assert stopper.step(0.74) is False
    assert not stopper.should_stop
    assert stopper.step(0.73) is False
    assert stopper.should_stop
    assert stopper.best == 0.75


def test_make_loaders_three_way_split_is_disjoint():
    n = 100
    ds = TensorDataset(torch.arange(n), torch.zeros(n), torch.ones(n, dtype=torch.bool))
    loaders, sizes = make_loaders(ds, val_split=0.15, test_split=0.15, batch_size=16, seed=0)

    assert sizes == {"train": 70, "val": 15, "test": 15}
    seen = {}
    for name, loader in loaders.items():
        seen[name] = {int(i) for x, _, _ in loader for i in x}
    assert len(seen["train"] | seen["val"] | seen["test"]) == n
    assert not (seen["train"] & seen["test"])
    assert not (seen["val"] & seen["test"])


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
