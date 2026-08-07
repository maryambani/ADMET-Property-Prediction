import torch

from admet.data import load_config
from admet.dataset import Tox21Dataset
from admet.model import Tox21Model, build_model


def test_forward_pass_shape():
    model = Tox21Model(input_dim=2048, n_tasks=12, hidden_dims=[512, 256], dropout=0.3)
    x = torch.randn(8, 2048)
    out = model(x)
    assert out.shape == (8, 12) # must be of the same shape


def test_build_model_from_config():
    config = load_config()
    model = build_model(input_dim=2048, n_tasks=12, config=config)
    out = model(torch.randn(4, 2048))
    assert out.shape == (4, 12)


def test_dataset_returns_mask():
    ds = Tox21Dataset()
    x, y, mask = ds[0]
    assert x.shape == (2048,)
    assert y.shape == (12,)
    assert mask.shape == (12,)
    assert mask.dtype == torch.bool
