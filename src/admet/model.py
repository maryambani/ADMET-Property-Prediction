import torch
import torch.nn as nn


class Tox21Model(nn.Module):
    # fingerprint in -> 12 toxicity logits out
    def __init__(self, input_dim: int, n_tasks: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = dim
        layers.append(nn.Linear(prev, n_tasks))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_dim: int, n_tasks: int, config: dict) -> Tox21Model:
    model_cfg = config["model"]
    return Tox21Model(
        input_dim=input_dim,
        n_tasks=n_tasks,
        hidden_dims=model_cfg["hidden_dims"],
        dropout=model_cfg["dropout"],
    )
