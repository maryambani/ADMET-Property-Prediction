import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from admet.train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb", dest="use_wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config, epochs=args.epochs, use_wandb=args.use_wandb)
