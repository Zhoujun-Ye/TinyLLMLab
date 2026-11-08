import argparse
from typing import Any

from utils import load_config
from trainer import run_training


def train(config: dict[str, Any]) -> None:
    run_training(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training placeholder; no algorithm is implemented.")
    parser.add_argument("--config", default="configs/grpo.toml")
    args = parser.parse_args()
    try:
        train(load_config(args.config))
    except NotImplementedError as error:
        parser.exit(2, f"{error}\n")
