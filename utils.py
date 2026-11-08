import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    algorithm = config.get("training", {}).get("algorithm")
    if algorithm not in {"grpo", "ppo"}:
        raise ValueError("training.algorithm must be grpo or ppo.")
    for split in ("train", "validation"):
        value = config.get("data", {}).get(split)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{split} must be a nonempty path string.")
    return config
