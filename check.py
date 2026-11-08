import argparse
import json

from utils import load_config
from datasets.dataset import read_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyLLMLab scaffold utilities (no algorithms bundled).")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-config")
    inspect.add_argument("--config", default="configs/grpo.toml")
    validate = commands.add_parser("validate-data")
    validate.add_argument("path")
    args = parser.parse_args()
    if args.command == "inspect-config":
        print(json.dumps(load_config(args.config), indent=2))
    elif args.command == "validate-data":
        print(f"Validated {len(read_examples(args.path))} RL prompts")


if __name__ == "__main__":
    main()
