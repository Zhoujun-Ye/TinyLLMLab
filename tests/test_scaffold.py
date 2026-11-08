import tempfile
import unittest
from pathlib import Path

from train import train
from utils import load_config
from datasets.dataset import RLExample, read_examples, write_examples
from models.policy import load_policy
from rollout import generate
from reward import score
from algorithms import grpo, ppo

ROOT = Path(__file__).resolve().parents[1]


class ScaffoldTests(unittest.TestCase):
    def test_configs_and_training_boundary(self):
        for algorithm in ("grpo", "ppo"):
            config = load_config(ROOT / f"configs/{algorithm}.toml")
            self.assertEqual(config["training"]["algorithm"], algorithm)
            with self.assertRaises(NotImplementedError):
                train(config)

    def test_bad_config_rejected(self):
        base = (ROOT / "configs/grpo.toml").read_text()
        for text in (base.replace('algorithm = "grpo"', 'algorithm = "unknown"'),
                     base.replace('train = "data/processed/train.jsonl"', 'train = ""')):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_roundtrip(self):
        examples = read_examples(ROOT / "data/examples/prompts.jsonl")
        self.assertEqual(len(examples), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            write_examples(path, examples)
            self.assertEqual(read_examples(path), examples)

    def test_duplicate_id_rejected(self):
        example = RLExample("duplicate", "A question")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            write_examples(path, [example, example])
            with self.assertRaises(ValueError):
                read_examples(path)

    def test_invalid_rows_report_location(self):
        for row in ('[]', '{}', '{"id":"x","prompt":""}',
                    '{"id":"x","prompt":1}', '{"id":"x","prompt":"q","metadata":[]}',
                    '{"id":"x","prompt":"q","response":"unexpected"}'):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.jsonl"
                path.write_text(row)
                with self.assertRaisesRegex(ValueError, r"bad.jsonl:1:"):
                    read_examples(path)

    def test_algorithms_and_components_are_unimplemented(self):
        calls = [
            lambda: load_policy("unused"),
            lambda: generate(None, ["question"], samples_per_prompt=2),
            lambda: score("question", "answer", "reference"),
            lambda: grpo.compute_advantages([0, 1], ["group", "group"]),
            lambda: grpo.compute_loss({}),
            lambda: ppo.compute_advantages({}),
            lambda: ppo.compute_loss({}),
        ]
        for call in calls:
            with self.assertRaises(NotImplementedError):
                call()

    def test_no_package_setup(self):
        self.assertFalse((ROOT / "pyproject.toml").exists())
        self.assertFalse(list((ROOT / "algorithms").rglob("__init__.py")))


if __name__ == "__main__":
    unittest.main()
