import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RLExample:
    id: str
    prompt: str
    ground_truth: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


def read_examples(path: str | Path) -> list[RLExample]:
    examples = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("Each row must be an object.")
                example = RLExample(**raw)
                if not isinstance(example.id, str) or not example.id.strip() or example.id in seen:
                    raise ValueError("IDs must be nonempty unique strings.")
                if not isinstance(example.prompt, str) or not example.prompt.strip():
                    raise ValueError("prompt must be a nonempty string.")
                if not isinstance(example.metadata, dict):
                    raise ValueError("metadata must be an object.")
                seen.add(example.id)
                examples.append(example)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return examples


def write_examples(path: str | Path, examples: Iterable[RLExample]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False, allow_nan=False) + "\n")
