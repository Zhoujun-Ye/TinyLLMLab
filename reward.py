"""Score an answer without exposing the reference answer to generation."""


def score(prompt: str, response: str, ground_truth) -> float:
    raise NotImplementedError("Implement your task reward; no default or synthetic score is provided.")
