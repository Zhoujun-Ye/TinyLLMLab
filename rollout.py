"""Single-turn generation only: one prompt produces one or more full answers."""


def generate(policy, prompts, *, samples_per_prompt: int):
    raise NotImplementedError(
        "Implement generation with prompt/group IDs, response tokens, masks, "
        "old-policy log probabilities and finish reasons."
    )
