"""Orchestrate generation, scoring, algorithm computation and parameter updates."""


def run_training(config):
    raise NotImplementedError(
        f"LLM RL training is not implemented (algorithm={config['training']['algorithm']}). "
        "Implement policy loading, rollout, reward, advantages/loss and updates first."
    )
