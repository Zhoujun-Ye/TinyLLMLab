"""Implement grouped advantages and the GRPO objective here."""


def compute_advantages(rewards, group_ids):
    raise NotImplementedError("Implement grouped advantage estimation and zero-variance handling.")


def compute_loss(batch):
    raise NotImplementedError("Define the tensor batch contract and implement your GRPO objective.")
