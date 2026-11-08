"""Optional later step: implement PPO, including value/return handling."""


def compute_advantages(batch):
    raise NotImplementedError("Implement PPO advantages/returns using the required value estimates.")


def compute_loss(batch):
    raise NotImplementedError("Implement PPO policy/value objectives and token masking.")
