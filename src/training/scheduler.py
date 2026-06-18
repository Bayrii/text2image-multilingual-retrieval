"""Linear warmup + cosine decay LR scheduler."""
from __future__ import annotations

import math
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int,
                           min_lr_ratio: float = 0.0) -> LambdaLR:
    """LR ramps 0 -> 1 over warmup_steps, then cosine decays to min_lr_ratio."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
