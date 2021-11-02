import torch
import numpy as np

from src.modules.stepwise_log import StepwiseLog


class ImportanceSampler:
    def __init__(self, diffusion_steps: int, loss_per_t: StepwiseLog, min_counts: int = 10):
        self.diffusion_steps = diffusion_steps
        self.loss_per_t = loss_per_t
        self._ready = False
        self.min_counts = min_counts

    def is_ready(self):
        if self._ready:
            return True
        else:
            if (self.loss_per_t.n_per_step >= self.min_counts).all():
                print("ImportanceSampler is warmed up now")
                self._ready = True
                return True
            return False

    def __call__(self, batch_size, device):
        if self.is_ready():
            p = self.loss_per_t.avg_sq_per_step + 1e-6 # in case of 0s
            p = p / p.sum()
            indices = np.random.choice(self.diffusion_steps-1, size=(batch_size,), p=p)
            weights = 1 / (p[indices] * batch_size)
            t = torch.from_numpy(indices).long().to(device) + 1 # +1 because of 1 indexed t
            return t, weights
        else:
            t = torch.randint(1, self.diffusion_steps, (batch_size,), device=device)
            return t, None
