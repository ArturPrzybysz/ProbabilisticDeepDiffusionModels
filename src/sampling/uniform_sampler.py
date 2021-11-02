import torch

class UniformSampler:
    def __init__(self, diffusion_steps):
        self.diffusion_steps = diffusion_steps

    def __call__(self, batch_size, device):
        t = torch.randint(1, self.diffusion_steps, (batch_size,), device=device)
        return t, None

