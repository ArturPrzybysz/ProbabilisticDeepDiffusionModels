import torch
import pytorch_lightning as pl

from src.modules import get_model
import numpy as np

from src.utils import mean_flat


def get_betas(b0, bmax, diffusion_steps, mode="linear"):
    if mode=="linear":
        return torch.linspace(b0, bmax, diffusion_steps)

class Engine(pl.LightningModule):
    def __init__(self, model_config, optimizer_config, diffusion_steps=100,
                 b0=1e-3, bmax=0.02, mode="linear", resolution=32):
        super(Engine, self).__init__()
        # create the model here
        self.model = get_model(dict(model_config))
        print(self.model)
        self.optimizer_config = optimizer_config
        self.diffusion_steps = diffusion_steps
        self.resolution = resolution

        print(self.device)
        self.betas = get_betas(b0, bmax, diffusion_steps, mode).to(self.device)
        self.alphas = 1 - self.betas
        print(self.alphas)
        self.alphas_hat = torch.cumprod(self.alphas, 0)
        print(self.alphas_hat)
        self.alphas_hat_sqrt = torch.sqrt(self.alphas_hat)
        self.one_min_alphas_hat_sqrt = torch.sqrt(1 - self.alphas_hat)
        # TODO: precompute sqrts etc.

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_config)


    def get_q_t(self, x, noise, t):
        return x * self.alphas_hat_sqrt[t-1].view((-1, 1, 1, 1)).to(self.device) \
               + self.one_min_alphas_hat_sqrt[t-1].view((-1, 1, 1, 1)).to(self.device) * noise

    def get_loss(self, predicted_noise, target_noise):
        # TODO: should batch be averaged or summed?
        return torch.mean(torch.square(target_noise - predicted_noise))

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # TODO: sample t
        # TODO get x_t
        x, y = batch
        batch_size = x.shape[0]
        t = torch.randint(1, self.diffusion_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x)
        x_t = self.get_q_t(x, noise, t)
        predicted_noise = self.model(x_t, t)
        loss = self.get_loss(predicted_noise, noise)

        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def generate_image(self, n=1):
        x_t = torch.randn((n, self.model.in_channels, self.resolution, self.resolution)).to(self.device)
        for t in range(self.diffusion_steps, 0, -1):
            if t > 1:
                z = torch.randn_like(x_t).to(self.device)
            else:
                z = 0
            epsilon = self.model(x_t, t * torch.ones(n).to(self.device))
            epsilon_scaled = (self.betas[t-1]/self.one_min_alphas_hat_sqrt[t-1]).to(self.device) * epsilon
            sigma = torch.sqrt(self.betas[t-1]).to(self.device)
            x_t = (x_t - epsilon_scaled) / self.alphas[t-1].to(self.device) - sigma * z

        return x_t.detach().cpu().numpy()
    # def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
    #     raise NotImplementedError

