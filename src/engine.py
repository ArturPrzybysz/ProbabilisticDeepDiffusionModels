import math
from collections import defaultdict
from contextlib import contextmanager

import torch
import pytorch_lightning as pl
# from torch_ema import ExponentialMovingAverage

from src.modules import get_model
import numpy as np

from src.modules.ema import Ema
from src.modules.stepwise_log import StepwiseLog
from src.sampling.importance_sampler import ImportanceSampler
from src.sampling.uniform_sampler import UniformSampler
from src.utils import mean_flat, get_generator_if_specified


# TODO: what is this
def alpha_bar(t):
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


def get_betas(
    beta_start=None, beta_end=None, diffusion_steps=1000, mode="linear", max_beta=0.999
):
    if mode == "linear":
        if beta_start is None or beta_end is None:
            # scale to the number of steps
            scale = 1000 / diffusion_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, diffusion_steps)
    elif mode == "cosine":
        # TODO: what is this
        betas = []
        for i in range(diffusion_steps):
            t1 = i / diffusion_steps
            t2 = (i + 1) / diffusion_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas)
    else:
        raise ValueError(f"Wrong beta mode: {mode}")


class Engine(pl.LightningModule):
    def __init__(
        self,
        model_config,
        optimizer_config,
        diffusion_steps=1000,
        beta_start=None,
        beta_end=None,
        mode="linear",
        sigma_mode="beta",
        resolution=32,
        clip_while_generating=True,
        sampling="uniform",
        ema=None
    ):
        super(Engine, self).__init__()
        self.save_hyperparameters()  # ??

        self.clip_while_generating = clip_while_generating

        # create the model here
        self.model = get_model(resolution, dict(model_config))

        # exponential moving average
        if ema is not None:
            # self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema)
            # self.ema.to(self.device)
            self.ema_model = get_model(resolution, dict(model_config))
            self.ema = Ema(self.ema_model, decay=ema)
        else:
            self.ema = None

        print(self.model)
        self.optimizer_config = optimizer_config
        self.diffusion_steps = diffusion_steps
        self.resolution = resolution

        self.sigma_mode = sigma_mode
        self.betas = get_betas(beta_start, beta_end, diffusion_steps, mode).to(
            self.device
        )
        self.alphas = 1 - self.betas
        # print(self.alphas)
        self.alphas_hat = torch.cumprod(self.alphas, 0)
        # print(self.alphas_hat)
        self.alphas_hat_sqrt = torch.sqrt(self.alphas_hat)
        self.one_min_alphas_hat_sqrt = torch.sqrt(1 - self.alphas_hat)

        self.alphas_hat_prev = np.append(1.0, self.alphas_hat[:-1])
        self.alphas_hat_next = np.append(self.alphas_hat[1:], 0.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_hat_prev) / (1.0 - self.alphas_hat)
        )

        self.loss_per_t = StepwiseLog(diffusion_steps, 10)
        self.loss_per_t_epoch = StepwiseLog(diffusion_steps)

        if sampling == "uniform":
            self.sampler = UniformSampler(diffusion_steps=diffusion_steps)
        elif sampling == "importance":
            self.sampler = ImportanceSampler(diffusion_steps=diffusion_steps, loss_per_t=self.loss_per_t, min_counts=20)
        else:
            raise ValueError(f"Unknown sampling option: \"{sampling}\"")

    @contextmanager
    def ema_on(self):
        if self.ema is None:
            yield
        else:
            try:
                self.original_model = self.model
                self.model = self.ema.module
                yield
            finally:
                self.model = self.original_model


    def on_epoch_end(self) -> None:
        # log loss per Q
        for i in range(4):
            self.log(
                f"loss_q{i+1}",
                self.loss_per_t_epoch.get_avg_in_range(max(1, int(i*self.diffusion_steps/4)),
                                                       int((i+1)*self.diffusion_steps/4)),
                on_step=False, on_epoch=True, prog_bar=False
            )
        self.loss_per_t_epoch.reset()



    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            # self.ema.to(self.device)
            # self.ema.update()
            self.ema.update(self.model)
            # self.ema.to('cpu')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_config)

    # ------------ Training and diffusion stuff ----------

    def get_q_t(self, x, noise, t):
        return (
            x * self.alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)
            + self.one_min_alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)
            * noise
        )

    def get_loss(self, predicted_noise, target_noise, t, weights=None, update_loss_log=True):
        loss = mean_flat(torch.square(target_noise - predicted_noise))
        if update_loss_log:
            losses = loss.detach().cpu().numpy().tolist()
            ts = t.detach().cpu().numpy().tolist()
            self.loss_per_t.update(ts, losses)
            self.loss_per_t_epoch.update(ts, losses)


        # TODO: should batch be averaged or summed?
        if weights is not None:
            return torch.sum(weights * loss)
        else:
            return torch.mean(loss)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        batch_size = x.shape[0]
        t, weights = self.sampler(batch_size, self.device)
        noise = torch.randn_like(x)
        x_t = self.get_q_t(x, noise, t)
        predicted_noise = self.model(x_t, t)
        loss = self.get_loss(predicted_noise, noise, weights=weights, t=t, update_loss_log=True)

        total_norm = self.compute_grad_norm(self.model.parameters())
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "total_grad_norm_L2",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        return loss

    # TODO: do some validation

    def compute_grad_norm(self, parameters, norm_type=2):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.0)
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
        return total_norm

    def get_sigma(self, t):
        if self.sigma_mode == "beta":
            return torch.sqrt(self.betas[t])
        elif self.sigma_mode == "beta_tilde":
            variance = self.posterior_variance[t]
            return torch.sqrt(variance)
        else:
            raise ValueError(f"Wrong sigma mode: {self.sigma_mode}")

    # ------------ Sampling and generation utils ----------

    def denoising_step(self, x_t, t, mean_only=False, generator=None):
        epsilon = self.model(x_t, t * torch.ones(x_t.shape[0]).to(self.device))
        epsilon *= (self.betas[t - 1] / self.one_min_alphas_hat_sqrt[t - 1]).to(
            self.device
        )
        sigma = self.get_sigma(t - 1).to(self.device)

        x_t -= epsilon
        x_t /= self.alphas[t - 1].to(self.device)

        if not mean_only:
            if t > 1:
                z = torch.randn(
                    x_t.shape, generator=generator, device=self.device, dtype=x_t.dtype
                )
            else:
                z = 0
            x_t -= sigma * z
        if self.clip_while_generating:
            x_t = x_t.clamp(-1, 1)
        return x_t

    def sample_from_step(self, x_t, t_start, mean_only=False, generator=None):
        for t in range(t_start, 0, -1):
            x_t = self.denoising_step(x_t, t, mean_only=mean_only, generator=generator)
        return x_t

    # ------------ Image generation endpoints ----------

    @torch.no_grad()
    def sample_and_return_steps(
        self,
        x_t,
        t_start=None,
        steps_to_return=(1,),
        mean_only=False,
        generator=None,
        seed=None,
        return_stds=False
    ):
        """Returns shape [B, STEPS, C, W, H]"""
        if t_start is None:
            t_start = self.diffusion_steps
        if generator is None:
            generator = get_generator_if_specified(seed, device=self.device)

        assert all(t < t_start for t in steps_to_return)

        self.eval()

        batch_size = x_t.shape[0]
        step_count = len(steps_to_return)
        image_shape = x_t.shape[1:]

        output = torch.zeros((batch_size, step_count) + image_shape)
        current_step_idx = 0

        if return_stds:
            stds = [torch.std(x_t).detach().cpu().item()]

        for t in range(t_start, 0, -1):
            x_t = self.denoising_step(x_t, t, mean_only=mean_only, generator=generator)

            if t in steps_to_return:
                output[:, current_step_idx] = x_t
                current_step_idx += 1

            if return_stds:
                stds.append(torch.std(x_t).detach().cpu().item())

        if return_stds:
            return output, stds

        return output

    @torch.no_grad()
    def generate_images(self, n=1, minibatch=4, mean_only=False, seed=None):
        self.eval()
        generator = get_generator_if_specified(seed, device=self.device)
        images = []

        for i in range(np.ceil(n / minibatch).astype(int)):
            x_t = torch.randn(
                (n, self.model.in_channels, self.resolution, self.resolution),
                generator=generator,
                device=self.device,
            )

            x_t = self.sample_from_step(
                x_t, self.diffusion_steps, mean_only=mean_only, generator=generator
            )
            images.append(x_t.detach().cpu().numpy())

        return np.concatenate(images, axis=0)

    @torch.no_grad()
    def generate_images_grid(
        self, steps_to_return, n=1, minibatch=4, mean_only=False, seed=None
    ):
        self.eval()
        generator = get_generator_if_specified(seed, device=self.device)
        starting_noise = []
        images = []

        for i in range(np.ceil(n / minibatch).astype(int)):
            x_t = torch.randn(
                (n, self.model.in_channels, self.resolution, self.resolution),
                generator=generator,
                device=self.device,
            )
            starting_noise.append(x_t.detach().cpu().numpy())

            x_t = self.sample_and_return_steps(
                x_t,
                self.diffusion_steps,
                steps_to_return=steps_to_return,
                mean_only=mean_only,
                generator=generator,
            )
            images.append(x_t.detach().cpu().numpy())

        return np.concatenate(starting_noise, axis=0), np.concatenate(images, axis=0)

    @torch.no_grad()
    def get_noised_representation(self, x0, t=None, seed=None, generator=None):
        """Will apply forward process to x0 up to t steps and then reconstruct."""
        if t is None:
            t = self.diffusion_steps
        if generator is None:
            generator = get_generator_if_specified(seed, device=self.device)
        x0 = x0.to(self.device)
        noise = torch.randn(
            x0.shape, generator=generator, device=self.device, dtype=x0.dtype
        )
        return self.get_q_t(x0, noise, t)

    @torch.no_grad()
    def diffuse_and_reconstruct(self, x0, t=None, seed=None):
        """Will apply forward process to x0 up to t steps and then reconstruct."""
        self.eval()
        if t is None:
            t = self.diffusion_steps
        generator = get_generator_if_specified(seed, device=self.device)
        x_t = self.get_noised_representation(x0, t, generator=generator)
        return self.sample_from_step(x_t.detach().clone(), t, generator=generator), x_t

    @torch.no_grad()
    def diffuse_and_reconstruct_grid(
        self, x0, t_start=None, steps_to_return=(1,), seed=None, mean_only=False, return_stds=False
    ):
        """Will apply forward process to x0 up to t steps and then reconstruct, finally return all selected steps."""
        self.eval()
        if t_start is None:
            t_start = self.diffusion_steps
        generator = get_generator_if_specified(seed, device=self.device)
        x0 = x0.to(self.device)
        noise = torch.randn(
            x0.shape, generator=generator, device=self.device, dtype=x0.dtype
        )
        x_t = self.get_q_t(x0, noise, t_start)
        return (
            self.sample_and_return_steps(
                x_t.detach().clone(), t_start, steps_to_return, generator=generator, mean_only=mean_only, return_stds=return_stds
            ),
            x_t,
        )
