import math
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Tuple

import torch
import torch as th

import pytorch_lightning as pl

# from torch_ema import ExponentialMovingAverage
import wandb
from tqdm import tqdm

from src.modules import get_model
import numpy as np
import pandas as pd

from src.modules.ema import Ema
from src.modules.stepwise_log import StepwiseLog
from src.sampling.importance_sampler import ImportanceSampler
from src.sampling.uniform_sampler import UniformSampler
from src.utils import mean_flat, get_generator_if_specified, normal_kl
import matplotlib.pyplot as plt


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
            ema=None,
            scheduler_name=None,
            scheduler_kwargs=None,
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
            # self.ema_model = get_model(resolution, dict(model_config))
            self.ema = Ema(self.model, decay=ema)
            self.ema.set(self.model)
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
        self.alphas_sqrt = th.sqrt(self.alphas)
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
            self.sampler = ImportanceSampler(
                diffusion_steps=diffusion_steps,
                loss_per_t=self.loss_per_t,
                min_counts=10,
            )
        else:
            raise ValueError(f'Unknown sampling option: "{sampling}"')

        self.val_sampler = UniformSampler(diffusion_steps=diffusion_steps)

        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs

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
        if isinstance(self.sampler, ImportanceSampler):
            print("self.sampler._ready: ", self.sampler._ready)
            if not self.sampler._ready:
                print(pd.Series(self.loss_per_t.n_per_step).value_counts())

        # log loss per Q
        for i in range(4):
            self.log(
                f"loss_q{i + 1}",
                self.loss_per_t_epoch.get_avg_in_range(
                    max(1, int(i * self.diffusion_steps / 4)),
                    int((i + 1) * self.diffusion_steps / 4),
                ),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        plt.figure(figsize=(10, 10))
        plt.plot(self.loss_per_t_epoch.avg_per_step)
        plt.xlabel("step")
        plt.ylabel("L_t")
        wandb.log({"loss_per_step": plt})

        plt.figure(figsize=(10, 10))
        plt.plot(self.loss_per_t_epoch.n_per_step)
        plt.xlabel("step")
        plt.ylabel("n_samples")
        wandb.log({"n_samples_per_step": plt})

        self.loss_per_t_epoch.reset()

    def optimizer_step(self, *args, **kwargs):
        self._log_grad_norm()
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            # self.ema.to(self.device)
            # self.ema.update()
            self.ema.update(self.model)
            # self.ema.to('cpu')

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.model.parameters():
            sqsum += (p.grad ** 2).sum().item()
        self.log(
            "grad_norm",
            np.sqrt(sqsum),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        # Consider weight decay
        # optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_config)
        if self.scheduler_name:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler_name)
            scheduler = scheduler_class(optimizer, **self.scheduler_kwargs)

            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return torch.optim.Adam(self.parameters(), **self.optimizer_config)

    # ------------ Training and diffusion stuff ----------

    def get_q_t(self, x, noise, t):
        return (
                x * self.alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)
                + self.one_min_alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)
                * noise
        )

    def get_loss(
            self, predicted_noise, target_noise, t, weights=None, update_loss_log=True
    ):
        loss = mean_flat(torch.square(target_noise - predicted_noise))
        if update_loss_log:
            losses = loss.detach().cpu().numpy().tolist()
            ts = t.detach().cpu().numpy().tolist()
            self.loss_per_t.update_multiple(ts, losses)
            self.loss_per_t_epoch.update_multiple(ts, losses)

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
        loss = self.get_loss(
            predicted_noise, noise, weights=weights, t=t, update_loss_log=True
        )

        total_norm = self.compute_grad_norm(self.model.parameters())
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # log loss per Q
        # for i in range(4):
        #     self.log(
        #         f"running_loss_q{i+1}",
        #         self.loss_per_t.get_avg_in_range(max(1, int(i*self.diffusion_steps/4)),
        #                                                int((i+1)*self.diffusion_steps/4)),
        #         on_step=True, on_epoch=False, prog_bar=False
        #     )
        self.log(
            "total_grad_norm_L2",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        batch_size = x.shape[0]
        t, weights = self.val_sampler(batch_size, self.device)
        noise = torch.randn_like(x)
        x_t = self.get_q_t(x, noise, t)
        predicted_noise = self.model(x_t, t)
        loss = self.get_loss(
            predicted_noise, noise, weights=weights, t=t, update_loss_log=False
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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
            return torch.sqrt(self.betas[t])  # TODO: Check indexing xD
        elif self.sigma_mode == "beta_tilde":
            variance = self.posterior_variance[t]  # TODO: Check indexing xD
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

    # ------------ TEST ----------

    def test_step(self, batch, batch_idx):
        x, _ = batch
        nll = self.calculate_likelihood(x)
        print(nll)
        self.log("test_L_0", nll["L_0"], on_step=True, on_epoch=False, prog_bar=False, )
        self.log("test_L_intermediate", nll["L_intermediate"], on_step=True, on_epoch=False, prog_bar=False, )
        self.log("test_L_T", nll["L_T"], on_step=True, on_epoch=False, prog_bar=False)
        self.log("test_nll", nll["nll"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("test_mse", nll["MSE"], on_step=True, on_epoch=False, prog_bar=False)

    def calculate_likelihood(self, x):
        """
        Implements eq. (5) from Denoising Diffusion Probabilistic Models
        """
        L_0 = self._calculate_L_0(x)
        L_intermediate_list, MSE_list = self._calculate_L_intermediate(x)
        L_T = self._calculate_L_T(x)
        L_intermediate = th.sum(th.stack(L_intermediate_list), dim=0)
        MSE = th.mean(th.stack(MSE_list))
        print("mse", MSE)
        print("L_0", th.mean(L_0, dim=0))
        print("L_intermediate_list", len(L_intermediate_list), L_intermediate_list[0].shape)
        print("L_intermediate", L_intermediate.shape)
        print("L_T", th.mean(L_T, dim=0))

        return {
            "MSE": MSE,  # TODO
            "MSE_list": MSE_list,  # TODO
            "L_0": th.mean(L_0, dim=0),
            "L_intermediate": L_intermediate,
            "L_T": th.mean(L_T, dim=0),
            "nll": th.mean(L_0 + L_intermediate + L_T, dim=0),
            "L_intermediate_list": L_intermediate_list,
        }

    def _calculate_L_T(self, x):
        """
        KL divergence of latent || prior: D_KL (q(x_T |x_0 ) || p(x_T )
        """
        q_mean, q_std = self.q_mean_std(x, self.diffusion_steps)
        p_mean, p_logvar = 0.0, 0.0
        return torch.mean(normal_kl(q_mean, 2 * th.log(q_std), p_mean, p_logvar), dim=[1, 2, 3]) / np.log(2.0)

    def _calculate_L_intermediate(self, x0) -> Tuple[List[th.Tensor], List[th.Tensor]]:
        """
        KL divergence of intermediate steps in [1, T-1] as:
        sum of KL divergences of (q(x_t−1 | x_t , x_0 ) || p_theta (x_t−1|x_t ))
        """
        L_intermediate_list = []
        MSE_list = []
        batch_size = x0.shape[0]
        batches = th.ones(batch_size, dtype=th.int64, device=self.device)
        for t_step in range(2, self.diffusion_steps + 1):
            t = batches * t_step
            noise = torch.randn_like(x0)
            x_t = self.get_q_t(x0, noise, t)
            mean_t, var_t = self.q_posterior(t, x0, x_t)
            predicted_noise = self.model(x_t, t)

            if t_step == -2:  # TODO: Remove - and fix
                L_i = mean_flat(self.discretized_gaussian_likelihood(x0, mean_t, var_t)) / np.log(2.0)
                print("!!!\nL_i", L_i, "\n")
            else:
                alpha_hat_sqrt_t = self.alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)
                one_minus_alpha_hat_sqrt_t = self.one_min_alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)
                predicted_mean = alpha_hat_sqrt_t * x0 + one_minus_alpha_hat_sqrt_t * predicted_noise
                predicted_std = self.get_sigma(t - 1).to(self.device)
                predicted_logvar = 2 * th.log(predicted_std).view((-1, 1, 1, 1)).to(self.device)

                logvar_1 = th.log(var_t) * th.ones_like(mean_t)
                logvar_2 = predicted_logvar * th.ones_like(mean_t)
                kl = normal_kl(mean1=mean_t, logvar1=logvar_1,
                               mean2=predicted_mean, logvar2=logvar_2)
                L_i = mean_flat(kl) / np.log(2.0)
            L_intermediate_list.append(L_i)
            mse_i = th.pow(predicted_noise - noise, 2)
            MSE_list.append(mse_i)
        return L_intermediate_list, MSE_list

    def discretized_gaussian_likelihood(self, x0, mean_t, var_t):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.

        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        log_stddev = 2 * th.log(var_t)
        print("\nx0.shape", x0.shape)
        print("mean_t.shape", mean_t.shape)
        print("var_t.shape", var_t.shape)
        print("var_t", var_t)
        print("log_stddev", log_stddev, "\n")
        assert x0.shape == mean_t.shape
        centered_x = x0 - mean_t
        inv_stdv = th.exp(-log_stddev)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))  # TODO: remove log
        log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))  # TODO: remove log
        cdf_delta = cdf_plus - cdf_min
        log_probs = th.where(
            x0 < -0.999,
            log_cdf_plus,
            th.where(x0 > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x0.shape
        print("log_probs.shape", log_probs.shape)
        return log_probs

    def approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))

    def q_posterior(self, t, x0, x_t):
        """
        Returns mean and variance of q(x_t-1 | x_t, x_0) following eq. (6) and (7)
        """
        alpha_hat_sqrt_t_min_1 = self.alphas_hat_sqrt[t - 2].view((-1, 1, 1, 1)).to(self.device)
        alpha_hat_t_min1 = self.alphas_hat[t - 2].view((-1, 1, 1, 1)).to(self.device)
        alpha_sqrt_t = self.alphas_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)
        alpha_hat_t = self.alphas_hat[t - 1].view((-1, 1, 1, 1)).to(self.device)
        beta_t = self.betas[t - 1].view((-1, 1, 1, 1)).to(self.device)
        mean_t = (
                x0 * alpha_hat_sqrt_t_min_1 * beta_t / (1 - alpha_hat_t)
                +
                x_t * alpha_sqrt_t * (1 - alpha_hat_t_min1) / (1 - alpha_hat_t)
        )

        var_t = beta_t * (
                (1 - alpha_hat_t_min1) / (1 - alpha_hat_t)
        )
        return mean_t, var_t

    def _calculate_L_0(self, x):
        """
        reconstruction likelihood: -log(p(x_0 | x_1))
        """
        return th.Tensor([0]).to(self.device) / np.log(2.0)

    def q_mean_std(self, x, t):
        """
        TODO: explain
        """
        mean = x * self.alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)  # eq. (4)
        std = self.one_min_alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)  # eq. (4)
        return mean, std

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
            return_stds=False,
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
            self,
            x0,
            t_start=None,
            steps_to_return=(1,),
            seed=None,
            mean_only=False,
            return_stds=False,
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
                x_t.detach().clone(),
                t_start,
                steps_to_return,
                generator=generator,
                mean_only=mean_only,
                return_stds=return_stds,
            ),
            x_t,
        )
