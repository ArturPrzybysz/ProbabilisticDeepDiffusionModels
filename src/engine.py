import math

import torch
import pytorch_lightning as pl

from src.modules import get_model
import numpy as np

from src.modules.gaussian_diffusion import _extract_into_tensor
from src.modules.losses import discretized_gaussian_log_likelihood, normal_kl
from src.utils import mean_flat, get_generator_if_specified
import torch as th


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
    ):
        super(Engine, self).__init__()
        self.save_hyperparameters()  # ??

        self.clip_while_generating = clip_while_generating

        # create the model here
        self.model = get_model(resolution, dict(model_config))
        print(self.model)
        self.optimizer_config = optimizer_config
        self.diffusion_steps = diffusion_steps
        self.resolution = resolution

        self.sigma_mode = sigma_mode
        self.betas = get_betas(beta_start, beta_end, diffusion_steps, mode).to(
            self.device
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.diffusion_steps,)
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

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
                self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            # np.array(1.0 - np.array(self.alphas_cumprod_prev))
            # *
                np.sqrt(self.alphas)
                / (1.0 - np.array(self.alphas_cumprod))
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_config)

    # ------------ Training and diffusion stuff ----------

    def get_q_t(self, x, noise, t):
        return (
                x * self.alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)
                + self.one_min_alphas_hat_sqrt[t - 1].view((-1, 1, 1, 1)).to(self.device)
                * noise
        )

    def get_loss(self, predicted_noise, target_noise):
        # TODO: should batch be averaged or summed?
        return torch.mean(torch.square(target_noise - predicted_noise))

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        batch_size = x.shape[0]
        t = torch.randint(1, self.diffusion_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x)
        x_t = self.get_q_t(x, noise, t)
        predicted_noise = self.model(x_t, t)
        loss = self.get_loss(predicted_noise, noise)

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

    # ------------ Model evaluation ----------

    def test_step(self, batch, batch_idx):
        x, _ = batch
        nll, vb, xstart_mse = self.calc_bpd_loop(x)
        self.log('nll/test', nll, on_step=False, on_epoch=True)
        self.log('vb/test', nll, on_step=False, on_epoch=True)
        self.log('xstart_mse/test', nll, on_step=False, on_epoch=True)

    def calc_bpd_loop(self, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        # device = 'cpu'
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in reversed(range(self.diffusion_steps)):
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.get_q_t(x=x_start, t=t_batch, noise=noise)

            with th.no_grad():
                out = self._vb_terms_bpd(
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.diffusion_steps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def _vb_terms_bpd(
            self, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # print("x_start.shape", x_start.shape)
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            self.model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)

        model_variance, model_log_variance = (
            np.append(self.posterior_variance[1], self.betas[1:]),
            np.log(np.append(self.posterior_variance[1], self.betas[1:])),
        )

        #     {
        #     # for fixedlarge, we set the initial (log-)variance like so
        #     # to get a better decoder log likelihood.
        #     ModelVarType.FIXED_LARGE: (
        #         np.append(self.posterior_variance[1], self.betas[1:]),
        #         np.log(np.append(self.posterior_variance[1], self.betas[1:])),
        #     ),
        #     ModelVarType.FIXED_SMALL: (
        #         self.posterior_variance,
        #         self.posterior_log_variance_clipped,
        #     ),
        # }[self.model_var_type]
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        )

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        return {
            "mean": model_output,  # there are more options! TODO: check if the enum is constant or changes over eval?
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,  # TODO: check if the enum is constant or changes over eval?
        }

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                       _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                       - pred_xstart
               ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

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

        for t in range(t_start, 0, -1):
            x_t = self.denoising_step(x_t, t, mean_only=mean_only, generator=generator)

            if t in steps_to_return:
                output[:, current_step_idx] = x_t
                current_step_idx += 1

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
            self, x0, t_start=None, steps_to_return=(1,), seed=None
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
                x_t.detach().clone(), t_start, steps_to_return, generator=generator
            ),
            x_t,
        )
