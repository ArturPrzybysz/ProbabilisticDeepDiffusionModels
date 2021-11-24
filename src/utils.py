"""
some code from https://github.com/openai/improved-diffusion
"""
import torch as th

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CONFIG_PATH = Path("../config")


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def save_img(x, path):
    plt.figure()
    x = x.transpose(1, 2, 0)
    if x.shape[-1] == 1:
        plt.imshow(x[:, :, 0], cmap="gray")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
    else:
        # plt.imshow(x)  # rows, columns, channels
        plt.imshow(x)  # rows, columns, channels
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.clf()


def model_output_to_image_numpy(x):
    x = x.transpose(1, 2, 0)
    if x.shape[0] == 1:
        return x[:, :, 0]
    else:
        return x


def get_generator_if_specified(seed=None, device="cpu"):
    if seed is None:
        return None
    else:
        generator = th.Generator(device=device)
        generator.manual_seed(seed)
        return generator


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + th.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


import torch
import matplotlib.pyplot as plt
import subprocess, json, time


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def save_img(x, path):
    plt.figure()
    x = x.transpose(1, 2, 0)
    if x.shape[-1] == 1:
        plt.imshow(x[:, :, 0], cmap="gray")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
    else:
        # plt.imshow(x)  # rows, columns, channels
        plt.imshow(x)  # rows, columns, channels
        plt.savefig(path, bbox_inches="tight", pad_inches=0)


def model_output_to_image_numpy(x):
    x = x.transpose(1, 2, 0)
    if x.shape[0] == 1:
        return x[:, :, 0]
    else:
        return x


def get_generator_if_specified(seed=None, device="cpu"):
    if seed is None:
        return None
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator


def free_GPUs():
    try:
        gpu_info = json.loads(subprocess.check_output(['gpustat', '--json']).decode("utf-8"))
    except:
        return [""]
    gpus = gpu_info["gpus"]
    return [str(gpu["index"]) for gpu in gpus if len(gpu["processes"]) == 0]


def wait_and_get_free_GPU_idx():
    i = 0
    wait_seconds = 0.5
    free_gpus_idx = free_GPUs()
    while not any(free_gpus_idx):
        if i % 10 == 0:
            print(f'Waiting for GPU: {i * wait_seconds} seconds')
        time.sleep(wait_seconds)
        free_gpus_idx = free_GPUs()
        i += 1
    return free_gpus_idx
