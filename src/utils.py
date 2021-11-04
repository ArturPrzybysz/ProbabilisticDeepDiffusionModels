import torch

th = torch
import matplotlib.pyplot as plt


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
