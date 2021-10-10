import torch
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
