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
