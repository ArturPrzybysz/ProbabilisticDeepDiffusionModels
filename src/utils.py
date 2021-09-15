from PIL import Image


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def save_img(x, path):
    if x.shape[0] == 1:
        img = Image.fromarray(x[0, :, :], "L")
    else:
        img = Image.fromarray(x, "RGB")
    img.save(path)
