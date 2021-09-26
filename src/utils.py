from PIL import Image
import matplotlib.pyplot as plt


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def save_img(x, path):
    x = x.transpose(1, 2, 0)
    if x.shape[0] == 1:
        # img = Image.fromarray(x[0, :, :], "L")
        plt.imshow(x[:, :, 0], cmap="gray")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
    else:
        # img = Image.fromarray(x, "RGB")
        plt.imshow(x)  # rows, columns, channels
        print("x.shape", x.shape)
        plt.show()
        # img.save(path)
        plt.savefig(path, bbox_inches="tight", pad_inches=0)


def model_output_to_image_numpy(x):
    x = x.transpose(1, 2, 0)
    if x.shape[0] == 1:
        return x[:, :, 0]
    else:
        return x
