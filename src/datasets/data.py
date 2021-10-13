from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms, models
from collections.abc import Iterable

# this shit seems to be required to download
from six.moves import urllib

from paths import DATA_DIR
import numpy as np

from src.datasets.celebahq import CelebAHQDataset

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

SPLIT_NAMES = {
    "CelebA": {True: "train", False: "valid"},
    "Cifar10": {True: "train", False: "valid"},
    "ImageNet": {True: "train", False: "val"},
    "SVHN": {True: "train", False: "test"},
}

NORMALIZATIONS = {
    "cifar": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "mnist": ((0.5,), (0.5,)),
    "oneone": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def get_dataloader(
    name,
    batch_size=128,
    download=False,
    train=True,
    num_workers=4,
    pin_memory=True,
    transformation_kwargs=None,
    num_samples_per_epoch=None,
):
    if transformation_kwargs is None:
        transformation_kwargs = {}
    transform = get_transformations(train=train, **transformation_kwargs)

    if name.lower() == "celebahq":
        dataset = CelebAHQDataset(train=train, transform=transform)
    else:
        dataset = getattr(datasets, name)
        dir = DATA_DIR / f"{name.lower()}_data"
        if name in SPLIT_NAMES:
            split = SPLIT_NAMES[name][train]
            dataset = dataset(dir, split=split, download=download, transform=transform)
        else:
            dataset = dataset(dir, train=train, download=download, transform=transform)

    if num_samples_per_epoch is not None:
        shuffle = False
        sampler = RandomSampler(dataset, num_samples=num_samples_per_epoch, replacement=True)
    else:
        shuffle = train
        sampler = None

    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sampler=sampler,
        pin_memory=pin_memory
    )


def get_transformations(
    train=True, flip=False, crop=False, crop_size=32, crop_padding=4, normalize=None
):
    transformations = []

    if flip and train:
        transformations.append(transforms.RandomHorizontalFlip())

    if crop and train:
        transformations.append(transforms.RandomCrop(crop_size, padding=crop_padding))

    if crop and not train: # TODO: deterministic crop?
        transformations.append(transforms.RandomCrop(crop_size, padding=crop_padding))

    # to tensor
    transformations.append(transforms.ToTensor())

    if normalize is not None:
        if isinstance(normalize, str):
            mean, std = NORMALIZATIONS[normalize]
        elif isinstance(normalize, Iterable):
            mean, std = normalize
        else:
            raise ValueError(f"Wrong normalization: {normalize}")

        transformations.append(transforms.Normalize(mean, std))

    return transforms.Compose(transformations)


def unnormalize(x, normalize=None, clip=False, channel_dim=0):
    """Reverts data normalization and clips"""
    if normalize is not None:
        if isinstance(normalize, str):
            mean, std = NORMALIZATIONS[normalize]
        elif isinstance(normalize, Iterable):
            mean, std = normalize
        else:
            raise ValueError(f"Wrong normalization: {normalize}")

        norm_shape = [1] * len(x.shape)
        channels = x.shape[channel_dim]
        norm_shape[channel_dim] = channels
        mean = np.array(mean).reshape(norm_shape)
        std = np.array(std).reshape(norm_shape)
        x = x * std + mean

    if clip:
        return np.clip(x, 0, 1)
    else:
        return x
