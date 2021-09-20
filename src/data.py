from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections.abc import Iterable

# this shit seems to be required to download
from six.moves import urllib

from paths import DATA_DIR

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

SPLIT_NAMES = {
    "CelebA": {True: "train", False: "valid"},
    "Cifar10": {True: "train", False: "valid"},
    "ImageNet": {True: "train", False: "val"},
    "SVHN": {True: "train", False: "test"},
}


def get_dataloader(
        name,
        batch_size=128,
        download=True,
        train=True,
        num_workers=4,
        pin_memory=True,
        transformation_kwargs=None,
):
    dataset = getattr(datasets, name)

    if transformation_kwargs is None:
        transformation_kwargs = {}
    transform = get_transformations(train=train, **transformation_kwargs)

    dir = DATA_DIR / f"{name.lower()}_data"
    if name in SPLIT_NAMES:
        split = SPLIT_NAMES[name][train]
        dataset = dataset(dir, split=split, download=download, transform=transform)
    else:
        dataset = dataset(dir, train=train, download=download, transform=transform)

    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=train
    )


def get_transformations(
        train=True, flip=False, crop=False, crop_size=32, crop_padding=4, normalize=None
):
    transformations = []

    if flip and train:
        transformations.append(transforms.RandomHorizontalFlip())

    if crop and train:
        transformations.append(transforms.RandomCrop(crop_size, padding=crop_padding))

    # to tensor
    transformations.append(transforms.ToTensor())

    if normalize == "cifar":
        transformations.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
    elif isinstance(normalize, Iterable):
        transformations.append(transforms.Normalize(normalize[0], normalize[1]))

    return transforms.Compose(transformations)
