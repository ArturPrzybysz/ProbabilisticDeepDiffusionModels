from pathlib import Path

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.data import unnormalize
from src.engine import Engine
from src.utils import save_img
from tempfile import TemporaryDirectory

from pytorch_fid import fid_score


def sample_from_model(engine: Engine, target_path: Path, mean_only, minibatch_size=256, image_count=8192):
    print("sample_from_model")
    images = engine.generate_images(n=image_count, minibatch=minibatch_size, mean_only=mean_only)

    for i in range(images.shape[0]):
        img = unnormalize(
            images[i], normalize=None, clip=True, channel_dim=0
        )
        print(img.shape)
        print(img)
        save_img(img, target_path / f"{i}.png")
        # images = wandb.Image(img, caption=f"{i}.png")
        wandb.log({"images": images})


count = 0


def save_dataloader_to_files(dataloader: DataLoader, path: Path, limit=16384):
    print("save_dataloader_to_files")
    import time
    global count
    t1 = time.time()
    for batch in dataloader:
        X = batch[0]
        for i in range(X.shape[0]):
            # img = X[i, :, :, :].detach().cpu().numpy()
            img = unnormalize(
                X[i].detach().cpu().numpy(),
                normalize=None,
                clip=True,
                channel_dim=0,
            )
            print(path / f"{count}.png")
            save_img(img, path / f"{count}.png")
            if count == limit: break
            count += 1
        if count == limit: break
    count = 0
    print("time", time.time() - t1, count)


def compute_FID_score(engine: Engine, dataloader):
    print("compute_FID_score")
    with TemporaryDirectory() as samples_dir, TemporaryDirectory() as dataset_dir:
        samples_path = Path(samples_dir)
        dataset_path = Path(dataset_dir)

        sample_from_model(engine=engine, target_path=samples_path, mean_only=False, image_count=10000)
        save_dataloader_to_files(dataloader, dataset_path)
        # save_dataloader_to_files(dataloader2, samples_path)

        FID = fid_score.calculate_fid_given_paths((str(dataset_path), str(samples_path)),
                                                  batch_size=64,
                                                  device=engine.device,
                                                  dims=2048)
        wandb.log("FID", FID)
        return FID


def compute_FID_score_for_loaders(dataloader1, dataloader2, device):
    print("compute_FID_score")
    with TemporaryDirectory() as samples_dir, TemporaryDirectory() as dataset_dir:
        samples_path = Path(samples_dir)
        dataset_path = Path(dataset_dir)

        save_dataloader_to_files(dataloader1, dataset_path)
        save_dataloader_to_files(dataloader2, samples_path)

        FID = fid_score.calculate_fid_given_paths((str(dataset_path), str(samples_path)),
                                                  batch_size=64,
                                                  device=device,
                                                  dims=2048)
        return FID
