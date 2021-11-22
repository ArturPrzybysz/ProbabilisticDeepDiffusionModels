from pathlib import Path

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.data import unnormalize
from src.engine import Engine
from src.utils import save_img
from tempfile import TemporaryDirectory

from pytorch_fid import fid_score


def sample_from_model(engine: Engine, target_path: Path, mean_only, minibatch_size=128, image_count=2048):
    print("sample_from_model")
    images = engine.generate_images(n=image_count, minibatch=minibatch_size, mean_only=mean_only)

    for i in range(images.shape[0]):
        img = unnormalize(
            images[i], normalize=None, clip=True, channel_dim=0
        )
        print(img.shape)
        print(img)
        save_img(img, target_path / f"{i}.png")
        images = wandb.Image(img, caption=f"{i}.png")
        wandb.log({"images": images})


def save_dataloader_to_files(dataloader: DataLoader, path: Path, lower_limit=0, limit=4096):
    print("save_dataloader_to_files")
    count = 0
    import time
    t1 = time.time()
    for batch in dataloader:
        X = batch[0]
        for i in tqdm(range(X.shape[0])):
            if i < lower_limit: continue
            # img = X[i, :, :, :].detach().cpu().numpy()
            img = unnormalize(
                X[i].detach().cpu().numpy(),
                normalize=None,
                clip=True,
                channel_dim=0,
            )
            save_img(img, path / f"{i}.png")
            if count == limit: break
            count += 1
    print("time", time.time() - t1, count)


def compute_FID_score(engine: Engine, dataloader, dir_to_save1=None, dir_to_save2=None, fid_batch_size=50):
    print("compute_FID_score")
    with TemporaryDirectory() as samples_dir, TemporaryDirectory() as dataset_dir:
        if dir_to_save1:
            target_path = dir_to_save1
        else:
            target_path = Path(samples_dir)

        if dir_to_save2:
            dataset_path = dir_to_save2
        else:
            dataset_path = Path(dataset_dir)

        sample_from_model(engine=engine, target_path=target_path, mean_only=True)
        save_dataloader_to_files(dataloader, dataset_path)

        FID = fid_score.calculate_fid_given_paths((str(dataset_path), str(samples_dir)),
                                                  batch_size=fid_batch_size,
                                                  device=engine.device,
                                                  dims=2048)
        return FID
