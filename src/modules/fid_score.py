from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.data import unnormalize, get_dataloader
from src.engine import Engine
from src.utils import save_img, CONFIG_PATH
from tempfile import TemporaryDirectory

from src.wandb_util import download_file
from pytorch_fid import fid_score


def sample_from_model(engine: Engine, target_path: Path, mean_only, minibatch_size=128, image_count=2048):
    images = engine.generate_images(n=image_count, minibatch=minibatch_size, mean_only=mean_only)

    for i in range(images.shape[0]):
        img = unnormalize(
            images[i], normalize=None, clip=False, channel_dim=0
        )
        save_img(img, target_path / f"{i}.png")


def save_dataloader_to_files(dataloader: DataLoader, path: Path, limit=4096):
    count = 0
    import time
    t1 = time.time()
    for batch in dataloader:
        X = batch[0]
        for i in tqdm(range(X.shape[0])):
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


def compute_FID_score(engine: Engine, dataloader, fid_batch_size=50):
    with TemporaryDirectory() as samples_dir, TemporaryDirectory() as dataset_dir:
        target_path = Path(samples_dir)
        dataset_path = Path(samples_dir)

        save_dataloader_to_files(dataloader, dataset_path)
        sample_from_model(engine=engine, target_path=target_path, mean_only=True)

        FID = fid_score.calculate_fid_given_paths((str(dataset_path), str(samples_dir)),
                                                  batch_size=fid_batch_size,
                                                  device=engine.device,
                                                  dims=2048)
        return FID


def main():
    run_id = "27z5khpa"
    checkpoint_path = download_file(run_id, "model.ckpt")
    engine = Engine.load_from_checkpoint(checkpoint_path)
    # engine = None

    cfg_path = download_file(run_id, "experiment_config.yaml")
    original_cfg = OmegaConf.load(cfg_path)
    dataloader = get_dataloader(
        train=False, pin_memory=True, **original_cfg["data"]
    )

    # dataset_path = "todo"
    FID_score = compute_FID_score(engine, dataloader)
    print("FID_score", FID_score)


if __name__ == '__main__':
    main()
