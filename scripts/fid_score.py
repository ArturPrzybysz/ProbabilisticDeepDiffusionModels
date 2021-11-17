import os
import tempfile
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from pytorch_fid import fid_score

from src.datasets.data import get_dataloader
from src.engine import Engine
from src.modules.fid_score import compute_FID_score, save_dataloader_to_files
from src.wandb_util import download_file
import pytorch_lightning as pl


def init_wandb(run_id):
    api = wandb.Api()
    wandb.init(project="diffusion", entity="ddpm", tags=["FID", run_id])
    run = api.run(f"ddpm/diffusion/{run_id}")
    run_name = "FID_" + run.name + "-" + wandb.run.name.split("-")[-1]
    wandb.run.name = run_name
    wandb.run.save()


def main():
    run_id = "1uk0nbqr"
    checkpoint_path = download_file(run_id, "model.ckpt")
    init_wandb(run_id)
    logger = pl.loggers.WandbLogger()

    engine = Engine.load_from_checkpoint(checkpoint_path)
    logger.watch(engine)

    ################
    cfg_path = download_file(run_id, "experiment_config.yaml")
    original_cfg = OmegaConf.load(cfg_path)

    dataloader = get_dataloader(
        train=False, pin_memory=True, download=True, **original_cfg["data"]
    )
    with tempfile.TemporaryDirectory() as _p1, tempfile.TemporaryDirectory() as _p2:
        p1 = Path(_p1)
        p2 = Path(_p2)
        save_dataloader_to_files(dataloader, p1, lower_limit=0, limit=2048)
        save_dataloader_to_files(dataloader, p2, lower_limit=2048, limit=4096)

        ################

        print("engine.device", engine.device)
        if torch.cuda.is_available():
            engine.cuda()
        print("engine.device", engine.device)

        cfg_file = os.path.join(wandb.run.dir, "config.yaml")
        wandb.save(cfg_file)

        cfg_path = download_file(run_id, "experiment_config.yaml")
        original_cfg = OmegaConf.load(cfg_path)
        print(original_cfg)
        dataloader = get_dataloader(
            train=False, pin_memory=True, download=True, **original_cfg["data"]
        )

        # dataset_path = "todo"
        # FID_score = compute_FID_score(engine, dataloader)
        # print("FID_score", FID_score)

        FID = fid_score.calculate_fid_given_paths((str(_p1), str(_p2)),
                                                  batch_size=100,
                                                  device=engine.device,
                                                  dims=2048)
        print("FID", FID)


if __name__ == '__main__':
    main()
