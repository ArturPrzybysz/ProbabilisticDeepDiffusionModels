import sys
import os
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf

from src.datasets.data import get_dataloader
from src.engine import Engine
from src.modules.fid_score import compute_FID_score, save_dataloader_to_files, sample_from_model
from src.wandb_util import download_file
import pytorch_lightning as pl


def init_wandb(run_id):
    api = wandb.Api()
    wandb.init(project="diffusion", entity="ddpm", tags=["FID", run_id])
    run = api.run(f"ddpm/diffusion/{run_id}")
    run_name = "FID_" + run.name + "-" + wandb.run.name.split("-")[-1]
    wandb.run.name = run_name
    wandb.run.save()
    wandb.save("*.png")
    wandb.save("images/*.png")
    wandb.save("images/*/*.png")


def main():
    run_id = sys.argv[1]
    clip_while_generating = sys.argv[2] == "True"
    print("run_id", run_id)
    checkpoint_path = download_file(run_id, "model.ckpt")
    init_wandb(run_id)
    logger = pl.loggers.WandbLogger()

    engine = Engine.load_from_checkpoint(checkpoint_path)
    engine.clip_while_generating = clip_while_generating

    logger.watch(engine)

    if torch.cuda.is_available():
        engine.cuda()
    print("engine.device =", engine.device)

    cfg_file = os.path.join(wandb.run.dir, "config.yaml")
    wandb.save(cfg_file)

    cfg_path = download_file(run_id, "experiment_config.yaml")
    original_cfg = OmegaConf.load(cfg_path)
    print("original_cfg =", original_cfg)
    dataloader = get_dataloader(
        train=False, pin_memory=True, download=True, **original_cfg["data"]
    )

    # sample_from_model(engine=engine, target_path=path1, mean_only=False, image_count=100, minibatch_size=50)

    FID_score = compute_FID_score(engine, dataloader)

    print("FID_score", FID_score)
    wandb.log({"FID_score": FID_score})


if __name__ == '__main__':
    main()
