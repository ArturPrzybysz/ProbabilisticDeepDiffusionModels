import sys
import os

from src.datasets.data import get_dataloader
from src.utils import wait_and_get_free_GPU_idx
from pathlib import Path

import torch
from src.engine import Engine
from src.modules.fid_score import compute_FID_score
import pytorch_lightning as pl

import wandb
from omegaconf import OmegaConf

from src.wandb_util import download_file


def init_wandb(run_id, clip_while_generating):
    api = wandb.Api()
    wandb.init(project="diffusion", entity="ddpm", tags=["FID", run_id, str(clip_while_generating)])
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
    print("clip_while_generating", clip_while_generating)
    checkpoint_path = download_file(run_id, "model.ckpt")
    init_wandb(run_id, clip_while_generating)
    logger = pl.loggers.WandbLogger()

    engine = Engine.load_from_checkpoint(checkpoint_path)
    engine.clip_while_generating = clip_while_generating
    logger.watch(engine)

    free_GPU_idx = wait_and_get_free_GPU_idx()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(free_GPU_idx)
    print("The free GPU is:", free_GPU_idx)
    if torch.cuda.is_available():
        engine.cuda()
    print("!!!!\n\n!!!!\n\n!!!!\n\nengine.device =", engine.device)

    cfg_file = os.path.join(wandb.run.dir, "config.yaml")
    wandb.save(cfg_file)

    cfg_path = download_file(run_id, "experiment_config.yaml")
    original_cfg = OmegaConf.load(cfg_path)
    print("original_cfg =", original_cfg)
    dataloader = get_dataloader(
        train=False, pin_memory=True, download=True, **original_cfg["data"]
    )

    path1 = Path("images/sample")
    path2 = Path("images/dataset")
    path1.mkdir(exist_ok=True, parents=True)
    path2.mkdir(exist_ok=True, parents=True)
    FID_score = compute_FID_score(engine, dataloader, dir_to_save1=path1, dir_to_save2=path2)
    wandb.save()
    print("FID_score", FID_score)


if __name__ == '__main__':
    main()
