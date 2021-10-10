# Run options
import json
import os
import subprocess
import time


def free_GPUs():
    try:
        gpu_info = json.loads(subprocess.check_output(['gpustat', '--json']).decode("utf-8"))
    except:
        return [""]
    gpus = gpu_info["gpus"]
    return [gpu["index"] for gpu in gpus if len(gpu["processes"]) == 0]


def wait_and_get_free_GPU_idx():
    i = 0
    wait_seconds = 0.5
    free_gpus_idx = free_GPUs()
    while not any(free_gpus_idx):
        if i % 10 == 0:
            print(f'Waiting for GPU: {i * wait_seconds} seconds')
        time.sleep(wait_seconds)
        free_gpus_idx = free_GPUs()
        i += 1
    return free_gpus_idx


free_GPU_idx = wait_and_get_free_GPU_idx()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(free_GPU_idx)

import traceback
import numpy as np

import torch
import wandb
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data import get_dataloader
from src.engine import Engine
from src.visualization_hooks import VisualizationCallback

wandb.init(project="diffusion", entity="ddpm")
wandb.config.update({"script": "train"})


@hydra.main(config_path="../config", config_name="default")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg["run_name"] is not None:
        wandb.run.name = cfg["run_name"]
        wandb.run.save()

    cfg_file = os.path.join(wandb.run.dir, "experiment_config.yaml")
    with open(cfg_file, "w") as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)
    # TODO: will this work?
    wandb.config.update(cfg)

    callbacks = []
    callbacks.append(pl.callbacks.EarlyStopping(patience=10, monitor="loss"))
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=wandb.run.dir,
            monitor="loss",
            filename="model",
            verbose=True,
            period=1,
        )
    )

    wandb.save("*.ckpt")  # should keep it up to date
    wandb.save("*.png")
    wandb.save("images/*.png")
    wandb.save("images/*/*.png")
    wandb.save("images/*/*/*.png")
    wandb.save("images/*/*/*/*.png")

    dataloader_train = get_dataloader(
        train=True, pin_memory=True, **cfg["data"]
    )

    engine = Engine(cfg["model"], **cfg["engine"])

    if engine.diffusion_steps <= 30:
        num_vis_steps = 5
    else:
        num_vis_steps = 10
    callbacks.append(
        VisualizationCallback(
            dataloader_train,
            img_path=os.path.join(wandb.run.dir, "images"),
            ts=np.linspace(0, engine.diffusion_steps, num=num_vis_steps + 1, dtype=int)[
               1:
               ],
            ts_interpolation=np.linspace(0, engine.diffusion_steps, num=5, dtype=int)[
                             1:
                             ],
            normalization=cfg["data"]["transformation_kwargs"].get("normalize"),
            **cfg['visualization']
        )
    )

    logger = pl.loggers.WandbLogger()
    logger.watch(engine)

    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        gpus=gpus,
        # limit_train_batches=10,
        # limit_test_batches=1,
        **cfg["trainer"],
    )

    try:
        trainer.fit(engine, train_dataloader=dataloader_train)
    except Exception as e:
        # for some reason errors get truncated here, so need to catch and raise again
        # probably hydra's fault
        print("Caught exception")
        print(e)
        traceback.print_exc(e)
        # raise e


if __name__ == "__main__":
    run_training()
