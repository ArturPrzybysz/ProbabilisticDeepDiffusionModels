# Run options
import os
import traceback

import numpy as np

import torch
import wandb
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.datasets.data import get_dataloader
from src.engine import Engine
from src.visualization_hooks import VisualizationCallback
from src.wandb_util import download_file


def init_wandb(cfg):
    tags = []

    tags.append(cfg["data"]["name"])
    effective_bs = cfg["data"]["batch_size"] * cfg["trainer"]["accumulate_grad_batches"]
    tags.append(f'BS_{effective_bs}')

    tags.append("train")
    if cfg["cont_run"]:
        tags.append("cont")
    else:
        tags.append(f'BLCK_{cfg["model"]["num_res_blocks"]}')
        if cfg["scheduler"]["scheduler_name"]:
            tags.append(cfg["scheduler"]["scheduler_name"])
        tags.append(f'LR_{cfg["engine"]["optimizer_config"]["lr"]}')
        tags.append(f'T_{cfg["engine"]["diffusion_steps"]}')
        tags.append(cfg["engine"]["mode"])
        if cfg["engine"].ema:
            tags.append(f'EMA_{cfg["engine"].ema}")
        if cfg["engine"].sampling == "importance":
            tags.append("importance")

    if "gradient_clip_val" in cfg["trainer"] and cfg["trainer"]["gradient_clip_val"] is not None:
        tags.append("grad_clip")


    wandb.init(project="diffusion", entity="ddpm", dir="/scratch/s193223/wandb/", tags=tags)
    wandb.config.update({"script": "train"})

    if cfg["run_name"] is not None:
        wandb.run.name = cfg["run_name"]
        wandb.run.save()


@hydra.main(config_path="../config", config_name="default")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    init_wandb(cfg)

    cfg_file = os.path.join(wandb.run.dir, "experiment_config.yaml")
    with open(cfg_file, "w") as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)
    wandb.config.update(cfg)
    wandb.config.update({"machine": os.uname()[1]})


    callbacks = []
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
    callbacks.append(pl.callbacks.EarlyStopping(patience=20, monitor="val_loss"))
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=wandb.run.dir,
            monitor="val_loss",
            filename="model",
            verbose=True,
            period=cfg.trainer.check_val_every_n_epoch,
        )
    )

    wandb.save("*.ckpt")  # should keep it up to date
    wandb.save("*.png")
    wandb.save("images/*.png")
    wandb.save("images/*/*.png")
    wandb.save("images/*/*/*.png")
    wandb.save("images/*/*/*/*.png")

    dataloader_train = get_dataloader(train=True, pin_memory=True, **cfg["data"])
    dataloader_val = get_dataloader(train=False, pin_memory=True, **cfg["data"])


    if cfg["cont_run"]:
        checkpoint_path = download_file(cfg["cont_run"], "model.ckpt")
        # TODO: allow override config ??
        engine = Engine.load_from_checkpoint(checkpoint_path)
    else:
        engine = Engine(cfg["model"], **cfg["scheduler"], **cfg["engine"])

    if engine.diffusion_steps <= 30:
        num_vis_steps = 5
    else:
        num_vis_steps = 10
    ts = np.linspace(0, engine.diffusion_steps, num=num_vis_steps + 1, dtype=int)[1:]
    ts_interpolation = np.linspace(0, engine.diffusion_steps, num=5, dtype=int)[1:]
    # callbacks.append(
    #     VisualizationCallback(
    #         dataloader_train,
    #         img_path=os.path.join(wandb.run.dir, "images"),
    #         ts=ts,
    #         ts_interpolation=ts_interpolation,
    #         normalization=cfg["data"]["transformation_kwargs"].get("normalize"),
    #         use_ema=True,
    #         **cfg["visualization"],
    #     )
    # )
    callbacks.append(
        VisualizationCallback(
            dataloader_val,
            img_path=os.path.join(wandb.run.dir, "images"),
            ts=ts,
            ts_interpolation=ts_interpolation,
            normalization=cfg["data"]["transformation_kwargs"].get("normalize"),
            use_ema=True,
            img_prefix="val_",
            **cfg["visualization"],
        )
    )

    logger = pl.loggers.WandbLogger()
    logger.watch(engine)

    print("DEVICES", torch.cuda.device_count())
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

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
        trainer.fit(
            engine, train_dataloader=dataloader_train, val_dataloaders=dataloader_val
        )
    except Exception as e:
        # for some reason errors get truncated here, so need to catch and raise again
        # probably hydra's fault
        print("Caught exception")
        print(e)
        traceback.print_exc(e)
        # raise e


if __name__ == "__main__":
    run_training()
