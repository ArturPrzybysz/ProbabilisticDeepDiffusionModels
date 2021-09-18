# Run options
import os
import traceback

from PIL import Image

import torch
import wandb
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data import get_dataloader
from src.engine import Engine
from src.visualization_hooks import VisualizationCallback

wandb.init(project="diffusion", entity="ddpm")


@hydra.main(config_path="../config", config_name="default")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

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

    dataloader_train = get_dataloader(
        download=True, train=True, num_workers=4, pin_memory=True, **cfg["data"]
    )

    callbacks.append(
        VisualizationCallback(
            dataloader_train,
            img_path=os.path.join(wandb.run.dir, "images"),
            run_every=10,
        )
    )

    engine = Engine(cfg["model"], **cfg["engine"])

    logger = pl.loggers.WandbLogger()
    logger.watch(engine)

    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        gpus=gpus,
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
