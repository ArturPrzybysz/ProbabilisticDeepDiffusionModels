# Run options
import os
import traceback

from PIL import Image

import torch
import wandb
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.datasets.data import get_dataloader
from src.engine import Engine

wandb.init(project="diffusion", entity="ddpm", dir="/scratch/s193223/wandb/")


@hydra.main(config_path="../config", config_name="default")
def run_training(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    cfg_file = os.path.join(wandb.run.dir, "config.yaml")
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

    dataloader_train = get_dataloader(train=True, pin_memory=True, **cfg["data"])

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

    # generate some images to check if it works
    images = engine.generate_image(16)
    img_path = os.path.join(wandb.run.dir, "images")
    os.mkdir(img_path)
    for i in range(16):
        # TODO: handle channels
        img = Image.fromarray(images[i, 0, :, :], "L")
        img.save(os.path.join(img_path, f"img_{i}.png"))


if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        print("Caught exception")
        print(e)
        traceback.print_exc(e)
