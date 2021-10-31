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
from src.modules import get_model

wandb.init(project="diffusion", entity="ddpm")

from src.wandb_util import download_file



@hydra.main(config_path="../config", config_name="eval")
def run_eval(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    cfg_file = os.path.join(wandb.run.dir, "config.yaml")
    with open(cfg_file, "w") as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)
    wandb.config.update(cfg)

    checkpoint_path = download_file(cfg["run_id"], "model.ckpt")

    engine = Engine.load_from_checkpoint(checkpoint_path)
    dataloader_train = get_dataloader(
        train=False, pin_memory=True, **cfg["data"]
    )

    logger = pl.loggers.WandbLogger()
    logger.watch(engine)
    gpus = 1 if torch.cuda.is_available() else 0

    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="training/logs",
        gpus=gpus,
        # limit_train_batches=10,
        limit_test_batches=1,
        # **cfg["trainer"],
    )

    try:
        trainer.test(engine, test_dataloaders=dataloader_train)
    except Exception as e:
        # for some reason errors get truncated here, so need to catch and raise again
        # probably hydra's fault
        print("Caught exception")
        print(e)
        traceback.print_exc(e)
        # raise e


if __name__ == '__main__':
    run_eval()
