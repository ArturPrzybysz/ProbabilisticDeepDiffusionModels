# Run options
import os

import torch
import wandb
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data import get_datasets
from src.engine import Engine

wandb.init(project='diffusion', entity='ddpm')

USE_CUDA = torch.cuda.is_available()

@hydra.main(config_path='config', config_name="default")
def run_training(cfg : DictConfig) -> dict:
    print(OmegaConf.to_yaml(cfg))

    cfg_file = os.path.join(wandb.run.dir, 'config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)
    # TODO: will this work?
    wandb.config.update(cfg)


    callbacks = []
    # TODO: is val_loss good?
    callbacks.append(pl.callbacks.EarlyStopping(patience=10, monitor='val_loss'))
    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=wandb.run.dir,
                                                  monitor='val_loss',
                                                  filename='model',
                                                  verbose=True,
                                                  period=1))

    wandb.save('*.ckpt') # should keep it up to date

    # TODO: data (see data.py
    dataloader_train = get_datasets()
    dataloader_val = get_datasets()


    # TODO: configure model
    model = Engine(cfg['model_config'], cfg['optimizer_config'])

    logger = pl.loggers.WandbLogger()
    logger.watch(model)


    gpus = 0
    if USE_CUDA:
        gpus = 1
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, default_root_dir="training/logs", max_epochs=cfg["max_epochs"], gpus=gpus)

    trainer.fit(model, train_dataloader=dataloader_train, val_dataloaders=dataloader_val)


if __name__ == '__main__':
    run_training()