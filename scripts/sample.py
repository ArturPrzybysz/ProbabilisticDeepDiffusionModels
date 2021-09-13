# Run options
import os
import traceback

from PIL import Image

from hydra.utils import get_original_cwd, to_absolute_path


import torch
import wandb
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data import get_dataloader
from src.engine import Engine
from src.modules import get_model
from src.wandb_util import download_file

wandb.init(project='diffusion', entity='ddpm')

@hydra.main(config_path='../config', config_name="sample")
def sample(cfg : DictConfig):

    print(OmegaConf.to_yaml(cfg))

    cfg_file = os.path.join(wandb.run.dir, 'config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)
    wandb.config.update(cfg)

    checkpoint_path = download_file(cfg['run_id'], 'model.ckpt')
    cfg_path = download_file(cfg['run_id'], 'model.ckpt')
    original_cfg = OmegaConf.load(cfg_path)

    engine = Engine.load_from_checkpoint(checkpoint_path, model_config=original_cfg["model"], **original_cfg["engine"])
    # TODO: load cfg_path

    images = engine.generate_image(cfg["num_samples"])
    img_path = os.path.join(wandb.run.dir, 'images')
    os.mkdir(img_path)
    for i in range(cfg["num_samples"]):
        if cfg["black_white"]:
            img = Image.fromarray(images[i,0,:,:], 'L')
        else:
            img = Image.fromarray(images[i], 'RGB')
        img.save(os.path.join(img_path, f'img_{i}.png'))

if __name__ == '__main__':
    sample()
