import sys
import os
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf

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
    wandb.save("*.png")
    wandb.save("images/*.png")
    wandb.save("images/*/*.png")


def main():
    run_id = sys.argv[1:][0]

    print("run_id", run_id)
    checkpoint_path = download_file(run_id, "model.ckpt")
    init_wandb(run_id)
    logger = pl.loggers.WandbLogger()

    engine = Engine.load_from_checkpoint(checkpoint_path)
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

    path1 = Path("images/sample")
    path2 = Path("images/dataset")
    path1.mkdir(exist_ok=True, parents=True)
    path2.mkdir(exist_ok=True, parents=True)
    FID_score = compute_FID_score(engine, dataloader, dir_to_save1=path1, dir_to_save2=path2)
    wandb.save()
    print("FID_score", FID_score)


original_cfg = {'model': {'name': 'unet', 'in_channels': 3, 'model_channels': 128, 'num_res_blocks': 3,
                          'attention_resolutions': [16, 8], 'dropout': 0.1, 'channel_mult': [1, 2, 2, 2],
                          'conv_resample': True, 'dims': 2, 'num_classes': None, 'use_checkpoint': False,
                          'num_heads': 4, 'num_heads_upsample': -1, 'use_scale_shift_norm': False},
                'data': {'name': 'CIFAR10', 'batch_size': 64, 'num_workers': 4,
                         'transformation_kwargs': {'normalize': 'oneone', 'flip': True}},
                'visualization': {'run_every': 10, 'n_images': 2, 'n_random': 4, 'n_interpolation_steps': 5,
                                  'n_interpolation_pairs': 2},
                'engine': {'resolution': 32, 'optimizer_config': {'lr': 0.0002}, 'diffusion_steps': 4000,
                           'beta_start': None, 'beta_end': None, 'clip_while_generating': False, 'sigma_mode': 'beta',
                           'mode': 'cosine', 'ema': 0.999, 'sampling': 'importance'},
                'scheduler': {'scheduler_name': None, 'scheduler_kwargs': {}},
                'run_name': 'cifar_4000_cosine_ema999_importance',
                'trainer': {'max_epochs': 4000, 'accumulate_grad_batches': 2, 'check_val_every_n_epoch': 2,
                            'limit_test_batches': 20}, 'cont_run': None, 'patience': 30}

if __name__ == '__main__':
    main()
