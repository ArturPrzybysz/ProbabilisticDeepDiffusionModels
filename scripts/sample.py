# Run options
import os

import torch
from PIL import Image

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

from src.datasets.data import get_dataloader
from src.engine import Engine
from src.visualization_hooks import VisualizationCallback
from src.wandb_util import download_file

wandb.init(project="diffusion", entity="ddpm")


@hydra.main(config_path="../config", config_name="sample")
def sample(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    cfg_file = os.path.join(wandb.run.dir, "config.yaml")
    with open(cfg_file, "w") as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)
    wandb.config.update(cfg)

    wandb.save("*.png")
    wandb.save("images/*.png")
    wandb.save("images/*/*.png")
    wandb.save("images/*/*/*.png")
    wandb.save("images/*/*/*/*.png")

    checkpoint_path = download_file(cfg["run_id"], "model.ckpt")
    cfg_path = download_file(cfg["run_id"], "experiment_config.yaml")
    original_cfg = OmegaConf.load(cfg_path)

    # engine = Engine.load_from_checkpoint(checkpoint_path, model_config=original_cfg["model"], **original_cfg["engine"])
    engine = Engine.load_from_checkpoint(checkpoint_path)

    dataloader_train = get_dataloader(train=True, pin_memory=True, **original_cfg["data"])
    dataloader_val = get_dataloader(train=False, pin_memory=True, **original_cfg["data"])


    if engine.diffusion_steps <= 30:
        num_vis_steps = 5
    else:
        num_vis_steps = 10
    ts = np.linspace(0, engine.diffusion_steps, num=num_vis_steps + 1, dtype=int)[1:]
    ts_interpolation = np.linspace(0, engine.diffusion_steps, num=5, dtype=int)[1:]

    vis_train = VisualizationCallback(
            dataloader_train,
            img_path=os.path.join(wandb.run.dir, "images"),
            ts=ts,
            ts_interpolation=ts_interpolation,
            normalization=original_cfg["data"]["transformation_kwargs"].get("normalize"),
            **original_cfg["visualization"],
        )

    vis_val = VisualizationCallback(
            dataloader_val,
            img_path=os.path.join(wandb.run.dir, "images"),
            ts=ts,
            ts_interpolation=ts_interpolation,
            normalization=original_cfg["data"]["transformation_kwargs"].get("normalize"),
            img_prefix="val_",
            **original_cfg["visualization"],
        )

    if torch.cuda.is_available():
        engine.cuda()

    for t0 in [engine.diffusion_steps, int(9*engine.diffusion_steps/10), int(4*engine.diffusion_steps/5), int(engine.diffusion_steps/2)]:
        ts = [t0 - i for i in range(7)] \
             + [t0 - i * 10 for i in range(1,6)] \
             + [int(t0 / 10), int(t0 / 5), int(t0 / 2)]
        ts = [t for t in sorted(set(ts)) if t > 0]
        print(ts)
        engine.clip_while_generating = False
        vis_val.visualize_single_reconstructions(engine, mean_only=False, ts=ts, img_prefix=f't{t0}_val_no_clip_')
        vis_val.visualize_single_reconstructions(engine, mean_only=True, ts=ts, img_prefix=f't{t0}_val_no_clip_')
        #
        # vis_train.visualize_single_reconstructions(engine, mean_only=False, ts=ts, img_prefix=f't{t0}_no_clip_')
        # vis_train.visualize_single_reconstructions(engine, mean_only=True, ts=ts, img_prefix=f't{t0}_no_clip_')

        engine.clip_while_generating = True
        vis_val.visualize_single_reconstructions(engine, mean_only=False, ts=ts, img_prefix=f't{t0}_val_')
        vis_val.visualize_single_reconstructions(engine, mean_only=True, ts=ts, img_prefix=f't{t0}_val_')
        #
        # vis_train.visualize_single_reconstructions(engine, mean_only=False, ts=ts, img_prefix=f't{t0}_')
        # vis_train.visualize_single_reconstructions(engine, mean_only=True, ts=ts, img_prefix=f't{t0}_')

    vis_val.run_visualizations(engine)
    vis_train.run_visualizations(engine)



if __name__ == "__main__":
    sample()
