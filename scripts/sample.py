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


def init_wandb(cfg):
    api = wandb.Api()
    tags = ["sample", cfg["run_id"]]
    if cfg["clip_while_generating"]:
        tags.append("clip")
    wandb.init(project="diffusion", entity="ddpm", tags=tags)
    run = api.run(f"ddpm/diffusion/{cfg['run_id']}")
    run_name = "SAMPLE_" + run.name + "-" + wandb.run.name.split("-")[-1]
    wandb.run.name = run_name
    wandb.run.save()


@hydra.main(config_path="../config", config_name="sample")
def sample(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    init_wandb(cfg)

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

    engine.clip_while_generating = cfg["clip_while_generating"]

    dataloader_train = get_dataloader(
        train=True, pin_memory=True, **original_cfg["data"]
    )
    dataloader_val = get_dataloader(
        train=False, pin_memory=True, **original_cfg["data"]
    )

    if cfg["num_vis_steps"] is None:
        if engine.diffusion_steps <= 30:
            num_vis_steps = 5
        else:
            num_vis_steps = 10
    else:
        num_vis_steps = cfg["num_vis_steps"]
    ts = np.linspace(0, engine.diffusion_steps, num=num_vis_steps + 1, dtype=int)[1:]
    ts_interpolation = np.linspace(0, engine.diffusion_steps, num=5, dtype=int)[1:]
    if cfg["use_train"]:
        prefix = "train_"
        dataloader = dataloader_train
    else:
        prefix = "val_"
        dataloader = dataloader_val

    vis = VisualizationCallback(
        dataloader,
        img_path=os.path.join(wandb.run.dir, "images"),
        ts=ts,
        ts_interpolation=ts_interpolation,
        normalization=original_cfg["data"]["transformation_kwargs"].get("normalize"),
        img_prefix=prefix,
        run_every=1,
        n_images=cfg["n_images"],
        n_random=cfg["n_random"],
        n_interpolation_steps=cfg["n_interpolation_steps"],
        n_interpolation_pairs=cfg["n_interpolation_pairs"],
        use_ema=cfg["use_ema"],
    )

    if torch.cuda.is_available():
        engine.cuda()

    if cfg["regular_viz"]:
        vis.run_visualizations(engine)

    if cfg["detailed_viz"]:
        run_detailed_viz(engine, vis, prefix)


def run_detailed_viz(engine, vis, prefix):
    for t0 in [
        engine.diffusion_steps,
        int(9 * engine.diffusion_steps / 10),
        int(4 * engine.diffusion_steps / 5),
        int(engine.diffusion_steps / 2),
    ]:
        ts = (
                [t0 - i for i in range(7)]
                + [t0 - i * 10 for i in range(1, 6)]
                + [int(t0 / 10), int(t0 / 5), int(t0 / 2)]
        )
        ts = [t for t in sorted(set(ts)) if t > 0]
        print(ts)

        engine.clip_while_generating = False
        images = []

        images.append(
            vis.visualize_single_reconstructions(
                engine, mean_only=False, ts=ts, img_prefix=f"t{t0}_{prefix}no_clip_"
            )
        )

        images.append(
            vis.visualize_single_reconstructions(
                engine, mean_only=True, ts=ts, img_prefix=f"t{t0}_{prefix}no_clip_"
            )
        )

        engine.clip_while_generating = True

        images.append(
            vis.visualize_single_reconstructions(
                engine, mean_only=False, ts=ts, img_prefix=f"t{t0}_{prefix}"
            )
        )

        images.append(
            vis.visualize_single_reconstructions(
                engine, mean_only=True, ts=ts, img_prefix=f"t{t0}_{prefix}"
            )
        )
        wandb.log({f"recon_{t0}": images})


if __name__ == "__main__":
    sample()
