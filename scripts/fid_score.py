from omegaconf import OmegaConf

from src.datasets.data import get_dataloader
from src.engine import Engine
from src.modules.fid_score import compute_FID_score
from src.wandb_util import download_file


def main():
    run_id = "27z5khpa"
    checkpoint_path = download_file(run_id, "model.ckpt")
    engine = Engine.load_from_checkpoint(checkpoint_path)
    # engine = None

    cfg_path = download_file(run_id, "experiment_config.yaml")
    original_cfg = OmegaConf.load(cfg_path)
    dataloader = get_dataloader(
        train=False, pin_memory=True, **original_cfg["data"]
    )

    # dataset_path = "todo"
    FID_score = compute_FID_score(engine, dataloader)
    print("FID_score", FID_score)


if __name__ == '__main__':
    main()
