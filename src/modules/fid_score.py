from pathlib import Path

from src.datasets.data import unnormalize
from src.engine import Engine
from src.utils import save_img
from tempfile import TemporaryDirectory


def sample_from_model(engine: Engine, dataset_path: Path, mean_only):
    minibatch_size = 32
    image_count = 512
    images = engine.generate_images(n=image_count, minibatch=minibatch_size, mean_only=mean_only)

    with TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir)
        for i in range(images.shape[0]):
            img = unnormalize(
                images[i], normalize=None, clip=True, channel_dim=0
            )
            save_img(img, target_path / f"{i}.png")

        # TODO: FID EVAL
