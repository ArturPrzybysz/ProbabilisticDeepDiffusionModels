import os

from pytorch_lightning import Callback

from src.utils import save_img


class VisualizationCallback(Callback):
    def __init__(self, dataloader, img_path, ts, run_every=None):
        self.dataloader = dataloader
        self.img_path = img_path
        os.mkdir(self.img_path)
        self.run_every = run_every
        self.n_images = 4
        self.ts = ts

    def run_visualizations(self, pl_module):
        self.visualize_random(pl_module)
        self.visualize_random(pl_module, mean_only=True)
        self.visualize_reconstructions(pl_module)

    def visualize_random(self, pl_module, mean_only=False):
        if mean_only:
            img_path = os.path.join(
                self.img_path, f"images_random_mean_{pl_module.current_epoch}"
            )
        else:
            img_path = os.path.join(
                self.img_path, f"images_random_{pl_module.current_epoch}"
            )

        if not os.path.exists(img_path):
            os.mkdir(img_path)
            images = pl_module.generate_images(self.n_images, mean_only=mean_only)
            for i in range(images.shape[0]):
                save_img(images[i], os.path.join(img_path, f"img_{i}.png"))

    def visualize_reconstructions(self, pl_module):
        img_path = os.path.join(
            self.img_path, f"images_reconstruct_{pl_module.current_epoch}"
        )

        if not os.path.exists(img_path):
            os.mkdir(img_path)
            batch = next(iter(self.dataloader))
            x, y = batch
            x = x[:self.n_images]

            for i in range(x.shape[0]):
                save_img(x[i].detach().cpu().numpy(), os.path.join(img_path, f"img_{i}_0.png"))

            for t in self.ts:
                images = pl_module.diffuse_and_reconstruct(x, t)
                for i in range(images.shape[0]):
                    save_img(images[i].detach().cpu().numpy(), os.path.join(img_path, f"img_{i}_{t}.png"))

    def on_train_epoch_end(self, trainer, pl_module):
        if self.run_every is not None and pl_module.current_epoch % self.run_every == 0:
            self.run_visualizations(pl_module)

    def on_train_end(self, trainer, pl_module):
        self.run_visualizations(pl_module)
