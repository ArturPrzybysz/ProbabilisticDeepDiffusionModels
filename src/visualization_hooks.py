import os

import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning import Callback

from src.utils import save_img, model_output_to_image_numpy


class VisualizationCallback(Callback):
    def __init__(self, dataloader, img_path, ts, run_every=None):
        self.dataloader = dataloader
        self.img_path = img_path
        os.mkdir(self.img_path)
        self.run_every = run_every
        self.n_images = 4
        self.ts = list(sorted(ts))

    def run_visualizations(self, pl_module):
        # self.visualize_random(pl_module)
        # self.visualize_random(pl_module, mean_only=True)
        # self.visualize_reconstructions(pl_module)
        self.visualize_reconstructions_grid(pl_module)

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
            x = x[: self.n_images]

            for i in range(x.shape[0]):
                save_img(
                    x[i].detach().cpu().numpy(),
                    os.path.join(img_path, f"img_{i}_0.png"),
                )
            for t in self.ts:
                images, images_with_noise = pl_module.diffuse_and_reconstruct(x, t)
                for i in range(images.shape[0]):
                    save_img(
                        images[i].detach().cpu().numpy(),
                        os.path.join(img_path, f"img_{i}_{t}.png"),
                    )
                    save_img(
                        images_with_noise[i].detach().cpu().numpy(),
                        os.path.join(img_path, f"img_{i}_{t}_noise.png"),
                    )

    def visualize_reconstructions_grid(self, pl_module):
        img_path = os.path.join(
            self.img_path, f"images_grid_{pl_module.current_epoch}"
        )
        image_count, step_count, image_shape, channels, width, height, target_image_shape = (None,) * 7

        if not os.path.exists(img_path):
            os.mkdir(img_path)
            batch = next(iter(self.dataloader))
            x, _ = batch
            x = x[: self.n_images]

            t_start_to_images = {}
            for t_start in self.ts:
                images, noisy_images = pl_module.diffuse_and_reconstruct_grid(x, t_start, self.ts)
                t_start_to_images[t_start] = (images, noisy_images, x)

                image_count = images.shape[0]
                step_count = images.shape[1]
                image_shape = images.shape[2:]

                channels = image_shape[0]
                width = image_shape[1]
                height = image_shape[2]
                # target_image_shape = (height, width, channels)

                # image_grid = np.ones((height * step_count, width * image_count, channels))
                # image_grid has following shape:
                # batch_size (different x_0's),
                # step_count (steps_len),
                # image_shape (2D + 1 or 3 channels)

                # for step in range(step_count):
                #     for img_idx in range(image_count):
                #         proper_img = model_output_to_image_numpy(images[img_idx, step].detach().cpu().numpy())
                #
                #         image_grid[width * step: width * (step + 1), height * img_idx: height * (img_idx + 1), :] \
                #             = proper_img
                #
                # plt.imshow(image_grid)
                # plt.show()

            for img_idx in range(image_count):
                image_grid = np.ones((height * step_count, width * (step_count + 2), channels))

                for t_start_idx, t in enumerate(self.ts):
                    start_grid_col_nr = len(self.ts) - t_start_idx - 1

                    images, noisy_images, x0 = t_start_to_images[t]

                    source_img = model_output_to_image_numpy(x0[img_idx].detach().cpu().numpy())
                    noisy_img = model_output_to_image_numpy(
                        noisy_images[img_idx].detach().cpu().numpy())

                    image_grid[height * (t_start_idx):height * (t_start_idx + 1),
                    width * start_grid_col_nr: width * (start_grid_col_nr + 1), :] \
                        = noisy_img
                    #
                    image_grid[height * t_start_idx: height * (t_start_idx + 1),
                    width * (step_count + 1): width * (step_count + 2), :] = source_img

                    for step in range(step_count - t_start_idx - 1, step_count):
                        img_to_display = model_output_to_image_numpy(
                            images[img_idx, step_count - step - 1].detach().cpu().numpy())
                        print("step", step, "t_start_idx", t_start_idx, "step_count", step_count)
                        l_border_idx = len(self.ts) - step + start_grid_col_nr
                        image_grid \
                            [height * (t_start_idx):height * (t_start_idx + 1),
                        width * (l_border_idx): width * (l_border_idx + 1), :] \
                            = img_to_display

                plt.imshow(image_grid)
                plt.xticks(np.arange(0, (step_count + 2)) * width + width // 2,
                           ["q"] + self.ts + ["$x_0$"])
                plt.xlabel("Denosing step")
                plt.yticks(np.arange(0, step_count) * height + height // 2)
                path = os.path.join(img_path, f"image_{img_idx}_epoch_{pl_module.current_epoch}.png")
                plt.savefig(path, bbox_inches="tight", pad_inches=0)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.run_every is not None and pl_module.current_epoch % self.run_every == 0:
            self.run_visualizations(pl_module)

    def on_train_end(self, trainer, pl_module):
        self.run_visualizations(pl_module)
