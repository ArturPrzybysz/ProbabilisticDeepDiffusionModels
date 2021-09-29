import os

import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning import Callback

from src.data import unnormalize
from src.utils import save_img, model_output_to_image_numpy


class VisualizationCallback(Callback):
    def __init__(self, dataloader, img_path, ts, run_every=None, normalization=None):
        self.dataloader = dataloader
        self.img_path = img_path
        os.mkdir(self.img_path)
        self.run_every = run_every
        self.n_images = 4
        self.ts = list(sorted(ts))
        self.normalization = normalization

    def run_visualizations(self, pl_module):
        self.visualize_random(pl_module)
        self.visualize_random_grid(pl_module)
        self.visualize_random(pl_module, mean_only=True)
        self.visualize_reconstructions_grid(pl_module)
        # self.visualize_reconstructions(pl_module)

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
                img = unnormalize(images[i], normalize=self.normalization, clip=True, channel_dim=0)
                save_img(img, os.path.join(img_path, f"img_{i}.png"))


    def visualize_random_grid(self, pl_module, mean_only=False):
        img_path = os.path.join(
            self.img_path, f"images_random_grid_{pl_module.current_epoch}"
        )
        if not os.path.exists(img_path):
            os.mkdir(img_path)
            noise, images = pl_module.generate_images_grid(steps_to_return=self.ts[:-1] + [1], n=self.n_images, mean_only=mean_only)

            channels = images.shape[2]
            width = images.shape[3]
            height = images.shape[4]
            step_count = len(self.ts)
            image_grid = np.ones((height * self.n_images, width * (step_count + 1), channels))

            # iterate through images
            for img_idx in range(self.n_images):
                # starting noise
                noisy_img = model_output_to_image_numpy(noise[img_idx])
                image_grid[
                    height * (img_idx):height * (img_idx + 1),
                    0 : width,
                    :
                ] = noisy_img

                # all the denoising steps
                for step in range(0, step_count):
                    img_to_display = model_output_to_image_numpy(
                        images[img_idx, step_count - step - 1])

                    image_grid[
                    height * (img_idx):height * (img_idx + 1),
                    width * (step + 1): width * (step + 2),
                    :
                    ] = img_to_display

            # plotting stuff
            img = unnormalize(image_grid, normalize=self.normalization, clip=True,
                              channel_dim=-1)
            if channels == 1:
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)

            plt.xticks(np.arange(0, (step_count + 1)) * width + width // 2,
                       list(reversed(self.ts)) + ['0'])
            plt.xlabel("Denoising step")

            # save image
            path = os.path.join(img_path, f"epoch_{pl_module.current_epoch}.png")
            plt.savefig(path, bbox_inches="tight", pad_inches=0)

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
                    img = unnormalize(images[i].detach().cpu().numpy(), normalize=self.normalization, clip=True, channel_dim=0)
                    save_img(
                        img,
                        os.path.join(img_path, f"img_{i}_{t}.png"),
                    )
                    img = unnormalize(images_with_noise[i].detach().cpu().numpy(), normalize=self.normalization, clip=True, channel_dim=0)
                    save_img(
                        img,
                        os.path.join(img_path, f"img_{i}_{t}_noise.png"),
                    )

    def visualize_reconstructions_grid(self, pl_module):
        img_path = os.path.join(
            self.img_path, f"images_grid_{pl_module.current_epoch}"
        )
        # image_count, step_count, image_shape, channels, width, height, target_image_shape = (None,) * 7

        if not os.path.exists(img_path):
            os.mkdir(img_path)
            batch = next(iter(self.dataloader))
            x, _ = batch
            x = x[: self.n_images]

            channels = x.shape[1]
            width = x.shape[2]
            height = x.shape[3]
            step_count = len(self.ts)

            t_start_to_images = {}

            # iterate through noise steps
            for i, t_start in enumerate(self.ts):
                # images: (B, Ts, C, W, H)
                # noisy_images: (B, C, W, H)
                images, noisy_images = pl_module.diffuse_and_reconstruct_grid(x, t_start, self.ts[:i] + [1])
                t_start_to_images[t_start] = (images, noisy_images, x)

            # iterate through images
            for img_idx in range(self.n_images):
                # initialize empty grid
                image_grid = np.ones((height * step_count, width * (step_count + 2), channels))

                # iterate through noise steps
                for t_start_idx, t_start in enumerate(self.ts):
                    # calculate column index
                    start_grid_col_nr = step_count - t_start_idx - 1

                    # convert and reshape images
                    images, noisy_images, x0 = t_start_to_images[t_start]
                    source_img = model_output_to_image_numpy(x0[img_idx].detach().cpu().numpy())
                    noisy_img = model_output_to_image_numpy(
                        noisy_images[img_idx].detach().cpu().numpy())

                    # starting image with noise
                    image_grid[
                        height * (t_start_idx):height * (t_start_idx + 1),
                        width * start_grid_col_nr: width * (start_grid_col_nr + 1),
                        :
                    ] = noisy_img
                    # source image without noise
                    image_grid[
                        height * t_start_idx: height * (t_start_idx + 1),
                        width * (step_count + 1): width * (step_count + 2),
                        :
                    ] = source_img

                    # all the denoising steps
                    for step in range(step_count - t_start_idx - 1, step_count):
                        img_to_display = model_output_to_image_numpy(
                            images[img_idx, step_count - step - 1].detach().cpu().numpy())

                        l_border_idx = len(self.ts) - step + start_grid_col_nr
                        image_grid[
                            height * (t_start_idx):height * (t_start_idx + 1),
                            width * (l_border_idx): width * (l_border_idx + 1),
                            :
                        ] = img_to_display

                # plotting stuff
                img = unnormalize(image_grid, normalize=self.normalization, clip=True,
                                  channel_dim=-1)
                if channels == 1:
                    plt.imshow(img, cmap="gray")
                else:
                    plt.imshow(img)

                plt.xticks(np.arange(0, (step_count + 2)) * width + width // 2,
                           list(reversed(self.ts)) + ['0', "$x_0$"])
                plt.xlabel("Denoising step")
                plt.yticks(np.arange(0, step_count) * height + height // 2,
                           self.ts
                           )
                plt.ylabel("Starting noise step")

                # save image
                path = os.path.join(img_path, f"image_{img_idx}_epoch_{pl_module.current_epoch}.png")
                plt.savefig(path, bbox_inches="tight", pad_inches=0)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.run_every is not None and pl_module.current_epoch % self.run_every == 0:
            self.run_visualizations(pl_module)

    def on_train_end(self, trainer, pl_module):
        self.run_visualizations(pl_module)
