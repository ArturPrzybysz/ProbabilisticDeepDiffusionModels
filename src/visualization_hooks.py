import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pytorch_lightning import Callback
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.data.data import unnormalize
from src.utils import save_img, model_output_to_image_numpy
import wandb


class VisualizationCallback(Callback):
    def __init__(
        self,
        dataloader,
        img_path,
        ts,
        ts_interpolation,
        run_every=None,
        normalization=None,
        seed=1234,
        n_images=4,
        n_random = 10,
        n_interpolation_steps = 10,
        n_interpolation_pairs = 5,
        same_class_interpolation=False,
        img_prefix='',
    ):
        self.dataloader = dataloader
        self.img_path = img_path
        if not os.path.exists(self.img_path):
            os.mkdir(self.img_path)
        self.run_every = run_every
        self.n_images = n_images
        self.n_random = n_random
        self.n_interpolation_steps = n_interpolation_steps
        self.n_interpolation_pairs = n_interpolation_pairs
        self.ts = list(sorted(ts))
        self.ts_interpolation = list(sorted(ts_interpolation))
        self.normalization = normalization
        self.seed = seed
        self.img_prefix = img_prefix
        self.same_class_interpolation = same_class_interpolation

    def run_visualizations(self, pl_module):
        # self.visualize_random(pl_module)
        if self.n_random > 0:
            self.visualize_random_grid(pl_module)
        self.visualize_interpolation(pl_module)
        # self.visualize_random(pl_module, mean_only=True)
        self.visualize_reconstructions_grid(pl_module)
        # self.visualize_reconstructions(pl_module)

    def visualize_random(self, pl_module, mean_only=False):
        if mean_only:
            img_path = os.path.join(
                self.img_path, f"{self.img_prefix}images_random_mean_{pl_module.current_epoch}"
            )
        else:
            img_path = os.path.join(
                self.img_path, f"{self.img_prefix}images_random_{pl_module.current_epoch}"
            )

        if not os.path.exists(img_path):
            os.mkdir(img_path)
            images = pl_module.generate_images(
                self.n_images, mean_only=mean_only, seed=self.seed
            )
            for i in range(images.shape[0]):
                img = unnormalize(
                    images[i], normalize=self.normalization, clip=True, channel_dim=0
                )
                save_img(img, os.path.join(img_path, f"img_{i}.png"))

    def show_full_reconstruction(self, images, noise, img_path, epoch, key):
        n_images = images.shape[0]
        channels = images.shape[2]
        width = images.shape[3]
        height = images.shape[4]
        step_count = images.shape[1]
        image_grid = np.ones(
            (height * n_images, width * (step_count + 1), channels)
        )

        # iterate through images
        for img_idx in range(n_images):
            # starting noise
            noisy_img = model_output_to_image_numpy(noise[img_idx])
            image_grid[
                height * (img_idx) : height * (img_idx + 1), 0:width, :
            ] = noisy_img

            # all the denoising steps
            for step in range(0, step_count):
                img_to_display = model_output_to_image_numpy(images[img_idx, step])

                image_grid[
                    height * (img_idx) : height * (img_idx + 1),
                    width * (step + 1) : width * (step + 2),
                    :,
                ] = img_to_display

        # plotting stuff
        img = unnormalize(
            image_grid, normalize=self.normalization, clip=True, channel_dim=-1
        )

        plt.figure()
        if channels == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        plt.xticks(
            np.arange(0, (len(self.ts) + 1)) * width + width // 2,
            list(reversed(self.ts)) + ["0"],
        )
        plt.xlabel("Denoising step")
        plt.gca().get_yaxis().set_visible(False)

        # save image
        path = os.path.join(img_path, f"{key}_epoch_{epoch}.png")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)

        images = wandb.Image(img, caption=f"{key}_epoch_{epoch}")
        wandb.log({key: images})

    def visualize_random_grid(self, pl_module, mean_only=False):
        img_path = os.path.join(
            self.img_path, f"{self.img_prefix}images_random_grid_{pl_module.current_epoch}"
        )
        if not os.path.exists(img_path):
            os.mkdir(img_path)
            print(f"Generating {self.n_random} random images")
            noise, images = pl_module.generate_images_grid(
                steps_to_return=self.ts[:-1] + [1],
                n=self.n_random,
                mean_only=mean_only,
                seed=self.seed,
            )
            self.show_full_reconstruction(
                images, noise, img_path, pl_module.current_epoch, key="random"
            )

    def get_first_batch(self):
        batch = [self.dataloader.dataset[i] for i in range(self.n_images)]
        return default_collate(batch)

    def get_img_pair(self, last_idx=0, last_class=None):
        if self.same_class_interpolation:
            i = last_idx + 1
            # get different class
            while self.dataloader.dataset[i][1] == last_class:
                i += 1
            d1 = self.dataloader.dataset[i]
            i += 1
            new_class = d1[1]
            # get another from the same
            while self.dataloader.dataset[i][1] != new_class:
                i += 1
            d2 = self.dataloader.dataset[i]
            return d1, d2, i, new_class
        else:
            d1 = self.dataloader.dataset[last_idx + 1]
            d2 = self.dataloader.dataset[last_idx + 2]
            return d1, d2, last_idx + 2, last_class

    def visualize_interpolation(self, pl_module):
        img_path = os.path.join(
            self.img_path, f"{self.img_prefix}images_interpolation_{pl_module.current_epoch}"
        )

        if not os.path.exists(img_path):
            os.mkdir(img_path)
            # batch = self.get_first_batch()
            # x, y = batch

            print("running interpolations for steps", self.ts_interpolation)
            for t in tqdm(self.ts_interpolation):
                last_idx = -1
                last_class = None
                image_rows = []

                for i in range(self.n_interpolation_pairs):
                    (x1, y1), (x2, y2), last_idx, last_class = self.get_img_pair(last_idx, last_class)
                    x1 = torch.unsqueeze(x1, 0)
                    x2 = torch.unsqueeze(x2, 0)
                    x_i0 = pl_module.get_noised_representation(
                        x1, seed=self.seed, t=t
                    )
                    x_j0 = pl_module.get_noised_representation(
                        x2, seed=self.seed + 1, t=t
                    )
                    x_0 = torch.ones(
                        [self.n_interpolation_steps] + list(x_i0.shape[1:]),
                        device=pl_module.device,
                        dtype=x_i0.dtype,
                    )

                    for k, a in enumerate(np.linspace(0, 1, self.n_interpolation_steps)):
                        x_0[k : k + 1] = (1 - a) * x_i0 + a * x_j0

                    if t == pl_module.diffusion_steps:
                        steps_to_return = self.ts[:-1] + [1]
                    else:
                        steps_to_return = [1]

                    images = pl_module.sample_and_return_steps(
                        x_0.detach().clone(),
                        t_start=t,
                        steps_to_return=steps_to_return,
                        seed=self.seed,
                    )
                    img_row = images[:, -1].detach().cpu().numpy()
                    xi_npy = x1.detach().cpu().numpy()
                    xj_npy = x2.detach().cpu().numpy()
                    img_row = np.concatenate(
                        [
                            xi_npy,
                            img_row,
                            xj_npy,
                        ],
                        axis=0,
                    )

                    originals_row = np.concatenate(
                        [
                            np.ones_like(xi_npy),
                            x_0.detach().cpu().numpy(),
                            np.ones_like(xi_npy),
                        ],
                        axis=0,
                    )
                    image_rows.append(originals_row)
                    image_rows.append(img_row)

                    if t == pl_module.diffusion_steps:
                        reference_column = np.ones(
                            [self.n_interpolation_steps, 1] + list(x_i0.shape[1:])
                        )
                        reference_column[0:1, 0] = xi_npy
                        reference_column[-1 : self.n_interpolation_steps, 0] = xj_npy

                        images = np.concatenate(
                            [images.detach().cpu().numpy(), reference_column],
                            axis=1,
                        )

                        self.show_full_reconstruction(
                            images,
                            x_0.detach().cpu().numpy(),
                            img_path,
                            pl_module.current_epoch,
                            key=f"{self.img_prefix}interpolation_{i}",
                        )

                # choose which images should have red border
                borders = [
                    (i, j)
                    for i in range(len(image_rows))
                    for j in range(self.n_interpolation_steps + 2)
                    if i % 2 == 0 or j == 0 or j == self.n_interpolation_steps + 1
                ]
                self.plot_grid(
                    image_rows,
                    img_path,
                    key=f"{self.img_prefix}all_interpolations_{t}",
                    epoch=pl_module.current_epoch,
                    border=borders,
                )

    def plot_grid(self, image_rows, path, key, epoch, border=tuple()):
        ncol = image_rows[0].shape[0]
        nrow = len(image_rows)
        channels = image_rows[0].shape[1]
        fig = plt.figure()
        grid = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(nrow, ncol),
            axes_pad=(0.03, 0.05),
        )
        for i, image_row in enumerate(image_rows):
            for j in range(ncol):
                ax = grid[i * ncol + j]

                img = unnormalize(
                    model_output_to_image_numpy(image_row[j]),
                    normalize=self.normalization,
                    clip=True,
                    channel_dim=-1,
                )

                if channels == 1:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)

                if (i, j) in border:
                    ax.patch.set_linewidth("2")
                    ax.patch.set_edgecolor("red")
                    ax.patch.set_facecolor("red")
                    ax.margins(0.2, 0.05)
                    ax.yaxis.set_ticks([])
                    ax.xaxis.set_ticks([])
                else:
                    ax.axis("off")

        # save image
        path = os.path.join(path, f"{self.img_prefix}{key}_epoch_{epoch}.png")
        plt.savefig(path, bbox_inches="tight", pad_inches=0.02)

        im = plt.imread(path)
        images = wandb.Image(im, caption=f"{self.img_prefix}{key}_epoch_{epoch}")
        wandb.log({key: images})

    def visualize_reconstructions(self, pl_module):
        img_path = os.path.join(
            self.img_path, f"{self.img_prefix}images_reconstruct_{pl_module.current_epoch}"
        )

        if not os.path.exists(img_path):
            os.mkdir(img_path)
            batch = self.get_first_batch()
            x, y = batch
            x = x[: self.n_images]

            for i in range(x.shape[0]):
                save_img(
                    x[i].detach().cpu().numpy(),
                    os.path.join(img_path, f"img_{i}_0.png"),
                )
            for t in self.ts:
                images, images_with_noise = pl_module.diffuse_and_reconstruct(
                    x, t, seed=self.seed
                )
                for i in range(images.shape[0]):
                    img = unnormalize(
                        images[i].detach().cpu().numpy(),
                        normalize=self.normalization,
                        clip=True,
                        channel_dim=0,
                    )
                    save_img(
                        img,
                        os.path.join(img_path, f"img_{i}_{t}.png"),
                    )
                    img = unnormalize(
                        images_with_noise[i].detach().cpu().numpy(),
                        normalize=self.normalization,
                        clip=True,
                        channel_dim=0,
                    )
                    save_img(
                        img,
                        os.path.join(img_path, f"img_{i}_{t}_noise.png"),
                    )

    def visualize_reconstructions_grid(self, pl_module):
        img_path = os.path.join(self.img_path, f"{self.img_prefix}images_grid_{pl_module.current_epoch}")
        # image_count, step_count, image_shape, channels, width, height, target_image_shape = (None,) * 7

        if not os.path.exists(img_path):
            os.mkdir(img_path)
            batch = self.get_first_batch()
            x, _ = batch
            x = x[: self.n_images]

            channels = x.shape[1]
            width = x.shape[2]
            height = x.shape[3]
            step_count = len(self.ts)

            t_start_to_images = {}

            # iterate through noise steps
            print('Running reconstructions for steps: ', self.ts)
            for i, t_start in tqdm(enumerate(self.ts)):
                # images: (B, Ts, C, W, H)
                # noisy_images: (B, C, W, H)
                images, noisy_images = pl_module.diffuse_and_reconstruct_grid(
                    x, t_start, self.ts[:i] + [1], seed=self.seed
                )
                t_start_to_images[t_start] = (images, noisy_images, x)

            # iterate through images
            for img_idx in range(self.n_images):
                # initialize empty grid
                image_grid = np.ones(
                    (height * step_count, width * (step_count + 2), channels)
                )

                # iterate through noise steps
                for t_start_idx, t_start in enumerate(self.ts):
                    # calculate column index
                    start_grid_col_nr = step_count - t_start_idx - 1

                    # convert and reshape images
                    images, noisy_images, x0 = t_start_to_images[t_start]
                    source_img = model_output_to_image_numpy(
                        x0[img_idx].detach().cpu().numpy()
                    )
                    noisy_img = model_output_to_image_numpy(
                        noisy_images[img_idx].detach().cpu().numpy()
                    )

                    # starting image with noise
                    image_grid[
                        height * (t_start_idx) : height * (t_start_idx + 1),
                        width * start_grid_col_nr : width * (start_grid_col_nr + 1),
                        :,
                    ] = noisy_img
                    # source image without noise
                    image_grid[
                        height * t_start_idx : height * (t_start_idx + 1),
                        width * (step_count + 1) : width * (step_count + 2),
                        :,
                    ] = source_img

                    # all the denoising steps
                    for step in range(step_count - t_start_idx - 1, step_count):
                        img_to_display = model_output_to_image_numpy(
                            images[img_idx, step_count - step - 1]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        l_border_idx = len(self.ts) - step + start_grid_col_nr
                        image_grid[
                            height * (t_start_idx) : height * (t_start_idx + 1),
                            width * (l_border_idx) : width * (l_border_idx + 1),
                            :,
                        ] = img_to_display

                # plotting stuff
                img = unnormalize(
                    image_grid, normalize=self.normalization, clip=True, channel_dim=-1
                )
                plt.figure()
                if channels == 1:
                    plt.imshow(img, cmap="gray")
                else:
                    plt.imshow(img)

                plt.xticks(
                    np.arange(0, (step_count + 2)) * width + width // 2,
                    list(reversed(self.ts)) + ["0", "$x_0$"],
                )
                plt.xlabel("Denoising step")
                plt.yticks(np.arange(0, step_count) * height + height // 2, self.ts)
                plt.ylabel("Starting noise step")

                # save image
                path = os.path.join(
                    img_path, f"{self.img_prefix}image_{img_idx}_epoch_{pl_module.current_epoch}.png"
                )
                plt.savefig(path, bbox_inches="tight", pad_inches=0)

                images = wandb.Image(
                    img, caption=f"{self.img_prefix}image_{img_idx}_epoch_{pl_module.current_epoch}"
                )
                wandb.log({f"{self.img_prefix}reconstructions_{img_idx}": images})

    def on_train_epoch_end(self, trainer, pl_module):
        if self.run_every is not None and (pl_module.current_epoch + 1) % self.run_every == 0:
            self.run_visualizations(pl_module)

    def on_train_end(self, trainer, pl_module):
        self.run_visualizations(pl_module)
