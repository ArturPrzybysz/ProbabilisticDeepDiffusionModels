import torch as th
import torch.nn as nn
import math



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DenseModel(nn.Module):

    def __init__(self, resolution=32, in_channels=3, num_hidden=[256, 256]):
        super().__init__()
        self.num_hidden = num_hidden
        self.resolution = resolution
        self.in_channels = in_channels

        time_embed_dim = num_hidden[0]
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(), # TODO: why?
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        print(self.time_embed)

        in_dim = resolution * resolution * in_channels + time_embed_dim
        layers = []
        for n in num_hidden:
            layers.append(nn.Linear(in_dim, n))
            in_dim = n
            layers.append(nn.ReLU()) # TODO: SiLU?
        layers.append(nn.Linear(in_dim, resolution * resolution * in_channels))
        self.dense = nn.Sequential(*layers)
        print(self.dense)



    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.num_hidden[0]))
        x = x.view((-1, self.resolution**2 * self.in_channels))
        out = self.dense(th.cat([emb, x], -1))
        return th.reshape(out, (-1, self.in_channels, self.resolution, self.resolution))