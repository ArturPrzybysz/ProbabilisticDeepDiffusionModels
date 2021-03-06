# https://fastai.github.io/timmdocs/training_modelEMA

from copy import deepcopy
import torch
from torch import nn


class Ema(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(Ema, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        # self.module = model
        self.module = deepcopy(model)
        self.module.eval()
        self.module.requires_grad_(False)
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
