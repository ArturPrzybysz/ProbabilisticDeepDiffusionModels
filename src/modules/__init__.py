from .dense import DenseModel
from .unet import UNetModel

MODELS = {
    "dense": DenseModel,
    "unet": UNetModel
}
def get_model(cfg):
    name = cfg.pop("name")
    return MODELS[name](**cfg)
