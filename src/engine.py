import torch
import pytorch_lightning as pl



class Engine(pl.LightningModule):
    def __init__(self, model_config, optimizer_config):
        super(Engine, self).__init__()
        # create the model here
        # self.model = ...
        self.train_acc = pl.metrics.Accuracy()
        self.optimizer_config = optimizer_config

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_config)


    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError

