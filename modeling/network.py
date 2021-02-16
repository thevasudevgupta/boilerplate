
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # put layers here

    def forward(self, batch):
        # put layers over batch
        return batch
