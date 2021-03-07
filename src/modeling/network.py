
import torch.nn as nn
from huggingface_hub import ModelHubMixin

class Model(nn.Module, ModelHubMixin):

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs.pop("config", None)
        # put layers here

    def forward(self, batch):
        # put layers over batch
        return batch
