
import torch
from .torch_trainer import TorchTrainer


class Trainer(TorchTrainer):

    def __init__(self, model, args):

        self.model = model
        self.args = args
        super().__init__(args)

    def fetch_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def training_step(self, batch, batch_idx):

        for k in batch:
          batch[k] = batch[k].to(self.device)

        out = self.model(**batch, return_dict=True)
        loss = out["loss"].mean()
        return loss

    @torch.no_grad()
    def validation_step(self, batch):

        for k in batch:
          batch[k] = batch[k].to(self.device)

        out = self.model(**batch, return_dict=True)
        loss = out["loss"].mean()
        return loss

    def training_epoch_end(self, epoch, losses):
        # saving state_dict at epoch level
        self.save_training_state_dict(self.args.base_dir)
        self.save_pretrained(self.args.save_path)

    def save_pretrained(self, path: str):
        module = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(module.state_dict(), path)
