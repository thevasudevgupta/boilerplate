
import os
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

    def training_epoch_end(self, epoch, tr_metric, val_metric):
        # saving state_dict at epoch level
        self.save_training_state_dict(self.args.base_dir)
        path = os.path.join(self.args.base_dir, self.args.save_epoch_dir+f'-e{epoch}')
        self.model.save_pretrained(path)

        if self.args.hub_id:
            self.model.push_to_hub(path, model_id=self.args.hub_id, commit_message=f"add epoch-{epoch} ckpt")
