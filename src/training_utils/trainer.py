
import quick

import os
import torch


class Trainer(quick.TorchTrainer):

    def __init__(self, model, args):
        super().__init__(args)
        self.setup(model)

        self.lr = args.lr

    def setup_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_on_batch(self, batch, batch_idx):

        batch = {batch[k].to(self.device) for k in batch}

        out = self.model(**batch, return_dict=True)
        loss = out["loss"].mean()
        return loss

    @torch.no_grad()
    def evaluate_on_batch(self, batch):

        batch = {batch[k].to(self.device) for k in batch}

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
