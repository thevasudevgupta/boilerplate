# torch-trainer

This is simple trainer for pytorch users (tested for pytorch 1.6).

**Note: Currently, this can't be used with models involving multiple optimizers (like GANs).**

```python

from torch_utils import TorchTrainer

class Trainer(TorchTrainer):

    def __init__(self, model, args):
        self.model = model
        self.lr = args.lr
        # define args whatever you need

        # call this at end only
        super().__init__(args)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # This method should look something like this

        batch = batch.to(self.device)

        out = self.model(batch)
        loss = out.mean()

        return loss

    def validation_step(self, batch):
        # This method should look something like this
        batch = batch.to(self.device)

        with torch.no_grad():
            out = self.model(batch)
            loss = out.mean()

        return loss

# define model architecture
model = .....

# define dataset
tr_dataset = .....
val_dataset = .....

trainer = Trainer(model, args)
trainer.fit(tr_dataset, val_dataset)
# Enjoy training......Yayyyyyyyyy
```

Currently only single GPU & Single TPU are supported. Don't forget to send your batch to `self.device`, model will be automatically transferred to `self.device` (you need not care that). `self.device` will be automatically set to GPU (when GPU is available) or to TPU (when tpu=1 in config-class).

## Configuration used in torch-trainer

Lots of args are involved in torch-trainer. You can either inherit your config-class from torch_utils or define your config-class containing default-args. **Note: your config-class must be decorated with `python dataclass`**

```python

from dataclasses import dataclass
from torch_utils import DefaultArgs

@dataclass
class Config(DefaultArgs):

    # pass your args
    lr: float = 2e-5
    ......

    # If want to update defaut_args; just pass it here only
    save_dir: str = 'weights'
    .......

# Default Arguments
"""
    base_dir (str) : root dir for any kind of saving (default = ".")
    map_location (torch.device) : argument used in torch.load() while loading model-state-dict (default = torch.device("cuda:0"))

    # model weights will be in `.pt` file 
    # while other training stuff will be in `.tar`
    save_dir (str) : If specified training stuff and model weights will be saved in this dir (default = None)
    load_dir (str) : If specified training stuff and model weights will be loaded from this dir (default = None)

    # wandb related args
    project_name (str) : Project name in wandb (default = None)
    wandb_run_name (str) : run name in wandb (default = None)
    wandb_off (bool) : If you want to disable wandb; useful for testing (default = False)

    max_epochs (int) : No of epochs (default = 5)

    accumulation_steps (int) : No of accumulation steps (default = 1)
    precision ('float32' or 'mixed16') : Precision during training (default = 'float32')

    # Run following command before hand and specify 1, incase you want to use TPUs
    # !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.6-cp36-cp36m-linux_x86_64.whl
    tpus (int) : specify 1 incase of using single tpu (default = 0)
"""
```

## TODO

- [ ] Enable training for models involving GANs
- [ ] Enable support for schedulers
- [ ] Enable support for multiple GPUs
- [ ] Enable support for multiple TPUs

**Its very boring to write your own trainer every time. So feel free to use it, customize it as per your own needs. Make a pull request if you want to contribute to this trainer.**
