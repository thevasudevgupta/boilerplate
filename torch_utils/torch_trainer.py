# __author__ = 'Vasudev Gupta'
# __author_email__ = '7vasudevgupta@gmail.com'

import torch
import numpy as np

import os
import wandb

if torch.cuda.is_available():
    print('GPUs available')
else:
    print("GPUs not available")

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    print("TPUs available")
except:
    print('TPUs not available')

try:
    from rich.progress import track
except:
    os.system("pip install rich")
    from rich.progress import track

from abc import ABC, abstractmethod
from dataclasses import dataclass


"""
USAGE:

    from torch_utils import TorchTrainer

    class Trainer(TorchTrainer):

        def __init__(self, model, args):
            self.model = model

            # call this at end only
            super().__init__(args)

        def forward(self, batch):
            '''[Optional]
                ....
            '''

        def configure_optimizers(self):
            '''
                ....
            '''

        def training_step(self, batch, batch_idx):
            '''
                ....
            '''

        def validation_step(self, batch):
            '''
                ....
            '''

    # define model architecture
    model = .....

    # define dataset
    tr_dataset = .....
    val_dataset = .....

    args = TrainerConfig(....)

    trainer = Trainer(model, args)
    trainer.fit(tr_dataset, val_dataset)

    Using TPU is pretty simple, run following command:
        !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.6-cp36-cp36m-linux_x86_64.whl
    & pass tpus=1 in TrainerConfig
        config = TrainerConfig(tpus=1, .....)

"""


class DefaultArgs:

    save_path: str = None

    fast_dev_run: bool = False

    project_name: str = 'Cool-Project'
    wandb_run_name: str = None

    # will be helpful in resuming
    wandb_resume: bool = False
    wandb_run_id: str = None

    max_epochs: int = 10
    load_path: str = None # 'resuming.tar'

    accumulation_steps: int = 1

    tpus: int = 0

    precision: str = 'float32' # or 'mixed16' or 'float16'


class TrainingLoop(ABC):

    def forward(self, **kwargs):
        """This method can be implemented in the some class inherited from this class"""

    @abstractmethod
    def configure_optimizers(self, **kwargs):
        """This method must be implemented in the some class inherited from this class"""

    @abstractmethod
    def training_step(self, **kwargs):
        """This method must be implemented in the some class inherited from this class"""

    @abstractmethod
    def validation_step(self, **kwargs):
        """This method must be implemented in the some class inherited from this class"""

    def __init__(self, args):
        super().__init__()

        # self.model = ?
        self.precision = args.precision

        self.device = self.configure_devices(args)
        if self.gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.max_epochs = args.max_epochs
        self.save_path = args.save_path
        self.accumulation_steps = args.accumulation_steps
        self.load_path = args.load_path

        self.wandb_resume = args.wandb_resume
        self.fast_dev_run = args.fast_dev_run

        self.project_name = args.project_name
        self.wandb_run_name = args.wandb_run_name
        self.wandb_run_id = args.wandb_run_id
        self.wandb_off = args.wandb_off

        self.optimizer = self.configure_optimizers()
        self.scaler = self._configure_scaler()

        # will help in resuming training
        self.start_epoch = 1
        self.start_batch_idx = 0

        if self.load_path:
            self.load_state_dict(self.load_path)

        self.model.to(self.device)

        if self.precision == 'float16':
            self._setup_half()

    def _setup_half(self):
        self.model.half()
        print('Training with float16')
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    def configure_devices(self, args):

        device = torch.device('cpu')

        # If gpu is available, its automatically getting used
        self.gpus = torch.cuda.device_count()
        if self.gpus > 0:
            device = torch.device('cuda')

        if self.precision == 'mixed_16':
            if args.tpus > 0:
                raise ValueError('Currently mixed_16 is not supported with TPUs')
            elif self.gpus == 0:
                raise ValueError("mixed_16 should not be used with cpu")

        self.tpus = args.tpus
        if args.tpus == 1:
            try:
                device = xm.xla_device()
                self.gpus = 0
            except:
                raise ValueError("Can't set device to TPUs")

        print(f"Using {device}")
        return device

    def fit(self, tr_dataset, val_dataset):

        self.setup_wandb()

        if self.fast_dev_run:
            print("fast_dev_run is set to True")
            tr_dataset = next(iter(tr_dataset))
            val_datset = next(iter(val_dataset))

        try:
            self.train(tr_dataset, val_dataset)
        except KeyboardInterrupt:
            print('Interrupting through keyboard ======= Saving model weights')
            torch.save(self.model.state_dict(), 'keyboard-interrupted_'+self.save_path)

    def setup_wandb(self):

        # useful for testing
        if self.wandb_off:
            try: 
                os.system('wandb off')
            except:
                raise ValueError("wandb not available")

        if self.wandb_resume:
            if self.wandb_run_id is None:
                raise ValueError('wandb-run-id must be mentioned for resuming training')
            wandb.init(resume=self.wandb_run_id)
        else:
            wandb.init(project=self.project_name, name=self.wandb_run_name, id=self.wandb_run_id)

    def _configure_scaler(self):
        if self.precision == 'mixed16':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is not available')
            print('Training with mixed16')
            return torch.cuda.amp.GradScaler()

    def empty_grad(self):
        for param in self.model.parameters():
            param.grad = None

    def train(self, tr_dataset, val_dataset):

        # activating layers like dropout, batch-normalization for training
        self.model.train(True)

        steps = 0 # updating under accumulation condition
        tr_loss = 0 # setting up tr_loss for accumulation

        # setting up epochs (handling resuming)
        epochs = range(self.start_epoch, self.max_epochs+1)
        for epoch in epochs:

            # accumulator of training-loss
            losses = [0]

            val_loss = 0

            # helping in resuming
            self.start_epoch = epoch

            # setting up progress bar to display
            pbar = track(enumerate(tr_dataset), description=f"running epoch-{epoch} | tr_loss-{np.mean(losses)} | val_loss-{val_loss}")
            for batch_idx, batch in pbar:

                # will help in resuming training from last-saved batch_idx
                if batch_idx != self.start_batch_idx:
                    steps += 1
                    print(f'Wasting this iteration-{batch_idx} to start training from batch_idx-{self.start_batch_idx}')
                    continue
                
                self.start_batch_idx += 1

                # simply doing forward-propogation
                loss = self.training_step(batch, batch_idx)

                # accumulating tr_loss for logging (helpful when accumulation-steps > 1)
                tr_loss += loss.item()

                # configuring for mixed-precision
                if self.precision == 'mixed_16':
                    self.scaler.scale(loss).backward()

                else:
                    loss.backward()

                # gradient accumulation handler
                if (batch_idx+1)%self.accumulation_steps == 0:

                    # configuring for mixed-precision
                    if self.precision == 'mixed_16':
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        if self.tpus == 1:
                            xm.optimizer_step(self.optimizer, barrier=True)
                        else:
                            self.optimizer.step()

                    steps += 1

                    wandb.log({
                    'global_steps': steps,
                    'step_tr_loss': tr_loss
                    }, commit=True)

                    # emptying gradients in very efficient way
                    self.empty_grad()

                    # accumulating losses for training-loss at epoch end
                    losses.append(tr_loss)

                    # emptying tr_loss
                    tr_loss = 0

                if self.save_path:
                    self.save_state_dict(self.save_path)

            # clearing batch_idx for next epoch
            self.start_batch_idx = 0

            # val_loss at training epoch end for logging
            val_loss = self.evaluate(val_dataset)
            val_loss = val_loss.item()

            wandb.log({
                'epoch': epoch,
                'tr_loss': np.mean(losses),
                'val_loss': val_loss
                }, commit=False)
            
        print("Saving final weights")
        self.save_state_dict(self.save_path)

    def evaluate(self, val_dataset):
        # disabling layers like dropout, batch-normalization
        self.model.train(False)

        with torch.no_grad():
            for batch in val_dataset:
                val_loss = self.validation_step(batch)

        return val_loss

    def save_state_dict(self, path: str):

        # handling data parallel case
        module = self.model.module if hasattr(self.model, "module") else self.model

        # defining what all to save
        state_dict = {
            'model': module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'start_epoch': self.start_epoch,
            'start_batch_idx':  self.start_batch_idx
            }

        # mixed-precision states saving, if saving enabled
        if self.precision == 'mixed_16':
            state_dict.update({
                'scaler': self.scaler.state_dict()
                })

        if self.tpu > 0:
            xm.save(state_dict, path)
        else:
            torch.save(state_dict, path)

    def load_state_dict(self, path: str):
        
        print(
            """loading:
                     1) model-state-dict
                     2) optimizer-state-dict
                     3) scaler-state-dict (if mixed-precision)
                     4) start_epoch
                     5) start_batch_idx
            """
            )

        # load checkpoint from local-dir
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

        # handling data-parallel case
        module = self.model.module if hasattr(self.model, "module") else self.model
        module.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.precision == 'mixed_16':
            self.scaler.load_state_dict(checkpoint['scaler'])

        # helpful in resuming training from particular step
        self.start_epoch = checkpoint['start_epoch']
        self.start_batch_idx = checkpoint['start_batch_idx']


class TorchTrainer(TrainingLoop):

    def __init__(self, args):
        TrainingLoop.__init__(self, args)

    def forward(self, batch):
        """
        defines how you want to call model
        """

    @abstractmethod
    def configure_optimizers(self):
        """
        Return:
            `torch.optim` object
        """

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        This method should look something like this

            batch = batch.to(self.device)
            out = self(batch)
            loss = out.mean()
            loss /= self.accumulation_steps

            return loss
        """

    @abstractmethod
    def validation_step(self, batch):
        """
        This method should look something like this

            batch = batch.to(self.device)
            out = self(batch)
            loss = out.mean()

            return loss
        """


if __name__ == '__main__':
    """
    Peace max .....
    """