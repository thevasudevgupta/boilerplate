# __author__ = 'Vasudev Gupta'
# __author_email__ = '7vasudevgupta@gmail.com'

import torch
import numpy as np

import wandb
from tqdm import tqdm

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace


"""
USAGE:

    import torch_utils
    from torch_utils import TrainerConfig

    class Trainer(torch_utils.Trainer):

        def __init__(self, model, config):
            super().__init__(args)
            self.model = model

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

    config = TrainerConfig(path='run-dir', max_epochs=10, load_path='resuming.tar',
                    project_name='Cool-Project', wandb_run_name='hello-world', accumulation_steps=8,
                    precision='float32', wandb_run_id='put-anything-unique')

    trainer = Trainer(model, config)
    trainer.fit(tr_dataset, val_dataset)
"""


@dataclass
class TrainerConfig:

    path: str = None

    project_name: str = 'Cool-Project'
    wandb_run_name: str = None

    resume_training: bool = False
    wandb_run_id: str = None # will be helpful in resuming

    max_epochs: int = 10
    load_path: str = None # 'resuming.tar'

    accumulation_steps: int = 1
    
    precision: str = 'float32' # or 'mixed_16'


class TrainingLoop(ABC):

    def forward(self, **kwargs):
        """This method is must be implemented in the some class inherited from this class"""

    @abstractmethod
    def configure_optimizers(self, **kwargs):
        """This method is must be implemented in the some class inherited from this class"""

    @abstractmethod
    def training_step(self, **kwargs):
        """This method is must be implemented in the some class inherited from this class"""

    @abstractmethod
    def validation_step(self, **kwargs):
        """This method is must be implemented in the some class inherited from this class"""

    def __init__(self, args):
        super().__init__()

        # self.model = ?

        # devices are automatically handled
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = torch.nn.DataParallel(self.model)

        self.precision = args.precision
        self.max_epochs = args.max_epochs
        self.path = args.path
        self.accumulation_steps = args.accumulation_steps
        self.load_path = args.load_path
        self.resume_training = resume_training

        self.optimizer = self.configure_optimizers()
        self.scaler = self._configure_precision()

        self.start_epoch = 1
        self.start_batch_idx = 0

        if self.load_path:
            self.load_state_dict(self.load_path)

        self.model.to(self.device)

    def fit(self, tr_dataset, val_dataset):

        self.setup_wandb()

        try:
            self.train(tr_dataset, val_dataset)
        except KeyboardInterrupt:
            print('Interrupting through keyboard ======= Saving model weights')
            torch.save(self.model.state_dict(), 'keyboard-interrupted_'+self.path)

    def setup_wandb(self):

        if self.resume_training:
            if self.wandb_run_id is None:
                raise ValueError('wandb-run-id must be mentioned for resuming training')
            wandb.init(resume=self.wandb_run_id)

        else:
            wandb.init(project=self.project_name, name=self.wandb_run_name, id=self.wandb_run_id)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def _configure_precision(self):
        if self.precision == 'mixed_16':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is not available')
            self.training_step = torch.cuda.amp.autocast(self.training_step, enabled=(self.precision=='mixed_16'))
            return torch.cuda.amp.GradScaler()

    def empty_grad(self):
        for param in self.model.parameters():
            param.grad = None

    def train(self, tr_dataset, val_dataset):

        # activating layers like dropout, batch-normalization for training
        self.model.train(True)

        steps = 0 # setting up global-steps for logging
        tr_loss = 0 # setting up tr_loss for accumulation

        # setting up progress-bar for epochs (handling resuming)
        epochs = range(self.start_epoch, self.max_epochs+1)
        for epoch in epochs:

            # accumulator of training-loss
            losses = []

            # helping in resuming
            self.start_epoch = epoch

            # setting up progress bar to display
            pbar = tqdm(enumerate(tr_dataset))
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
                        self.optimizer.step()

                    # emptying gradients in very efficient way
                    self.empty_grad()
                    
                    # emptying tr_loss
                    tr_loss = 0

                    steps += 1

                    wandb.log({
                    'global_steps': steps,
                    'step_tr_loss': tr_loss
                    }, commit=True)

                    # accumulating losses for training-loss at epoch end
                    losses.append(loss.item())
                
                if self.path:
                    self.save_state_dict(self.path)

            # clearing batch_idx for next epoch
            self.start_batch_idx = 0

            # val_loss at training epoch end for logging
            val_loss = self.evaluate(val_dataset)

            wandb.log({
                'epoch': epoch,
                'tr_loss': np.mean(losses),
                'val_loss': val_loss.item()
                }, commit=False)

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
        checkpoint = torch.load(path, map_location=self.device)

        # handling data-parallel case
        module = self.model.module if hasattr(self.model, "module") else self.model
        module.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.precision == 'mixed_16':
            self.scaler.load_state_dict(checkpoint['scaler'])

        # helpful in resuming training from particular step
        self.start_epoch = checkpoint['start_epoch']
        self.start_batch_idx = checkpoint['start_batch_idx']


class Trainer(TrainingLoop):

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
    
    # code working on cpu
    # gradient accumulation working
    # resume-training is working

    config = TrainerConfig()
    args = config
    print(args)
    
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 1)
    )

    # trainer = Trainer(model, args)
    # trainer.fit(torch.ones(3200, 4, dtype=torch.float), torch.ones(3, 4, dtype=torch.float))
