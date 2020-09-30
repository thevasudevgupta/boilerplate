# __author__ = 'Vasudev Gupta'
# __author_email__ = '7vasudevgupta@gmail.com'

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
    from tqdm import tqdm
except:
    os.system("pip install tqdm")
    from tqdm import tqdm

from abc import ABC, abstractmethod
from dataclasses import dataclass


"""
USAGE:

    from torch_utils import TorchTrainer, DefaultArgs

    class Trainer(TorchTrainer):

        def __init__(self, model, args):
            self.model = model

            # call this at end only
            super().__init__(args)

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

    @dataclass
    class Config(DefaultArgs):

        # pass your args
        lr: float = 2e-5
        ......

        # If want to update defaut_args; just pass it here only
        save_dir: str = 'weights'
        .......

    trainer = Trainer(model, args)
    trainer.fit(tr_dataset, val_dataset)

    Using TPU is pretty simple, run following command:
        !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.6-cp36-cp36m-linux_x86_64.whl
    & pass tpus=1 in DefaultArgs

"""

@dataclass
class DefaultArgs:

    # root dir for any kind of saving
    base_dir: str = "."

    # args used in TorchTrainer
    map_location: torch.device = torch.device("cuda:0")

    # model weights will be in `.pt` file 
    # while other training stuff will be in `.tar`
    save_dir: str = "resuming"
    load_dir: str = None

    fast_dev_run: bool = False

    project_name: str = None
    wandb_run_name: str = None
    wandb_off: bool = False

    # will be helpful in resuming
    wandb_resume: bool = False
    wandb_run_id: str = None
    
    max_epochs: int = 5
    
    accumulation_steps: int = 1
    tpus: int = 0
    precision: str = 'float32'
    

class TrainingLoop(ABC):

    @abstractmethod
    def configure_optimizers(self, **kwargs):
        """This method must be implemented in the class inherited from this class"""

    @abstractmethod
    def training_step(self, **kwargs):
        """This method must be implemented in the class inherited from this class"""

    @abstractmethod
    def validation_step(self, **kwargs):
        """This method must be implemented in the class inherited from this class"""

    def training_batch_end(self, batch_idx):
        """This method is called at the end of batch-{batch_idx}"""

    def training_epoch_end(self, epoch, losses):
        """This method is called at the end of epoch"""

    def training_end(self):
        """This method is called at the end of complete training"""

    def after_backward(self, batch_idx):
        """This method is called just after `loss.backward()`"""

    def __init__(self, args):
        super().__init__()

        self._sanity_check(args)

        self.args_dictn = args.__dict__

        # self.model = ?
        self.base_dir = self._setup_basedir(args.base_dir)

        self.precision = args.precision
        self.load_dir = args.load_dir
        self.save_dir = self._setup_savedir(args.save_dir)

        self.device = self.configure_devices(args)
        if self.gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.load_dir:
            self.map_location = args.map_location
            self.load_model_state_dict(f"{self.base_dir}/{self.load_dir}")
        self.model.to(self.device)

        self.optimizer = self.configure_optimizers()
        self.scaler = self._configure_scaler()
        self.start_epoch = 0
        self.start_batch_idx = 0
        if self.load_dir:
            self.load_training_state_dict(self.load_dir)

        self.max_epochs = args.max_epochs        
        self.accumulation_steps = args.accumulation_steps

        self.wandb_resume = args.wandb_resume
        self.fast_dev_run = args.fast_dev_run

        self.project_name = args.project_name
        self.wandb_run_name = args.wandb_run_name
        self.wandb_run_id = args.wandb_run_id
        self.wandb_off = args.wandb_off
        self.wandb_dir = self.base_dir

        if self.precision == 'float16':
            self._setup_half()

    def _setup_savedir(self, save_dir):
        if save_dir:
            if save_dir not in os.listdir(self.base_dir):
                os.mkdir(f"{self.base_dir}/{save_dir}")
            return save_dir

    def _setup_basedir(self, base_dir):
        if base_dir is None:
            return "."
        elif base_dir == ".":
            return base_dir
        elif base_dir not in os.listdir():
            os.mkdir(base_dir)
            print(f"training stuff will be saved in {base_dir}")
            return base_dir

    def _sanity_check(self, args):
        if not hasattr(args, "__dict__"):
            raise ValueError("Your argument class must have `dataclass` decorator")

        for arg in DefaultArgs().__dict__:
            if not hasattr(args, arg):
                raise ValueError(f"Your config must have `{arg}`")

    def train_step(self, batch, batch_idx):

        if self.precision == 'mixed16':
            return torch.cuda.amp.autocast(self.training_step)(batch, batch_idx)

        return self.training_step(batch, batch_idx)

    def val_step(self, batch):

        if self.precision == 'mixed16':
            return torch.cuda.amp.autocast(self.validation_step)(batch, batch_idx)

        return self.validation_step(batch)

    def _setup_half(self):
        raise ValueError('Need to implement float16 with apex')

    def configure_devices(self, args):

        device = torch.device('cpu')

        # If gpu is available, its automatically getting used
        self.gpus = torch.cuda.device_count()
        if self.gpus > 0:
            device = torch.device('cuda')

        if self.precision == 'mixed16':
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
            wandb.init(project=self.project_name, name=self.wandb_run_name, id=self.wandb_run_id, config=self.args_dictn, dir=self.wandb_dir)

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
        epochs = range(self.start_epoch, self.max_epochs)
        for epoch in epochs:

            # accumulator of training-loss
            losses = []

            # helping in resuming
            self.start_epoch = epoch

            # setting up progress bar to display
            desc = f"running epoch-{epoch}"
            pbar = tqdm(enumerate(tr_dataset), total=len(tr_dataset), desc=desc, initial=0, leave=True)
            for batch_idx, batch in pbar:

                # will help in resuming training from last-saved batch_idx
                if batch_idx != self.start_batch_idx:
                    steps += 1
                    pbar.write(f'training will start from batch_idx-{self.start_batch_idx}')
                    continue

                self.start_batch_idx += 1

                # simply doing forward-propogation
                loss = self.train_step(batch, batch_idx)
                loss /= self.accumulation_steps

                # accumulating tr_loss for logging (helpful when accumulation-steps > 1)
                tr_loss += loss.item()

                # configuring for mixed-precision
                if self.precision == 'mixed16':
                    loss = self.scaler.scale(loss)

                loss.backward()

                self.after_backward(batch_idx)

                # gradient accumulation handler
                if (batch_idx+1)%self.accumulation_steps == 0:

                    # configuring for mixed-precision
                    if self.precision == 'mixed16':
                        self.mixed_optimizer_step(self, self.optimizer)

                    else:
                        if self.tpus == 1:
                            xm.optimizer_step(self.optimizer, barrier=True)
                        else:
                            self.optimizer.step()

                    wandb.log({
                    'global_steps': steps,
                    'step_tr_loss': tr_loss
                    }, commit=True)

                    steps += 1
                    pbar.set_postfix(tr_loss=tr_loss)

                    # emptying gradients in very efficient way
                    self.empty_grad()

                    # accumulating losses for training-loss at epoch end
                    losses.append(tr_loss)

                    # emptying tr_loss
                    tr_loss = 0

                self.training_batch_end(batch_idx)

            # clearing batch_idx for next epoch
            self.start_batch_idx = 0

            # val_loss at training epoch end for logging
            val_loss = self.evaluate(val_dataset)

            wandb.log({
                'epoch': epoch,
                'tr_loss': np.mean(losses),
                'val_loss': val_loss.item()
                }, commit=False)

            self.training_epoch_end(epoch, losses)

        self.start_epoch += 1

        if self.save_dir:
            print("Saving model and training related stuff")
            self.save_model_state_dict(f"{self.base_dir}/{self.save_dir}")
            self.save_training_state_dict(f"{self.base_dir}/{self.save_dir}")
        
        self.training_end()

    def mixed_optimizer_step(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def evaluate(self, val_dataset):
        # disabling layers like dropout, batch-normalization
        self.model.train(False)

        desc = 'Validating ....'
        pbar = tqdm(val_dataset, total=len(val_dataset), desc=desc, initial=0, leave=False)
        for batch in pbar:
            val_loss = self.val_step(batch)
            pbar.set_postfix(val_loss=val_loss.item())

        return val_loss

    def save_training_state_dict(self, save_dir: str):

        path = f"{save_dir}/model.pt"

        # defining what all to save
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'start_epoch': self.start_epoch,
            'start_batch_idx':  self.start_batch_idx
            }

        # mixed-precision states saving, if saving enabled
        if self.precision == 'mixed16':
            state_dict.update({
                'scaler': self.scaler.state_dict()
                })

        torch.save(state_dict, path)

    def save_model_state_dict(self, save_dir: str):

        path = f"{save_dir}/model.pt"

        module = self.model.module if hasattr(self.model, "module") else self.model
        state_dict = module.state_dict()

        if self.tpus > 0:
            xm.save(state_dict, path)
        else:
            torch.save(state_dict, path)

    def load_model_state_dict(self, load_dir: str):

        path = f"{load_dir}/model.pt"
        """
        Note:
            `map_function` is very memory expensive if you are changing the device
        """

        print(
            """loading:
                1) model state_dict
            """
        )

        model = torch.load(path, map_location=self.map_location)

        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model)
        else:
            self.model.load_state_dict(model)        

    def load_training_state_dict(self, load_dir: str):
        
        path = f"{load_dir}/training.tar"

        print(
            """loading:
                1) optimizer-state-dict
                2) scaler-state-dict (if mixed-precision)
                3) start_epoch
                4) start_batch_idx
            """
            )

        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint.pop('optimizer'))

        if self.precision == 'mixed16':
            self.scaler.load_state_dict(checkpoint.pop('scaler'))

        # helpful in resuming training from particular step
        self.start_epoch = checkpoint.pop('start_epoch')
        self.start_batch_idx = checkpoint.pop('start_batch_idx')

        print(f'loading successful (start-epoch-{self.start_epoch}, start_batch_idx-{self.start_batch_idx})')


class TorchTrainer(TrainingLoop):

    def __init__(self, args):
        TrainingLoop.__init__(self, args)

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
            # with torch.cuda.amp.autocast((self.precision=='mixed_16')):
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
            with torch.no_grad():
                # with torch.cuda.amp.autocast((self.precision=='mixed_16')):
                out = self(batch)
                loss = out.mean()

            return loss
        """

    def training_batch_end(self, batch_idx):
        """This method is called at the end of batch-{batch_idx}"""

    def training_epoch_end(self, epoch, losses):
        """This method is called at the end of epoch"""

    def training_end(self):
        """This method is called at the end of complete training"""

    def after_backward(self, batch_idx):
        """This method is called just after `loss.backward()`"""

    def histogram_params(self, logdir="tb_params"):
        """
        You need to call this method yourself
        """
        writer = SummaryWriter(log_dir=f"{self.base_dir}/{logdir}")

        params = self.model.named_parameters()
        for n, param in params:
            writer.add_histogram(n, param)
        
        writer.close()
        # tensorboard --logdir "{tb_params}"
        
    def histogram_grads(self, logdir="tb_grads"):
        """
        You need to call this method yourself

        Remember to call this only after `backward`
        """

        writer = SummaryWriter(log_dir=f"{self.base_dir}/{logdir}")

        params = self.model.named_parameters()
        for n, param in params:
            if param.grad is not None:
                writer.add_histogram(n, param.grad)
            else:
                writer.add_scalar(n, 0.0)

        writer.close()
        # tensorboard --logdir "{tb_grads}"


if __name__ == '__main__':
    """
    Peace max .....
    """