# define training arguments here
import quick
from dataclasses import dataclass, field, replace

@dataclass
class TrainingArgs(quick.TorchTrainingArgs):

    lr: float = 1.e-3
    batch_size: int = 8
    num_workers: int = 2

    tr_file_path: str = None
    val_file_path: str = None
    hub_id: str = None

    # inside args
    base_dir: str = None
    save_epoch_dir: str = None
    gradient_accumulation_steps: str = 1
    max_epochs: str = 5
    project_name: str = None
    wandb_run_name: str = None

baseline = TrainingArgs()
