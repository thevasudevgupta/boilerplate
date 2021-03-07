# define training arguments here
from dataclasses import dataclass, field, replace

@dataclass
class TrainingArgs:

    lr: float = 1.e-3
    batch_size: int = 8
    num_workers: int = 2
    
    tr_file_path: str = None
    val_file_path: str = None

    save_path: str = None
    base_dir: str = None

baseline = TrainingArgs()
