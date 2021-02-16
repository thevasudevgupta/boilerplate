
import yaml

from data_utils import DataLoader
from training_utils import Trainer, TrainerConfig
import training_utils
from modeling import Model

if __name__ == '__main__':

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = Model(config)

    args = TrainerConfig.from_default()
    args.update(getattr(training_utils, "baseline").__dict__)

    dl = DataLoader(args)
    tr_dataset, val_dataset = dl.setup()
    tr_dataset = dl.train_dataloader(tr_dataset)
    val_dataset = dl.val_dataloader(val_dataset)

    trainer = Trainer(model, args)
    trainer.fit(tr_dataset, val_dataset)
