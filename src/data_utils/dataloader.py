
import torch


class DatasetReader(torch.utils.data.Dataset):

    def __init__(self, file_path: str):
        self.file_path = file_path
        # put your reading strategy here
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[0]

class DataLoader(object):

    def __init__(self, args, **kwargs):

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.tr_file_path = args.tr_file_path
        self.val_file_path = args.val_file_path

    def setup(self):
        tr_dataset = DatasetReader(self.tr_file_path)
        val_dataset = DatasetReader(self.val_file_path)
        return tr_dataset, val_dataset

    def train_dataloader(self, tr_dataset):
        return torch.utils.data.DataLoader(tr_dataset,
                                            pin_memory=True,
                                            shuffle=True,
                                            batch_size=self.batch_size,
                                            collate_fn=self.collate_fn,
                                            num_workers=self.num_workers)

    def val_dataloader(self, val_dataset):
        return torch.utils.data.DataLoader(val_dataset,
                                            pin_memory=True,
                                            shuffle=False,
                                            batch_size=self.batch_size,
                                            collate_fn=self.collate_fn,
                                            num_workers=self.num_workers)

    def collate_fn(self, features):
        # update your collate function here
        return features
