import random
from typing import Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

def worker_init_fn(id: int):
    """
    DataLoaders workers init function.
    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.
    If a global seed is set, the augmentations are deterministic.
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    # Recently a PR fixed this in Lightning, but an override of the worker_init_fn
    # without these fixes would break it again.
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)
    # Implementation by the numpy author from here
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562


class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset : Dataset, test_dataset : Dataset, val_dataset : Optional[Dataset] = None, val_size : float = 0.1, batch_size : int = 32, num_workers : int = 1, pin_memory : bool = False, persistent_workers : bool = False, class_balancing : bool = False):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.val_size = val_size

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.class_balancing = class_balancing

        self.setup()

    def prepare_data(self) -> None:
        # Download the dataset here
        # pylogger.info("Downloading data")
        pass

    def setup(self, stage: Optional[str] = None):        
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if stage is None or stage == "fit":
            if self.val_dataset is None:
                train_length = int(len(self.train_dataset) * (1.0 - self.val_size))
                val_length = int(len(self.train_dataset) - train_length)
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_length, val_length])

        #if stage is None or stage == "test":
        #    pass

    def train_dataloader(self) -> DataLoader:
        sampler = None
        return DataLoader(
            self.train_dataset,
            shuffle=not self.class_balancing,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                worker_init_fn=worker_init_fn,
            )
        ]
