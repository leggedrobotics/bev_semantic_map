import torch
from lightning import LightningDataModule
from perception_bev_learning.dataset import (
    BevDataset,
    collate_fn,
    BevDatasetTemporal,
)
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
import random
from torch.utils.data import Sampler
from itertools import chain


class BEVBatchSampler(Sampler):
    def __init__(self, index_list, batch_size, shuffle=False):
        """
        Batch Sampler for returning the indices of the batch
        index_list: list of sequences where each item contains n = sequence_length indices
        batch_size: batch size
        shuffle: Whether to shuffle the sequences or not
        """
        self.sequence_length = len(index_list[0])
        self.data = index_list
        self.batch_size = batch_size
        self.num_sequences = len(self.data)
        self.num_samples = (
            self.num_sequences // self.batch_size
        ) * self.sequence_length
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)

        indices = list(chain(*self.data))
        indices = indices[: self.num_samples * self.batch_size]
        print(f"Number of Samples are {len(indices)}")
        print(f"Number of batches are {self.num_samples}")
        # print(indices)

        for i in range(0, len(indices), self.batch_size * self.sequence_length):
            for j in range(i, i + self.sequence_length):
                yield [
                    indices[j + k * self.sequence_length]
                    for k in range(self.batch_size)
                ]

    def __len__(self):
        return self.num_samples


class BEVDataModuleTemporal(LightningDataModule):
    """`LightningDataModule for the Bev Temporal Dataset with batch sampler for returning consecutive
    samples in subsequent iterations for N = sequence length iterations.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.cfg_dataset = self.hparams.dataset
        self.cfg_dataloader = self.hparams.dataloader

        self.data_train: Optional[BevDatasetTemporal] = None
        self.data_val: Optional[BevDatasetTemporal] = None
        self.data_test: Optional[BevDatasetTemporal] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        self.data_train = BevDatasetTemporal(self.cfg_dataset, mode="train")
        self.data_val = BevDatasetTemporal(self.cfg_dataset, mode="val")
        print("Created Val and Training dataset")
        if self.cfg_dataset.return_test:
            self.data_test = BevDatasetTemporal(self.cfg_dataset, mode="test")
        else:
            self.data_test = self.data_val

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        batch_sampler = BEVBatchSampler(
            index_list=self.data_train.idx_sequences,
            batch_size=self.cfg_dataloader.train_dataloader.batch_size,
            shuffle=self.cfg_dataloader.train_dataloader.shuffle,
        )

        return DataLoader(
            dataset=self.data_train,
            persistent_workers=self.cfg_dataloader.train_dataloader.persistent_workers,
            prefetch_factor=self.cfg_dataloader.train_dataloader.prefetch_factor,
            num_workers=self.cfg_dataloader.train_dataloader.num_workers,
            pin_memory=self.cfg_dataloader.train_dataloader.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        batch_sampler = BEVBatchSampler(
            index_list=self.data_val.idx_sequences,
            batch_size=self.cfg_dataloader.val_dataloader.batch_size,
            shuffle=self.cfg_dataloader.val_dataloader.shuffle,
        )

        return DataLoader(
            dataset=self.data_val,
            persistent_workers=self.cfg_dataloader.val_dataloader.persistent_workers,
            prefetch_factor=self.cfg_dataloader.val_dataloader.prefetch_factor,
            num_workers=self.cfg_dataloader.val_dataloader.num_workers,
            pin_memory=self.cfg_dataloader.val_dataloader.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        batch_sampler = BEVBatchSampler(
            index_list=self.data_test.idx_sequences,
            batch_size=self.cfg_dataloader.test_dataloader.batch_size,
            shuffle=self.cfg_dataloader.test_dataloader.shuffle,
        )

        return DataLoader(
            dataset=self.data_test,
            persistent_workers=self.cfg_dataloader.test_dataloader.persistent_workers,
            prefetch_factor=self.cfg_dataloader.test_dataloader.prefetch_factor,
            num_workers=self.cfg_dataloader.test_dataloader.num_workers,
            pin_memory=self.cfg_dataloader.test_dataloader.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
