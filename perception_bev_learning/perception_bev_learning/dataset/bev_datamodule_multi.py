import torch
from lightning import LightningDataModule
from perception_bev_learning.dataset import (
    BevDatasetMultiTemporalTruncated,
    collate_fn_temporal_multi,
    BEVTruncatedBatchSampler,
)
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
import copy


class BEVDataModuleMulti(LightningDataModule):
    """`LightningDataModule` for the Bev dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
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

        self.data_train: Optional[BevDatasetMultiTemporalTruncated] = None
        self.data_val: Optional[BevDatasetMultiTemporalTruncated] = None
        self.data_test: Optional[BevDatasetMultiTemporalTruncated] = None

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

        self.data_train = BevDatasetMultiTemporalTruncated(
            self.cfg_dataset, mode="train"
        )
        self.data_val = BevDatasetMultiTemporalTruncated(self.cfg_dataset, mode="val")
        print("Created Val and Training dataset")
        if self.cfg_dataset.return_test:
            self.data_test = BevDatasetMultiTemporalTruncated(
                self.cfg_dataset, mode="test"
            )
        else:
            self.data_test = self.data_val

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        batch_sampler = BEVTruncatedBatchSampler(
            index_list=self.data_train.idx_sequences,
            batch_size=self.cfg_dataloader.train_dataloader.batch_size,
            shuffle=self.cfg_dataloader.train_dataloader.shuffle,
            cfg=self.cfg_dataset,
        )

        return DataLoader(
            dataset=self.data_train,
            persistent_workers=self.cfg_dataloader.train_dataloader.persistent_workers,
            prefetch_factor=self.cfg_dataloader.train_dataloader.prefetch_factor,
            num_workers=self.cfg_dataloader.train_dataloader.num_workers,
            pin_memory=self.cfg_dataloader.train_dataloader.pin_memory,
            collate_fn=collate_fn_temporal_multi,
            batch_sampler=batch_sampler,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        batch_sampler = BEVTruncatedBatchSampler(
            index_list=self.data_val.idx_sequences,
            batch_size=self.cfg_dataloader.val_dataloader.batch_size,
            shuffle=self.cfg_dataloader.val_dataloader.shuffle,
            cfg=self.cfg_dataset,
        )

        return DataLoader(
            dataset=self.data_val,
            persistent_workers=self.cfg_dataloader.val_dataloader.persistent_workers,
            prefetch_factor=self.cfg_dataloader.val_dataloader.prefetch_factor,
            num_workers=self.cfg_dataloader.val_dataloader.num_workers,
            pin_memory=self.cfg_dataloader.val_dataloader.pin_memory,
            collate_fn=collate_fn_temporal_multi,
            batch_sampler=batch_sampler,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        batch_sampler = BEVTruncatedBatchSampler(
            index_list=self.data_test.idx_sequences,
            batch_size=self.cfg_dataloader.test_dataloader.batch_size,
            shuffle=self.cfg_dataloader.test_dataloader.shuffle,
            cfg=self.cfg_dataset,
        )

        return DataLoader(
            dataset=self.data_test,
            persistent_workers=self.cfg_dataloader.test_dataloader.persistent_workers,
            prefetch_factor=self.cfg_dataloader.test_dataloader.prefetch_factor,
            num_workers=self.cfg_dataloader.test_dataloader.num_workers,
            pin_memory=self.cfg_dataloader.test_dataloader.pin_memory,
            collate_fn=collate_fn_temporal_multi,
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
