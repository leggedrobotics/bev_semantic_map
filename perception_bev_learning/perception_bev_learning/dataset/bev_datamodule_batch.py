import torch
from lightning import LightningDataModule
from perception_bev_learning.dataset import (
    BevDataset,
    BevDatasetTemporalBatch,
    collate_fn_temporal,
)
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
import copy


class BEVDataModuleTemporalBatch(LightningDataModule):
    """`LightningDataModule` for the BevDatasetTemporalBatch.
    Used for returning dictionary of BxNxitems
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

        self.data_train: Optional[BevDatasetTemporalBatch] = None
        self.data_val: Optional[BevDatasetTemporalBatch] = None
        self.data_test: Optional[BevDatasetTemporalBatch] = None

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

        self.data_train = BevDatasetTemporalBatch(self.cfg_dataset, mode="train")
        self.data_val = BevDatasetTemporalBatch(self.cfg_dataset, mode="val")
        print("Created Val and Training dataset")
        if self.cfg_dataset.return_test:
            self.data_test = BevDatasetTemporalBatch(self.cfg_dataset, mode="test")
        else:
            self.data_test = self.data_val

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            **(self.cfg_dataloader.train_dataloader),
            collate_fn=collate_fn_temporal,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            **(self.cfg_dataloader.val_dataloader),
            collate_fn=collate_fn_temporal,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            **(self.cfg_dataloader.test_dataloader),
            collate_fn=collate_fn_temporal,
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
