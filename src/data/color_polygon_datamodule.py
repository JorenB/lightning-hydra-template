from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.components.color_polygon_dataset import ColorPolygonDataset


class ColorPolygonDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Any,
        batch_size: int = 10,
        num_workers: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_train: ColorPolygonDataset = dataset(num_samples=5 * 1280)
        self.data_val: ColorPolygonDataset = dataset(num_samples=1280)
        self.data_test: ColorPolygonDataset = dataset(num_samples=1280)

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up."""
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
