from typing import Any, List, Optional, Sequence, Union

import hydra
import lightning as pl
from litdata import StreamingDataLoader
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from goldenretriever.common.log import get_logger
from goldenretriever.data.datasets import GoldenRetrieverDataset

# from goldenretriever.data.streaming_dataset import GoldenRetrieverCollator, StreamingGoldenRetrieverDataset
from goldenretriever.data.lit_dataset import GoldenStreamingDataset
from goldenretriever.data.streaming_dataset import GoldenRetrieverCollator, GoldenRetrieverStreamingDataset
from goldenretriever.data.utils import GoldenDistributedSampler

logger = get_logger()


class GoldenRetrieverPLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Optional[GoldenRetrieverDataset] = None,
        val_datasets: Optional[Sequence[GoldenRetrieverDataset]] = None,
        test_datasets: Optional[Sequence[GoldenRetrieverDataset]] = None,
        num_workers: Optional[Union[DictConfig, int]] = None,
        datasets: Optional[DictConfig] = None,
        tokenizer=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.datasets = datasets
        if num_workers is None:
            num_workers = 0
        if isinstance(num_workers, int):
            num_workers = DictConfig(
                {"train": num_workers, "val": num_workers, "test": num_workers}
            )
        self.num_workers = num_workers
        # data
        self.train_dataset: Optional[GoldenRetrieverDataset] = train_dataset
        self.val_datasets: Optional[Sequence[GoldenRetrieverDataset]] = val_datasets
        self.test_datasets: Optional[Sequence[GoldenRetrieverDataset]] = test_datasets
        self.tokenizer = tokenizer

    def prepare_data(self, *args, **kwargs):
        """
        Method for preparing the data before the training. This method is called only once.
        It is used to download the data, tokenize the data, etc.
        """
        pass

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            # usually there is only one dataset for train
            # if you need more train loader, you can follow
            # the same logic as val and test datasets
            # if self.train_dataset is None:
            #     self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            #     self.val_datasets = [
            #         hydra.utils.instantiate(dataset_cfg)
            #         for dataset_cfg in self.datasets.val
            #     ]
            self.train_dataset = GoldenRetrieverStreamingDataset(
                name="aida_train",
                question_tokenizer=self.tokenizer,
                local="/home/ric/Projects/golden-retriever/data/dpr-like/el/mosaic/train",
                batch_size=32,
                predownload=64*64,
                shuffle_seed=42,
            )
            # self.train_dataset = GoldenStreamingDataset(
            #     name="streaming_train",
            #     question_tokenizer=self.tokenizer,
            #     input_dir="/home/ric/Projects/golden-retriever/data/dpr-like/el/litdata/train",
            # )
            self.val_dataset = GoldenRetrieverStreamingDataset(
                name="aida_val",
                question_tokenizer=self.tokenizer,
                local="/home/ric/Projects/golden-retriever/data/dpr-like/el/mosaic/val",
                batch_size=32,
                predownload=64*64,
                shuffle_seed=42,
            )
            # self.val_dataset = GoldenStreamingDataset(
            #     name="streaming_val",
            #     question_tokenizer=self.tokenizer,
            #     input_dir="/home/ric/Projects/golden-retriever/data/dpr-like/el/litdata/val",
            # )
            self.val_datasets = [self.val_dataset]
        if stage == "test":
            if self.test_datasets is None:
                self.test_datasets = [
                    hydra.utils.instantiate(dataset_cfg)
                    for dataset_cfg in self.datasets.test
                ]

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        # torch_dataset = self.train_dataset.to_torch_dataset()
        return DataLoader(
            # self.train_dataset.to_torch_dataset(),
            # torch_dataset,
            self.train_dataset,
            shuffle=False,
            batch_size=32,
            num_workers=self.num_workers.train,
            pin_memory=False,
            # prefetch_factor=2,
            # persistent_workers=True,
            # collate_fn=lambda x: x,
            collate_fn=GoldenRetrieverCollator(tokenizer=self.train_dataset.question_tokenizer),
            # user a custom distributed sampler
            # sampler=GoldenDistributedSampler
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        dataloaders = []
        for dataset in self.val_datasets:
            # torch_dataset = dataset.to_torch_dataset()
            dataloaders.append(
                DataLoader(
                    # torch_dataset,
                    dataset,
                    shuffle=False,
                    batch_size=32,
                    num_workers=self.num_workers.val,
                    pin_memory=False,
                    # prefetch_factor=2,
                    # persistent_workers=True,
                    # collate_fn=lambda x: x,
                    collate_fn=GoldenRetrieverCollator(tokenizer=dataset.question_tokenizer),
                    # sampler=GoldenDistributedSampler
                )
            )
        return dataloaders

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        dataloaders = []
        for dataset in self.test_datasets:
            torch_dataset = dataset.to_torch_dataset()
            dataloaders.append(
                DataLoader(
                    torch_dataset,
                    shuffle=False,
                    batch_size=None,
                    num_workers=self.num_workers.test,
                    pin_memory=False,
                    collate_fn=lambda x: x,
                )
            )
        return dataloaders

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, "
        )
