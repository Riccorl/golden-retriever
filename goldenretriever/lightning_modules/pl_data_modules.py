import os
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Union

import hydra
import lightning as pl
import torch
import transformers as tr
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from goldenretriever.common.data_utils import preprocess_to_mds
from goldenretriever.common.log import get_logger

from goldenretriever.data.datasets import (
    GoldenRetrieverCollator,
    GoldenRetrieverStreamingDataset,
    GoldenStreamingDataLoader,
)

from omegaconf import OmegaConf, open_dict
import goldenretriever.common.dist_utils as dist

logger = get_logger(__name__)


class GoldenRetrieverPLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Optional[GoldenRetrieverStreamingDataset] = None,
        train_dataset_kwargs: dict = None,
        val_datasets: Optional[Sequence[GoldenRetrieverStreamingDataset]] = None,
        val_datasets_kwargs: List[dict] = None,
        test_datasets: Optional[Sequence[GoldenRetrieverStreamingDataset]] = None,
        test_datasets_kwargs: List[dict] = None,
        num_workers: Optional[Union[DictConfig, int]] = None,
        datasets: Optional[DictConfig] = None,
        tokenizer=None,
        question_tokenizer=None,
        passage_tokenizer=None,
        preprocess: bool = True,
        seed: int = 42,
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
        # kwargs
        self.train_dataset_kwargs = train_dataset_kwargs or {}
        if val_datasets is not None:
            self.val_datasets_kwargs = val_datasets_kwargs or [{} for _ in val_datasets]
        self.val_datasets_kwargs = val_datasets_kwargs
        if test_datasets is not None:
            self.test_datasets_kwargs = test_datasets_kwargs or [
                {} for _ in test_datasets
            ]
        # data
        self.train_dataset: Optional[GoldenRetrieverStreamingDataset] = train_dataset
        self.val_datasets: Optional[Sequence[GoldenRetrieverStreamingDataset]] = (
            val_datasets
        )
        self.test_datasets: Optional[Sequence[GoldenRetrieverStreamingDataset]] = (
            test_datasets
        )
        self.train_dataloader_obj = None
        self.val_dataloader_obj = []
        self.test_dataloader_obj = []
        self.tokenizer = tokenizer
        self.question_tokenizer = question_tokenizer or tokenizer
        self.passage_tokenizer = passage_tokenizer or tokenizer

        # other stuff
        self.preprocess = preprocess
        self.seed = seed

    def prepare_data(self, *args, **kwargs):
        """
        Method for preparing the data before the training. This method is called only once.
        It is used to download the data, tokenize the data, etc.
        """
        # preprocess dataset
        if self.train_dataset is not None:
            data_path = None
            if isinstance(self.train_dataset, (str, os.PathLike)):
                data_path = self.train_dataset
                kwargs = self.train_dataset_kwargs
            elif isinstance(self.train_dataset, DictConfig):
                # TODO
                data_path = self.train_dataset["local"]
                kwargs = self.train_dataset
            else:
                logger.debug("No data path found, skipping preprocessing")

            if data_path:
                logger.debug(f"Building train dataset from {data_path}")

                dataset_class = kwargs.get("_target_", GoldenRetrieverStreamingDataset)
                preprocessing_fn = None
                if kwargs.get("preprocess", False):
                    preprocessing_fn = partial(
                        dataset_class.tokenize,
                        **{
                            "question_tokenizer": self.question_tokenizer,
                            "passage_tokenizer": self.passage_tokenizer,
                            **kwargs,
                        },
                    )
                preprocess_to_mds(data_path, preprocessing_fn)

        if self.val_datasets is not None:
            for i, dataset in enumerate(self.val_datasets):
                data_path = None
                if isinstance(dataset, (str, os.PathLike)):
                    data_path = dataset
                    kwargs = self.val_datasets_kwargs[i]
                elif isinstance(dataset, DictConfig):
                    # TODO
                    data_path = dataset["local"]
                    kwargs = dataset
                else:
                    logger.debug("No data path found, skipping preprocessing")

                if data_path:
                    logger.debug(f"Building val dataset from {data_path}")

                    dataset_class = kwargs.get(
                        "_target_", GoldenRetrieverStreamingDataset
                    )
                    preprocessing_fn = None
                    if kwargs.get("preprocess", False):
                        preprocessing_fn = partial(
                            dataset_class.tokenize,
                            **{
                                "question_tokenizer": self.question_tokenizer,
                                "passage_tokenizer": self.passage_tokenizer,
                                **kwargs,
                            },
                        )
                    preprocess_to_mds(data_path, preprocessing_fn)

        if self.test_datasets is not None:
            for dataset in self.test_datasets:
                data_path = None
                if isinstance(dataset, (str, os.PathLike)):
                    data_path = dataset
                    kwargs = self.test_datasets[i]
                elif isinstance(dataset, DictConfig):
                    # TODO
                    data_path = dataset["local"]
                    kwargs = dataset
                else:
                    logger.debug("No data path found, skipping preprocessing")

                if data_path:
                    logger.debug(f"Building test dataset from {data_path}")

                    dataset_class = kwargs.get(
                        "_target_", GoldenRetrieverStreamingDataset
                    )
                    preprocessing_fn = None
                    if kwargs.get("preprocess", False):
                        preprocessing_fn = partial(
                            dataset_class.tokenize,
                            **{
                                "question_tokenizer": self.question_tokenizer,
                                "passage_tokenizer": self.passage_tokenizer,
                                **self.test_datasets_kwargs[i],
                            },
                        )
                    preprocess_to_mds(data_path, preprocessing_fn)

    def setup(self, stage: str | None = None):

        os.environ["WORLD_SIZE"] = str(dist.get_world_size())
        os.environ["LOCAL_WORLD_SIZE"] = str(dist.get_local_world_size())
        os.environ["RANK"] = str(dist.get_rank())

        logger.debug(f"World size: {os.environ['WORLD_SIZE']}")
        logger.debug(f"Local world size: {os.environ['LOCAL_WORLD_SIZE']}")
        logger.debug(f"Rank: {os.environ['RANK']}")

        if stage == "fit" or stage is None:
            # usually there is only one dataset for train
            # if you need more train loader, you can follow
            # the same logic as val and test datasets
            if isinstance(self.train_dataset, (str, os.PathLike)):
                self.train_dataset, self.train_dataset_kwargs = self.dataset_builder(
                    dataset=self.train_dataset,
                    name="train_dataset",
                    question_tokenizer=self.question_tokenizer,
                    passage_tokenizer=self.passage_tokenizer,
                    shuffle=True,  # force shuffle True for training
                    shuffle_seed=self.seed,
                    dataset_kwargs=self.train_dataset_kwargs,
                )
            elif isinstance(self.train_dataset, DictConfig):
                if (
                    "question_tokenizer" not in self.train_dataset
                    and self.question_tokenizer is None
                ):
                    raise ValueError(
                        "The question tokenizer is required for the dataset."
                    )
                if (
                    "passage_tokenizer" not in self.train_dataset
                    and self.passage_tokenizer is None
                ):
                    raise ValueError(
                        "The passage tokenizer is required for the dataset."
                    )
                self.train_dataset = hydra.utils.instantiate(
                    self.train_dataset,
                    question_tokenizer=self.question_tokenizer,
                    passage_tokenizer=self.passage_tokenizer,
                )
            else:
                self.train_dataset = self.train_dataset

            _val_dataset = []
            # keep track also of the kwargs
            _val_dataset_kwargs = []
            for i, dataset in enumerate(self.val_datasets):
                ds_kwargs = None
                if isinstance(dataset, (str, os.PathLike)):
                    val_dataset, ds_kwargs = self.dataset_builder(
                        dataset=dataset,
                        name=f"val_dataset_{i}",
                        question_tokenizer=self.question_tokenizer,
                        passage_tokenizer=self.passage_tokenizer,
                        shuffle=False,
                        shuffle_seed=self.seed,
                        dataset_kwargs=self.val_datasets_kwargs[i],
                    )
                elif isinstance(dataset, DictConfig):
                    if (
                        "question_tokenizer" not in dataset
                        and self.question_tokenizer is None
                    ):
                        raise ValueError(
                            "The question tokenizer is required for the dataset."
                        )
                    if (
                        "passage_tokenizer" not in dataset
                        and self.passage_tokenizer is None
                    ):
                        raise ValueError(
                            "The passage tokenizer is required for the dataset."
                        )
                    val_dataset = hydra.utils.instantiate(
                        dataset,
                        question_tokenizer=self.question_tokenizer,
                        passage_tokenizer=self.passage_tokenizer,
                    )
                else:
                    val_dataset = dataset

                _val_dataset.append(val_dataset)
                # keep track of the kwargs
                if ds_kwargs is not None:
                    _val_dataset_kwargs.append(ds_kwargs)

            # update val_dataset with the new datasets
            self.val_datasets = _val_dataset
            # update val_dataset_kwargs with the new kwargs
            self.val_datasets_kwargs = _val_dataset_kwargs

        if stage == "test":
            if self.test_datasets is None:
                self.test_datasets = [
                    hydra.utils.instantiate(dataset_cfg)
                    for dataset_cfg in self.datasets.test
                ]

            _test_dataset = []
            # keep track also of the kwargs
            _test_dataset_kwargs = []
            for i, dataset in enumerate(self.test_datasets):
                if isinstance(dataset, (str, os.PathLike)):
                    ds_kwargs = None
                    test_dataset, ds_kwargs = self.dataset_builder(
                        dataset=dataset,
                        name=f"test_dataset_{i}",
                        question_tokenizer=self.question_tokenizer,
                        passage_tokenizer=self.passage_tokenizer,
                        shuffle=False,
                        shuffle_seed=self.seed,
                        dataset_kwargs=self.test_datasets_kwargs[i],
                    )
                elif isinstance(dataset, DictConfig):
                    test_dataset = hydra.utils.instantiate(
                        dataset,
                        question_tokenizer=self.question_tokenizer,
                        passage_tokenizer=self.passage_tokenizer,
                    )
                else:
                    test_dataset = dataset

                _test_dataset.append(test_dataset)
                # keep track of the kwargs
                if ds_kwargs is not None:
                    _test_dataset_kwargs.append(ds_kwargs)

            # update val_dataset with the new datasets
            self.test_datasets = _test_dataset
            # update val_dataset_kwargs with the new kwargs
            self.test_datasets_kwargs = _test_dataset_kwargs

    def train_dataloader(self, train_dataset=None, *args, **kwargs) -> DataLoader:
        train_dataset = train_dataset or self.train_dataset
        logger.debug(f"Building train dataloader for {self.train_dataset.name}")
        train_dataloader_obj = GoldenStreamingDataLoader(
            train_dataset,
            collate_fn=GoldenRetrieverCollator(
                pad_token_type_id=train_dataset.question_tokenizer.pad_token_type_id,
            ),
            batch_size=train_dataset.batch_size,
            num_workers=self.num_workers.train,
            pin_memory=True,
            prefetch_factor=(
                max(1, 8 * train_dataset.batch_size // self.num_workers.train)
                if self.num_workers.train > 0
                else None
            ),
            # persistent_workers=True if self.num_workers.train > 0 else False,
            timeout=0,
        )
        self.train_dataloader_obj = train_dataloader_obj
        return train_dataloader_obj

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        dataloaders = []
        for i, dataset in enumerate(self.val_datasets):
            dataloaders.append(
                GoldenStreamingDataLoader(
                    dataset,
                    collate_fn=GoldenRetrieverCollator(
                        pad_token_type_id=dataset.question_tokenizer.pad_token_type_id,
                    ),
                    batch_size=dataset.batch_size,
                    num_workers=self.num_workers.val,
                    pin_memory=True,
                    prefetch_factor=(
                        max(1, 8 * dataset.batch_size // self.num_workers.val)
                        if self.num_workers.val > 0
                        else None
                    ),
                    # persistent_workers=True if self.num_workers.val > 0 else False,
                    timeout=0,
                )
            )
        self.val_dataloader_obj = dataloaders
        return dataloaders

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        dataloaders = []
        # test datasets are optional
        test_datasets = self.test_datasets or []
        for i, dataset in enumerate(test_datasets):
            dataloaders.append(
                GoldenStreamingDataLoader(
                    dataset,
                    collate_fn=GoldenRetrieverCollator(
                        pad_token_type_id=dataset.question_tokenizer.pad_token_type_id,
                    ),
                    batch_size=dataset.batch_size,
                    num_workers=self.num_workers.val,
                    pin_memory=True,
                    prefetch_factor=(
                        max(
                            1,
                            8 * dataset.batch_size // self.num_workers.test,
                        )
                        if self.num_workers.test > 0
                        else None
                    ),
                    persistent_workers=True if self.num_workers.test > 0 else False,
                    timeout=0,
                )
            )
        self.test_dataloader_obj = dataloaders
        return dataloaders

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        state_dict = {
            "train_dataloader": self.train_dataloader_obj.state_dict(),
            "val_dataloader": [
                dataloader.state_dict() for dataloader in self.val_dataloader_obj
            ],
        }
        if self.test_dataloader_obj:
            state_dict["test_dataloaders"] = [
                dataloader.state_dict() for dataloader in self.test_dataloader_obj
            ]
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule state_dict.

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        print("Loading state dict")
        print(state_dict.keys())
        exit

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, "
        )

    @staticmethod
    def dataset_builder(
        dataset: str | GoldenRetrieverStreamingDataset | None = None,
        name: str | None = None,
        batch_size: int | None = None,
        question_tokenizer: tr.PreTrainedTokenizerBase | None = None,
        passage_tokenizer: tr.PreTrainedTokenizerBase | None = None,
        shuffle: bool | None = None,
        shuffle_seed: int | None = None,
        dataset_kwargs: dict | None = None,
    ):
        dataset = dataset or dataset_kwargs.get("local", None)
        if dataset is None:
            raise ValueError("The dataset is required.")
        if isinstance(dataset, str):
            # check if all the necessary parameters are provided
            if name is None and "name" not in dataset_kwargs:
                raise ValueError("The dataset name is required.")
            if batch_size is None and "batch_size" not in dataset_kwargs:
                raise ValueError("The batch size is required.")
            if (
                question_tokenizer is None
                and "question_tokenizer" not in dataset_kwargs
            ):
                raise ValueError("The question_tokenizer is required.")
            if passage_tokenizer is None and "passage_tokenizer" not in dataset_kwargs:
                raise ValueError("The passage_tokenizer is required.")
            if shuffle is None and "shuffle" not in dataset_kwargs:
                raise ValueError("The shuffle parameter is required.")
            if shuffle_seed is None and "shuffle_seed" not in dataset_kwargs:
                raise ValueError("The shuffle_seed parameter is required.")

            if name:
                dataset_kwargs["name"] = name
            if "local" not in dataset_kwargs:
                dataset_kwargs["local"] = dataset
            if question_tokenizer:
                dataset_kwargs["question_tokenizer"] = question_tokenizer
            if passage_tokenizer:
                dataset_kwargs["passage_tokenizer"] = passage_tokenizer
            if batch_size:
                dataset_kwargs["batch_size"] = batch_size
            if shuffle:
                dataset_kwargs["shuffle"] = shuffle
            if shuffle_seed:
                dataset_kwargs["shuffle_seed"] = shuffle_seed
            dataset = GoldenRetrieverStreamingDataset(**dataset_kwargs)

        return dataset, dataset_kwargs
