import abc
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Optional, Iterable

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from sentify.data.collator import CustomDataCollatorWithPadding
from sentify.datasets.dataset import AugmentationType, SentimentDataset, SampleType


class BaseDataModule(LightningDataModule, abc.ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        dataset_path: Optional[Path] = None,
        augmentations: Optional[AugmentationType] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
    ):
        super().__init__()
        self.augmentations = augmentations
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._seed = seed
        self._max_length = max_length
        self._tokenizer = tokenizer
        self._train_samples: Optional[dict, Any] = None
        self._val_samples = None
        self._test_samples = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(self._train_samples, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._val_samples, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._val_samples, shuffle=False)

    @property
    def samples(self) -> Iterable[tuple[SampleType, str]]:
        yield from chain(
            zip(self._train_samples, repeat('train')),
            zip(self._val_samples, repeat('val')),
            zip(self._test_samples, repeat('test')),
        )

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        pass

    def _create_dataloader(
        self,
        samples: list[dict[str, Any]],
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset=SentimentDataset(
                samples=samples,
                tokenizer=self._tokenizer,
                max_length=self._max_length,
                augmentations=self.augmentations,
            ),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            collate_fn=CustomDataCollatorWithPadding(self._tokenizer),
            shuffle=shuffle,
        )
