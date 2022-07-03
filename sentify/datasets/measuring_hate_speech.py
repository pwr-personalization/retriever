import abc
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from toolz import groupby
from toolz.sandbox import unzip
from transformers import PreTrainedTokenizerFast

from sentify.datasets.base import AugmentationType, BaseDataModule
from sentify.datasets.dataset import SampleType
from sentify.utils.seed import set_seeds

MHS = [
    "sentiment",
    "hatespeech",
]


class MeasuringHateSpeechDataModule(BaseDataModule, abc.ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task: str,
        dataset_path: Optional[Path] = None,
        augmentations: Optional[AugmentationType] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_size: float = 0.6,
        min_texts_per_user: int = 20,
    ):
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
        )
        self._train_size = train_size
        self._min_texts_per_user = min_texts_per_user
        self._task = task

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        set_seeds(self._seed)

        dataset = self._load_dataset()
        train_samples, val_samples, test_samples = unzip(
            self._split(group)
            for group in groupby(itemgetter('username'), dataset).values()
            if len(group) >= self._min_texts_per_user
        )
        self._train_samples = self._add_index_attribute(
            list(chain.from_iterable(train_samples)),
            split='train',
        )
        self._val_samples = self._add_index_attribute(
            list(chain.from_iterable(val_samples)),
            split='val',
        )
        self._test_samples = self._add_index_attribute(
            list(chain.from_iterable(test_samples)),
            split='test',
        )

    def _load_dataset(self) -> list[SampleType]:
        if self._task not in MHS:
            raise ValueError("Incorrect MHS task name")

        dataset = load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')['train']
        column_mapping = {
            'annotator_id': 'username',
            'text': 'text',
            self._task: 'label',
        }
        dataset = dataset.rename_columns(column_mapping)
        return (
            dataset.remove_columns(set(dataset.column_names).difference(column_mapping.values()))
                .to_pandas().astype({'label': 'int'})
                .to_dict(orient='records')
        )

    def _split(self, group):
        train, val_test = train_test_split(
            group,
            train_size=self._train_size,
            random_state=self._seed,
        )
        val, test = train_test_split(
            val_test,
            train_size=0.5,
            random_state=self._seed,
        )
        return train, val, test

    @staticmethod
    def _add_index_attribute(samples: list[dict], split: str) -> list[dict]:
        new_samples = []
        for i, item in enumerate(samples):
            item['index'] = f'{split}_{i}'
            new_samples.append(item)

        return new_samples


class MHSSentimentDataModule(MeasuringHateSpeechDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task: str = 'sentiment',
        dataset_path: Optional[Path] = None,
        augmentations: Optional[AugmentationType] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_size: float = 0.6,
        min_texts_per_user: int = 20,
    ):
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            task=task,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_size=train_size,
            min_texts_per_user=min_texts_per_user,
        )

    @property
    def name(self) -> str:
        return 'MHS_sentiment'

    @property
    def num_classes(self) -> int:
        return 5


class MHSHatespeechDataModule(MeasuringHateSpeechDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task: str = 'hatespeech',
        dataset_path: Optional[Path] = None,
        augmentations: Optional[AugmentationType] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        train_size: float = 0.6,
        min_texts_per_user: int = 20,
    ):
        super().__init__(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            task=task,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            max_length=max_length,
            train_size=train_size,
            min_texts_per_user=min_texts_per_user,
        )

    @property
    def name(self) -> str:
        return 'MHS_hatespeeech'

    @property
    def num_classes(self) -> int:
        return 3
