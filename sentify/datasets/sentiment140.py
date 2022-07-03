import json
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Optional

from sklearn.model_selection import train_test_split
from toolz import groupby
from toolz.sandbox import unzip
from transformers import PreTrainedTokenizerFast

from sentify import DATASETS_PATH
from sentify.datasets.base import BaseDataModule
from sentify.datasets.dataset import SampleType, AugmentationType
from sentify.utils.seed import set_seeds

_LABEL_MAPPING = {0: 0, 4: 1}


class Sentiment140DataModule(BaseDataModule):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            dataset_path: Path = DATASETS_PATH.joinpath('sentiment140'),
            augmentations: Optional[AugmentationType] = None,
            batch_size: int = 16,
            num_workers: int = 8,
            seed: int = 2718,
            max_length: int = int(1e10),
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

    @property
    def name(self) -> str:
        return 'Sentiment140'

    @property
    def num_classes(self) -> int:
        return 2

    def setup(self, stage: Optional[str] = None) -> None:
        set_seeds(self._seed)
        train_dataset_path = self._dataset_path / 'all_data_niid_4_keep_30_train_8.json'
        test_dataset_path = self._dataset_path / 'all_data_niid_4_keep_30_test_8.json'
        train_dataset = self._load_dataset(train_dataset_path)

        train_samples, val_samples = unzip(
            train_test_split(
                group,
                train_size=0.8,
                random_state=self._seed,
            )
            for username, group in groupby(itemgetter('username'), train_dataset).items()
        )
        self._train_samples = list(chain.from_iterable(train_samples))
        self._val_samples = list(chain.from_iterable(val_samples))
        self._test_samples = self._load_dataset(test_dataset_path)

    @staticmethod
    def _load_dataset(path: Path) -> Iterable[SampleType]:
        with path.open('r') as file:
            user_data = json.load(file)['user_data']

        for username, data in user_data.items():
            for row, label in zip(data['x'], data['y']):
                yield {
                    'index': row[0],
                    'username': username,
                    'text': row[-2],
                    'label': _LABEL_MAPPING[int(label)],
                }
