from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import PreTrainedTokenizerFast

from sentify import DATASETS_PATH
from sentify.datasets.base import AugmentationType, BaseDataModule
from sentify.datasets.dataset import SampleType
from sentify.utils.seed import set_seeds


class YelpDataModule(BaseDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        dataset_path: Path = DATASETS_PATH.joinpath('yelp'),
        augmentations: Optional[AugmentationType] = None,
        split_type: str = 'b',
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
        min_user_post_num: int = 50,
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
        self._split_type = split_type
        self._min_user_post_num = min_user_post_num

    @property
    def name(self) -> str:
        return 'Yelp'

    @property
    def num_classes(self) -> int:
        return 5

    def setup(self, stage: Optional[str] = None) -> None:
        set_seeds(self._seed)

        self._train_samples = self._load_dataset(
            self._dataset_path.joinpath(self._split_type, f'train-{self._split_type}-subset.tsv'),
        )
        self._val_samples = self._load_dataset(
            self._dataset_path.joinpath(self._split_type, f'dev-{self._split_type}-subset.tsv'),
        )
        self._test_samples = self._load_dataset(
            self._dataset_path.joinpath(self._split_type, f'test-{self._split_type}-subset.tsv'),
        )

    @staticmethod
    def _load_dataset(path: Path) -> list[SampleType]:
        # df = pd.read_csv(path, sep='\t+', names=['username', 'index', 'label', 'text'], engine='python')
        df = pd.read_csv(path, sep='\t')
        # df['label'] = df['label'].apply(lambda x: x - 1)
        split = path.name.split('-')[0]
        df['index'] = [f'{split}_{i}' for i in range(len(df))]
        return df.to_dict(orient='records')
