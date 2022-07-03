import abc
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import PreTrainedTokenizerFast

from sentify import DATASETS_PATH
from sentify.datasets.base import BaseDataModule
from sentify.datasets.dataset import AugmentationType, SampleType
from sentify.utils.seed import set_seeds

WIKI_DETOX = [
    "attack",
    "aggression",
    "toxicity",
]


class WikiDetoxDataModule(BaseDataModule, abc.ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task: str,
        dataset_path: Path = DATASETS_PATH.joinpath('wiki_detox'),
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
        self._train_samples = self._get_split(dataset, split_name='train')
        self._val_samples = self._get_split(dataset, split_name='dev')
        self._test_samples = self._get_split(dataset, split_name='test')

    def _load_dataset(self) -> pd.DataFrame:
        if self._task not in WIKI_DETOX:
            raise ValueError("Incorrect WikiDetox task name")

        # comments = pd.read_csv(
        #     self._dataset_path.joinpath(f'{self._task}_annotated_comments.tsv'),
        #     sep='\t',
        #     index_col=0,
        # )
        # annotations = pd.read_csv(
        #     self._dataset_path.joinpath(f'{self._task}_annotations.tsv'),
        #     sep='\t',
        # )
        # comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
        # comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
        #
        # df = annotations.join(comments, on='rev_id')
        #
        # column_mapping = {
        #     'worker_id': 'username',
        #     'comment': 'text',
        #     self._task: 'label',
        # }
        # select = list(column_mapping.keys())
        # select.append('split')
        # df = df[select]
        # df = df.rename(columns=column_mapping)

        df = pd.read_csv(self._dataset_path.joinpath(f'{self._task}-subset.tsv'), sep='\t')
        df = df.astype({'label': 'int'})

        return df

    @staticmethod
    def _get_split(df: pd.DataFrame, split_name: str) -> list[SampleType]:
        split_df = df.query(f"split=='{split_name}'")
        split_df = split_df.drop(columns=['split'])
        split_df['index'] = [f'{split_name}_{i}' for i in range(len(split_df))]
        return split_df.to_dict(orient='records')


class AggressionWikiDataModule(WikiDetoxDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task: str = 'aggression',
        dataset_path: Path = DATASETS_PATH.joinpath('wiki_detox'),
        augmentations: Optional[AugmentationType] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
    ):
        super().__init__(
            task=task,
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
        return "wiki_aggression"

    @property
    def num_classes(self) -> int:
        return 2


class ToxicityWikiDataModule(WikiDetoxDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task: str = 'toxicity',
        dataset_path: Path = DATASETS_PATH.joinpath('wiki_detox'),
        augmentations: Optional[AugmentationType] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
    ):
        super().__init__(
            task=task,
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
        return "wiki_toxicity"

    @property
    def num_classes(self) -> int:
        return 2


class AttackWikiDataModule(WikiDetoxDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task: str = 'attack',
        dataset_path: Path = DATASETS_PATH.joinpath('wiki_detox'),
        augmentations: Optional[AugmentationType] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_length: int = int(1e10),
    ):
        super().__init__(
            task=task,
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
        return "wiki_attack"

    @property
    def num_classes(self) -> int:
        return 2
