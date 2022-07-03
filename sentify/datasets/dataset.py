from typing import Any, Callable, Optional, TypedDict

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

AugmentationType = Callable[[dict[str, Any]], dict[str, Any]]


class SampleType(TypedDict):
    index: str
    username: str
    text: str
    label: int


class SentimentDataset(Dataset):
    def __init__(
        self,
        samples: list[SampleType],
        tokenizer: PreTrainedTokenizerFast,
        augmentations: Optional[AugmentationType] = None,
        max_length: int = int(1e10),
    ):
        self._samples = samples
        self._tokenizer = tokenizer
        self._augmentations = augmentations or (lambda x: x)
        self._max_length = max_length

    def __getitem__(self, idx: int):
        item = self._samples[idx]
        item = self._augmentations(item)

        tokenized_text = self._tokenizer(
            text=item['text'],
            truncation=True,
            max_length=self._max_length,
            return_tensors='pt',
        )
        return {
            **item,
            'input_ids': tokenized_text['input_ids'].squeeze(0),
            'attention_mask': tokenized_text['attention_mask'].squeeze(0),
        }

    def __len__(self):
        return len(self._samples)
