from typing import Any

from transformers import DataCollatorWithPadding


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        padded = super().__call__(
            [
                {
                    'input_ids': f['input_ids'],
                    'attention_mask': f['attention_mask'],
                    'label': f['label'],
                }
                for f in features
            ]
        )
        for key in features[0]:
            if key not in padded:
                padded[key] = [f[key] for f in features]

        return dict(padded)
