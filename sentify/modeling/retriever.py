from typing import Optional, Literal

import torch
from torch import nn, Tensor
from torch.nn.functional import cosine_similarity, normalize, one_hot
from transformers import PreTrainedModel


class Retriever(nn.Module):
    def __init__(
        self,
        backbone: PreTrainedModel,
        num_labels: int,
        normalize_weights: bool = False,
        feature_normalization: Literal['none', 'before', 'after', 'learnable'] = 'none',
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self._classifier = RetrieverClassificationHead(self.backbone.config, num_labels, feature_normalization)
        self._num_labels = num_labels
        self._normalize_weights = normalize_weights
        self._top_k = top_k

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        user_texts_similarities: list[Tensor],
        user_texts_labels: list[Tensor],
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]

        aggregated_labels = self._compute_aggregated_labels(
            user_texts_similarities=user_texts_similarities,
            user_texts_labels=user_texts_labels,
        )
        return self._classifier(sequence_output, aggregated_labels)

    def _compute_aggregated_labels(
        self,
        user_texts_similarities: list[Tensor],
        user_texts_labels: list[Tensor],
    ):
        return torch.stack(
            [
                self._compute_labels(similarities, labels)
                for similarities, labels in zip(user_texts_similarities, user_texts_labels)
            ]
        )

    def _compute_labels(self, similarities: Tensor, labels: Tensor):
        weights = similarities.unsqueeze(1)
        labels_one_hot = one_hot(labels, self._num_labels).type(weights.dtype)

        if self._top_k is None:
            labels_one_hot *= weights
        else:
            indices = torch.topk(
                weights.flatten(),
                k=min(self._top_k, len(weights)),
            ).indices
            labels_one_hot = labels_one_hot[indices]

        aggregated = labels_one_hot.mean(dim=0)

        # By default, aggregated labels do not sum to 1. I think it should stay
        if self._normalize_weights:
            aggregated /= weights.mean()

        return aggregated


class RetrieverClassificationHead(nn.Module):
    """Copied from RobertaClassificationHead"""

    def __init__(self, config, num_labels, feature_normalization):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size + num_labels, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
        self._feature_normalization = feature_normalization
        self._labels_proj = nn.Linear(num_labels, num_labels) if self._feature_normalization == 'learnable' else None

    def forward(self, features, aggregated_labels):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        if self._feature_normalization == 'before':
            x = normalize(x, dim=1)
            aggregated_labels = normalize(aggregated_labels, dim=1)
        elif self._feature_normalization == 'learnable':
            aggregated_labels = self._labels_proj(aggregated_labels)

        x = torch.cat((x, aggregated_labels), dim=1)

        if self._feature_normalization == 'after':
            x = normalize(x, dim=1)

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)
