import abc
import math
from operator import itemgetter
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from nltk import StanfordPOSTagger, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from toolz import groupby
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from sentify import EMBEDDINGS_PATH, SENTILARE_DIR
from sentify.datasets.base import BaseDataModule
from sentify.datasets.dataset import SampleType


class SentiLAREInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, pos_ids, senti_ids, polarity_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        self.senti_ids = senti_ids
        self.polarity_ids = polarity_ids
        self.label_id = label_id


class SentiLAREUserTextEmbeddings:
    def __init__(
        self,
        datamodule: BaseDataModule,
        tokenizer,
        model,
        batch_size: int,
        mean_center_embeddings: bool = False,
        save_path: Optional[Path] = None,
        sentinet_path: Path = SENTILARE_DIR.joinpath('SentiWordNet_3.0.0.txt'),
        gloss_path: Path = SENTILARE_DIR.joinpath('gloss_embedding.npy'),
        max_seq_length: int = 512,
        device: str = 'cpu',
    ):
        # Refer to https://nlp.stanford.edu/software/tagger.shtml to download the tagging model
        self._eng_tagger = StanfordPOSTagger(
            model_filename=str(SENTILARE_DIR.joinpath('corenlp/models/english-left3words-distsim.tagger')),
            path_to_jar=str(SENTILARE_DIR.joinpath('corenlp/stanford-postagger-3.9.2.jar')),
        )
        # verb(v), adjective(a), adverb(r), noun(n), others(u)
        self._pos_tag_ids_map = {'v': 0, 'a': 1, 'r': 2, 'n': 3, 'u': 4}
        self._lemmatizer = WordNetLemmatizer()

        self._sbert_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)
        self._save_path = save_path or EMBEDDINGS_PATH.joinpath(f'{datamodule.name}_sentiLARE.pt')
        self._sentinet, self._gloss_embedding, self._gloss_emb_norm = self._load_sentinet(sentinet_path, gloss_path)

        self._tokenizer = tokenizer
        self._model = model
        self._batch_size = batch_size
        self._device = device
        self._max_seq_length = max_seq_length

        self._samples = {
            sample['index']: {**sample, 'split': split} for sample, split in datamodule.samples
        }
        self._embeddings = self._create_embeddings()

        if mean_center_embeddings:
            embs = torch.stack(tuple(self._embeddings.values()))
            norm_embs = embs - embs.mean(dim=0)
            self._embeddings = {k: e for k, e in zip(self._embeddings.keys(), norm_embs)}

        self._username_to_ids = {
            user: [sample['index'] for sample in samples]
            for user, samples in groupby(itemgetter('username'), self._samples.values()).items()
        }

    def __call__(self, item: SampleType) -> SampleType:
        item_index = item['index']
        username = item['username']

        embeddings, labels = [self._embeddings[item_index]], []
        for index in self._username_to_ids[username]:
            # ignore itself and non-training samples
            if index == item_index or self._samples[index]['split'] != 'train':
                continue

            embeddings.append(self._embeddings[index])
            labels.append(self._samples[index]['label'])

        item['user_texts_embeddings'] = torch.stack(embeddings)
        item['user_texts_labels'] = torch.tensor(labels)
        return item

    def _create_embeddings(
        self,
    ):
        if self._save_path.exists():
            return torch.load(self._save_path, map_location='cpu')

        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        texts = [sample['text'] for sample in self._samples.values()]
        labels = [sample['label'] for sample in self._samples.values()]

        tokenized_texts, pos_list, sentiments, labels = self.process_texts(texts, labels)
        examples = list(zip(tokenized_texts, pos_list, sentiments, labels))
        tensor_dataset = self.convert_examples_to_features_roberta(
            tokenizer=self._tokenizer,
            examples=examples,
            max_seq_length=self._max_seq_length,
            sep_token=self._tokenizer.sep_token,
            cls_token=self._tokenizer.cls_token,
            pad_token=self._tokenizer.convert_tokens_to_ids([self._tokenizer.pad_token])[0],
        )
        dataloader = DataLoader(tensor_dataset, batch_size=self._batch_size)

        # code adapted from
        # https://github.com/thu-coai/SentiLARE/blob/5f1243788fb872e56b5e259939b932346b378419/finetune/run_sent_sentilr_roberta.py#L21
        output_vectors = []
        for batch in tqdm(dataloader, desc="Calculating embeddings by SentiLARE Roberta model"):
            self._model.eval()
            batch = tuple(t.to(self._device) for t in batch)
            with torch.no_grad():
                attention_mask = batch[1]
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': attention_mask,
                    'pos_ids': batch[2],
                    'senti_word_ids': batch[3],
                    'polarity_ids': batch[4],
                }
                outputs = self._model(**inputs)
                output_tokens = outputs[0]

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(output_tokens.size()).float()
                sum_embeddings = torch.sum(output_tokens * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                embedding = sum_embeddings / sum_mask
                output_vectors.append(embedding.detach().cpu())

        output_vectors = torch.cat(output_vectors, dim=0)
        print(output_vectors.size())
        ids_embeddings = {idx: emb for idx, emb in zip(self._samples, output_vectors)}

        self._save_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(ids_embeddings, self._save_path)
        return ids_embeddings

    @staticmethod
    def convert_examples_to_features_roberta(
        tokenizer,
        examples: list,
        max_seq_length: int = 512,
        cls_token='[CLS]',
        sep_token='[SEP]',
        sep_token_extra: bool = True,  # True for Roberta
        pad_token=0,
        mask_padding_with_zero=True,
    ):
        """
        Code adapted from
        https://github.com/thu-coai/SentiLARE/blob/5f1243788fb872e56b5e259939b932346b378419/finetune/sent_data_utils_sentilr.py

         Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        """

        features = []
        for example in examples:
            tokenized_text, pos, sentiment, label = example

            tokens, poses, sentiments = [], [], []
            for i, tok in enumerate(tokenized_text):
                tok_list = tokenizer.tokenize(tok)
                tokens.extend(tok_list)
                poses.extend([pos[i]] * len(tok_list))
                sentiments.extend([sentiment[i]] * len(tok_list))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2

            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
                poses = poses[:(max_seq_length - special_tokens_count)]
                sentiments = sentiments[:(max_seq_length - special_tokens_count)]

            # The convention in BERT is for single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            # 4 in POS tags means others, 2 in word-level polarity labels means neutral, and
            # 5 in sentence-level sentiment labels means unknown sentiment
            tokens = tokens + [sep_token]
            pos_ids = poses + [4]
            senti_ids = sentiments + [2]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                pos_ids += [4]
                senti_ids += [2]

            tokens = [cls_token] + tokens
            pos_ids = [4] + pos_ids
            senti_ids = [2] + senti_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            pos_ids = pos_ids + ([4] * padding_length)
            senti_ids = senti_ids + ([2] * padding_length)

            # During fine-tuning, the sentence-level label is set to unknown
            polarity_ids = [5] * max_seq_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(pos_ids) == max_seq_length
            assert len(senti_ids) == max_seq_length
            assert len(polarity_ids) == max_seq_length

            features.append(
                SentiLAREInputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=[0] * max_seq_length,
                    pos_ids=pos_ids,
                    senti_ids=senti_ids,
                    polarity_ids=polarity_ids,
                    label_id=label,
                )
            )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
        all_senti_ids = torch.tensor([f.senti_ids for f in features], dtype=torch.long)
        all_polarity_ids = torch.tensor([f.polarity_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_pos_ids,
            all_senti_ids,
            all_polarity_ids,
            all_label_ids,
        )
        return dataset

    def process_texts(self, text_list: list[str], label_list: list[int]):
        """
        Code adapted from:
        https://github.com/thu-coai/SentiLARE/blob/5f1243788fb872e56b5e259939b932346b378419/preprocess/prep_sent.py
        """
        sent_list = []
        sent_list_str = []
        data_cnt = 0

        # Tokenization with NLTK
        for text in text_list:
            try:
                token_list = word_tokenize(text.strip())
            except:
                token_list = text.strip().split()
            if len(token_list) == 0:
                continue
            sent_list.append(token_list)
            sent_list_str.append(text.strip())
            data_cnt += 1

        print('original number of data = ', data_cnt)

        # pos tagging with Stanford Corenlp
        sent_split = self._eng_tagger.tag_sents(sent_list)

        # sentence embedding
        corpus_embedding = self._sbert_model.encode(
            sent_list_str,
            # convert_to_tensor=True,
            show_progress_bar=True,
        )
        corpus_embedding = np.array(corpus_embedding)
        corpus_emb_norm = [np.linalg.norm(corpus_embedding[id]) for id in range(len(corpus_embedding))]
        corpus_emb_norm = np.array(corpus_emb_norm)
        assert len(corpus_embedding) == len(sent_split)

        # get pos tags and sentiment polarities for each word
        clean_sent_list, pos_list, senti_list, clean_label_list = [], [], [], []
        for sent_id in tqdm(range(len(sent_split))):
            sent_list_ele, pos_list_ele, senti_list_ele = [], [], []
            for pair in sent_split[sent_id]:
                if len(pair[0]) != 0:
                    word, pos = pair[0], self.convert_postag(pair[1])
                    sent_list_ele.append(word)
                    pos_list_ele.append(pos)
                    if pos != 'u':
                        word = self._lemmatizer.lemmatize(word.lower(), pos=pos)

                    # gloss-aware sentiment attention
                    if word in self._sentinet and pos in self._sentinet[word]:
                        sim_list = []
                        score_list = []
                        for ele_term in self._sentinet[word][pos]:
                            gloss_line = ele_term[4]
                            gloss_emb, gloss_norm = self._gloss_embedding[gloss_line], self._gloss_emb_norm[gloss_line]
                            sent_emb, sent_norm = corpus_embedding[sent_id], corpus_emb_norm[sent_id]
                            sim_score = self.cos_sim(gloss_emb, sent_emb, gloss_norm, sent_norm)
                            sim_list.append((sim_score + 1) / (2 * ele_term[0]))
                            score_list.append(ele_term[1] - ele_term[2])

                        sim_exp = [math.exp(sim_list[id]) for id in range(len(sim_list))]
                        sum_sim_exp = sum(sim_exp)
                        sim_exp = np.array([sim_exp[id] / sum_sim_exp for id in range(len(sim_exp))])
                        score_list = np.array(score_list)
                        final_score = np.dot(sim_exp, score_list)
                        senti_list_ele.append(final_score)
                    else:
                        senti_list_ele.append(0.0)

            assert len(sent_list_ele) == len(pos_list_ele)
            assert len(sent_list_ele) == len(senti_list_ele)

            if len(sent_list) != 0:
                clean_sent_list.append(sent_list_ele)
                # transform pos_tag (str) to integer
                pos_list.append([self._pos_tag_ids_map[ele] for ele in pos_list_ele])
                # transform sentiment score (float) to integer
                senti_list.append([1 if ele > 0 else 0 if ele < 0 else 2 for ele in senti_list_ele])
                clean_label_list.append(label_list[sent_id])

        assert len(clean_sent_list) == len(clean_label_list)
        assert len(clean_sent_list) == len(pos_list)
        assert len(clean_sent_list) == len(senti_list)

        print('number after processing = ', len(clean_label_list))

        return clean_sent_list, pos_list, senti_list, clean_label_list

    @staticmethod
    def _load_sentinet(sentiwordnet_path: Path, gloss_path: Path):
        f = open(sentiwordnet_path, 'r')
        line_id = 0
        sentinet = {}

        for line in f.readlines():
            if line_id < 26:
                line_id += 1
                continue
            if line_id == 117685:
                break
            line_split = line.strip().split('\t')
            pos, pscore, nscore, term, gloss = (
                line_split[0], float(line_split[2]), float(line_split[3]), line_split[4], line_split[5]
            )

            if "\"" in gloss:
                shop_pos = gloss.index('\"')
                gloss = gloss[: shop_pos - 2]
            each_term = term.split(' ')
            for ele in each_term:
                ele_split = ele.split('#')
                assert len(ele_split) == 2
                word, sn = ele_split[0], int(ele_split[1])
                if word not in sentinet:
                    sentinet[word] = {}
                if pos not in sentinet[word]:
                    sentinet[word][pos] = []
                sentinet[word][pos].append([sn, pscore, nscore, gloss, line_id - 26])
            line_id += 1
        f.close()

        # load gloss embedding (which is calculated by sentence-transformers in advance)
        gloss_embedding = np.load(str(gloss_path))
        gloss_emb_norm = [np.linalg.norm(gloss_embedding[id]) for id in range(len(gloss_embedding))]
        gloss_emb_norm = np.array(gloss_emb_norm)

        return sentinet, gloss_embedding, gloss_emb_norm

    @staticmethod
    def convert_postag(pos):
        """Convert NLTK POS tags to SentiWordNet's POS tags."""
        if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return 'v'
        elif pos in ['JJ', 'JJR', 'JJS']:
            return 'a'
        elif pos in ['RB', 'RBR', 'RBS']:
            return 'r'
        elif pos in ['NNS', 'NN', 'NNP', 'NNPS']:
            return 'n'
        else:
            return 'u'

    @staticmethod
    def cos_sim(a, b, norm_a, norm_b):
        dot_prod = np.dot(a, b)
        return dot_prod / (norm_a * norm_b)


class BaseComputeSimilarity:
    def __init__(
        self,
        encoder: Union[SentenceTransformer, CrossEncoder],
        datamodule: BaseDataModule,
        name: str,
    ):
        self._encoder = encoder
        name = name.replace('/', '_')
        save_path = EMBEDDINGS_PATH.joinpath(f'{datamodule.name}_{name}.pt')
        self._samples = {
            sample['index']: {**sample, 'split': split} for i, (sample, split) in enumerate(datamodule.samples)
        }

        if save_path.exists():
            self._username_to_similarity = torch.load(save_path, map_location='cpu')
        else:
            self._username_to_similarity = self._create_user_to_similarities_mapping(self._samples)
            torch.save(self._username_to_similarity, save_path)

    @abc.abstractmethod
    def _create_user_to_similarities_mapping(self, samples):
        pass

    def __call__(self, item: SampleType) -> SampleType:
        item_index = item['index']
        username = item['username']

        similarity_matrix = self._username_to_similarity[username]
        similarities, labels = [], []
        for index, similarity in similarity_matrix[item_index].iteritems():
            # ignore itself and non-training samples
            if index == item_index or self._samples[index]['split'] != 'train':
                continue
            similarities.append(similarity)
            labels.append(self._samples[index]['label'])

        item['user_texts_similarities'] = torch.tensor(similarities)
        item['user_texts_labels'] = torch.tensor(labels)
        return item


class BIEncoderComputeSimilarity(BaseComputeSimilarity):
    def _create_user_to_similarities_mapping(self, samples):
        embeddings = self._create_embeddings()
        return {
            user: self._compute_similarities(samples, embeddings)
            for user, samples in groupby(itemgetter('username'), self._samples.values()).items()
        }

    def _create_embeddings(self):
        texts = [sample['text'] for sample in self._samples.values()]
        embeddings = self._encoder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        ids_embeddings = {idx: emb for idx, emb in zip(self._samples, embeddings)}
        return ids_embeddings

    def _compute_similarities(
        self,
        samples: list[SampleType],
        embeddings: dict[str, Tensor]
    ) -> pd.DataFrame:
        ids = [s['index'] for s in samples]
        text_embeddings = torch.stack([embeddings[i] for i in ids])
        similarities = self._compute_cosine_similarity_matrix(text_embeddings).detach().cpu().numpy()
        return pd.DataFrame(similarities, index=ids, columns=ids)

    @staticmethod
    def _compute_cosine_similarity_matrix(embeddings, eps=1e-8):
        norm = embeddings.norm(dim=1)[:, None]
        embeddings = embeddings / torch.clamp(norm, min=eps)
        similarities = torch.mm(embeddings, embeddings.transpose(0, 1))
        return (similarities + 1.0) / 2.0


class CrossEncoderComputeSimilarity(BaseComputeSimilarity):
    def _create_user_to_similarities_mapping(self, samples):
        return {
            user: self._compute_similarities(samples)
            for user, samples in tqdm(groupby(itemgetter('username'), self._samples.values()).items())
        }

    def _compute_similarities(
        self,
        samples: list[SampleType],
    ) -> pd.DataFrame:
        ids = [s['index'] for s in samples]
        range_ids = np.arange(len(ids))
        indices_pairs = []
        text_pairs = []
        for i in range_ids:
            for j in range_ids:
                if j >= i or not ((samples[i]['split'] == 'train') or (samples[j]['split'] == 'train')):
                    continue
                text_pairs.append([samples[i]['text'], samples[j]['text']])
                indices_pairs.append((i, j))

        similarities = self._encoder.predict(text_pairs)
        matrix = np.zeros((len(ids), len(ids)), dtype=np.float32)
        for (i, j), sim in zip(indices_pairs, similarities):
            matrix[i, j] = sim
            matrix[j, i] = sim
        matrix = matrix + np.eye(len(ids))
        return pd.DataFrame(matrix, index=ids, columns=ids)
