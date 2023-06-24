import copy
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB

from .augmented import AugmentedDataset
from .creation import datasets
from .data_module import DataModule
from .text import TextDataset
from .utils import add_label_noise, get_vocab


class IMDBDataModule(DataModule):
    def __init__(self, data_dir: str, train_size: int, val_size: int, extra_size: int, batch_size: int,
                 random_state: int, noise: float):
        super().__init__(data_dir, train_size, val_size, extra_size, batch_size, random_state, noise)

        self.data_dir = data_dir
        self.prepare_data()
        self.setup(None)

    def prepare_data(self):
        # download
        if not self.train_data:
            IMDB(self.data_dir, split="train")
            IMDB(self.data_dir, split="test")

    def setup(self, stage: Optional[str] = None):
        if not self.train_data:
            full_data = IMDB(self.data_dir, split="train")
            vocab = get_vocab(full_data)
            self.vocab = vocab
            tokenizer = get_tokenizer("basic_english")

            full_data = raw_to_processed(full_data, vocab, tokenizer)

            if self.random_state is not None:
                r = np.random.RandomState(self.random_state)
                all_indices = r.choice(np.arange(len(full_data)),
                                       size=self.train_size + self.val_size + self.extra_size,
                                       replace=False)
            else:
                all_indices = np.random.choice(np.arange(len(full_data)),
                                               size=self.train_size + self.val_size + self.extra_size,
                                               replace=False)

            if self.val_size == 0:
                train_indices = all_indices
                val_indices = np.array([]).astype(int)
            else:
                train_indices, val_indices = train_test_split(all_indices, test_size=self.val_size,
                                                              random_state=self.random_state)

            train_data = copy.deepcopy(full_data)
            val_data = copy.deepcopy(full_data)

            train_data = AugmentedDataset(train_data, train_indices, 0)
            train_data.data = train_data.data[train_indices]
            train_data.targets = torch.tensor(train_data.targets)
            train_data.targets = train_data.targets[train_indices]

            val_data = AugmentedDataset(val_data, val_indices, 0)
            val_data.data = val_data.data[val_indices]
            val_data.targets = torch.tensor(val_data.targets)
            val_data.targets = val_data.targets[val_indices]

            if self.extra_size == 0:
                train_indices = np.arange(len(train_data))
                extra_indices = np.array([]).astype(int)
            else:
                train_indices, extra_indices = train_test_split(np.arange(len(train_data)), test_size=self.extra_size,
                                                                random_state=self.random_state)

            extra_data = copy.deepcopy(train_data)

            train_data.data = train_data.data[train_indices]
            train_data.targets = train_data.targets[train_indices]
            train_data.targets = add_label_noise(train_data.targets, self.noise)
            train_data.indices = train_data.indices[train_indices]

            extra_data.data = extra_data.data[extra_indices]
            extra_data.targets = extra_data.targets[extra_indices]
            extra_data.targets = add_label_noise(extra_data.targets, self.noise)
            extra_data.indices = extra_data.indices[extra_indices]
            extra_data.source = 1

            self.train_data = train_data
            self.val_data = val_data
            self.extra_data = extra_data
            self.orig_train_data = copy.deepcopy(self.train_data)

            test_data = IMDB(self.data_dir, split="test")
            test_data = raw_to_processed(test_data, vocab, tokenizer)
            self.test_data = AugmentedDataset(test_data, np.arange(len(test_data)), 0)
            predict_data = IMDB(self.data_dir, split="test")
            predict_data = raw_to_processed(predict_data, vocab, tokenizer)
            self.predict_data = AugmentedDataset(predict_data, np.arange(len(predict_data)), 0)

    def merge_train_and_extra_data(self):
        self.train_data.data = np.concatenate([self.train_data.data, self.extra_data.data])
        self.train_data.targets = torch.cat([self.train_data.targets, self.extra_data.targets])
        self.train_data.indices = np.concatenate([self.train_data.indices, self.extra_data.indices])

    @property
    def num_classes(self):
        return 2

    @property
    def max_len(self):
        return 200

    @property
    def pad_index(self):
        return self.vocab.get_stoi()["<PAD>"]

    @property
    def vocab_size(self):
        return 20000

    @property
    def stats(self):
        return {"max_len": self.max_len,
                "pad_index": self.pad_index,
                "vocab_size": self.vocab_size}

    @property
    def train_labels(self):
        return np.array(self.orig_train_data.targets)

    @property
    def test_labels(self):
        return np.array(self.test_data.targets)

    @property
    def val_labels(self):
        return np.array(self.val_data.targets)


def raw_to_processed(data, vocab, tokenizer):
    labels, sentences = [], []

    for batch in data:
        labels.append(batch[0])
        sentences.append((torch.tensor(vocab(tokenizer(batch[1])[:200]))))

    sentences = pad_sequence(sentences, padding_value=vocab.get_stoi()["<PAD>"])
    sentences = torch.tensor(sentences)
    sentences = torch.permute(sentences, (1, 0))

    labels = np.array(labels)
    labels[labels == "neg"] = 0
    labels[labels == "pos"] = 1
    labels = labels.astype("long")

    data = TextDataset(sentences, labels)

    return data


datasets.register_builder("imdb", IMDBDataModule)
