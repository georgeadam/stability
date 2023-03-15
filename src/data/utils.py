import copy

import numpy as np

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


def add_label_noise(labels, noise):
    if noise == 0:
        return labels

    labels = copy.deepcopy(labels)

    num_classes = max(labels) + 1
    num_noisy_samples = int(len(labels) * noise)
    noisy_indices = np.random.choice(np.arange(len(labels)), num_noisy_samples, replace=False)

    for noisy_index in noisy_indices:
        labels[noisy_index] = np.random.choice(np.setdiff1d(np.arange(num_classes), labels[noisy_index]))

    return labels


def yield_tokens(data_iter):
    tokenizer = get_tokenizer("basic_english")
    for _, text in data_iter:
        yield tokenizer(text)

def get_vocab(train_datapipe):
    vocab = build_vocab_from_iterator(yield_tokens(train_datapipe),
                                    specials=["<UNK>", "<PAD>"],
                                    max_tokens=20000)
    vocab.set_default_index(vocab['<UNK>'])
    return vocab