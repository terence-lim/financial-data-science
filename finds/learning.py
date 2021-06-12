"""Text mining and machine learning helpers

- relativized word embeddings, word2idx
- stratified train-test split, minibatches

Author: Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandas.api import types
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import json
import os
import random

def form_batches(batch_size, idx):
    """Shuffles idx list into minibatches of size batch_size"""
    idxs = [i for i in idx]
    random.shuffle(idxs)
    return [idxs[i:(i+batch_size)] for i in range(0,len(idxs),batch_size)]

def form_splits(labels, test_size=0.2, random_state=42):
    """Randomly stratifies labels into train-test split indexes"""
    if type.is_list_like(labels):
        return train_test_split(np.arange(len(labels)), stratify=labels,
                                random_state=random_state, test_size=test_size)
    else:
        return train_test_split(np.arange(labels), stratify=False,
                                random_state=random_state, test_size=test_size)

class TextualData:
    """Class for pre-processing textual data

    Attributes
    ----------
    train_idx, test_idx : list of int
        computed row indexes from stratified train-test sample split
    word2idx : dict of {word str : index int}
        dict to lookup index of a word str ('UNK' has index=0)
    """
    def __init__(self, regex=r"\b[^\d\W][^\d\W][^\d\W]+\b"):
        """Initialize class for pre-processing textual data"""
        self.tokenizer = RegexpTokenizer(regex)

    def __call__(self, words, pos=0):
        """Reads (list of) words to form vocabulary"""
        self.word2idx = {(w if pos is None or isinstance(w, str) else w[pos]):
                          i+1 for i, w in enumerate(words)}
        self.word2idx['UNK'] = 0
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        self.n = len(self.word2idx)

    def __getitem__(self, word):
        """Return int index of (list of) word; 0 if not found"""
        return ([self[w] for w in word] if types.is_list_like(word)
                else self.word2idx.get(word, 0))

    def __contains__(self, word):
        """Returns True (False) if word is in (not in) vocab"""
        return word in self.word2idx

    def __len__(self):
        """Returns length of vocab"""
        return len(self.word2idx)

    def __iter__(self):
        """Returns vocab as dict of str word: int index"""
        return self.word2idx

    def dump(self, filename, outdir='./'):
        """Save vocab to local file"""
        with open(os.path.join(outdir, filename), 'wt') as f:
            f.write(json.dumps(self.word2idx))

    def load(self, filename, outdir='./'):
        """Load vocab from local file"""
        with open(os.path.join(outdir, filename), 'rt') as f:
            self.word2idx = json.loads(f.read())
        self.word2idx['UNK'] = 0
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        self.n = len(self.word2idx)

    def relativize(self, glove):
        """Reduces word embeddings to rows in vocab"""
        if isinstance(glove, str):
            glove = pd.read_csv(glove, sep=" ", quoting=3, header=None,
                                index_col=0)
        weights = np.random.normal(scale=0.6, size=(self.n, glove.shape[1]))
        for k in self.word2idx:
            if k in glove.index:
                weights[self[k]] = glove.loc[k].values
        return weights

    def form_input(self, docs, word2idx=False, tensor=True):
        """Returns lists of word indexes, with padding

        Parameters
        ----------
        docs : list of lists of str or int
            Input documents
        word2idx : bool (default is False)
            If False, inputs are int indexes; else convert str to indexes
        tensor : bool (default is True)
            Whether to return result as torch LongTensor, or list

        Notes
        -----
        Short lists padded with words randomly chosen with replacement from list
        """
        if word2idx:
            docs = [[self[w] for w in doc if w in self] for doc in docs]
        else:
            docs = [[w for w in doc if w] for doc in docs]
        lengths = [len(doc) for doc in docs]
        max_length = max(lengths)
        if max_length:
            out = [doc + random.choices(doc, k=max_length-n) if n else
                   [0] * max_length for doc, n in zip(docs, lengths)]
        else:
            out = [[0]] * len(lengths)
        return torch.LongTensor(out) if tensor else out

    def tokenize(self, docs):
        """Returns list of (list of) tokens from input (list of) sentences"""
        return (self.tokenizer.tokenize(docs.lower()) if isinstance(docs, str)
                else [self.tokenize(doc) for doc in docs])

    def counter(self, lines):
        """Returns Counter from words in (list of) lines"""
        vocab = Counter()
        if isinstance(lines, str):
            vocab.update(lines)
        else:
            for line in lines:
                vocab.update(line)
        return vocab   # most_common(20000)

    def form_splits(self, labels, test_size=0.2, random_state=42):
        """Creates stratified training and test indexes given list of labels"""
        self.train_idx, self.test_idx = form_splits(
            labels, random_state=random_state, test_size=test_size)
        return self.train_idx, self.test_idx

    def form_batches(self, batch_size):
        """Shuffles training indexes into minibatches"""
        return form_batches(batch_size, self.train_idx)
