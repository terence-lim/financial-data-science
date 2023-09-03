"""Classes for textual datasets

Copyright 2022, Terence Lim

MIT License
"""
from typing import Dict, Iterable, List, Any, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types
import random
import os
import json
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter, namedtuple
import torch
_VERBOSE = 1

def form_batches(batch_size: int, idx: List) -> List[List[int]]:
    """Shuffles idx list into minibatches each of size batch_size

    Args:
        batch_size: Size of each minibatch
        idx: List of indexes

    Returns:
        List of batches of shuffled indexes 
    """
    idxs = [i for i in idx]
    random.shuffle(idxs)
    return [idxs[i:(i+batch_size)] for i in range(0, len(idxs), batch_size)]

def form_splits(labels: List[str | int] | int,
                test_size: float | int = 0.2,
                random_state: int = 42) -> List[List]:
    """Randomly stratifies labels into train-test split indexes

    Args:
        labels: Labels of series to shuffle, or length of series
        test_size: Desired size of test set as fraction or number of samples
        random_state: Set random seed

    Returns:
        List of stratified train indexes, List of stratified test indexes
    """
    if types.is_list_like(labels):
        return train_test_split(np.arange(len(labels)),
                                stratify=labels,
                                random_state=random_state,
                                test_size=test_size)
    else:
        return train_test_split(np.arange(labels),
                                stratify=False,
                                random_state=random_state,
                                test_size=test_size)


class Textual:
    """Class for pre-processing textual data for deep learning

    Args:
        regex: regular expression for RegexpTokenizer

    Attributes:
        train_idx, test_idx: computed row indexes from stratified split
        word2idx: dict {word str : index int} for lookup ('UNK' has index=0)
    """
    def __init__(self, regex: str =r"\b[^\d\W][^\d\W][^\d\W]+\b"):
        """Initialize class for pre-processing textual data

        Args:
            regex: Regular expression to tokenize
        """
        self.tokenizer = RegexpTokenizer(regex)

    def __call__(self, words: List, field: str = ''):
        """Reads (list of) words to form vocabulary
        
        Args:
            words: List of words
            field: Optional name of field of words with word string

        Returns:
            Length of computed word indexes
        """
        self.word2idx = {(w[field] if field else w): i + 1
                         for i, w in enumerate(words)}
        self.word2idx['UNK'] = 0
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        self.n = len(self.word2idx)

    def __getitem__(self, word: str | List[str]) -> int | List[int]:
        """Return int index of (list of) word; 0 if not found"""
        return ([self[w] for w in word] if types.is_list_like(word)
                else self.word2idx.get(word, 0))

    def __contains__(self, word: str) -> bool:
        """Returns True (False) if word is in (not in) vocab"""
        return word in self.word2idx

    def __len__(self) -> int:
        """Returns length of vocab"""
        return len(self.word2idx)

#    def __iter__(self):
#        """Returns vocab as dict of str word: int index"""
#        return self.word2idx

    def dump(self, filename: str, outdir: str = ''):
        """Save vocab to json file"""
        with open(os.path.join(outdir, filename), 'wt') as f:
            f.write(json.dumps(self.word2idx))

    def load(self, filename: str, outdir : str = ''):
        """Load vocab from json file"""
        with open(os.path.join(outdir, filename), 'rt') as f:
            self.word2idx = json.loads(f.read())
        self.word2idx['UNK'] = 0
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        self.n = len(self.word2idx)

    def relativize(self, filename: str) -> np.ndarray:
        """Reduces glove word embeddings in filename to rows in vocab"""
        if isinstance(glove, str):
            glove = pd.read_csv(filename,
                                sep=" ",
                                quoting=3,
                                header=None,
                                index_col=0)
        weights = np.random.normal(scale=0.6, size=(self.n, glove.shape[1]))
        for k in self.word2idx:
            if k in glove.index:
                weights[self[k]] = glove.loc[k].values
        return weights


    def form_input(self, docs: List[List[str | int]], as_tensor: bool = True,
                   is_str: bool = False) -> List:
        """Returns lists of word indexes, with padding

        Args:
            docs: Input documents as list of str or int
            as_tensor: Whether to return result as torch LongTensor, or List 
            is_str: If True, convert str to indexes; else inputs already int

        Returns:
            List of list of ints, optionally as torch.LongTensor

        Notes:
            Short docs randomly padded with replacement with words in doc
        """
        if is_str:
            docs = [[self[w] for w in doc if w in self] for doc in docs]
        else:
            docs = [[w for w in doc if w] for doc in docs]
        lengths = [len(doc) for doc in docs]   # length of each doc
        max_length = max(lengths)              # pad so all lengths equal max
        if max_length:
            out = [doc + random.choices(doc, k=max_length-n) if n else
                   [0] * max_length for doc, n in zip(docs, lengths)]
        else:
            out = [[0]] * len(lengths)
        return torch.LongTensor(out) if as_tensor else out

    def tokenize(self, docs: str | List[str]) -> List[List[int] | int]:
        """Returns list of (list of) tokens from input (list of) sentences"""
        if isinstance(docs, str):
            return self.tokenizer.tokenize(docs.lower())
        else:
            return [self.tokenize(doc) for doc in docs]

    def counter(self, lines: str | List[str]) -> Counter:
        """Returns Counter from words in (list of) lines"""
        vocab = Counter()
        if isinstance(lines, str):
            vocab.update(lines)
        else:
            for line in lines:
                vocab.update(line)
        return vocab   # .most_common(20000)


    def form_splits(self, labels: List[str | int] | int,
                    test_size: float | int = 0.2,
                    random_state: int = 42) -> List[List]:
        """Randomly stratifies labels into train-test split indexes

        Args:
            labels: Labels of series to shuffle, or length of series
            test_size: Desired size of test set as fraction or number of samples
            random_state: Set random seed

        Returns:
            List of stratified train indexes, List of stratified test indexes
        """
        self.train_idx, self.test_idx = form_splits(labels,
                                                    random_state=random_state,
                                                    test_size=test_size)
        return self.train_idx, self.test_idx

    def form_batches(self, batch_size: int) -> List[List[int]]:
        """Shuffles idx list into minibatches each of size batch_size

        Args:
            batch_size: Size of each minibatch
            idx: List of indexes

        Returns:
            List of batches of shuffled indexes 
        """
        return form_batches(batch_size, self.train_idx)

