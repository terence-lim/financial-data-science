"""Class to manage words vocabulary

Copyright 2022, Terence Lim

MIT License
"""
from typing import Dict, Iterable, List, Any, Tuple, Self, Set
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types
import random
import os
import json
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from pathlib import Path
_VERBOSE = 0

class Vocab():
    """Class for managing a vocabulary of words
        
    Args:
      words : List of words to create index
      unk : str representation of unknown word
    """

    tokenize = RegexpTokenizer(r"\b[^\d\W][^\d\W][^\d\W]+\b").tokenize
    """a default tokenizer, wraps nltk RegexpTokenizer"""

    def __init__(self, words: List = [], unk: str = '<UNK>'):
        """Initialize class for managing words vocabulary

        Examples:
        >>> Vocab(['hello', 'world'])
        """

        # create bidirectional mapping of words and indexes
        self.word2idx = {unk: 0}
        self.idx2word = {0: unk}
        self.unk = unk
        self.update(words)
        self.embeddings = []

    def update(self, words: List):
        """update words in vocab, in lower case"""
        idx = len(self.word2idx)
        for w in words:
            w = w.lower()
            if w not in self.word2idx:
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                idx += 1

    @property
    def dim(self) -> int:
        """returns the dimensionality of the embeddings vector"""
        return self.embeddings.shape[1]

    def dump(self, filename: str) -> Self:
        """Dump vocab to file"""
        with open(filename, "wb") as f:
            pickle.dump([self.word2idx, self.idx2word, self.embeddings], f)
    
    def load(self, filename: str) -> Self:
        """Load vocab from file"""
        with open(filename, "rb") as f:
            self.word2idx, self.idx2word, self.embeddings = pickle.load(f)

    def __getitem__(self, item: str | int) -> int | str:
        """Return index of str item or word of int item"""
        if isinstance(item, str):
            return self.word2idx.get(item.lower(), 0)
        elif isinstance(item, int):
            return self.idx2word.get(item, self.unk)
        else:
            raise Exception("item must be str or int")
    
    def get_index(self, words: str | List) -> int | List:
        """Return indexes of words list, optionally drop unknown words"""
        return ([self.get_index(w) for w in words] if types.is_list_like(words)
                else self[words])

    def get_word(self, index: int | List) -> str | List:
        """Return words of indexes"""
        return ([self.get_word(k) for k in index] if types.is_list_like(index)
                else self[index])

    def __contains__(self, word: str) -> bool:
        """Returns True (False) if word is in (not in) vocab"""
        return word in self.word2idx

    def __len__(self) -> int:
        """Returns length of vocab"""
        return len(self.word2idx)

    def set_embeddings(self, embeddings: DataFrame) -> DataFrame:
        """Relativize and index embeddings to words in vocab"""
        # default embeddings vector values
        vectors = np.random.normal(scale=0.6,
                                   size=(len(self.word2idx), embeddings.shape[1]))
        vectors[0] = np.zeros((1, embeddings.shape[1]))   # values for unknown word
        words = Series(self.word2idx)
        common = list(set(words.index).intersection(embeddings.index)\
                      .difference(['nan']))
        vectors[words[common].values] = embeddings.loc[common].values
        self.embeddings = vectors

    def get_embeddings(self, word: str | List) -> np.array:
        """Return embedding vector of a (list of) word"""
        return (np.vstack([self.get_embeddings(w) for w in word])
                if types.is_list_like(word)
                else self.embeddings[self[word]])


if __name__ == "__main__":
    from collections import Counter
    from secret import paths
    
    text = ['The quick brown fox jumps over the lazy dog',
            'The cow jumps over the moon']
    lines = [Vocab.tokenize(line.lower()) for line in text]

    # Count words for vocab
    counts = Counter()
    for line in lines:
        counts.update(line)
    words = [w[0] for w in counts.most_common(5)]
    vocab = Vocab(words)

    # test it
    print(vocab['the'], vocab['unk'])
    print(vocab[2], vocab[0], vocab[1000])
    print(vocab.get_index(lines))
    print(vocab.get_words([1, 2, 3, 4, 5, 6, 7]))
    

    # Read word embeddings vectors as a DataFrame
    filename = paths['scratch'] / 'glove.6B.300d.txt'
    sep = " "
    quoting = 3
    df = pd.read_csv(filename, sep=sep, quoting=quoting,
                     header=None, index_col=0, low_memory=True)
    df.index = df.index.astype(str).str.lower()   # convert to lower case

    # Relativize to vocab
    vocab.set_embeddings(df)
