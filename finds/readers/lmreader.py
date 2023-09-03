"""Retrieves Loughran-MacDonald word lists

MIT License

Copyright 2022-2023 Terence Lim
"""
import requests
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pandas_datareader as pdr
from typing import List, Dict
from finds.database.mongodb import MongoDB
_VERBOSE = 1

class LMReader:
    """Class to retrieve LoughranMcDonald dictionary and stopwords"""
    
    master_url = 'https://sraf.nd.edu/loughranmcdonald-master-dictionary/'
    stopwords_url = 'https://sraf.nd.edu/textual-analysis/stopwords/'

    @staticmethod
    def fetch(source: str, stopword: str = '', sep=','):
        """Helper to retrieve and parse LoughranMcDonald dictionary or stopwords

        Args:
            source: URL or full pathname of csv file 
            stopword: if source is a stopword file, specifies its category

        Returns:
            word lists keyed by associated label, e.g. 'positive', 'generic' etc

        Notes:

        - https://sraf.nd.edu/textual-analysis/
        """
        if not stopword:
            master = pd.read_csv(source, sep=sep)        # main csv file
            master.columns = master.columns.str.lower()  # column names to lower
            master['word'] = master['word'].str.lower()  # all strings to lower
            results = dict()
            for s in ['negative', 'positive', 'uncertainty', 'litigious', 
                      'constraining', 'strong_modal', 'weak_modal']:
                results[s] = master['word'][master[s].ne(0)].tolist()
        else:
            words = pd.read_csv(source, sep='|', encoding='latin_1')
            results = {stopword:
                       words.iloc[:,0].str.lower().str.rstrip().to_list()}
        return results

    @staticmethod
    def update(wordlists: MongoDB | None = None, stopfiles: List[str] = [],
               verbose=_VERBOSE) -> Dict:
        """Update Loughran McDonald stopword and sentiment words to MongoDB"""

        def _print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        def get_filenames(url):
            """helper scrape LM's web pages for all google drive filenames"""
            prefix = "https://drive.google.com/uc?export=download&id="
            r = requests.get(url).text  
            files = {
                re.search('>(.*)?</a>', h).group(1):
                prefix + re.search('/file/d/(.*)?/view', h).group(1)
                for h in re\
                .findall('<a\s+?href.*?https://drive.google.com/file.*?</a>', r)
            }
            return files

        # determine master dictionary filename, then fetch
        master = get_filenames(LMReader.master_url)
        masterfile = [v for k,v in master.items() if k.endswith('.csv')][0]
        words = LMReader.fetch(masterfile)
        _print(masterfile, words.keys())
        
        # determine stopword filenames, then fetch
        if not stopfiles:
            stopfiles = get_filenames(LMReader.stopwords_url)
            f = {re.search('StopWords_(.*)\.txt', k, re.I).group(1).lower(): v
                     for k,v in stopfiles.items() if 'stopwords_' in k.lower()}
            _print(Series(f))
        else:
            f = {re.search('StopWords_(.*)\.txt', k, re.I).group(1).lower(): k
                 for k in stopfiles if 'stopwords_' in k.lower()}
        for stopword, filename in f.items():
            words.update(LMReader.fetch(filename, stopword))
        _print(Series({k: len(v) for k,v in words.items()},  name='count'))

        if wordlists:  # insert each wordlist as a document in collection 'lm'
            for k,v in words.items():       
                wordlists.insert('lm', {k:v}, keys=[k])  
            _print(Series({k: len(wordlists['lm', k]) for k in words.keys()}))
        return words


if __name__ == "__main__":  
    import os
    import time
    from pathlib import Path
    from finds.database import MongoDB
    from finds.unstructured import Unstructured
    from secret import credentials, paths

    # Update Loughran-MacDonald word lists
    mongodb = MongoDB()
    datadir = paths['downloads'] / 'stocks2022/LM'
    wordlists = Unstructured(mongodb, 'WordLists')
    stopfiles = list(downloads.glob("*"))
    LMReader.update(wordlists, stopfiles)

