"""Classes for unstructured and textual datasets

- FOMC minutes
- Loughran and McDonald words
- S&P CapitalIQ key developments situations text

Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import re
import requests
import io
import csv
import time
import json
import gzip
import pickle
import random
import os
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter, namedtuple
import torch
from bs4 import BeautifulSoup
from functools import reduce
from pandas.api import types
from finds.database import MongoDB
from typing import Dict, Iterable, List, Any, Tuple

_VERBOSE = 1

#
# Helper to store key-value attributes as namedtuple
#
#TODO: call this NamedStore
class Store:
    """Store key-value attributes as namedtuple
    Args:
        path: Local folder to store in
        filetype: 'pickle' or 'gzip' or 'json'
        name: Optional name of NamedTuple
        verbose: Debug messages

    Examples:
    >>> store = Store('Downloads')
    >>> store.dump(mydict, 'varname')
    >>> mydict = store.load('varname')

    >>> store['tuplename'] = dict(a=1, b=2)
    >>> mytuple = store['tuplename']
    """
    def __init__(self, path: str = "", filetype: str = 'pickle',
                 name: str = 'NamedTuple', verbose: int = 0):
        self.path_ = str(path)
        self.name_ = name
        self.filetype_ = filetype[0].lower()
        self.verbose_ = verbose

    def pathjoin(self, *p) -> str:
        return os.path.join(self.path_, *p)

    def pathname(self, filename: str, filetype: str):
        filename = os.path.join(self.path_, filename)
        if filetype[0].lower() == 'p' and not filename.endswith('.pkl'):
            filename += '.pkl'
        if filetype[0].lower() == 'g' and not filename.endswith('.gz'):
            filename += '.gz'
        if filetype[0].lower() == 'j' and not filename.endswith('.json'):
            filename += 'json'
        return filename

    @staticmethod
    def gzip_dump(obj: Any, filename: str):
        with gzip.open(filename, 'wt') as fp:
            json.dump(obj, fp)

    @staticmethod
    def gzip_load(filename: str) -> Any:
        with gzip.open(filename, 'rt') as fp:
            return json.load(fp)

    @staticmethod
    def json_dump(obj: Any, filename: str):
        with open(filename, 'wt') as fp:
            json.dump(obj, fp)

    @staticmethod
    def json_load(filename: str) -> Any:
        with open(filename, 'rt') as fp:
            return json.load(fp)

    @staticmethod
    def pickle_dump(obj: Any, filename: str):
        with open(filename, 'wb') as fp:
            pickle.dump(obj, fp)

    @staticmethod
    def pickle_load(filename: str) -> Any:
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    def load(self, filename: str):
        load_ = dict(p=Store.pickle_load,
                     j=Store.json_load,
                     g=Store.gzip_load).get(self.filetype_)
        filename = self.pathname(filename, self.filetype_)
        return load_(filename)
        

    def dump(self, obj: Any, filename: str):
        """ TODO should initialized self.dump_ so that no need to lookup again"""
        
        dump_ = dict(p=Store.pickle_dump,
                     j=Store.json_dump,
                     g=Store.gzip_dump).get(self.filetype_)
        filename = self.pathname(filename, self.filetype_)
        dump_(obj, filename)
    
    def __contains__(self, filename: str) -> bool:
        """Check if filename (after path prepend and suffix append) exists"""
        filename = self.pathname(filename, filetype=self.filetype_)        
        return os.path.exists(filename)

    def __call__(self, **kwargs) -> namedtuple:
        """Convert keyword args dict to namedtuple"""
        NamedTuple = namedtuple(self.name_, list(kwargs.keys()))
        return NamedTuple(**kwargs)

    def __setitem__(self, filename: str, items: Dict | namedtuple):
        """Stores keywords args dict or namedtuple object to file

        should initialized self.dump_ so that no need to lookup again
        """
        if issubclass(type(items), tuple):
            items = items._asdict()
        assert isinstance(items, dict)
        dump_ = dict(p=Store.pickle_dump,
                     j=Store.json_dump,
                     g=Store.gzip_dump).get(self.filetype_)
        filename = self.pathname(filename, self.filetype_)
        dump_(items, filename)

    def __getitem__(self, filename: str) -> namedtuple:
        """Loads namedtuple attribute values from file"""
        load_ = dict(p=self.pickle_load,
                     j=self.json_load,
                     g=self.gzip_load).get(self.filetype_)
        filename = self.pathname(filename, self.filetype_)
        return self(**load_(filename))


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

class TextualData:
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



def parse_where(where: Dict | str | List) -> Dict:
    """Helper to parse dict or list-like where clause to pymongo command str

    Args:
        where: keyword name may have operator suffix:

        - name_eq:     be equal to the value
        - name_ne:     be equal to the value
        - name_lt:     less than the value
        - name_le:     less than or equal to the value
        - name_gt:     greater than the value
        - name_ge:     greater than or equal to the value
        - name_in:     be included in the value
        - name_notin:  not be included in the value

    Returns:
        Dictionary of conditions in pymongo format
    """
    result = dict()
    if isinstance(where, dict):    # where dict of {key: condition}'s =>
        for k,v in where.items():    # parse condition to test key value:
            if isinstance(v, dict):     # if dict: 
                result[k] = v
            elif isinstance(v, set):    # if set: key isin set
                result[k] = {'$in' : list(v)}
            elif types.is_list_like(v): # if tuple:
                assert(v[0] <= v[1])      # inclusive bounds on key value
                result[k] = {'$gte': v[0], '$lte': v[1]}
            else:                       # if scalar: key value equals
                result[k] = {'$eq': v}
    elif isinstance(where, str):     # where string => test if key name exists
        result = {where : {'$exists' : True}}
    elif where:                      # where list-like => test if all keys exist
        try:
            result = {k : {'$exists' : True} for k in where}
        except:
            raise Exception('[where] must be a dict, array-like or str')
    return result
                        

class Unstructured(object):
    """Base class for unstructured datasets

    Args:
        mongod: connection to MongoClient where data collection is stored
        database: name of the database in MongoDB


    Attributes:
        db : pymongo.database.Database connection

    Examples:

    >>> fomc = Unstructured(mongodb, 'fomc')  # connect to client named 'fomc'
    >>> fomc.show()
    >>> fomc.select('minutes', where_clause)
    >>> fomc.delete('minutes', where_clause)
    >>> fomc.insert('minutes', doc)
    >>> fomc['minutes'].estimated_document_count() # count docs in collection
    >>> fomc['minutes', 'field']

    Notes:
    - sudo apt-get install -y mongodb-org  # install latest community version
    - sudo systemctl start mongod     # start and stop mongodb server
    - sudo systemctl status mongod
    - sudo systemctl restart mongod
    - sudo systemctl stop mongod

    """
    def __init__(self, mongodb: MongoDB, database: str):
        self.mongodb = mongodb
        self.database = database
        self.db = mongodb.client[database]  # make Database operations available
        #self._c = self.c
        #self.collections = self._c

    def __getitem__(self, args: Tuple | Any) -> Any:
        """Access a collection by name, or optionally by field"""
        if isinstance(args, tuple):
            return self.get(*args)
        else:
            return self.db[args]
    
    def delete(self, collection: str, where: str | Dict | List) -> int:
        """Delete all docs in collection satisfying where clause

        Args:
            collection: name of collection in database to delete
            where: where clause describing documents to delete

        Returns:
            number of documents deleted, -1 if collection not in database

        Notes:

        - str filter (passed on directly to pymongo)
        - dict of {keys:values}
        - list of key names (to delete if key name $exists)

        """
        if collection not in self.db.list_collection_names():
            return -1
        result = self[collection].delete_many(parse_where(where))
        return result.deleted_count

    def insert(self, collection: str, doc: Dict, keys: List[str] = []):
        """Insert one doc; optionally remove existing duplicate document first

        Args:
            collection: name of collection in database to insert into
            doc: dict of {key:value} representing document
            keys: list of field names, to delete existing docs with same values

        Returns:
            number of existing documents (with same key values) deleted
        """
        deleted = self.delete(collection, keys) if [] else 0
        self[collection].insert_one({k:v for k,v in doc.items() if k != '_id'})
        return deleted

    def get(self, collection: str, field: str) -> Any:
        """Return value of field of first doc containing key field name

        Args:
            collection: name of collection in database to retrieve from
            field: key field name

        Returns:
            value of key field of first document where key field name exists
        """
        return self[collection].find_one({field : {'$exists' : True}})[field]

    def select(self, collection, where: str | List | Dict = [],
               include_id: bool = False) -> List:
        """Iterator to retrieve docs in collection satisfying where clause

        Args:
            collection: Name of collection in database to delete
            where: Where clause describing documents to retrieve
            include_id: If True, then include _id field in return

        Returns:
            Document selecting where clause in a list of dict
        """
        include_id = dict() if include_id else {'projection': {'_id': 0}}
        return self[collection].find(parse_where(where), **include_id)

    def show(self, collection: str = ''):
        """Return list of collections; or key names in all docs in collection"""
        if not collection:
            #: self[c].index_information()
            return {c for c in self.db.list_collection_names()}
        else:
            return reduce(lambda all_keys, rec_keys: all_keys | set(rec_keys),
                          map(lambda d: d.keys(), self[collection].find()),
                          set())

    def load_dataframe(self, collection: str, df: DataFrame,
                       keys: List[str] = [], update: bool = False):
        """Insert_many records from rows of dataframe to a collection

        Args:
            collection: Name of collection in database to delete
            df: Each row of DataFrame is document, column names as key fields
            keys: Fields names to update or replace if same values
            update: If key fields have same value, update if True. Else replace
        """
        if not keys:
            self[collection].insert_many(df.to_dict(orient='records'))
        else:
            for doc in df.to_dict(orient='records'):
                if update:
                    self[collection].update_one({k: doc[k] for k in keys},
                                                doc,
                                                upsert=True)
                else:
                    self[collection].replace_one({k: doc[k] for k in keys},
                                                 doc,
                                                 upsert=True)

class FOMCReader:
    """Class to retrieve FOMC minutes"""
    
    fed_url = 'https://www.federalreserve.gov/'  # Else catalog from main site
    
    @staticmethod
    def fetch(url: str = '') -> str | Dict[int, str]:
        """Retrieve FOMC minutes or catalog from Fed website

        Args:
            url: Optional webpage url to retrieve text from

        Returns:
            text of minutes, or dict of all dates and urls from Fed site
        """

        if url:                # Retrieve FOMC minutes from input url
            raw = BeautifulSoup(markup=requests.get(url).content,
                                features='html.parser')
            minutes = "\n\n".join([p.get_text().strip()
                                   for p in raw.findAll('p')])
            return re.sub('\n+','\n', re.sub('[\r\t]',' ', minutes))

        dateOf = lambda s: int(re.sub('\D', '', s)[-8:]) 
        
        # latest five years' minutes can be linked from a main page
        new_url = FOMCReader.fed_url + 'monetarypolicy/fomccalendars.htm'
        raw = BeautifulSoup(markup=requests.get(new_url).content,
                            features='html.parser')
        hrefs = raw.find_all(name='a',
                             href=re.compile('\S+minutes\S+.htm$', re.I))
        links = [FOMCReader.fed_url + m.attrs['href'] for m in hrefs]

        # earlier years' minutes are linked from annual pages with this format
        old_url = FOMCReader.fed_url + 'monetarypolicy/fomchistorical%d.htm'
        for year in range(1993, min([dateOf(m) for m in links]) // 10000):
            raw = BeautifulSoup(markup=requests.get(old_url % year).content,
                                features='html.parser')
            hrefs = raw.find_all(name='a',
                                 href=re.compile('\S+minutes\S+.htm$', re.I))
            links += [FOMCReader.fed_url
                      + m.attrs['href'].replace(FOMCReader.fed_url,'')
                      for m in hrefs]
        return {dateOf(link) : link for link in links}


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
    from finds.database import MongoDB
    from conf import credentials, paths, VERBOSE

    if False:
        mongodb = MongoDB()

    def update_situations(csvfile: str, sep: str = '\t'):
        """Helper method to parse situations text from key developments file
        
        Args:
            csvfile: name of delimited text file, may be .gz
            sep: delimiter used in input file

        Returns:
            DataFrame with overflowing 'situation' text corrected, 
            and unique 'keydevid'
        """
        tic = time.time()
        open_ = gzip.open if csvfile.endswith('.gz') else open
        with open_(csvfile, mode = "rt", encoding="latin-1") as f:
            lines = f.readlines()                # "ISO-8859-1" "latin-1")
            nsep = lines[0].count(sep)           # infer number of delimiters
        for i in range(len(lines)-1, 0, -1):     # merge overflow text from end
            lines[i] = lines[i].encode('ascii', 'ignore').decode('ascii')
            if lines[i].count(sep) < nsep:
                lines[i-1] += lines[i]
                del lines[i]
            else:
                lines[i] = re.sub('\n', ' ', re.sub('\x1a', ' ', lines[i]))
        print(round(time.time() - tic, 0), 'secs',
              len(lines), min([line.count('\t') for line in lines]), lines[0])

        tic = time.time()
        df = DataFrame(data=list(csv.reader(io.StringIO("\n".join(lines[1:])),
                                            quotechar=None,
                                            delimiter=sep)),
                       columns=lines[0].lower().rstrip().split(sep))
        print(round(time.time() - tic, 0), 'secs', len(df), df.columns)

        df = df.sort_values(['keydevid', 'keydeveventtypeid'])\
               .drop_duplicates(['keydevid'])      # keep unique 'keydevid'
        df['keydevid'] = df['keydevid'].astype(int)
        df['keydeveventtypeid'] = df['keydeveventtypeid'].astype(int)
        df.index = np.arange(len(df))
        return df.loc[:, ['keydeveventtypeid', 'keydevid',
                          'headline', 'situation']]


    def update_keydev(): # Sample code to read keydev situations text file
        keydev = Unstructured(mongodb, 'KeyDev')
        keydev['events'].create_index('keydevid', unique=True)

        downloads = os.path.join(paths['downloads'], 'stocks2020', 'PSTAT')
        for year in [2018, 2019]:
            tic = time.time()
            csvfile = os.path.join(downloads, f"situations{year}.txt.gz")
            df = read_situations(csvfile)
            keydev.load_dataframe('events', df)
            print(time.time() - tic)
        
        counts = Series({t: keydev['events']\
                         .count_documents({'keydeveventtypeid': t})
                         for t in keydev['events']\
                         .distinct('keydeveventtypeid')})

    def update_lm():  # Sample code to update from LoughranMcDonald word files
        wordlists = Unstructured(mongodb, 'WordLists')
        from glob import glob
        stopfiles = glob(os.path.join(paths['downloads'], 'stocks2022/LM/*'))
        LMReader.update(wordlists, stopfiles)

    print("unstructured")
