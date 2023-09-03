"""Helper class to accumulate and locally store logs

Copyright 2022, Terence Lim

MIT License
"""
import re
import csv
import json
import gzip
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Any, Tuple, Iterator
from collections import namedtuple
_VERBOSE = 0

#
# Helper to store key-value attributes as namedtuple
#
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

    def __init__(self, folder: str, ext: str = 'pkl', verbose: int = _VERBOSE):
        """Initialize a store instance with folder name and file extension

        Args:
          folder: name of folder to store items
          ext: output file format {'pkl', 'gz', 'json'}
        """
        self.folder_ = Path(str(folder))
        self.ext_ = ext.lower()
        assert self.ext_ in ['pkl', 'gz', 'json'], "ext must be pkl, gz, json"
        self.verbose_ = verbose
        if not self.folder_.exists():
            self.folder_.mkdir()
        if verbose:
            print("Store in", folder, "as", self.ext_)
            
    def pathname(self, key: str) -> str:
        """Return full path name for key name"""
        return str(self.folder_ / (key + '.' + self.ext_))

    def __contains__(self, key: str) -> bool:
        """Check if object named as key exists in store"""
        return Path(self.pathname(key)).exists()

    def __call__(self, *args, **kwargs) -> namedtuple:
        """Convert keyword arguments to namedtuple

        Args:
           args : key name
           kwargs : data items as keyword argument pairs
        """
        NamedTuple = namedtuple(re.sub('[^0-9a-zA-Z]+', '_', args[0]),
                                list(kwargs.keys()))
        return NamedTuple(**kwargs)

    def dump(self, obj: Any, key: str):
        """Dumps object, named as key, to file"""
        if self.verbose_:
            print("Store is dumping", key, "to", self.folder_)
        _dump = dict(p=Store.pickle_dump, j=Store.json_dump, g=Store.gzip_dump)
        _dump[self.ext_[0]](obj, self.pathname(key))        
    
    def __setitem__(self, key: str, items: Dict | namedtuple):
        """Dumps args dict or namedtuple object, named as key, to file

        Args:
          key : key name
          items : dict or namedtuple of items keywords and values

        Examples:

        >>> store['point1'] = dict(x=1, y=2)
        """
        if issubclass(type(items), tuple):
            items = items._asdict()
        assert isinstance(items, dict), "items must be dict or namedtuple"
        self.dump(items, key=key)

    def load(self, key: str):
        """Loads object, named by key, from store"""
        if self.verbose_:
            print("Store is loading", key, "from", self.folder_)
        _load = dict(p=Store.pickle_load, j=Store.json_load, g=Store.gzip_load)
        return _load[self.ext_[0]](self.pathname(key))

    def __getitem__(self, key: str) -> namedtuple:
        """Loads object, named by key, from store as namedtuple

        Args:
          key : key name

        Returns:
          namedtuple retrieved by input key name
        """
        return self(key, **self.load(key))

    def __iter__(self) -> Iterator:
        """Iterates over all objects' key names in store's folder"""
        for filename in self.folder_.glob('*.' + self.ext_):
            yield filename.stem

if __name__ == "__main__":
    store = Store('/home/terence/Downloads/store', 'pkl')
    store.dump(dict(key='1', value='2'), key='test1')
    print(store.load('test1'))
    store['test2'] = dict(key='1', value='2')
    print(store['test2'])
    
    
