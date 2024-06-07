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
from typing import Dict, Iterable, List, Any, Iterator
_VERBOSE = 0

#
# Helper to store named objects to disk
#
class Store:
    """Wrapper to serialize and deserialize named objects to disk

    Args:
        path: Local folder to store in
        filetype: 'pickle' or 'gzip' or 'json'
        verbose: Debug messages

    Examples:
    >>> store = Store('Downloads')
    >>> store.dump(mydict, 'varname')
    >>> mydict = store.load('varname')

    >>> store['dictname'] = dict(a=1, b=2)
    >>> mydict = store['dictname']
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
            
    def pathname(self, name: str) -> str:
        """Return full path name for object name"""
        return str(self.folder_ / (name + '.' + self.ext_))

    def __contains__(self, name: str) -> bool:
        """Check if object name exists in store"""
        return Path(self.pathname(name)).exists()

    def dump(self, obj: Any, name: str):
        """Helper to dump object, named as name, to file"""
        if self.verbose_:
            print("Store is dumping", name, "to", self.folder_)
        _dump = dict(p=Store.pickle_dump, j=Store.json_dump, g=Store.gzip_dump)
        _dump[self.ext_[0]](obj, self.pathname(name))        
    
    def __setitem__(self, name: str, item: Any):
        """Dumps items to disk, as name

        Args:
          name : name to give to object
          items : dict of items keywords and values

        Examples:

        >>> store['point1'] = (x, y)
        """
        self.dump(item, name=name)

    def load(self, name: str):
        """Helper to oad object, named by name, from store"""
        if self.verbose_:
            print("Store is loading", name, "from", self.folder_)
        _load = dict(p=Store.pickle_load, j=Store.json_load, g=Store.gzip_load)
        return _load[self.ext_[0]](self.pathname(name))

    def __getitem__(self, name: str) -> Any:
        """Loads object, named by name, from store

        Args:
          name : name of object
        """
        return self.load(name)

    def __iter__(self) -> Iterator:
        """Iterates over all object name in store's folder"""
        for filename in self.folder_.glob('*.' + self.ext_):
            yield filename.stem

if __name__ == "__main__":
    store = Store('/home/terence/Downloads/store', 'pkl')
    store.dump(dict(key='1', value='2'), name='test1')
    print(store.load('test1'))
    store['test2'] = dict(key='1', value='2')
    print(store['test2'])
    
    
