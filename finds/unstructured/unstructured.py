"""Classes for unstructured and textual datasets

Copyright 2022, Terence Lim

MIT License
"""
from typing import Dict, Iterable, List, Any, Tuple
import re
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from functools import reduce
from finds.database.mongodb import MongoDB, parse_where
_VERBOSE = 1

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

if __name__ == "__main__":  
    from pathlib import Path
    import time
    import csv
    import io
    from finds.database import MongoDB
    from secret import credentials, paths

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
    if False:
        mongodb = MongoDB()

    print("unstructured")
