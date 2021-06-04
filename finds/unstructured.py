"""Implements interface for unstructured data sets

- pymongo, S&P CapitalIQ key developments

Author: Terence Lim
License: MIT
"""
import pandas as pd
from pandas import DataFrame, Series
import re
import requests
from bs4 import BeautifulSoup
from functools import reduce
import gzip
import io
import csv
from pandas.api import types

def parse_where(where):
    """Helper to parse dict or list-like where clause to pymongo command str

    Parameters
    ----------
    where : str or dict or iterable, keyword name may have operator suffix:
        name_eq     be equal to the value
        name_ne     be equal to the value
        name_lt     less than the value
        name_le     less than or equal to the value
        name_gt     greater than the value
        name_ge     greater than or equal to the value
        name_in     be included in the value
        name_notin  not be included in the value

    Examples
    --------

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
    """Base class for unstructured data, with convenience CRUD methods

    Parameters
    ----------
    mongodb : MongoClient object
        connection to underlying mongodb where collection is stored
    database : str
        name of the database


    Attributes
    ----------
    db : pymongo.database.Database object
        pymongo object for this client instance and database name

    Examples
    --------
    fomc = Unstructured(mongodb, 'fomc')       # connect to client named 'fomc'
    fomc.show()
    fomc.select('minutes', where_clause)
    fomc.delete('minutes', where_clause)
    fomc.insert('minutes', doc)
    fomc['minutes'].estimated_document_count() # count docs in collection
    fomc['minutes', 'field']
    Notes
    -----
    
    """
    def __init__(self, mongodb, database):
        """Initialize with database name and mongodb client connection"""
        self.mongodb = mongodb
        self.database = database
        self.db = mongodb.client[database]  # make Database operations available
        #self._c = self.c
        #self.collections = self._c

    def __getitem__(self, args):
        """Access a collection by name, or optionally by field"""
        return self.get(*args) if isinstance(args, tuple) else self.db[args]
    
    def delete(self, collection, where):
        """Delete all docs in collection satisfying where clause

        Parameters
        ----------
        collection : str
           name of collection
        where : str or dict or iterable
           where clause describing documents to delete, as either: 
               str filter (passed on directly to pymongo), or 
               dict of {keys:values}, or
               list of key names (to delete if key name simply $exists)

        Returns
        -------
        count: int
            number of documents deleted
        """
        if collection not in self.db.list_collection_names():
            return None
        result = self[collection].delete_many(parse_where(where))
        return result.deleted_count

    def insert(self, collection, doc, keys=None):
        """Insert one doc; optionally remove existing duplicate document first

        Parameters
        ----------
        collection : str
            name of collection
        doc : dict
            dict of {key:value} representing document
        keys : list of str, optional (default None)
            list of key field names, to delete existing docs with same values

        Returns
        -------
        count: int
            number of existing documents (with same key values) deleted
        """
        deleted = self.delete(collection, keys) if keys else 0
        self[collection].insert_one({k:v for k,v in doc.items() if k != '_id'})
        return deleted

    def get(self, collection, field):
        """Return value of field in first doc containing key field name

        Parameters
        ----------
        collection : str
            name of collection
        field : str
            key field name

        Returns
        -------
        value : value
            value of key field of first document where key field name exists
        """
        return self[collection].find_one({field : {'$exists' : True}})[field]

    def select(self, collection, where=None, include_id=False):
        """Iterator to retrieve docs in collection satisfying where clause

        Parameters
        ----------
        collection : str
            name of collection
        where : str or dict or iterable, optional (default is None)
            where clause describing documents to retrieve
        include_id: bool, optional
            if True, then return _id (default is False)
        """
        include_id = dict() if include_id else {'projection': {'_id': 0}}
        return self[collection].find(parse_where(where), **include_id)

    def show(self, collection=None):
        """Return list of collections; or key names in all docs in collection"""
        if collection is None:
            #: self[c].index_information()
            return {c for c in self.db.list_collection_names()}
        else:
            return reduce(lambda all_keys, rec_keys: all_keys | set(rec_keys),
                          map(lambda d: d.keys(), self[collection].find()),
                          set())

    def load_dataframe(self, collection, df, keys=None, update=False):
        """Insert_many records from rows of dataframe to a collection

        Parameters
        ----------
        collection : str
            name of collection
        df : DataFrame
            each row is a document, with column names as its key fields
        keys : list of str, optional (default is None)
            key field names, to take action if existing doc has same value
        update : bool, optional
            if True, update (default) existing docs if same key values,
            else replace. Simply upsert if no such existing matching doc.
        """
        if keys is None:
            self[collection].insert_many(df.to_dict(orient='records'))
        else:
            for doc in df.to_dict(orient='records'):
                if update:
                    self[collection].update_one({k : doc[k] for k in keys},
                                                doc, upsert=True)
                else:
                    self[collection].replace_one({k : doc[k] for k in keys},
                                                 doc, upsert=True)

def read_situations(csvfile, sep='\t'):
    """Helper method to parse situations text from key developments input file
        
    Parameters
    ----------
    csvfile : str
        name of delimited text file, may be .gz
    sep : str
        delimiter used in input file

    Returns
    -------
    df : DataFrame
        with overflowing 'situation' text corrected, and unique 'keydevid'
    """
    open_ = gzip.open if csvfile.endswith('.gz') else open
    with open_(csvfile, mode = "rt", encoding="latin-1") as f:
        lines = f.readlines()  # "ISO-8859-1" "latin-1")
    nsep = lines[0].count(sep)   # infer number of delimiters
    for i in range(len(lines)-1, 0, -1):  # merge overflow text from end
        lines[i] = lines[i].encode('ascii', 'ignore').decode('ascii')
        if lines[i].count(sep) < nsep:
            lines[i-1] += lines[i]
            del lines[i]
        else:
            lines[i] = re.sub('\n', ' ', re.sub('\x1a', ' ', lines[i]))
    print(len(lines), min([line.count('\t') for line in lines]), lines[0])
    df = DataFrame(data=list(csv.reader(io.StringIO("\n".join(lines[1:])),
                                        quotechar=None, delimiter=sep)),
                   columns=lines[0].lower().rstrip().split(sep))
    print(df.columns)
    df = df.sort_values(['keydevid', 'keydeveventtypeid'])\
           .drop_duplicates(['keydevid'])        # keep unique 'keydevid'
    df['keydevid'] = df['keydevid'].astype(int)
    df['keydeveventtypeid'] = df['keydeveventtypeid'].astype(int)
    df.index = np.arange(len(df))
    return df.loc[:, ['keydeveventtypeid', 'keydevid', 'headline', 'situation']]


if False:  # Read keydevelopment situations text file
    from finds.database import MongoDB
    mongodb = MongoDB()

    from settings import settings
    import os
    import time
    
    keydev = Unstructured(mongodb, 'KeyDev')
    keydev['events'].create_index('keydevid', unique=True)

    downloads = settings['remote']
    for year in [2018, 2019]:
        tic = time.time()
        csvfile = os.path.join(downloads, 'PSTAT', f"situations{year}.txt.gz")
        df = read_situations(csvfile)
        keydev.load_dataframe('events', df)
        print(time.time() - tic)
        
    counts = Series(
        {t: keydev['events'].count_documents({'keydeveventtypeid': t})
         for t in keydev['events'].distinct('keydeveventtypeid')})

