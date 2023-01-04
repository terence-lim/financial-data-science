"""Wrappers for database engines

- SQL: sqlalchemy
- MongoDB: pymongo
- NoSQL store: redis

Convenience methods to:

- Load, store and manipulate pandas DataFrames that match SQL database schemas

- Serialize DataFrame to Redis key-value store (cache SQL query results)


Copyright 2022, Terence Lim

MIT License
"""
from typing import List, Dict, Mapping, Any
import random
import sys
import os
import time
import io
import requests
import zipfile
import gzip
import csv
import json
import unicodedata
import glob
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import redis
from pymongo import MongoClient
from typing import Dict, Any, List

_VERBOSE = 0   # default verbose level
_headers = {'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36'
            'OPR/38.0.2220.41'}

def requests_get(url: str, params: Dict = None, retry: int = 7,
                 sleep: float = 2., timeout: float = 3., delay: float = 0.25,
                 trap: bool = False, headers: str = _headers,
                 verbose: int = _VERBOSE) -> requests.Response | None:
    """Wrapper over requests.get, with retry loops and delays

    Args:
      url: URL address to request
      params: Payload of &key=value to append to url
      headers: User-Agent, Connection and other headers parameters
      timeout: Number of seconds before timing out one request try
      retry: Number of times to retry request
      sleep: Number of seconds to wait between retries
      trap: On timed-out: if True raise exception, else return False
      delay: Number of seconds to wait initially
      verbose: Whether to display verbose debugging messages

    Returns:
      requests.Response or None if timed-out or status_code != 200
    """
    def _print(*args, **kwargs):
        """helper to print verbose messages"""
        if verbose > 0:
            print(*args, **kwargs)
            
    _print(url)
    if delay:
        time.sleep(delay + (delay * np.random.rand()))
    for i in range(retry):
        try:
            r = requests.get(url,
                             headers=headers,
                             timeout=timeout,
                             params=params)
            assert(r.status_code >= 200 and r.status_code <= 404)
            break
        except Exception as e:
            _print(f"(requests_url {i}/{retry})", e)
            time.sleep(sleep * (2 ** i) + sleep*np.random.rand())
            r = None
    if r is None:  # likely timed-out after retries:
        if trap:     # raise exception if trap, else silently return None
            raise Exception(f"requests_get: {url} {time.time()}")
        return None
    if r.status_code != 200:
        _print(r.status_code, r.content)
        return None
    return r


class SQL:
    """Provide convenience interface to sqlalchemy engine"""

    def __init__(self, user: str, password: str, host: str = 'localhost',
                 port: str = '3306', database: str = '',
                 autocommit: str = 'true', charset: str = 'utf8',
                 temp: str = f"temp{random.randint(0, 8192)}",
                 verbose: int = _VERBOSE):
        self.url = \
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?" \
            + f"charset={charset}&local_infile=1&autocommit={autocommit}"
        self._verbose = verbose
        self._t = temp  # name of temp table for this process
        self.create_engine()
        
    def _print(self, *args, verbose: int = _VERBOSE, level: int = 0, **kwargs):
        """helper to print verbose messages"""
        if max(verbose, self._verbose) > 0:
            print(*args, **kwargs)
        
    def create_engine(self):
        """Wrap sqlalchemy.create_engine() and MetaData(); store attributes"""
        self.engine = sqlalchemy.create_engine(self.url, echo=self._verbose>0)
        self.metadata = sqlalchemy.MetaData(self.engine)

    def rollback(self):
        """Wraps sessionmaker() to rollback current transaction in progress"""
        Session = sessionmaker(self.engine)
        with Session() as session:
            session.rollback()
    
    def Table(self, key: str, *args, **kwargs) -> sqlalchemy.Table:
        """Wrap sqlalchemy.Table() to first remove table entry from metadata"""
        if key in self.metadata.tables:
            self.metadata.remove(self.metadata.tables[key])
        return sqlalchemy.Table(key, self.metadata, *args, **kwargs)

    @classmethod
    def Index(cls, *args) -> sqlalchemy.Index:
        """Wrap sqlalchemy.Index(), auto-generates index name from args"""
        return sqlalchemy.Index("_".join(args), *args)

    def remove(self, key: str):
        """Remove a table, by its key name, from metadata instance"""
        if key in self.metadata.tables:
            self.metadata.remove(self.metadata.tables[key])

    def run(self, q, *args, **kwargs) -> Dict | None:
        """Execute an sql command

        Args:
            q: query string
            *args: argument list for query
            **kwargs: keyword arguments for query

        Returns:
            The result set {'data', 'columns'}, or None.

        Raises:
            RuntimeError: failed to run query

        Examples:
            >>> sql.run("show databases")
            >>> sql.run("show tables")
            >>> sql.run('select * from testing')
            >>> sql.run('select distinct permno from benchmarks')
            >>> sql.run("show create table _")
            >>> sql.run("describe _")
            >>> sql.run("truncate table _", fetch=False)
        """

        for _ in range(2):
            try:
                with self.engine.begin() as conn:
                    try:
                        r = conn.execute(q, *args, **kwargs)
                        return {'data': r.fetchall(), 'columns': r.keys()}
                    except Exception:
                        return None
                break
            except Exception as e:
                self._print(e)
                self.create_engine()
        raise RuntimeError('(sql.run) ' + q)

    def summary(self, table: str, val: str, key: str = '') -> DataFrame:
        """Return summary statistics for a field, optionally grouped-by

        Args:
            table: Physical name of table
            val: Field name to summarise
            key: Field to group by

        Returns:
            DataFrame with columns (count, average, max, min)

        Examples:
            >>> sql.summary('annual', 'revt', 'sic')
        """
        if key:
            q = (f"SELECT {key}, COUNT(*) as count, AVG({val}) as avg, "
                 f" STD({val}) as std, MAX({val}) as max, MIN({val}) as min "
                 f" FROM {table} GROUP BY {key}")
            return self.read_dataframe(q).set_index(key).sort_index()
        else:
            q = (f"SELECT COUNT(*) as count, AVG({val}) as avg, "
                 f"  MAX({val}) as max, MIN({val}) as min FROM {table}")
            return DataFrame(index=[val], **self.run(q))

    def load_infile(self, table: str, csvfile: str, options: str =''):
        """Load table from csv file, using mysql's load data local infile

        Args:
            table: Physical name of table to load into
            csvfile: CSV filename
            options: String appended to SQL load infile query
        """
        q = (f"LOAD DATA LOCAL INFILE '{csvfile}' INTO TABLE {table} "
             f" FIELDS TERMINATED BY ',' ENCLOSED BY '\"'"
             f" LINES TERMINATED BY '\\n' IGNORE 1 ROWS {options};")
        try:
            self._print("(load_infile)", q)
            self.run(q)
        except Exception as e:
            print("(load_infile) Got exception = ", e, " Query = ", q)
            raise e

    def load_dataframe(self, table: str, 
                             df: DataFrame,
                             index_label: str = '', 
                             to_sql: bool = True,
                             replace: bool = False):
        """Load dataframe into sql table, ignoring duplicate primary keys

        Args:
            table: Physical name of table to insert into
            df: Source dataframe
            index_label: Column name to load index as, None (default) to ignore
            to_sql: first attempt pandas.to_sql(), which may fail if duplicate
               keys; then/else insert ignore from temp table instead.
            replace: set True to overwrite table, else append (default)
        """

        df.columns = df.columns.map(str.lower).map(str.rstrip)
        chunksize = int(1024*1024*32 // len(df.columns))
        try:     # to_sql raises exception if exist duplicate keys
            assert(to_sql)
            df.to_sql(table,
                      self.engine,
                      if_exists=('replace' if replace else 'append'),
                      chunksize=chunksize,
                      index=bool(index_label),
                      index_label=index_label)
        except Exception as e:  # duplicates exist
            self._print("(load_dataframe) Retrying insert ignore", table)
            self.run('drop table if exists ' + self._t)
            df.to_sql(self._t,
                      self.engine,
                      if_exists='replace',
                      chunksize=chunksize,
                      index=bool(index_label),
                      index_label=index_label)
            # warnings.filterwarnings("ignore", category=pymysql.Warning)
            columns = ", ".join(df.columns)
            q = (f"INSERT IGNORE INTO {table} ({columns})"
                 f" SELECT {columns} FROM {self._t}")
            self.run(q)
            # warnings.filterwarnings("default", category=pymysql.Warning)
            self.run('drop table if exists ' + self._t)

    def read_dataframe(self, q: str):
        """Return sql query result as data frame

        Args:
            q: query string or SQLAlchemy Selectable

        Returns:
            DataFrame of results

        Raises:
            RuntimeError: Failed to run query
        """
        result = self.run(q)
        if result is None:
            raise RuntimeError('read_dataframe error in database: ', str(q))
        return DataFrame(**result)

    def pivot(self, table: str, 
                    index: str, 
                    columns: str, 
                    values: str, 
                    where: str = '', 
                    limit: int | None = None,
                    chunksize: int | None = None) -> DataFrame:
        """Return sql query result as pivoted data frame

        Args:
            table: Physical name of table to retrieve from
            index: Field name to select as dataframe index
            columns: Field name to select as column labels
            values: Field name to select as values
            where: Where clause, optional
            limit: Maximum optional number of rows or chunks to return
            chunksize: To optionally buildup results in chunks of this size

        Returns:
            Query result as a pivoted (wide) DataFrame
        """

        if where:  # pre-prend where clause with keyword
            where = 'WHERE ' + where

        if isinstance(chunksize, int):  # execute in chunks
            rows = self.read_dataframe(
                f"SELECT DISTINCT {index} FROM {table} {where}")
            rows = np.array(rows[index].astype(str))
            out = DataFrame()
            n_features = len(rows)
            n_splits = n_features // chunksize
            if n_splits * chunksize < n_features:
                n_splits += 1
            for i in range(n_splits):
                row = slice(chunksize * i, min(n_features, chunksize * (i+1)))
                self._print('slice #', i, 'of', n_splits)
                if isinstance(limit, int) and i >= limit:
                    break
                where += " AND " if where else " WHERE "
                indexes = "','".join(rows[row])
                q = (f"SELECT {index}, {columns}, {values} FROM {table} "
                     f" {where} {index} in ('{indexes}')")
                df = self.read_dataframe(q)
                out = out.append(
                    df.pivot(index=index, columns=columns, values=values),
                    sort=True)
            return out
        else:  # execute as single chunk
            where += " LIMIT " + str(limit) if limit else ''
            q = f"SELECT {index}, {columns}, {values} FROM {table} {where}"
            if self._verbose:
                print('(pivot)', q)
            return self.read_dataframe(q).pivot(
                index=index, columns=columns, values=values)

class Redis:
    """Provide DataFrames interface with parquet to redis key-value store

    Args:
       host: Hostname
       port: Port number
       charset: Character set
       decode_responses: Set to False to zlib dataframe

    Attributes:
        redis: Redis client instance providing interface to all Redis commands

    Redis built-in methods:

        - r.delete(key)      -- delete an item
        - r.get(key)         -- get an item
        - r.exists(key)      -- does item exist
        - r.set(key, value)  -- set an item
        - r.keys()           -- get keys

    Examples:
        ::

            $ ./redis-5.0.4/src/redis-server
            $ ./redis-cli --scan --pattern '*CRSP_2020*' | xargs ./redis-cli del
            CLI> keys *
            CLI> flushall
            CLI> info memory
    """

    def __init__(self, host: str, port: int, charset: str = 'utf-8',
                 decode_responses: bool = False, **kwargs):
        """Open a Redis connection instance"""
        self.redis = redis.StrictRedis(host=host, port=port, charset=charset,
                                       decode_responses=decode_responses,
                                       **kwargs)
        
    def dump(self, key: str, df: DataFrame):
        """Saves dataframe, serialized to parquet, by key name to redis

        Args:
            key: Name of key in the store
            df: DataFrame to store, serialized with to_parquet
        """
        #self.r.set(key, pa.serialize(df).to_buffer().to_pybytes())
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype('string')  # parquet fails object
        self.redis.set(key, df.to_parquet())

    def load(self, key: str) -> DataFrame:
        """Return and deserialize dataframe given its key from redis store

        Args:
            key: Name of key in the store
        """
        df = pd.read_parquet(io.BytesIO(self.redis.get(key)))
        return df.copy()   # return copy lest flag.writable is False


class MongoDB:
    """Provides convenience interface to pymongo database

    Args:
        database: Name of database in MongoDB
        host: IP address of server
        port: Port number

    Attributes:
        client: MongoClient instance providing MongoDB interface

    Examples:
        >>> mdb = MongoDB()
        >>> serverStatusResult = mdb.client.admin.command("serverStatus")
        >>> pprint(serverStatusResult)
        >>> collections = mdb.client['database'].list_collection_names()
        >>> mdb.client[database][collections[0]].estimated_document_count()

    Other pymongo MongoClient methods for a collection object:

    - count_documents(self, filter, session=None, limit=None)
    - create_index(self, keys, unique=False)
    - create_indexes(self, indexes)
    - delete_one(self, filter)
    - distinct(self, key, filter=None)
    - drop(self)
    - drop_index(self, index_or_name)
    - drop_indexes(self)
    - estimated_document_count(self)
    - find(self, filter={}, projection=[], limit=None)
    - find_one(self, filter=None)
    - insert_many(self, documents, ordered=True)
    - insert_one(self, document)
    - list_indexes(self)
    - replace_one(self, filter, replacement, upsert=False)
    - update_many(self, filter, update, upsert=False)
    - update_many(self, filter, update, upsert=False)
    - update_one(self, filter, update, upsert=False)

    MongoDB Operators:
    ::

    $eq     Matches values that are equal to a specified value.
    $gt     Matches values that are greater than a specified value.
    $gte    Matches values that are greater than or equal to a specified value.
    $in     Matches any of the values specified in an array.
    $lt     Matches values that are less than a specified value.
    $lte    Matches values that are less than or equal to a specified value.
    $ne     Matches all values that are not equal to a specified value.
    $nin    Matches none of the values specified in an array.
    $and    Joins query clauses with a logical AND
    $not    Inverts the effect of a query expression
    $nor    Joins query clauses with a logical NOR returns
    $or     Joins query clauses with a logical OR returns
    $exists Matches documents that have the specified field.
    $type   Selects documents if a field is of the specified type.

    Unix Installation:

    https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/

    ::

        sudo systemctl start mongod
        sudo systemctl restart mongod
        sudo systemctl stop mongod
        sudo systemctl enable mongod
        sudo service mongod stop
        sudo apt-get purge mongodb-org* 

    /etc/mongod.conf - configuration file for MongoDB:

    - dbPath -  where the database files stored (/var/lib/mongodb)
    - systemLog - logging options (/var/log/mongodb/mongod.log)
    """

    def __init__(self, host: str = 'localhost', port: int = 27017, 
                 verbose = _VERBOSE):
        self.client = MongoClient(host=host, port=port)
        if verbose:
            result = self.client.admin.command("serverStatus")
            print(result)

    def show(self, database: str = ''):
        """List all database or collection (table) names

        Args:
            database: List collections in (blank to list all databases)
        """
        if not database:
            return self.client.list_database_names()
        return self.client[database].list_collection_names()

    def drop(self, database: str, collection: str = ''):
        """Drop a database or collection (table) by name

        Args:
            database: Name of database to drop collection
            collection: Name of collection to drop (blank to drop database)
        """
        if not collection:
            self.client.drop_database(database)
        self.client[database][collection].drop()


if __name__ == "__main__":
    #    from os.path import dirname, abspath
    #    sys.path.insert(0, dirname(dirname(abspath(__file__))))
    from conf import credentials, VERBOSE
    VERBOSE = 0

    def test_mdb():
        mdb = MongoDB(verbose = VERBOSE)
        mdb.show()
        db = mdb.client['test']
        if 'test' not in db.list_collection_names():
            db.create_collection('test')
        c = db['collection']
        c.insert_one({'hello': 'world'})
        found = c.find_one({'hello' : {'$exists' : True}})
        print(found)

    def test_rdb():
        rdb = Redis(**credentials['redis'])
        df = DataFrame(data=[[1, 1.5, 'a'], [2, '2.5', None]],
                    columns=['a', 'b', 'c'],
                    index=['d', 'e'])
        rdb.dump('my_key', df)
        print(rdb.load('my_key'))

    def update_sql(*databases):
        """Create initial raw and user databases"""
        query = "mysql+pymysql://{user}:{password}@{host}:{port}"\
            .format(**credentials['sql'])
        engine = sqlalchemy.create_engine(query)
        for database in databases:
            print('Creating Database: ' + database)
            engine.execute("CREATE DATABASE " + database)

    def test_sql():
        sql = SQL(**credentials['sql'], verbose=VERBOSE)
        print(sql.run('show tables'))
        user = SQL(**credentials['user'], verbose=VERBOSE)
        print(user.run('show tables'))

        df = DataFrame(data=[[1, 1.5, 'a'], [2, '2.5', None]],
                       columns=['a', 'b', 'c'],
                       index=['d', 'e'])
        user.run('drop table if exists test')
        user.load_dataframe('test', df)
        s = user.run('select * from test')
        print('test:', s)


    test_sql()
        
    """Update
    update_sql(credentials['sql']['database']))
    """
