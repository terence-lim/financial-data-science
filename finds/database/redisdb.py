"""Redis class wrapper, with convenience methods for pandas DataFrames

Copyright 2022, Terence Lim

MIT License
"""
from typing import List, Dict, Mapping, Any
import io
import random
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import redis
from .database import Database

class RedisDB(Database):
    """Interface to redis, with convenience functions for dataframes

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

    def __init__(self,
                 host: str,
                 port: int,
                 charset: str = 'utf-8',
                 decode_responses: bool = False,
                 **kwargs):
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

if __name__ == "__main__":
    from env.conf import credentials
    
    rdb = RedisDB(**credentials['redis'])
    df = DataFrame(data=[[1, 1.5, 'a'], [2, '2.5', None]],
                   columns=['a', 'b', 'c'],
                   index=['d', 'e'])
    rdb.dump('my_key', df)
    print(rdb.load('my_key'))
