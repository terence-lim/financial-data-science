"""Benchmarks dataset for index returns

Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sqlalchemy import Table, Column, Index
from sqlalchemy import Integer, String, Float, SmallInteger, Boolean, BigInteger
from finds.database.sql import SQL
from finds.database.redisdb import RedisDB
from finds.structured.busday import BusDay
from finds.structured.stocks import Stocks
_VERBOSE = 1

class Benchmarks(Stocks):
    """Provide Structured Stocks interface to benchmark and index returns"""

    def __init__(self, sql: SQL, bd: BusDay, verbose: int = _VERBOSE):
        """Initialize connection to a benchmark index returns dataset"""
        tables = {
            'daily': sql.Table('benchmarks',
                               Column('permno', String(32), primary_key=True),
                               Column('date', Integer, primary_key=True),
                               Column('ret', Float)),
            'ident': sql.Table('benchident',
                               Column('permno', String(32), primary_key=True),
                               Column('name', String(64)),
                               Column('item', String(8)))}
        tables['monthly'] = tables['daily']
        super().__init__(sql, bd, tables, identifier='permno',
                         name='benchmarks', verbose=verbose)
    
    def load_series(self, df: DataFrame, name: str, item: str = '', 
                    monthly: bool = False) -> DataFrame:
        """Loads a Series containing benchmark returns to sql

        Args:
            df : DataFrame with time-series in each column to load to sql
            name: Primary label for this source to insert into ident table
            item: Secondary label for this source to insert into ident table
            monthly: if True: convert index to business calendar endmo dates

        Returns:
            DataFrame of identifiers metadata for series successfully loaded

        Notes:

        - Each column of input data frame is loaded to sql table 'daily',
          with its series name as 'permno' field, values as 'ret' field,
          and series index as 'date' field.
        - 'idents' table in sql is also updated with identifier and metadata    
        """
        self.sql.create_all()
        #self['daily'].create(checkfirst=True)
        permno = df.name
        df = df.rename('ret').to_frame()
        df['permno'] = permno
        self.sql.run(self['daily'].delete().where(self['daily'].c['permno'] == permno))
        self.sql.load_dataframe(self['daily'].key, df=df, index_label='date')
        
        #self['ident'].create(checkfirst=True)
        self.sql.run(self['ident'].delete().where(self['ident'].c['permno'] == permno))
        ident = DataFrame.from_dict({0: {'permno': permno, 'name': name, 'item':item}},
                                    orient='index')
        self.sql.load_dataframe(self['ident'].key, df=ident)
        return ident

if __name__ == "__main__":
    from secret import credentials
    from finds.readers import FFReader
    VERBOSE = 1

    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    user = SQL(**credentials['user'], verbose=VERBOSE)
    rdb = RedisDB(**credentials['redis'])
    bd = BusDay(sql)
    bench = Benchmarks(sql, bd)

    downloads = paths['data'] / 'CRSP'
    df = pd.read_csv(downloads / 'sp500.txt.gz', header=0, sep='\t').set_index('caldt')
    for col in df.columns:
        print(bench.load_series(df[col], name=col))
    
    # load benchmarks (mostly FamaFrench)
    for datasets, date_formatter in zip(
            [FFReader.monthly, FFReader.daily],[bd.endmo, bd.offset]):
        for name, item, suffix in datasets:
            df = FFReader.fetch(name=name, 
                                item=item,
                                suffix=suffix,
                                date_formatter=date_formatter)
            for col in df.columns:
                print(bench.load_series(df[col], name=name, item=str(item)))
            print(DataFrame(**sql.run('select * from ' + bench['ident'].key)))

    print(bench.get_series('CMA', 'ret'))
    print(bench.get_series(['CMA', 'HML'], 'ret'))
