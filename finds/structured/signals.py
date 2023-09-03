"""Signals implements structured dataset interface for derived signal values

Copyright 2022, Terence Lim

MIT License
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_list_like, is_integer_dtype
from sqlalchemy import Table, Column, Index, Integer, Float
from finds.database.sql import SQL
from .stocks import Stocks
from .structured import as_dtypes
_VERBOSE = 1

class Signals(Stocks):
    """Provide structured stocks data interface to derived signal values 

    Args:
        sql: Connection to SQL database
    """
    def __init__(self, sql: SQL, verbose=_VERBOSE):
        """Initalize a connection to derived Signals values datasets"""
        super().__init__(sql=sql, bd= None, tables={}, identifier='permno', 
                         name='signals', verbose=verbose)

    def __call__(self,
                 label: str, 
                 date: int, 
                 start: int = -1, 
                 rebaldate: str = 'rebaldate') -> DataFrame:
        """Return cross-section of signal values available as of a date

        Args:
            label: Name of signal to retrieve
            date : Rebalance date
            start : Non-inclusive start of date range; -1 means exact date
            rebaldate: Name of rebalance date column

        Returns:
            DataFrame of signal values prevailing as of input date
        """
        return self.get_section(dataset=label, fields=[rebaldate, label],
                date_field=rebaldate, date=date, start=start)

    def table_key(self, label: str) -> str:
        """Helper method generates a table key name for the input label"""
        return '__' + label     # prefix with "__"

    def __getitem__(self, label) -> Table:
        """Overrides parent class method to get Table schema of label"""
        return self.sql.Table(self.table_key(label),
                              Column('permno', Integer, primary_key=True),
                              Column('rebaldate', Integer, primary_key=True),
                              Column(label, Float))

    def summary(self, label: str) -> DataFrame:
        """Perform a 'proc summary' by rebaldate on a signal's values"""
        return self.sql.summary(self.table_key(label), label, key='rebaldate')

    def write(self,
              data: DataFrame,
              label: str,
              overwrite: bool = True, 
              rebaldate: str = 'rebaldate',
              permno: str = 'permno') -> int:
        """Saves a new sql table from dataframe of signal values

        Args:
            data: Signal values, with columns ['permno', 'rebaldate', label]
            label: Signal name of column and table (prefixed '__')
            overwrite: If False, append to table ignoring dups. Else recreate
            rebaldate: Column name of rebalance dates in input dataframe
            permno: Column name of permno identifiers in input dataframe

        Returns:
            Number of rows saved

        Notes:

        - first removes dup keys, then drops null rows before saving to table
        """

        df = data[[permno, rebaldate, label]].copy()
        df.index.name = None # 'permno' may be both index level or column label
        df = df.rename(columns={permno: 'permno', rebaldate: 'rebaldate'})
        table = self[label]
        df = as_dtypes(df=df, columns={k.lower(): v.type
                                       for k, v in table.columns.items()})
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.sort_values(by=['permno', 'rebaldate', label])
        df.drop_duplicates(['permno', 'rebaldate'], keep='first', inplace=True)
        df = df.dropna()        # NaN's last
        if overwrite:
            table.drop(self.sql.engine, checkfirst=True)
        table.create(self.sql.engine, checkfirst=True)
        #self.sql.create_all()
        self.sql.load_dataframe(table=table.key, df=df, index_label=None)
        self._print("(signals_write)", label, len(df))
        return len(df)

    def read(self, label: str, where: str = '') -> DataFrame:
        """Read signal values from sql and return as data frame

        Args:
            label: Name of signal
            where: Where clause for sql select

        Returns:
            DataFrame of query with columns = ['permno', 'rebaldate', label]
        """
        if where:
            where = 'WHERE' + where
        table = self.table_key(label)
        q = f"SELECT permno, rebaldate, {label} FROM {table} {where}"
        return self.sql.read_dataframe(q).sort_values(['permno', 'rebaldate'])


class SignalsFrame(Signals):
    """Cache dataframe of signals values, provide Signals-like interface"""

    def __init__(self, df: DataFrame, identifier: str = 'permno'):
        """Initialize instance from input dataframe"""
        self.data = df
        self.identifier = identifier

    def __call__(self,
                 label: str, 
                 date: int, 
                 start: int = -1, 
                 rebaldate: str = 'rebaldate') -> DataFrame:
        """Select from rebaldates that fall between start and date, keep latest

        Args:
            label: Name of column to return
            date: As of this date or possibly earlier
            start: Non-inclusive start date. Set to 0 for all, -1 for exact 
            rebaldate: Column name containing rebaldate
        """
        if start < 0:
            start = date - 1
        df = self.data.loc[self.data[rebaldate].le(date)
                           & self.data[rebaldate].gt(start),
                           [self.identifier, rebaldate, label]]
        df = df.sort_values([self.identifier, rebaldate], na_position='first')\
               .drop_duplicates([self.identifier], keep='last')\
               .dropna()
        return df.set_index(self.identifier)
