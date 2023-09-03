"""Stocks subclass for stocks datasets

Copyright 2022, Terence Lim

MIT License
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_list_like, is_integer_dtype
from sqlalchemy import Table
from finds.busday import BusDay
from finds.database.sql import SQL
from finds.database.redisdb import RedisDB
from .structured import Structured
_VERBOSE = 1

class Stocks(Structured):
    """Provide interface to structured stock price datasets"""

    def __init__(self,
                 sql: SQL, 
                 bd: BusDay, 
                 tables: Dict[str, Table], 
                 identifier: str, name: str, 
                 rdb: RedisDB | None = None,
                 verbose: int = _VERBOSE):   
        """Initialize a connection to Stocks structured datasets"""
        super().__init__(sql, bd, tables, identifier=identifier, name=name,
                         rdb=rdb, verbose=verbose)

    def get_series(self,
                   permnos: int | str | List[str | int], 
                   field: str = 'ret', 
                   date_field: str = 'date',
                   dataset: str = 'daily', 
                   beg: int = 19000000, 
                   end: int = 29001231) -> DataFrame | Series:
        """Return time series of a field for multiple permnos as DataFrame

        Args:
            permnos: Identifiers to filter
            field: Name of column to extract
            beg: Inclusive start date (YYYYMMDD)
            end: Inclusive end date (YYYYMMDD)
            dataset: Name of dataset to retrieve `ret` (default is `daily`)

        Returns:
            DataFrame indexed by date with permnos in columns
        """
        assert self[dataset] is not None
        if isinstance(permnos, (int, str)) :
            q = ("SELECT {date_field}, {field}"
                 "  FROM {table}"
                 "  WHERE {date_field} >= {beg} AND {date_field} <= {end} "
                 "    AND {permno} = '{permnos}'").format(
                     permno=self.identifier,
                     field=field,
                     date_field=date_field,
                     table=self[dataset].key,
                     beg=int(beg),
                     end=int(end),
                     permnos=permnos)
            self._print('(get_series single)', q)
            return self.sql.read_dataframe(q)\
                .set_index(date_field)[field].sort_index().rename(permnos)
        else:
            q = ("SELECT {date_field}, {permno}, {field} "
                 "  FROM {table}"
                 "  WHERE {date_field} >= {beg} AND {date_field} <= {end} "
                 "    AND {permno} IN ('{permnos}')").format(
                     permno=self.identifier,
                     field=field,
                     date_field=date_field,
                     table=self[dataset].key,
                     beg=int(beg),
                     end=int(end),
                     permnos="', '".join([str(p) for p in permnos]))
            self._print('(get_series many)', q)
            return self.sql.read_dataframe(q)\
                    .pivot(index='date', 
                           columns=self.identifier, 
                           values=field)[permnos].sort_index()

    def get_ret(self,
                beg: int, 
                end: int, 
                dataset: str = 'daily', 
                field: str = 'ret', 
                date_field: str = 'date',
                cache_mode: str = 'rw') -> Series:
        """Compounded returns between beg and end dates of all stocks

        Args:
            beg: Inclusive start date (YYYYMMDD)
            end: Inclusive end date (YYYYMMDD)
            dataset: Name of dataset to retrieve (default is `daily`)
            field: Name of returns field
            date_field: Name of date field
            cache_mode: 'r' to try read from cache first, 'w' to write to cache

        Series:
            DataFrame with prod(min_count=1) of returns in column `ret`, 
            with rows indexed by permno
        """
        rkey = "_".join([field, str(self), str(beg), str(end)])
        if 'r' in cache_mode and self.rdb and self.rdb.redis.exists(rkey):
            self._print('(get_ret load)', rkey)
            return self.rdb.load(rkey)[field]    # use cache

        q = ("SELECT {field}, {identifier} FROM {table} "
             " WHERE date >= {beg} AND date <= {end}").format(
                 table=self[dataset].key,
                 field=field,
                 identifier=self.identifier,
                 beg=beg,
                 end=end)
        self._print('(get_ret)', q)
        df = self.sql.read_dataframe(q).sort_values(self.identifier)

        # compute compounded returns
        df[field] += 1
        df = (df.groupby(self.identifier).prod(min_count=1)-1).dropna()

        if 'w' in cache_mode and self.rdb and beg != end:  # if write cache
            self._print('(get_ret dump)', rkey)
            self.rdb.dump(rkey, df)
        return df[field]

    def get_compounded(self,
                       periods: List[Tuple[int, int]], 
                       permnos: List[int], 
                       cache_mode: str = "rw") -> DataFrame:
        """Compound returns within list of periods, for given permnos

        Args:
            periods: Tuples of inclusive begin and end dates of returns period
            permnos: List of permnos
            cache_mode: 'r' to try read from cache first, 'w' to write to cache

        Returns:
            DataFrame of compounded returns in rows, for permnos in cols
        """
        # accumulate horizontally, then finally transpose
        r = DataFrame(index=permnos)
        for beg, end in periods:
            r[end] = self.get_ret(beg, end, cache_mode=cache_mode)\
                         .reindex(permnos)
        return r.transpose()

    def cache_ret(self,
                  dates: List[Tuple[int, int]], 
                  replace: bool, 
                  field: str = 'ret', 
                  date_field: str ='date',
                  dataset: str = 'daily'):
        """Pre-generate compounded returns from daily for redis store"""
        assert self.rdb is not None
        q = ("SELECT {field}, {identifier}, {date_field} FROM {table} "
             " WHERE {date_field} >= {beg} "
             "   AND {date_field} <= {end}").format(
                 table=self[dataset].key,
                 field=field,
                 identifier=self.identifier,
                 date_field=date_field,
                 beg=dates[0][0],
                 end=dates[-1][-1])
        self._print('(cache_ret)', q)
        rets = self.sql.read_dataframe(q).sort_values(self.identifier)
        rets[field] += 1
        
        for beg, end in dates:
            rkey = "_".join([field, str(self), str(beg), str(end)])
            if not replace and self.rdb.redis.exists(rkey):
                self._print('(cache_ret exists)', rkey)
                continue
            df = rets[rets['date'].ge(beg) & rets['date'].le(end)]\
                 .drop(columns='date')
            df = (df.groupby(self.identifier).prod(min_count=1) - 1).dropna()
            self._print('(cache_ret dump)', rkey, beg, end, len(df))
            self.rdb.dump(rkey, df)
    
    
    def get_window(self,
                   dataset: str, 
                   field: str, 
                   permnos: List[Any], 
                   date_field: str, 
                   dates: List[int], 
                   left: int, 
                   right: int, 
                   avg: bool = False) -> DataFrame:
        """Retrieve field values for permnos in window centered around dates

        Args:
            dataset: Name of dataset
            field: Name of field to retrieve
            permnos: List of identifiers to retrieve
            date_field: Name of date field in database
            dates : List of corresponding dates of center of event window
            left : Relative (inclusive) offset of start of event window
            right : Relative (inclusive) offset of end of event window

        Returns:
            DataFrame columns [0:(right-left)] of field values in event window
        """
        dates = list(dates)
        permnos = list(permnos)
        if avg:
            # Generate and save dates to sql temp
            df = DataFrame({'a': self.bd.offset(dates, left),
                            'b': self.bd.offset(dates, right),
                            self.identifier: permnos},
                           index=np.arange(len(dates)))
            self.sql.load_dataframe(table=self.sql._t,
                                    df=df,
                                    index_label='n',
                                    replace=True)
            if is_integer_dtype(df[self.identifier].dtype):
                q = f"CREATE INDEX a on {self.sql._t} ({self.identifier},a,b)"
                self.sql.run(q)
                q = f"CREATE INDEX b on {self.sql._t} ({self.identifier},b,a)"
                self.sql.run(q)

            # join on (permno, date) and retrieve from target table
            q = ("SELECT {temp}.n, "
                 " AVG({field}) as {field} FROM {temp} LEFT JOIN {table}"
                 " ON {temp}.{identifier} = {table}.{identifier} "
                 " WHERE {table}.{date_field} >= {temp}.a "
                 " AND {table}.{date_field} <= {temp}.b"
                 " GROUP BY {temp}.n").format(
                     temp=self.sql._t,
                     identifier=self.identifier,
                     field=field,
                     date_field=date_field,
                     table=self[dataset].key)
            df = self.sql.read_dataframe(q).drop_duplicates(subset=['n'])\
                                           .set_index('n')
            result = DataFrame({'permno': permnos, 'date': dates},
                               index=np.arange(len(dates)))\
                        .join(df, how='left')
        else:
            # Generate and save dates to sql temp
            cols = ["day" + str(i) for i in range(1 + right - left)]
            df = DataFrame(data=self.bd.offset(dates, left, right), 
                           columns=cols)
            df[self.identifier] = permnos
            self.sql.load_dataframe(self.sql._t, df, replace=True)

            # Loop over each date, and join as columns of result
            result = DataFrame({'permno': permnos, 'date': dates})
            for col in cols:
                # create index on date to speed up join with target table
                if is_integer_dtype(df[self.identifier].dtype):
                    q = "CREATE INDEX {col} on {temp} ({ident}, {col})".format(
                        temp=self.sql._t, ident=self.identifier, col=col)
                    self.sql.run(q)

                # join on (permno, date) and retrieve from target table
                q = ("SELECT {temp}.{identifier}, {field}"
                     " FROM {temp} LEFT JOIN {table}"
                     " ON {table}.{identifier} = {temp}.{identifier} "
                     "  AND {table}.{date_field} = {temp}.{col}").format(
                         temp=self.sql._t,
                         identifier=self.identifier,
                         field=field,
                         date_field=date_field,
                         table=self[dataset].key,
                         col=col)
                df = self.sql.read_dataframe(q)
                # left join, so assume same order
                result[col] = df[field].values
        self.sql.run('drop table if exists ' + self.sql._t)
        result.columns = [int(c[3:]) if c.startswith('day') else c
                          for c in result.columns]
        return result.reset_index(drop=True)

    def get_many(self,
                 dataset: str, 
                 permnos: List[str | int], 
                 fields: List[str], 
                 date_field: str, 
                 dates: List[int], 
                 exact: bool = True) -> DataFrame:
        """Retrieve multiple fields for lists of permnos and dates

        Args:
            dataset: Name of dataset
            permnos: List of identifiers to retrieve
            dates: List of corresponding dates of center of event window
            field: Names of fields to retrieve
            date_field: Names of date field in database
            exact: Whether require exact date match, or allow most recent

        Returns:
            DataFrame with permno, date, and retrieved fields across columns
        """
        field = "`, `".join(list(fields))
        self.sql.load_dataframe(table=self.sql._t,
                                df=DataFrame({self.identifier: list(permnos),
                                              'date': list(dates)},
                                             index=np.arange(len(permnos))),
                                index_label='_seq',
                                replace=True)
        if exact:
            q = ("SELECT {temp}._seq, {temp}.{identifier}, "
                 "  {temp}.date AS date, `{field}` "
                 "  FROM {temp} LEFT JOIN {table}"
                 "    ON {table}.{identifier} = {temp}.{identifier} "
                 "    AND {table}.{date_field} = {temp}.date").format(
                     temp=self.sql._t,
                     identifier=self.identifier,
                     date_field=date_field,
                     field=field,
                     table=self[dataset].key)
            df = self.sql.read_dataframe(q).set_index('_seq').sort_index()
            df.index.name = None
        else:
            q = ("SELECT {temp}._seq, {temp}.{identifier}, "
                 "  {temp}.date AS date, `{field}` "
                 "  FROM {temp} LEFT JOIN {table}"
                 "    ON {table}.{identifier} = {temp}.{identifier} "
                 "    AND {table}.{date_field} <= {temp}.date").format(
                     temp=self.sql._t,
                     identifier=self.identifier,
                     field=field,
                     date_field=date_field,
                     table=self[dataset].key)
            df = self.sql.read_dataframe(q)\
                         .sort_values(['_seq', 'date'], na_position='first')\
                         .drop_duplicates(subset=['_seq'], keep='last')\
                         .set_index('_seq').sort_index()
        self.sql.run('drop table if exists ' + self.sql._t)
        return df

    def get_section(self,
                    dataset: str, 
                    fields: List[str], 
                    date_field: str,
                    date: int, 
                    start: int = -1) -> DataFrame:
        """Return a cross-section of values of fields as of a single date

        Args:
            dataset: Dataset to extract from
            fields: list of columns to return
            date_field: Name of date column in the table
            date: Desired date in YYYYMMDD format
            start: Non-inclusive date of starting range; if -1 then exact date

        Returns:
            Most recent row within date range, indexed by permno

        Note:

        - If start is not -1, then the latest prevailing record for each
          between (non-inclusive) start and (inclusive) date is returned

        Examples:

        >>> t = crsp.get_section('shares', ['shrenddt','shrout'], 'shrsdt', dt)
        >>> u = crsp.get_section('names', ['comnam'], 'date', dt-10000)
        """

        assert is_list_like(fields)
        if self.identifier not in fields:
            fields += [self.identifier]
        if start < 0:
            q = ("SELECT {fields} FROM {table} "
                 " WHERE {date_field} = {date}").format(
                     fields=", ".join(fields),
                     table=self[dataset].key,
                     date_field=date_field,
                     date=date)
        else:
            q = ("SELECT {fields} FROM {table} JOIN"
                 "  (SELECT {permno}, MAX({date_field}) AS {date_field} "
                 "   FROM {table} "
                 "     WHERE {date_field} <= {date} AND {date_field} > {start}"
                 "     GROUP BY {permno}) as a "
                 "  USING ({permno}, {date_field})").format(
                     fields=", ".join(fields),
                     table=self[dataset].key,
                     permno=self.identifier,
                     date_field=date_field,
                     date=date,
                     start=start)
        self._print('(get_section)', q)
        return self.sql.read_dataframe(q).set_index(self.identifier)

    def get_range(self,
                  dataset: str, 
                  fields: List[str] | Dict[str, str],
                  date_field: str, 
                  beg: int, 
                  end: int,
                  cache_mode: str = "rw") -> DataFrame:
        """Return field values within a date range

        Args:
            dataset: Name of dataset to extract from
            fields: Names of columns to return (and optionally rename as)
            date_field: Name of date column in the table
            beg: Inclusive start date in YYYYMMDD format
            end: Inclusive end date in YYYYMMDD format
            cache_mode: 'r' to try read from cache first, 'w' to write to cache

        Returns:
            DataFrame multi-indexed by permno, date
        """
        assert(fields)
        if isinstance(fields, dict):
            rename = fields
            fields = list(fields.keys())
        else:
            rename = {k:k for k in fields}
        if self.identifier not in fields:
            fields += [self.identifier]

        rkey = f"{str(self)}_{'_'.join(fields)}_{beg}_{end}"

        if self.rdb and 'r' in cache_mode and self.rdb.redis.exists(rkey):
            self._print('(get_range load)', rkey)
            return self.rdb.load(rkey)
        q = ("SELECT {fields}, {date_field} FROM {table} WHERE "
             " {date_field} >= {beg} AND {date_field} <= {end}").format(
                 fields=", ".join(fields),
                 table=self[dataset].key,
                 date_field=date_field,
                 beg=beg,
                 end=end)
        self._print('(get_range)', q)
        r = self.sql.read_dataframe(q).set_index([self.identifier, date_field])
        r = r.rename(columns=rename) if rename else r.iloc[:,0]
        if 'w' in cache_mode and self.rdb:
            self._print('(get_range dump)', rkey)
            self.rdb.dump(rkey, r)
        return r


class StocksBuffer(Stocks):
    """Cache daily returns into memory, and provide Stocks-like interface"""
    
    def __init__(self,
                 stocks: Stocks, 
                 beg: int, 
                 end: int,
                 dataset: str = 'daily',
                 fields: List[str] = ['ret', 'retx'], 
                 identifier: str = 'permno'):
        """Create object and load daily returns into its cache

        Args:
            stocks: Stocks structured data object to access stock returns data
            beg: Earliest date of daily stock returns to pre-load
            end: Latest date of daily stock returns to pre-load
            fields: Column names of returns fields to load
            dataset: Name of dataset to extract from
            identifier: Field name of stocks identifier
        """
        self.fields = fields
        self.identifier = identifier
        self.bd = stocks.bd
        self._dataset = dataset
        q = (f"SELECT permno, date, {', '.join(fields)} "
             f"  FROM {stocks[dataset].key}"
             f"  WHERE date>={beg} AND date<={end}")
        self.rets = stocks.sql.read_dataframe(q).sort_values(['permno', 'date'])

    def get_section(self,
                    dataset: str, 
                    fields: List[str], 
                    date_field: str,
                    date: int, 
                    start: int = -1) -> DataFrame:
        """Return a cross-section of values of fields as of a single date

        Args:
            dataset: Dataset to extract from
            fields: list of columns to return
            date_field: Name of date column in the table
            date: Desired date in YYYYMMDD format
            start: Non-inclusive date of starting range (ignored)

        Returns:
            Most recent row within date range, indexed by permno
        """
        df = self.rets[self.rets['date'].eq(date)]\
                 .drop(columns=['date'])\
                 .set_index('permno')\
                 .dropna()
        return df[fields]
        
    def get_ret(self, beg: int, end: int, field: str = 'ret') -> Series:
        """Return compounded stock returns between beg and end dates

        Args:
            beg: Begin date to compound returns
            end: End date (inclusive) to compound returns
            field: Name of returns field in dataset, in {'ret', 'retx')
        """
        df = self.rets.loc[self.rets['date'].between(beg, end),
                           [self.identifier, field]].dropna()
        df.loc[:, field] += 1
        df = (df.groupby(self.identifier).prod(min_count=1) - 1).fillna(0)
        return df[field]


class StocksFrame(Stocks):
    """Mimic Stocks object given an input DataFrame of returns
    
    Args:
        df: DataFrame of returns with date in index and permno in columns
        rsuffix: replicate output columns and append rsuffix to column name
        identifier: name of identifier column

    Notes:

    - limited interface to manipulate DataFrame of asset returns as Stocks-like
    """

    class bd:
        """Class to mimic basic behavior of BusDay object"""
        @staticmethod
        def begmo(date: int | List[int]) -> int | List[int]:
            """Returns same date"""
            return date

        @staticmethod
        def endmo(date: int | List[int]) -> int | List[int]:
            """Returns same date"""
            return date

        @staticmethod
        def date_tuples(dates: List[int]) -> List[Tuple[int, int]]:
            """Returns adjacent dates as the holding date tuples"""
            return list(zip(dates[1:], dates[1:]))

    def __init__(self,
                 df: DataFrame, 
                 rsuffix: str = '', 
                 identifier: str = 'permno'):
        self.data = DataFrame(df)
        if rsuffix is not None:
            self.data = self.data.join(self.data, how='left', rsuffix=rsuffix)
        self.identifier = identifier

    def get_series(self, permnos: str | int | List[str | int],
                   *arg, **kwarg) -> Series:
        """Return the series for target permnos"""
        return self.data[permnos]

    def get_ret(self, beg: int, end: int, *args, **kwargs) -> Series:
        """Compounded returns between beg and end (inclusive) dates"""
        df = DataFrame((self.data.loc[(self.data.index >= beg)
                                      & (self.data.index <= end)] + 1).prod() - 1)
        df.columns = ['ret']
        df.index.name = self.identifier
        return df['ret']

