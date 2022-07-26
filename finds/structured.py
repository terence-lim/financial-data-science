"""Classes to implement interface for structured data sets

- CRSP (daily, monthly, names, delistings, distributions, shares outstanding)
- S&P/CapitalIQ Compustat (Annual, Quarterly, Key Development, customers)
- IBES Summary

Redis store: SQL query results are (optionally) cached in in Redis

Signals class to store and retrieve derived signal values

Subclasses to mimic parent class interfaces with pre-loaded batch in memory

Lookup identifiers within and across data sets



Copyright 2022, Terence Lim

MIT License
"""
from typing import Iterable, List, Dict, Mapping, Any, Callable, Tuple
import random
import sys
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_list_like, is_integer_dtype
from sqlalchemy import Table, Column, Index, Integer, String, Float, \
    SmallInteger, Boolean, BigInteger
from datetime import datetime
from finds.database import SQL, Redis
from finds.busday import BusDay, to_date
from finds.recipes import fractiles
from finds.busday import to_datetime

_VERBOSE = 1

def as_dtypes(df: DataFrame, 
              columns: Dict, 
              drop_duplicates: List[str] = [], 
              sort_values: List[str] = [], 
              keep: str ='first',
              replace : Dict[str, Tuple[Any, Any]] = {}) -> DataFrame:
    """Convert DataFrame dtypes to the given sqlalchemy Column types

    Args:
        df: Input DataFrame to apply new data types from target columns
        columns: Target sqlalchemy column types as dict of {column: type}
        sort_values: List of column names to sort by
        drop_duplicates: list of fields if all duplicated to drop rows
        keep : 'first' or 'last' row to keep if drop duplicates
        replace : dict of {column label: tuple(old, replacement) values}

    Returns:
        DataFrame with columns and rows transformed

    Notes:

    - Columns of DataFrame are dropped if not specified in columns input
    - If input is None, then return empty DataFrame with given column types
    - Blank values in boolean and int fields are set to False/0.
    - Invalid/blank values in double field are coerced to NaN.
    - Invalid values in int field are coerced to 0
    """

    if df is None:
        df = DataFrame(columns=list(columns))
    df.columns = df.columns.map(str.lower).map(str.rstrip) # clean column names
    df = df.reindex(columns=list(columns))  # reorder and only keep columns
    if len(sort_values):
        df.sort_values(sort_values)
    if len(drop_duplicates):
        df.drop_duplicates(subset=drop_duplicates, keep=keep, inplace=True)
    for col, v in columns.items():
        if col in replace:
            df[col] = df[col].replace(*replace[col])
        if isinstance(v, Integer) or isinstance(v, SmallInteger):
            df[col] = df[col].replace('', 0).astype(int)
        elif isinstance(v, Boolean):
            df[col] = df[col].replace('', False).astype(bool)
        elif isinstance(v, Float):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        elif isinstance(v, String):
            df[col] = df[col].astype(str).str.encode(
                'ascii', 'ignore').str.decode('ascii')
        else:
            raise Exception('Unknown type for column ' + col)
    return df


class Structured(object):
    """Base class for interface to structured datasets, stored in SQL

    Args:
        sql: Connection instance to mysql database
        bd: Custom business calendar instance
        tables: Sqlalchemy Tables and names defined for this datasets group
        identifier: Name of field of unique identifier key
        name: Display name for this datasets group
        rdb: Connector to Redis cache store, if desired

    Attributes:
        identifier: Field name of identifier key by this datasets group
        name: Display name for this datasets group
    """

    def __init__(self, sql: SQL, 
                       bd: BusDay, 
                       tables: Dict[str, Table], 
                       identifier: str,
                       name: str, 
                       rdb: Redis | None = None, 
                       verbose: int = _VERBOSE):
        """Initialize a connection to structured datasets"""
        self.bd = bd
        self.sql = sql
        self.tables_ = tables
        self.rdb = rdb
        self.identifier = identifier
        self.name = name
        self._verbose = verbose
        
    def _print(self, *args, verbose: int = _VERBOSE, level: int = 0, **kwargs):
        """helper to print verbose messages"""
        if max(verbose, self._verbose) > 0:
            print(*args, **kwargs)

    def __str__(self) -> str:
        """String to identify this class of datasets group"""
        return self.name

    def __getitem__(self, dataset: str) -> Table:
        """Return the table object corresponding to a dataset name"""
        assert dataset in self.tables_
        return self.tables_[dataset]

    def create_all(self):
        """Create all tables and indexes in SQL using associated schemas"""
        if self.tables_:
            for table in self.tables_.values():
                table.create(checkfirst=True)

    def drop_all(self):
        """Drop all associated tables from SQL database"""
        if self.tables_:
            for table in self.tables_.values():
                table.drop()

    def load_dataframe(self, df: DataFrame, table: Table,
                       to_replace: Any = None, value: Any = None,
                       overwrite: bool = True) -> int:
        """Load dataframe to SQLAlchemy table object using associated schema

        Args:
            df: DataFrame to load from
            table: Destination Table object
            to_replace: Original value or list of values to replace
            value: Value to replace with
            overwrite: Whether to overwrite or append

        Returns:
            Number of rows loaded

        Notes:

        - DataFrame should contain same column names as Table,
        - DataFrame columns types are converted to Table column types
        - Duplicate primary fields are removed, keeping first        
        """
        df = as_dtypes(df=df,
                       columns={k.lower(): v.type
                                for k, v in table.columns.items()})
        if to_replace is not None and value is not None:
            df = df.replace(to_replace, value)
        primary = table.primary_key.columns.keys()
        if primary:
            df = df.sort_values(by=primary)
            df.drop_duplicates(primary, keep='first', inplace=True)
        df = df.dropna()        # NaN's last
        if overwrite:
            table.drop(checkfirst=True)
        table.create(checkfirst=True)
        self.sql.load_dataframe(table=table.key, df=df, index_label=None)
        self._print("(structured store)", table.key, len(df))
        return len(df)
                
    def read_dataframe(self, table: str, where: str = '') -> DataFrame:
        """Read signal values from sql and return as data frame

        Args:
            table: Table to read from
            where: Where clause str for sql select

        Returns:
            DataFrame of query
        """
        where = bool(where)*'WHERE ' + where
        return self.sql.read_dataframe(f"SELECT * FROM {table} {where}")
    
    def load_csv(self, dataset: str, 
                       csvfile: str, 
                       drop: Dict[str, List[Any]] = {}, 
                       replace: Dict[str, Tuple[Any, Any]] = {}, 
                       sep: str = ',', 
                       encoding: str = 'latin-1', 
                       header: Any = 0, 
                       low_memory: bool = False, 
                       na_filter: bool = False, 
                       **kwargs) -> DataFrame:
        """Insert ignore into SQL table from csvfile, and return as DataFrame

        Args:
            dataset: dataset name
            csvfile: csv file name
            drop: {column: value} specifies rows with value in column
            replace: {column: [old,new]} specifies values to replace in column
            sep, encoding, header, low_memory, na_filter: args for pd.read_csv

        Returns:
            DataFrame containing loaded data

        Notes:

        - Create new table, if not exists, using associated schema
        - New records with duplicate key are dropped (insert ignore used)
        """

        # Load csv to DataFrame
        table = self[dataset]    # Table object for dataset
        assert table is not None
        df = pd.read_csv(csvfile, sep=sep, encoding=encoding, header=header,
                         low_memory=low_memory, na_filter=na_filter)
        # 'utf-8' codec can't decode byte 0xf6 => encoding='latin-1'
        df.columns = df.columns.map(str.lower).map(str.rstrip)
        self._print('(read_csv)', len(df), csvfile)

        # clean up column dtypes and rows
        for col, vals in drop.items():  # drop rows where col has value val
            rows = df.index[df[col].isin(vals)]
            self._print('Dropping', len(rows), 'rows with', col, 'in', vals)
            df.drop(index=rows, inplace=True)
        df = as_dtypes(
            df=df,
            columns={k.lower(): v.type for k, v in table.columns.items()},
            drop_duplicates=[p.key.lower() for p in table.primary_key],
            replace=replace)
        for col, vals in drop.items():  # drop rows where col has value in val
            df.drop(index=df.index[df[col].isin(vals)], inplace=True)
        self._print("(load_csv)", len(df), table)

        # Create sql table and load from DataFrame
        table.create(checkfirst=True)
        self.sql.load_dataframe(table=table.key, df=df, index_label=None)
        return df

    class Lookup:
        """Loads dated identifier mappings to memory, to lookup by date

            Args:
                sql: SQL connection instance
                source: Name of source identifier key
                target: Name of target identifier key to return
                date_field: Name of date field in database table
                table: Physical SQL table name containing identifier mappings
                fillna: Value to return if not found
        """
        def __init__(self, sql: SQL, source: str, target: str, 
                     date_field: str, table: str, fillna: Any):
            lookups = sql.read_dataframe(
                f"SELECT {source} as source,"
                f"  {target} AS target,"
                f"  {date_field} AS date"
                f"  FROM {table}"
                f"  WHERE {source} IS NOT NULL"
                f"    AND {target} IS NOT NULL")
            lookups = lookups.sort_values(['source', 'target', 'date'])
            try:
                lookups = lookups.loc[lookups['source'] > 0]
            except:
                lookups = lookups.loc[lookups['source'].str.len() > 0]
            self.lookups = lookups.groupby('source')
            self.keys = set(self.lookups.indices.keys())
            self.source = source
            self.target = target
            self.fillna = fillna

        def __call__(self, label: str, date: int = 99999999) -> Any:
            """Return target identifiers matched to source as of date"""
            if is_list_like(label):
                return [self(b) for b in label]
            if label in self.keys:
                a = self.lookups.get_group(label)
                b = a[a['date'] <= date]  # prevailing date, else first
                return (b.iloc[-1] if len(b) else a.iloc[0]).at['target']
            return self.fillna

        def __getitem__(self, labels):
            return self(labels)

    def build_lookup(self, source: str, target: str, date_field: str,
                     dataset: str, fillna: Any) -> Any:
        """Helper to build lookup of target from source identifiers"""
        table = self[dataset]    # Table object for dataset
        assert table is not None
        assert source in table.c
        assert target in table.c
        return self.Lookup(sql=self.sql, source=source, target=target,
                           date_field=date_field, table=table.key,
                           fillna=fillna)

    def get_permnos(self, keys: List[str], 
                          date: int, 
                          link_perm: str,
                          link_date: str, 
                          permno: str) -> DataFrame:
        """Returns matching permnos as of a prevailing date from 'links' table

        Args:
            keys: Input list of identifiers to lookup
            date: Prevailing date of link
            link_perm: Name of permno field in 'links' table
            link_date: Name of link date field in 'links' table
            permno: Name of field to output permnos to

        Returns:
            List of Linked permnos, as of prevailing date; missing set to 0
        """
        assert self['links'] is not None
        key = self.identifier
        s = ("SELECT {links}.{key},"
             "       {links}.{link_perm} AS {permno}, "
             "       {links}.{link_date} FROM"
             "  (SELECT {key}, MAX({link_date})"
             "    AS dt FROM {links} "
             "    WHERE {link_date} <= {date} "
             "      AND {link_perm} > 0 "
             "      GROUP BY {key}, {link_date}) AS a"
             "    INNER JOIN {links} "
             "      ON {links}.{key} = a.{key} "
             "         AND {links}.{link_date} = a.dt").format(
                key=self.identifier,
                date=date,
                links = self['links'].key,   # Table object for links dataset
                link_perm=link_perm,   
                link_date=link_date,   
                permno=permno)
        permnos = self.sql.read_dataframe(s).set_index(self.identifier)
        permnos = permnos[~permnos.index.duplicated(keep='last')]
        keys = DataFrame(keys)
        keys.columns = [self.identifier]
        result = keys.join(permnos, on=self.identifier, how='left')[permno]
        return result.fillna(0).astype(int).to_list()

    def get_linked(self, dataset: str, 
                         fields: List[str], 
                         date_field: str, 
                         link_perm: str, 
                         link_date: str, 
                         where: str = '', 
                         limit: int | str | None = None) -> DataFrame:
        """Query a dataset, and join 'links' table to return data with permno

        Args:
            dataset: Name internal Table to query data from
            fields: Data fields to retrieve
            date_field: Name of date field in data table
            link_date: Name of link date field in 'links' table
            link_perm: Name of permno field in 'links' table
            where: Where clause (optional)
            limit: Maximum rows to return (optional)

        Returns:
            DataFrame containing result of query
        """

        if where:
            where = ' AND ' + where
        limit = " LIMIT " + str(limit) if limit else ''
        fields = ", ".join([f"{self[dataset].key}.{f.lower()}"
                            for f in set([self.identifier, date_field]
                                         + fields)])        
        q = ("SELECT {links}.{link_perm} as {permno}, "
             "       {links}.{link_date}, "
             "       {fields} "
             "FROM {table} "
             "LEFT JOIN {links} "
             "  ON {table}.{key} = {links}.{key} "
             "     AND {links}.{link_date} = "
             "         (SELECT MAX(c.{link_date}) AS {link_date} "
             "            FROM {links} AS c "
             "            WHERE c.{key} = {table}.{key} "
             "              AND (c.{link_date} <= {table}.{date_field} "
             "                   OR c.{link_date} = 0)) "
             "WHERE {links}.{link_perm} IS NOT NULL AND {links}.{link_perm} > 0"
             "  AND {table}.{key} IS NOT NULL {where} {limit}").format(
                links=self['links'].key,
                permno='permno',
                link_perm=link_perm,
                link_date=link_date,
                fields=fields,
                table=self[dataset].key,
                key=self.identifier,
                date_field=date_field,
                where=where,
                limit=limit)
        self._print("(get_linked)", q)
        return self.sql.read_dataframe(q)

class Stocks(Structured):
    """Provide interface to structured stock price datasets"""

    def __init__(self, sql: SQL, 
                       bd: BusDay, 
                       tables: Dict[str, Table], 
                       identifier: str, name: str, 
                       rdb: Redis | None = None,
                       verbose: int = _VERBOSE):   
        """Initialize a connection to Stocks structured datasets"""
        super().__init__(sql, bd, tables, identifier=identifier, name=name,
                         rdb=rdb, verbose=verbose)

    def get_series(self, permnos: int | str | List[str | int], 
                         field: str = 'ret', 
                         date_field: str = 'date',
                         dataset: str = 'daily', 
                         start: int = 19000000, 
                         end: int = 29001231) -> DataFrame | Series:
        """Return time series of a field for multiple permnos as DataFrame

        Args:
            permnos: Identifiers to filter
            field: Name of column to extract
            start: Inclusive start date (YYYYMMDD)
            end: Inclusive end date (YYYYMMDD)
            dataset: Name of dataset to retrieve `ret` (default is `daily`)

        Returns:
            DataFrame indexed by date with permnos in columns
        """
        assert self[dataset] is not None
        if isinstance(permnos, (int, str)) :
            q = ("SELECT {date_field}, {field}"
                 "  FROM {table}"
                 "  WHERE {date_field} >= {start} AND {date_field} <= {end} "
                 "    AND {permno} = '{permnos}'").format(
                     permno=self.identifier,
                     field=field,
                     date_field=date_field,
                     table=self[dataset].key,
                     start=int(start),
                     end=int(end),
                     permnos=permnos)
            self._print('(get_series single)', q)
            return self.sql.read_dataframe(q)\
                .set_index(date_field)[field].sort_index().rename(permnos)
        else:
            q = ("SELECT {date_field}, {permno}, {field} "
                 "  FROM {table}"
                 "  WHERE {date_field} >= {start} AND {date_field} <= {end} "
                 "    AND {permno} IN ('{permnos}')").format(
                     permno=self.identifier,
                     field=field,
                     date_field=date_field,
                     table=self[dataset].key,
                     start=int(start),
                     end=int(end),
                     permnos="', '".join([str(p) for p in permnos]))
            self._print('(get_series many)', q)
            return self.sql.read_dataframe(q)\
                    .pivot(index='date', 
                           columns=self.identifier, 
                           values=field)[permnos].sort_index()

    def get_ret(self, start: int, 
                      end: int, 
                      dataset: str = 'daily', 
                      field: str = 'ret', 
                      date_field: str = 'date',
                      use_cache: bool | None = True) -> Series:
        """Compounded returns between start and end dates of all stocks

        Args:
            start: Inclusive start date (YYYYMMDD)
            end: Inclusive end date (YYYYMMDD)
            dataset: Name of dataset to retrieve (default is `daily`)
            field: Name of returns field
            date_field: Name of date field
            use_cache: True to read and and write cache; 
                False to write but not read cache; None to ignore cache

        Series:
            DataFrame with prod(min_count=1) of returns in column `ret`, 
            with rows indexed by permno

        Notes:

        - If start and end are first and last business dates of a month, then
          search range is expanded to include first and last calendar dates
          of respective months, and `monthly` table is queried
        """
        if self._use_monthly(start, end):  # use monthly if input dates align
            dataset = 'monthly'
            start = (start // 100) * 100
            end = (end // 100 * 100) + 99
        rkey = "_".join([field, str(self), str(start), str(end)])
        if use_cache and self.rdb and self.rdb.redis.exists(rkey):
            self._print('(get_ret load)', rkey)
            return self.rdb.load(rkey)[field]    # use cache

        if dataset == 'monthly' and (start // 100) == (end // 100):
            q = ("SELECT {field}, {identifier} FROM {table} "
                 " WHERE {date_field} >= {start} "
                 "   AND {date_field} <= {end}").format(
                     table=self[dataset].key,
                     field=field,
                     date_field=date_field,
                     identifier=self.identifier,
                     start=start,
                     end=end)
        else:
            q = ("SELECT {field}, {identifier} FROM {table} "
                 " WHERE date >= {start} AND date <= {end}").format(
                     table=self[dataset].key,
                     field=field,
                     identifier=self.identifier,
                     start=start,
                     end=end)
        self._print('(get_ret)', q)
        df = self.sql.read_dataframe(q).sort_values(self.identifier)

        # computed compounded returns
        df[field] += 1
        df = (df.groupby(self.identifier).prod(min_count=1)-1).dropna()

        if use_cache is not None and self.rdb and start != end:  # if cache
            self._print('(get_ret dump)', rkey)
            self.rdb.dump(rkey, df)
        return df[field]

    def get_compounded(self, periods: List[Tuple[int, int]], 
                             permnos: List[int], 
                             use_cache: bool | None = True) -> DataFrame:
        """Compound returns within list of periods, for given permnos

        Args:
            periods: Tuples of inclusive begin and end dates of returns period
            permnos: List of permnos
            use_cache: If True, then read and write cache; If False, then
                write but not read cache.  None to ignore cache

        Returns:
            DataFrame of compounded returns in rows, for permnos in cols
        """
        # accumulate horizontally, then finally transpose
        r = DataFrame(index=permnos)
        for beg, end in periods:
            r[end] = self.get_ret(beg, end, use_cache=use_cache)\
                                 .reindex(permnos)
        return r.transpose()

    def cache_ret(self, dates: List[Tuple[int, int]], 
                        replace: bool, 
                        field: str = 'ret', 
                        date_field: str ='date',
                        dataset: str = 'daily'):
        """Pre-generate compounded returns from daily for redis store"""
        assert self.rdb is not None
        q = ("SELECT {field}, {identifier}, {date_field} FROM {table} "
             " WHERE {date_field} >= {start} "
             "   AND {date_field} <= {end}").format(
                 table=self[dataset].key,
                 field=field,
                 identifier=self.identifier,
                 date_field=date_field,
                 start=dates[0][0],
                 end=dates[-1][-1])
        self._print('(cache_ret)', q)
        rets = self.sql.read_dataframe(q).sort_values(self.identifier)
        rets[field] += 1
        
        for start, end in dates:
            rkey = "_".join([field, str(self), str(start), str(end)])
            if not replace and self.rdb.redis.exists(rkey):
                self._print('(cache_ret exists)', rkey)
                continue
            df = rets[rets['date'].ge(start) & rets['date'].le(end)]\
                 .drop(columns='date')
            df = (df.groupby(self.identifier).prod(min_count=1) - 1).dropna()
            self._print('(cache_ret dump)', rkey, start, end, len(df))
            self.rdb.dump(rkey, df)
    
    
    def get_window(self, dataset: str, 
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

    def get_many(self, dataset: str, 
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

    def get_section(self, dataset: str, 
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

    def get_range(self, dataset: str, 
                        fields: List[str] | Dict[str, str],
                        date_field: str, 
                        beg: int, 
                        end: int, 
                 use_cache: bool | None = None) -> DataFrame:
        """Return field values within a date range

        Args:
            dataset: Name of dataset to extract from
            fields: Names of columns to return (and optionally rename as)
            date_field: Name of date column in the table
            beg: Inclusive start date in YYYYMMDD format
            end: Inclusive end date in YYYYMMDD format
            use_cache: True to read and and write cache; 
                False to write but not read cache; None to ignore cache

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

        rkey = f"CRSP_{'_'.join(fields)}_{beg}_{end}"

        if self.rdb and use_cache and self.rdb.redis.exists(rkey):
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
        if use_cache is not None and self.rdb:
            self._print('(get_range dump)', rkey)
            self.rdb.dump(rkey, r)
        return r

    def _use_monthly(self, beg, end):
        """Check beg and end align with bus month, and monthly table exists"""
        if 'monthly' in self.tables_:
            return beg <= self.bd.begmo(beg) and end >= self.bd.endmo(end)
        return False


class Benchmarks(Stocks):
    """Provide Structured Stocks interface to benchmark and index returns"""

    def __init__(self, sql: SQL, 
                       bd: BusDay, 
                       verbose: int = _VERBOSE):
        """Initialize connection to a benchmark index returns dataset"""
        tables = {
            'daily': sql.Table(
                'benchmarks',
                Column('permno', String(32), primary_key=True),
                Column('date', Integer, primary_key=True),
                Column('ret', Float)),
            'ident': sql.Table(
                'benchident',
                Column('permno', String(32), primary_key=True),
                Column('name', String(64)),
                Column('item', String(8)))}
        tables['monthly'] = tables['daily']
        super().__init__(sql, bd, tables, identifier='permno',
                         name='benchmarks', verbose=verbose)
    
    def load_series(self, df: DataFrame, 
                          name: str = '', 
                          item: str = '', 
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
        self['daily'].create(checkfirst=True)
        permno = df.name
        df = df.rename('ret').to_frame()
        df['permno'] = permno
        self.sql.run(self['daily'].delete()\
                     .where(self['daily'].c['permno'] == permno))
        self.sql.load_dataframe(self['daily'].key, df=df, index_label='date')

        self['ident'].create(checkfirst=True)
        self.sql.run(self['ident'].delete()\
                    .where(self['ident'].c['permno'] == permno))
        ident = DataFrame.from_dict({0: {'permno': permno,
                                         'name': name,
                                         'item':item}},
                                    orient='index')
        self.sql.load_dataframe(self['ident'].key, df=ident)
        return ident

import pandas_datareader as pdr

class FFReader:
    """Wraps over pandas_datareader to extract FamaFrench factors

    Attributes:
        _datasets: List of common FF factors/industries
    """
    _datasets: List[Tuple[str, int, str]] = [
        ('F-F_Research_Data_5_Factors_2x3_daily', 0, ''),
        ('F-F_Research_Data_5_Factors_2x3', 0, '(mo)'),
        ('F-F_Research_Data_Factors_daily', 0, ''),
        ('F-F_Research_Data_Factors', 0, '(mo)'),   # "(mo)" for monthly
        ('F-F_Momentum_Factor_daily', 0, ''),
        ('F-F_Momentum_Factor', 0, '(mo)'),
        ('F-F_LT_Reversal_Factor_daily', 0, ''),
        ('F-F_LT_Reversal_Factor', 0, '(mo)'),
        ('F-F_ST_Reversal_Factor_daily', 0, ''),
        ('F-F_ST_Reversal_Factor', 0, '(mo)'),
        ('49_Industry_Portfolios_daily', 0, '49vw'), # append suffix
        ('48_Industry_Portfolios_daily', 0, '48vw'), #  to differentiate
        ('49_Industry_Portfolios_daily', 1, '49ew'), #  value-weighted vs
        ('48_Industry_Portfolios_daily', 1, '48ew')] #  equal-weighted

    @staticmethod
    def fetch(name: str, 
              item: int = 0, 
              suffix: str = '', 
              start: int = 19260101, 
              end: int = 20271231, 
              date_formatter = lambda x: x) -> DataFrame:
        """Retrieve item and return as DataFrame

        Args:
            name: Name of research factor in Ken French website
            item: Index of item to research (e.g. 0 is usually value-weighted)
            suffix: Suffix string to append to name when stored in sql
            start: earliest date to retrieve
            end: latest date to retrieve 
            date_formatter: to reformat dates, e.g. bd.offset or bd.endmo
        """
        start = to_datetime(start)
        end = to_datetime(end)
        df = pdr.data.DataReader(name=name,
                                 data_source='famafrench',
                                 start=start,
                                 end=end)[item]
        try:
            df.index = df.index.to_timestamp()
        except:
            pass     # else invalid comparison error!
        df = df[(df.index >= start) & (df.index <= end)]
        df.index = [date_formatter(d) for d in df.index]
        df.columns = [c.rstrip() + suffix for c in df.columns]
        df.where(df > -99.99, other=np.nan, inplace=True)  # replace NaNs
        df = df / 100   # change percentage returns in source to decimals
        return df

class CRSP(Stocks):
    """Implements an interface to CRSP structured stocks datasets

    Args:
        sql: Connection to mysql database
        dates: Business dates object
        rdb: Optional connection to Redis for caching selected query results

    Notes:

    - Earliest CRSP prc is 19251231, FF is 19260701 
      (except STRev daily is 19260126)
    """

    def __init__(self, sql: SQL, 
                       bd: BusDay, 
                       rdb: Redis | None = None, 
                       verbose: int = _VERBOSE):
        """Initialize connection to CRSP datasets"""
        tables = {
            'daily': sql.Table(
                'daily',
                Column('permno', Integer, primary_key=True),
                Column('date', Integer, primary_key=True),
                Column('bidlo', Float),
                Column('askhi', Float),
                Column('prc', Float),
                Column('vol', Float),
                Column('ret', Float),
                Column('retx', Float),   # need retx!
                Column('bid', Float),
                Column('ask', Float),
                Column('shrout', Integer, default=0),
#                Column('shrout', Float),
                Column('openprc', Float),
            ),
            'shares': sql.Table(
                'shares',
                Column('permno', Integer, primary_key=True),
                Column('shrout', Integer, default=0),
                Column('shrsdt', Integer, primary_key=True),
                Column('shrenddt', Integer, primary_key=True),
            ),
            'delist': sql.Table(
                'delist',
                Column('permno', Integer, primary_key=True),
                Column('dlstdt', Integer, primary_key=True),
                Column('dlstcd', SmallInteger, primary_key=True),
                Column('nwperm', Integer, default=0), # '0' - '99841'  (int64)
                Column('nwcomp', Integer, default=0), # '0' - '90044'  (int64)
                Column('nextdt', Integer, default=0), # 'String(8)' '19870612' @0
                Column('dlamt', Float),    # '0' - '2349.5'  (float64)
                Column('dlretx', Float),    # 'Float' '-0.003648' @ 3
                Column('dlprc', Float),    # '-1315' - '2349.5'  (float64)
                Column('dlpdt', Integer, default=0),  # 'String(8)' '19870612' @0
                Column('dlret', Float),    # 'Float' '-0.003648' @ 3
            ),
            'dist': sql.Table(
                'dist',
                Column('permno', Integer, primary_key=True),
                Column('distcd', SmallInteger, primary_key=True),
                Column('divamt', Float),
                Column('facpr', Float),
                Column('facshr', Float),
                Column('dclrdt', Integer, default=0),
                Column('exdt', Integer, primary_key=True),
                Column('rcrddt', Integer, default=0),
                Column('paydt', Integer, default=0),
                Column('acperm', Integer, default=0),
                Column('accomp', Integer, default=0),
            ),
            'names': sql.Table(
                'names',
                Column('date', Integer, primary_key=True),
                Column('comnam', String(32)),
                Column('ncusip', String(8)),
                Column('shrcls', String(1)),
                Column('ticker', String(5)),
                Column('permno', Integer, primary_key=True),
                Column('nameendt', Integer, default=0),
                Column('shrcd', SmallInteger, default=0),
                Column('exchcd', SmallInteger, default=0),
                Column('siccd', SmallInteger, default=0),
                Column('tsymbol', String(7)),
                Column('naics', Integer, default=0),
                Column('primexch', String(1)),
                Column('trdstat', String(1)),
                Column('secstat', String(4)),
                Column('permco', Integer, default=0),
                sql.Index('ncusip', 'date')
            ),
            'monthly': sql.Table(
                'monthly',
                Column('permno', Integer, primary_key=True),
                Column('date', Integer, primary_key=True),
                Column('prc', Float),
                Column('ret', Float),
                Column('retx', Float)
            )
        }
        super().__init__(sql, bd, tables, identifier='permno', name='CRSP',
                         rdb=rdb, verbose=verbose)

    def build_lookup(self, source: str, target: str, date_field='date', 
                     dataset: str = 'names', fillna: Any = 0) -> Any:
        """Build lookup function to return target identifier from source"""
        return super().build_lookup(source=source, target=target,
                                    date_field=date_field, dataset=dataset,
                                    fillna=fillna)

    def get_cap(self, date: int, 
                      use_cache: bool | None = True, 
                      use_daily: bool = True, 
                      use_permco: bool = True) -> Series:
        """Compute a cross-section of market capitalization values

        Args:
            date: YYYYMMDD int date of market cap
            use_cache: Is True, then both read and write cache; if False, then
                write but not read cache; If None, then ignore cache
            use_daily: If True, use shrout from 'daily' table, else 'shares'
            use_permco: If True, sum caps by permco, else by permno

        Returns:
            Series of market cap indexed by permno
        """
        rkey = f"cap{'co' if use_permco else ''}_{str(self)}_{date}"
        if self.rdb and use_cache and self.rdb.redis.exists(rkey):
            self._print('(get_cap load)', rkey)
            return self.rdb.load(rkey)['cap']
        if use_daily:   # where 'daily' table contains 'shrout'
            cap = self.get_section(dataset='daily', 
                                   fields=['prc', 'shrout'],
                                   date_field='date', 
                                   date=date)
            df = DataFrame(cap['shrout'] * cap['prc'].abs(), columns=['cap'])
        else:   # else get 'shrout' from 'shares' table
            permnos = list(self.get_section(dataset='daily',
                                            fields=[self.identifier],
                                            date_field='date',
                                            date=date).index)
            self._print('LENGTH PERMNOS =', len(permnos))

            prc = self.get_section(dataset='daily',
                                   fields=['prc'],
                                   date_field='date',
                                   date=date).reindex(permnos)
            self._print('NULL PRC =', prc['prc'].isna().sum())

            shr = self.get_section(dataset='shares',
                                   fields=['shrout'],
                                   date_field='shrsdt',
                                   date=date,
                                   start=0).reindex(permnos)
            self._print('NULL SHR =', shr['shrout'].isna().sum())

            df = DataFrame(shr['shrout'] * prc['prc'].abs(), columns=['cap'])
        if use_permco:
            df = df.join(self.get_section(dataset='names',
                                          fields=['permco'],
                                          date_field='date',
                                          date=date,
                                          start=0).reindex(df.index))
            sumcap = df.groupby(['permco'])[['cap']].sum()
            df = df[['permco']].join(sumcap, on='permco')[['cap']]
        self._print('NULL CAP =', sum(df['cap'].isna()))
        df = df[df > 0].dropna()
        if self.rdb and use_cache is not None:
            self._print('(get_cap dump)', rkey)
            self.rdb.dump(rkey, df)
        return df['cap']

    def get_universe(self, date: int, 
                           minprc: float = 0.0, 
                           use_cache : bool | None = True) -> DataFrame:
        """Return standard CRSP universe of US-domiciled common stocks

        Args:
            date: Rebalance date (YYYYMMDD)
            minprc: Minimum share price filter
            use_cache: If True, then read and write cache; if False, then
                write but not read cache; If None then ignore cache

        Returns:
            DataFrame of screened universe, indexed by permno, with columns: 
            market cap "decile" (1..10), "nyse" bool, "siccd", "prc", "cap"

        Notes:

        - Market cap must be available on date, with prc > 0.0
        - shrcd in [10, 11], exchcd in [1, 2, 3]
        - TODO: market cap by permco
        """
        rkey = "_".join(["universe", str(self), str(date)])
        if use_cache and self.rdb and self.rdb.redis.exists(rkey):
            self._print('(get_universe load)', rkey)
            df = self.rdb.load(rkey)
        else: 
            df = self.get_section(dataset='daily',
                                  fields=['prc', 'shrout'],
                                  date_field='date',
                                  date=date)
            df['cap'] = df['shrout'] * df['prc'].abs()

            #
            # TODO: market cap by permco
            #
            
            df = df.join(self.get_section(dataset='names',
                                          fields=['shrcd', 'exchcd',
                                                  'siccd', 'naics'],
                                          date_field='date',
                                          date=date,
                                          start=0),
                         how='left')
            self._print('LENGTH PERMNOS', str(len(df)))
            self._print('PRC NULL:', df['prc'].isna().sum(),
                        'NEG:', df['prc'].le(0).sum())
            self._print('SHR ZERO:', df['shrout'].le(0).sum())
            self._print('CAP NON-POSITIVE:', len(df) - df['cap'].gt(0).sum())
            
            df = df[df['cap'].gt(0) & 
                    df['shrcd'].isin([10, 11]) &
                    df['exchcd'].isin([1, 2, 3])]
            df['nyse'] = df['exchcd'].eq(1)                    # nyse indicator
            df['decile'] = fractiles(values=df['cap'], # size deciles 
                                     pct=np.arange(10, 100, 10),
                                     keys=df.loc[df['nyse'], 'cap'],
                                     ascending=False)
            df = df[['cap', 'decile', 'nyse', 'siccd', 'prc', 'naics']]
            if use_cache is not None and self.rdb:
                self._print('(get_universe dump)', rkey)
                self.rdb.dump(rkey, df)
        return df[df['prc'].abs().gt(minprc)] if minprc > 0.0 else df

    def get_divamt(self, start: int, 
                         end: int) -> DataFrame:
        """Accmumulates total dollar dividends between start and end dates

        Args:
            start: Inclusive start date (YYYYMMDD)
            end: Inclusive end date (YYYYMMDD)

        Returns:
            DataFrame with accumulated divamts = per share divamt * shrout
        """
        q = ("SELECT {dist}.{identifier} AS {identifier}, "
             " SUM({table}.shrout * {dist}.divamt) AS divamt "
             "FROM {dist} INNER JOIN {table} "
             " ON {table}.{identifier} = {dist}.{identifier} AND "
             "    {table}.date = {dist}.exdt "
             " WHERE {dist}.divamt > 0 AND {dist}.exdt >= {start} "
             "   AND {dist}.exdt <= {end} GROUP BY {identifier} ").format(
                 dist=self['dist'].key,
                 identifier=self.identifier,
                 table=self['daily'].key,
                 start=start,
                 end=end)
        return self.sql.read_dataframe(q).set_index(self.identifier)

    def get_dlstret(self, start: int, 
                          end: int, 
                          use_cache: bool | None = True) -> Series:
        """Compounded delisting returns from start to end dates for all permnos

        Args:
            start: Inclusive start date (YYYYMMDD)
            end: Inclusive end date (YYYYMMDD)
            use_cache: If True, then read and write cache; if False, then
                write but not read cache; If None then ignore cache

        Returns:
            Series of compounded returns
        """
        rkey = "_".join(["dlst", str(self), str(start), str(end)])
        if use_cache and self.rdb and self.rdb.redis.exists(rkey):
            self._print("(get_dlstret load)", rkey, str(self))
            return self.rdb.load(rkey)['ret']

        q = ("SELECT (1+dlret) AS ret, {identifier} FROM {table} "
             "  WHERE dlstdt >= {start} AND dlstdt <= {end}").format(
                 table=self['delist'].key,
                 identifier=self.identifier,
                 start=start,
                 end=end)
        self._print('(get_dlst)', q)
        df = self.sql.read_dataframe(q).sort_values(self.identifier)
        if len(df):
            df = (df.groupby(self.identifier).prod(min_count=1)-1).dropna()
        if use_cache is not None and self.rdb:
            self._print("(get_dlstret dump)", rkey, str(self))
            self.rdb.dump(rkey, df)
        return df['ret']

    def get_ret(self, start: int, end: int, *args, 
                      delist: bool = False, **kwargs) -> Series:
        """Get compounded returns, with option to include delist returns"""
        ret = super().get_ret(start, end, *args, **kwargs)
        if (delist and 'delist' in self.tables_        # if using delist and
                and self._use_monthly(start, end)):    #   monthly tables
            dlst = self.get_dlstret(start, end)
            permnos = ret.index.intersection(dlst.index)
            if len(permnos):
                ret[permnos] = (1+ret[permnos]) * (1+dlst[permnos]) - 1
        return ret

class PSTAT(Structured):
    """Provide interface to Compustat structured data sets

    Args:
        sql: Connection to mysql database
        bd: Custom business days object

    Attributes:
        _role: Reference Series mapping keydev role id to description
        _event: Reference Series mapping keydev event id to description

    Notes:

    - Screen on (INDFMT= 'INDL', DATAFMT='STD', POPSRC='D', and CONSOL='C') 
      keeps majority of records and uniquely identifies GVKEY, DATADATE.
    """

    _role = Series({   # Key Development role id labels
        1: 'Target',
        2: 'Advisor',
        3: 'Buyer',
        4: 'Seller',
        5: 'Transaction',
        6: 'Transaction Consideration',
        7: 'Lender',
        8: 'Participant',
        9: 'TradingItemId',
        10: 'Auditor',
        11: 'Sponsor' }, name='role')

    _event = Series({   # Key Development event id labels
        1: 'Seeking to Sell/Divest',            # may be "not sell"
        3: 'Seeking Acquisitions/Investments',
        5: 'Seeking Financing/Partners', # too general, mentions banks
        7: 'Bankruptcy - Other',  # good: includes contemplates and motions
        11: 'Delayed SEC Filings',   # good
        12: 'Delistings',            # good, but beware of microcap
        16: 'Executive/Board Changes - Other',
        21: 'Discontinued Operations/Downsizings',
        22: 'Strategic Alliances',
        23: 'Client Announcements',
        24: 'Regulatory Agency Inquiries',
        25: 'Lawsuits & Legal Issues',
        26: 'Corporate Guidance - Lowered',
        27: 'Corporate Guidance - Raised',
        28: 'Announcements of Earnings',
        29: 'Corporate Guidance - New/Confirmed',
        31: 'Business Expansions',
        32: 'Business Reorganizations',
        36: 'Buybacks',
        41: 'Product-Related Announcements',
        42: 'Debt Financing Related',
        43: 'Restatements of Operating Results',
        44: 'Labor-related Announcements',
        45: 'Dividend Affirmations',
        46: 'Dividend Increases',
        47: 'Dividend Decreases',
        48: 'Earnings Calls',
        49: 'Guidance/Update Calls',
        50: 'Shareholder/Analyst Calls',
        51: 'Company Conference Presentations',
        52: 'M&A Calls',
        53: 'Stock Splits & Significant Stock Dividends',
        54: 'Stock Dividends (<5%)',
        55: 'Earnings Release Date',
        56: 'Name Changes',
        57: 'Exchange Changes',
        58: 'Ticker Changes',
        59: 'Auditor Going Concern Doubts',
        60: 'Address Changes',
        61: 'Delayed Earnings Announcements',
        62: 'Annual General Meeting',
        63: 'Considering Multiple Strategic Alternatives',
        64: 'Ex-Div Date (Regular)',
        65: 'M&A Rumors and Discussions',
        #    68 : 'Credit Rating - S&P - Upgrade',
        #    69 : 'Credit Rating - S&P - Downgrade',
        #    70 : 'Credit Rating - S&P - Not-Rated Action',
        #    71 : 'Credit Rating - S&P - New Rating',
        #    72 : 'Credit Rating - S&P - CreditWatch/Outlook Action',
        73: 'Impairments/Write Offs',
        74: 'Debt Defaults',
        75: 'Index Constituent Drops',
        76: 'Legal Structure Changes',
        77: 'Changes in Company Bylaws/Rules',
        78: 'Board Meeting',
        79: 'Fiscal Year End Changes',
        80: 'M&A Transaction Announcements',
        81: 'M&A Transaction Closings',
        82: 'M&A Transaction Cancellations',
        83: 'Private Placements',
        85: 'IPOs',
        86: 'Follow-on Equity Offerings',
        87: 'Fixed Income Offerings',
        88: 'Derivative/Other Instrument Offerings',
        89: 'Bankruptcy - Filing',
        90: 'Bankruptcy - Conclusion',
        91: 'Bankruptcy - Emergence/Exit',
        92: 'End of Lock-Up Period',
        93: 'Shelf Registration Filings',
        94: 'Special Dividend Announced',
        95: 'Index Constituent Adds',
        97: 'Special/Extraordinary Shareholders Meeting',
        99: 'Potential Privatization of Government Entities',
        100: 'Ex-Div Date (Special)',
        101: 'Executive Changes - CEO',
        102: 'Executive Changes - CFO',
        #    103 : 'LCD Institutional Loan News',
        #    104 : 'LCD Trend News',
        #    105 : 'LCD Fallen Angel News',
        #    106 : 'LCD Debtor-in-possession News',
        #    107 : 'LCD Middle Market News',
        #    108 : 'LCD High-Yield Bond Story News',
        #    109 : 'LCD Leveraged Buyout News',
        #    110 : 'LCD People Story News',
        #    111 : 'LCD Sponsored Deal News',
        #    112 : 'LCD M&A News',
        #    113 : 'LCD Distressed News',
        #    114 : 'LCD Break Price News',
        #    115 : 'LCD Investment Grade Loan News',
        #    116 : 'LCD Repricing News',
        #    117 : 'LCD Dividend News',
        #    118 : 'LCD Repayment News',
        #    119 : 'LCD Mezzanine Debt News',
        #    120 : 'LCD Second-lien News',
        #    121 : 'LCD High-yield Europe News',
        #    122 : 'LCD Covenant-lite News',
        #    123 : 'LCD Cross-border Deal News',
        #    124 : 'LCD CLO News',
        #    125 : 'LCD Secondary Story News',
        #    127 : 'LCD Amendment News',
        #    128 : 'LCD Communications News',
        #    129 : 'LCD European News',
        #    130 : 'LCD Price-flex News',
        #    131 : 'LCD Global News',
        #    132 : 'LCD Ratings News',
        134: 'Composite Units Offerings',
        135: 'Structured Products Offerings',
        136: 'Public Offering Lead Underwriter Change',
        137: 'Spin-Off/Split-Off',
        138: 'Announcements of Sales/Trading Statement',
        139: 'Sales/Trading Statement Calls',
        140: 'Sales/Trading Statement Release Date',
        #    141 : 'LCD Bids Wanted in Competition',
        #    142 : 'LCD Company Buys Back Outstanding Bank Debt',
        #    143 : 'LCD Debt Exchange',
        144: 'Estimated Earnings Release Date (CIQ Derived)',
        #    145 : 'LCD Loan Credit Default Swap News',
        #    146 : 'LCD Credit Defaults Swap News',
        #    147 : 'LCD Default News',
        #    148 : 'LCD Deal Launch News',
        149: 'Conferences',
        150: 'Auditor Changes',
        151: 'Buyback Update',
        152: 'Potential Buyback',
        153: 'Bankruptcy - Asset Sale/Liquidation',
        154: 'Bankruptcy - Financing',
        155: 'Bankruptcy - Reorganization',
        156: 'Investor Activism - Proposal Related',
        157: 'Investor Activism - Activist Communication',
        160: 'Investor Activism - Target Communication',
        163: 'Investor Activism - Proxy/Voting Related',
        164: 'Investor Activism - Agreement Related',
        172: 'Investor Activism - Nomination Related',
        177: 'Investor Activism - Financing Option from Activist',
        187: 'Investor Activism - Supporting Statements',
        192: 'Analyst/Investor Day',
        194: 'Special Calls',
        205: 'Regulatory Authority - Regulations',
        206: 'Regulatory Authority - Compliance',
        207: 'Regulatory Authority - Enforcement Actions',
        #    208 : 'Macro: Releases',
        #    209 : 'Macro: General',
        #    210 : 'Macro: Auctions',
        #    211 : 'Macro: Seminars',
        #    212 : 'Macro: Holidays',
        213: 'Dividend Cancellation',
        214: 'Dividend Initiation',
        215: 'Preferred Dividend',
        #    216 : 'S&P Events',
        #    217 : "Not a Keydev - Only for Timeline"
        218: "Announcement of Interim Management Statement",
        219: "Operating Results Release Date",
        220: "Interim Management Statement Release Date",
        221: "Operating Results Calls",
        222: "Interim Management Statement Calls",
        223: "Fixed Income Calls",
        224: "Halt/Resume of Operations - Unusual Events",
        225: "Corporate Guidance - Unusual Events",
        226: "Announcement of Operating Results",
        230: "Buyback - Change in Plan Terms",
        231: "Buyback Tranche Update",
        232: "Buyback Transaction Announcements",
        233: "Buyback Transaction Cancellations",
        234: "Buyback Transaction Closings"}, name='event')
    
    def __init__(self, sql: SQL, 
                       bd: BusDay, 
                       name: str = 'PSTAT', 
                       verbose = _VERBOSE):
        """Initialize connection to Compustat datasets"""
        tables = {
            'links': sql.Table(
                'links',
                Column('gvkey', Integer, primary_key=True),
                Column('conm', String(30)),
                Column('tic', String(8)),
                Column('cusip', String(9)),
                Column('cik', Integer, default=0),
                Column('sic', SmallInteger, default=0),
                Column('naics', Integer, default=0),
                Column('linkprim', String(1)),
                Column('liid', String(3)),
                Column('linktype', String(2)),
                Column('lpermno', Integer, default=0),
                Column('lpermco', Integer, default=0),
                Column('linkdt', Integer, default=0, primary_key=True),
                Column('linkenddt', Integer, default=0),
                sql.Index('cusip', 'linkdt'),
                sql.Index('cik', 'linkdt'),
                sql.Index('lpermno', 'linkdt'),
            ),
            'annual': sql.Table(
                'annual',
                Column('gvkey', Integer, primary_key=True),
                Column('datadate', Integer, primary_key=True),
                Column('indfmt', String(4), primary_key=True),
                Column('consol', String(1), primary_key=True),
                Column('popsrc', String(1), primary_key=True),
                Column('datafmt', String(3), primary_key=True),
                Column('curcd', String(3), primary_key=True),
                Column('costat', String(1)),
                Column('cusip', String(9)),
                Column('cik', BigInteger, default=0),
                Column('fyr', SmallInteger, default=0),
                Column('naics', Integer, default=0),
                Column('sic', SmallInteger, default=0),
                Column('fyear', SmallInteger, default=0),
                Column('prcc_f', Float),
                Column('sich', SmallInteger, default=0),
                *(Column(key, Float) for key in [
                    'aco', 'acox', 'act', 'ao', 'aox',
                    'ap', 'aqc', 'aqi', 'aqs', 'at',
                    'caps', 'capx', 'capxv', 'ceq', 'ceql',
                    'ceqt', 'ch', 'che', 'chech', 'cogs',
                    'cshfd', 'csho', 'cshrc', 'dc', 'dclo',
                    'dcpstk', 'dcvsr', 'dcvsub', 'dcvt', 'dd',
                    'dd1', 'dd2', 'dd3', 'dd4', 'dd5',
                    'dlc', 'dltis', 'dlto', 'dltp', 'dltt',
                    'dm', 'dn', 'do', 'dp', 'dpact',
                    'dpc', 'dpvieb', 'ds', 'dv', 'dvc',
                    'dvp', 'dvt', 'ebit', 'ebitda', 'emp',
                    'epsfx', 'epspx', 'esub', 'esubc', 'fatb',
                    'fatl', 'fca', 'fopo', 'gdwl', 'gp',
                    'gwo', 'ib', 'ibadj', 'ibc', 'ibcom',
                    'icapt', 'idit', 'intan', 'intc', 'invfg',
                    'invrm', 'invt', 'invwip', 'itcb', 'itci',
                    'ivaeq', 'ivao', 'ivch', 'ivst', 'lco',
                    'lcox', 'lct', 'lifr', 'lifrp', 'lo',
                    'lse', 'lt', 'mib', 'mibt', 'mii',
                    'mrc1', 'mrc2', 'mrc3', 'mrc4', 'mrc5',
                    'mrct', 'msa', 'ni', 'niadj', 'nopi',
                    'nopio', 'np', 'oancf', 'ob', 'oiadp',
                    'oibdp', 'pi', 'ppegt', 'ppent', 'ppeveb',
                    'prstkc', 'pstk', 'pstkc', 'pstkl', 'pstkn',
                    'pstkr', 'pstkrv', 'rea', 'reajo', 'recco',
                    'recd', 'rect', 'recta', 'rectr', 'reuna',
                    'revt', 'sale', 'scstkc', 'seq', 'spi',
                    'sppe', 'sstk', 'tlcf', 'tstk', 'tstkc',
                    'tstkn', 'txc', 'txdb', 'txdi', 'txditc',
                    'txfed', 'txfo', 'txp', 'txr', 'txs',
                    'txt', 'txw', 'wcap', 'xacc', 'xad',
                    'xido', 'xidoc', 'xint', 'xlr', 'xopr',
                    'xpp', 'xpr', 'xrd', 'xrdp', 'xrent', 'xsga'
                ]),
            ),
            'quarterly': sql.Table(
                'quarterly',
                Column('gvkey', Integer, primary_key=True),
                Column('datadate', Integer, primary_key=True),
                Column('fyearq', SmallInteger, primary_key=True),
                Column('fqtr', SmallInteger, primary_key=True),
                Column('indfmt', String(4), primary_key=True),
                Column('consol', String(1), primary_key=True),
                Column('popsrc', String(1), primary_key=True),
                Column('datafmt', String(3), primary_key=True),
                Column('cusip', String(9)),
                Column('datacqtr', String(6)),
                Column('datafqtr', String(6)),
                Column('rdq', Integer, default=0),
                Column('cik', BigInteger, default=0),
                Column('costat', String(1)),
                Column('prccq', Float),
                Column('naics', Integer, default=0),
                Column('sic', SmallInteger, default=0),
                *(Column(key, Float) for key in
                  ['actq', 'atq', 'ceqq', 'cheq', 'cogsq',
                   'cshoq', 'dlcq', 'ibq', 'lctq', 'ltq',
                   'ppentq', 'pstkq', 'pstkrq', 'revtq', 'saleq',
                   'seqq', 'txtq', 'xsgaq']),
            ),
            'keydev': sql.Table(
                'keydev',
                Column('keydevid', Integer, primary_key=True),
                Column('companyid', Integer, default=0),
                Column('companyname', String(100)),
                Column('keydeveventtypeid', SmallInteger, primary_key=True),
                Column('keydevstatusid', SmallInteger, default=0),
                Column('keydevtoobjectroletypeid', SmallInteger, primary_key=True),
                Column('announcedate', Integer, primary_key=True),
                Column('enterdate', Integer, default=0),
                Column('gvkey', Integer, primary_key=True),
            ),
            'customer': sql.Table(
                'customer',
                Column('gvkey', Integer, primary_key=True),  # Supplier GVKEY
                Column('conm', String(29)),                  # Supplier Name
                Column('cgvkey', Integer, primary_key=True), # Customer GVKEY
                Column('cconm', String(28)),            #Cust Current Name
                Column('cnms', String(50)),                     # Customer Name
                Column('srcdate', Integer, primary_key=True),   # Source Date
                Column('cid', SmallInteger, default=0), #Cust Identifier
                Column('sid', SmallInteger, default=0), #Cust Segment Ident Link
                Column('ctype', String(7)),  # Customer Type
                Column('salecs', Float),     # Customer Sales
                Column('scusip', String(9)), # Supplier CUSIP
                Column('stic', String(5)),   # Supplier Ticker Symbol
                Column('ccusip', String(9)), # Customer CUSIP
                Column('ctic', String(5)),   # Customer Ticker Symbol
            ),
        }
        super().__init__(sql, bd, tables, identifier='gvkey', name=name,
                         verbose=verbose)

    def build_lookup(self, source: str, target: str, date_field='linkdt', 
                     dataset: str = 'links', fillna: Any = None) -> Any:
        """Build lookup function to return target identifier from source"""
        return super().build_lookup(source=source, target=target,
                                    date_field=date_field, dataset=dataset,
                                    fillna=fillna) 

    def get_permnos(self, keys: List[str], date: int, link_perm='lpermno', 
                    link_date: str ='date', permno='permno') -> DataFrame:
        """Return list of permnos mapped to gvkeys as of a date

        Args:
            keys: Input list of gvkeys to lookup
            date: Prevailing date of link        
        """

        return super().get_permnos(keys, date, link_perm='lpermno', 
                                   link_date='date', permno='permno')

    def get_linked(self, dataset: str, fields: List[str],
                   date_field: str = 'datadate', link_perm: str = 'lpermno', 
                   link_date: str = 'linkdt', where: str = '', 
                   limit: int | str | None = None) -> DataFrame:
        """Query a pstat table, and return with linked crsp permno

        Args:
            dataset: pstat dataset to query
            fields : Names of fields to return
            date_field: Name of date field in pstat table to query
            link_date: Name of link date field in 'links' table
            link_perm: Name of permno field in 'links' table
            where : Sql where clause, as sql string (optional)
            limit : Maximum number of records to return (optional)

        Returns:
            DataFrame containing result of query

        Examples:

        >>> df = pstat.get_linked(dataset='annual', date_field='datadate',
                   fields=['ceq','pstkrv','pstkl','pstk'],
                   where='ceq > 0 and datadate>=19930104 and datadate<=20991231')
        >>> df = keydev.get_linked(dataset='keydev', date_field='announcedate',
                   fields=['companyname', 'keydeveventtypeid',
                   'keydevtoobjectroletypeid'],
                   where='', limit=''):

        Notes:

        ::

            select keydev.companyname, keydev.keydeveventtypeid,
            keydev.keydevtoobjectroletypeid,
            keydev.announcedate, keydev.gvkey, lpermno as permno
            from keydev left join links
            on keydev.gvkey = links.gvkey and links.linkdt =
                (select max(c.linkdt) as linkdt from links c
                where c.gvkey = keydev.gvkey and c.linkdt <= keydev.announcedate)
            where lpermno is not null and keydev.gvkey > 0 and links.gvkey > 0
            and announcedate >= 20180301
            limit 100;

        """

        return super().get_linked(dataset=dataset, date_field=date_field,
                fields=fields, link_perm='lpermno', link_date='linkdt', 
                where=where, limit=limit)

class IBES(Structured):
    """Provide interface to IBES analyst estimates structured datasets
    
    Args:
        sql: Connection to SQL database
        bd: Custom business day calendar object
    """

    def __init__(self, sql: SQL, 
                       bd: BusDay, 
                       name='IBES', 
                       verbose=_VERBOSE):
        """Initialize a connection to IBES datasets"""
        tables = {
            'summary': sql.Table(
                'summary',
                Column('ticker', String(6), primary_key=True),
                Column('fpedats', Integer, primary_key=True),
                Column('statpers', Integer, primary_key=True),
                Column('measure', String(3)), #### NEXT TIME: primary_key=True
                Column('fpi', String(1), primary_key=True),
                Column('numest', SmallInteger, default=0),
                Column('numup', SmallInteger, default=0),
                Column('numdown', SmallInteger, default=0),
                Column('medest', Float),
                Column('meanest', Float),
                Column('stdev', Float),
                Column('highest', Float),
                Column('lowest', Float),
                Column('actual', Float),
                Column('anndats_act', Integer, default=0),
            ),
            'ident': sql.Table(
                'ident',
                Column('ticker', String(6), primary_key=True),
                Column('cusip', String(8)),
                Column('oftic', String(8)),
                Column('cname', String(32)),
                Column('dilfac', SmallInteger, default=0),
                Column('pdi', String(1)),
                #Column('ccopcf', String(1)),
                Column('tnthfac', SmallInteger, default=0),
                #Column('instrmnt', String(1)),
                #Column('exchcd', String(2)),
                #Column('country', String(1)),
                #Column('compflag', String(1)),
                #Column('usfirm', SmallInteger, default=0),
                Column('sdates', Integer, primary_key=True),
            ),
            'adjust': sql.Table(
                'adjust',
                Column('ticker', String(6), primary_key=True),
                Column('oftic', String(6)),
                Column('statpers', Integer, primary_key=True),
                Column('adjspf', Float),
            ),
            'surprise': sql.Table(
                'surprise',
                Column('ticker', String(6), primary_key=True),
                Column('oftic', String(6)),
                Column('measure', String(3)),
                Column('fiscalp', String(3), primary_key=True),
                Column('pyear', SmallInteger, default=0),
                Column('pmon', SmallInteger, default=0),
                Column('anndats', Integer, primary_key=True),
                Column('actual', Float),
                Column('surpmean', Float),
                Column('surpstdev', Float),
                Column('suescore', Float),
            ),
            'links': sql.Table(
                'identlink',
                Column('ticker', String(6), primary_key=True),
                Column('sdates', Integer, primary_key=True),
                Column('permno', Integer, default=0),
                Column('date', Integer, default=0),
                Column('cname', String(32)),
                Column('comnam', String(32)),
                Column('cusip', String(8)),
            ),
        }
        super().__init__(sql, bd, tables, identifier='ticker', name=name,
                         verbose=verbose)

    def build_lookup(self, source: str, target: str, date_field='sdates', 
                     dataset: str = 'ident', fillna: Any = None) -> Any:
        """Build lookup function to return target identifier from source"""
        return super().build_lookup(source=source, target=target,
                                    date_field=date_field, dataset=dataset,
                                    fillna=fillna) 

    def write_links(self):
        """Create links table by merging 'ident' and CRSP 'names' on cusip-8"""
        self['links'].create(checkfirst=True)
        q = ("INSERT INTO {links}"
             "  SELECT {ident}.ticker, {ident}.sdates, permno, date, comnam, "
             "  cname, {ident}.cusip FROM {ident} LEFT JOIN names "
             "    ON {ident}.cusip = names.ncusip AND names.date = "
             "      (SELECT MAX(date) FROM names c WHERE c.ncusip={ident}.cusip"
             "       AND c.date<={ident}.sdates)").format(
                 links=self['links'].key,
                 ident=self['ident'].key)
        self._print("(write_links) ", q)
        self.sql.run(q)
        q = (f"SELECT SUM(ISNULL(permno)) AS missing, "
             f"  COUNT(*) AS count FROM {self['links'].key}")
        return self.sql.read_dataframe(q)

    def get_permnos(self, keys: List[str], 
                          date: int, 
                          link_perm: str = 'permno', 
                          link_date: str = 'sdates', 
                          permno: str = 'permno') -> DataFrame:
        """Return list of permnos mapped to IBES tickers as of a date

        Args:
            keys: Input list of IBES tickers to lookup
            date: Prevailing date of link        
        """
        return super().get_permnos(keys, date, link_perm='lpermno', 
                link_date='date', permno='permno')
                
    def get_linked(self, dataset: str, 
                         fields: List[str], 
                         date_field: str = 'statpers', 
                         link_perm: str = 'permno', 
                         link_date: str = 'sdates', 
                         where: str = '', 
                         limit: int | str | None = None) -> DataFrame:
        """Query an ibes table, and return with linked crsp permnos

        Args:
            dataset: Dataset to query
            fields : Fields to return
            date_field: Name of date field in ibes table to query
            link_perm: Name of permno field in links table
            link_date: Name of match date in links table
            where : Sql where clause, as sql string
            limit : Max number of records to return

        Examples:

        >>> ibes.get_linked('ident', fields=['cname'], date_field='statpers'):

        Notes:

        ::

            where fpi='6'  /* 1 is for annual forecasts, 6 is for quarterly */
            and statpers < ANNDATS_ACT /* forecasts prior to earnings annoucement
            and measure='EPS' and not missing(medest)
            and not missing(fpedats)  and (fpedats-statpers)>=0;
            (fpedats-statpers)>=0;
        """

        return super().get_linked(dataset=dataset, fields=fields, 
                date_field=date_field, link_perm='permno', link_date=link_date,
                where=where, limit=limit)

class StocksBuffer(Stocks):
    """Cache daily returns into memory, and provide Stocks-like interface"""
    
    def __init__(self, stocks: Stocks, 
                       beg: int, 
                       end: int, 
                       fields: List[str] = ['ret', 'retx'], 
                       identifier: str = 'permno'):
        """Create object and load daily returns into its cache

        Args:
            stocks: Stocks structured data object to access stock returns data
            beg: Earliest date of daily stock returns to pre-load
            end: Latest date of daily stock returns to pre-load
            fields : Column names of returns fields to load
        """
        q = (f"SELECT permno, date, {', '.join(fields)} "
             f"  FROM {stocks['daily'].key}"
             f"  WHERE date>={beg} AND date<={end}")
        self.rets = stocks.sql.read_dataframe(q).sort_values(['permno', 'date'])
        self.fields = fields
        self.identifier = identifier
        self.bd = stocks.bd

    def get_ret(self, beg: int, end: int, field: str = 'ret') -> Series:
        """Return compounded stock returns between beg and end dates

        Args:
            beg: Begin date to compound returns
            end: End date (inclusive) to compound returns
            field: Name of returns field in dataset, in {'ret', 'retx')
        """
        df = self.rets[self.rets['date'].between(beg, end)]\
            .drop(columns=['date'])
        df.loc[:, self.fields] += 1
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

    def __init__(self, df: DataFrame, 
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

    def get_ret(self, start: int, end: int, *args, **kwargs) -> Series:
        """Compounded returns between start and end (inclusive) dates"""
        df = DataFrame((self.data.loc[(self.data.index >= start)
                                      & (self.data.index <= end)]
                        + 1).prod() - 1)
        df.columns = ['ret']
        df.index.name = self.identifier
        return df['ret']

class Signals(Stocks):
    """Provide structured stocks data interface to derived signal values 

    Args:
        sql: Connection to SQL database
    """
    def __init__(self, sql: SQL, verbose=_VERBOSE):
        """Initalize a connection to derived Signals values datasets"""
        super().__init__(sql=sql, bd= None, tables={}, identifier='permno', 
                         name='signals', verbose=_VERBOSE)

    def __call__(self, label: str, 
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

    def write(self, data: DataFrame, label: str, overwrite: bool = True, 
              rebaldate: str = 'rebaldate', permno: str = 'permno') -> int:
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
            table.drop(checkfirst=True)
        table.create(checkfirst=True)
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

    def __call__(self, label: str, 
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
        df = self.data.loc[self.data[rebaldate].le(date) &
                           self.data[rebaldate].gt(start),
                           [self.identifier, rebaldate, label]]
        df = df.sort_values([self.identifier, rebaldate],
                            na_position='first')\
               .drop_duplicates([self.identifier],
                                keep='last')\
               .dropna()
        return df.set_index(self.identifier)

def famafrench_sorts(stocks: Stocks, 
                    label: str, 
                    signals: Signals, 
                    rebalbeg: int, 
                    rebalend: int,
                    window: int = 0, 
                    pct: Tuple[float, float] = (30., 70.), 
                    leverage: float = 1.,
                    months: List[int] = [], 
                    minobs: int = 100, 
                    minprc: float = 0., 
                    mincap: float = 0., 
                    maxdecile: int = 10) -> Dict[str, Any]:
    """Generate monthly time series of holdings by two-way sort procedure

    Args:
        stocks: Stocks object for accessing stock returns and price data
        label: Name of signal to retrieve
        signals: Call to extract cross section of values for the signal
        rebalbeg: First rebalance date (YYYYMMDD)
        rebalend: Last holding date (YYYYMMDD)
        pct: Percentile breakpoints to sort high, medium and low buckets
        window: No. of months to look back for signal values; 0 is exact month
        months: Months (e.g. 6=June) to retrieve univ; empty for all months
        maxdecile: Include largest stocks decile from 1 through maxdecile
        mincap: Minimum market cap
        minobs: Minimum required sample size with non-missing signal values
        leverage: Multiplier for leverage or shorting

    Notes:

    - Independent sort by median (NYSE) mkt cap and 30/70 (NYSE) HML percentiles
    - Subportfolios of the intersections are value-weighted; 
    - Spread portfolios are equal-weighted of subportfolios
    - Portfolio are resorted every June; and other months' holdings are 
      adjusted by monthly realized retx (i.e. dividends not reinvested)
    """
    rebaldates = stocks.bd.date_range(rebalbeg, rebalend, 'endmo')
    holdings = {label: dict(), 'smb': dict()}  # to return two sets of holdings
    sizes = {h : dict() for h in ['HB','HS','MB','MS','LB','LS']}
    for rebaldate in rebaldates:  #[:-1]

        # check if this is a rebalance month
        if not months or (rebaldate//100)%100 in months or not holdings[label]:
            
            # rebalance: get this month's universe of stocks with valid data
            df = stocks.get_universe(rebaldate)
            
            # get signal values within lagged window
            if window:
                start = stocks.bd.endmo(rebaldate, months=-abs(window))
            else:
                start = stocks.bd.offset(rebaldate, offsets=-1)
            signal = signals(label=label,
                             date=rebaldate,
                             start=start)
            df[label] = signal[label].reindex(df.index)

            df = df[df['prc'].abs().gt(minprc)
                    & df['cap'].gt(mincap)
                    & df['decile'].le(maxdecile)].dropna()
            if (len(df) < minobs):  # skip if insufficient observations
                continue

            # split signal into desired fractiles, and assign to subportfolios
            df['fractile'] = fractiles(df[label],
                                       pct=pct,
                                       keys=df[label][df['nyse']],
                                       ascending=False)
            subs = {'HB' : (df['fractile'] == 1) & (df['decile'] <= 5),
                    'MB' : (df['fractile'] == 2) & (df['decile'] <= 5),
                    'LB' : (df['fractile'] == 3) & (df['decile'] <= 5),
                    'HS' : (df['fractile'] == 1) & (df['decile'] > 5),
                    'MS' : (df['fractile'] == 2) & (df['decile'] > 5),
                    'LS' : (df['fractile'] == 3) & (df['decile'] > 5)}
            weights = {label: dict(), 'smb': dict()}
            for subname, weight in zip(['HB','HS','LB','LS'],
                                       [0.5, 0.5, -0.5, -0.5]):
                cap = df.loc[subs[subname], 'cap']
                weights[label][subname] = leverage * weight * cap / cap.sum()
                sizes[subname][rebaldate] = sum(subs[subname])
            for subname, weight in zip(['HB','HS','MB','MS','LB','LS'],
                                       [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]):
                cap = df.loc[subs[subname], 'cap']
                weights['smb'][subname] = leverage * weight * cap / cap.sum()
                sizes[subname][rebaldate] = sum(subs[subname])
            #print("(famafrench_sorts)", rebaldate, len(df))
            
        else:  # else not a rebalance month, so simply adjust holdings by retx
            retx = 1 + stocks.get_ret(stocks.bd.begmo(rebaldate),
                                      rebaldate,
                                      field='retx')
            for port, subports in weights.items():
                for subport, old in subports.items():
                    new = old * retx.reindex(old.index, fill_value=1)
                    weights[port][subport] = new / (abs(np.sum(new))
                                                    * len(subports) / 2)

        # combine this month's subportfolios
        for h in holdings:
            holdings[h][rebaldate] = pd.concat(list(weights[h].values()))
    return {'holdings': holdings, 'sizes': sizes}
    
class Finder:
    """Builds a class to lookup identifiers from multiple datasets"""

    def __init__(self, sql: SQL, identifier: str = '', table: str = ''):
        """Initialize lookup method with preferred identifier type and table

        Args:
            sql: SQL connection instance
            identifier: Type of input identifier for this Finder instance
            table: Physical name of table to query

        Examples:

        >>> find = Find(sql, identifier='comnam', table='names')
        """

        self.sql = sql
        self.identifier = identifier
        self.table = table

    def __call__(self, label: str = '', 
                       identifier: str = '', 
                       table: str = '', 
                       **kwargs) -> DataFrame:
        """Lookup an identifier

        Args:
            label: Input label to lookup
            identifier: Identifier type of input label
            table: Physical name of table to query
            kwargs: Alternate method to specify identifier=label

        Notes:

        Guesses identifier type and table if not specified or initialized

        Examples:

        >>> find('ALPHABET', 'comnam')
        >>> find('ALPHABET', 'conm')
        >>> find('ALPHABET', 'cname')
        >>> find(18144)
        >>> find(328795, 'gvkey')
        >>> find('0011', 'ticker', 'ident')
        >>> find('aapl')
        >>> find('03783310')
        >>> find('03783310','cusip','links')
        >>> find('03783310','cusip','ident')
        >>> find('45483', 'permco', 'names')
        """

        if len(kwargs):
            for k, v in kwargs.items():
                identifier = k
                label = v
        label = str(label).upper()
        assert label
        
        if not identifier:   # guess identifier if not specified
            if len(label) == 5 and label.isnumeric():
                identifier = 'permno'
                label = int(label)
            elif label.isnumeric():
                identifier = 'gvkey'
                label = int(label)
            elif len(label) == 8 or len(label) == 9:
                identifier = 'ncusip'
                label = label[:8]
            else:
                identifier = 'tsymbol'
                
        if not table:   # guess table if not specified
            if identifier in ['permno', 'ncusip', 'tsymbol', 'comnam']:
                table = 'names'
            elif identifier in ['gvkey', 'conm', 'cik']:
                table = 'links'
            else:
                table = 'ident'
                
        like = '='
        if identifier in ['comnam', 'conm', 'cname']:
            label = '%' + label.upper() + '%'
            like = 'LIKE'  # for identifiers of str type, match with wildcard
        elif identifier in ['permno', 'gvkey', 'cik']:
            label = int(label)
        elif identifier in ['ncusip', 'cusip']:
            label = label[:8]
        q = "SELECT * FROM {table} WHERE {identifier} {like} %s".format(
            table=table, identifier=identifier, like=like)
        result = self.sql.run(q, label)
        return DataFrame(**result) if result is not None else None

if __name__ == "__main__":
#    from os.path import dirname, abspath
#    sys.path.insert(0, dirname(dirname(abspath(__file__))))
    from conf import credentials, _VERBOSE

    import glob
    import time
    from pandas import DataFrame, Series
    from finds.database import SQL, Redis
    from finds.busday import BusDay, WeeklyDay

    VERBOSE = 1

    # open all structured datasets
    if False:
        VERBOSE = 0
        sql = SQL(**credentials['sql'], verbose=VERBOSE)
        user = SQL(**credentials['user'], verbose=VERBOSE)
        rdb = Redis(**credentials['redis'])
        bd = BusDay(sql)
        bench = Benchmarks(sql, bd)
        find = Finder(sql)
        print(find('GOOG'))

    # load benchmarks (mostly FamaFrench)
    def update_FamaFrench():
        print("\n".join(f"[{i}] {d}" 
              for i, d in enumerate(FFReader.datasets)))
        for name, item, suffix in FFReader.datasets:
            date_formatter = (bd.endmo if suffix == '(mo)' else bd.offset)
            df = FFReader.fetch(name=name, 
                                item=item,
                                suffix=suffix,
                                date_formatter=date_formatter)
            for col in df.columns:
                print(bench.load_series(df[col], name=name, item=str(item)))
        print(DataFrame(**sql.run('select * from ' + bench['ident'].key)))

    def test_bench():
        print(bench.get_series('CMA', 'ret'))
        print(bench.get_series(['CMA', 'HML'], 'ret'))
    #update_FamaFrench()


    # load CRSP: TODO handle missing return codes (< -1, see below)
    def update_crsp():
        downloads = '/home/terence/Downloads/stocks2022/'
        downloads = '/home/terence/Downloads/stocks2021/'
        dir = os.path.join(downloads, 'CRSP') + '/'
        crsp.load_csv('names', dir + 'names.txt.gz', sep='\t')   # 103383
        crsp.load_csv('shares', dir + 'shares.txt.gz', sep='\t') # 2346131
        crsp.load_csv('dist', dir + 'dist.txt.gz', sep='\t') # 935880
        crsp.load_csv('delist', dir + 'delist.txt.gz', sep='\t')  # 33584
        crsp.load_csv('monthly', dir + 'monthly.txt.gz', sep='\t') #4606907
        for i, s in enumerate(sorted(glob.glob(dir + 'stocks*.txt.gz'))):
        # for s in [dir + 'daily2021.txt.gz']:
            tic = time.time()
            crsp.load_csv('daily',
                          csvfile=s,
                          sep='\t', 
                          drop={'permno': ['PERMNO', '.'],
                                'date': ['.'],
                                'shrout':['.']})
            print(s, round(time.time() - tic, 0), 'secs')
    # def_update_crsp()


    # Pre-generate weekly returns and save in Redis cache
    begweek = 19730629  # increased stocks coverage in CRSP around this date
    middate = 19850628  # increased stocks traded in CRSP around this date
    endweek = 20211231  # is a Friday
    def update_weekly():
        wd = WeeklyDay(sql, 'Fri')   # Generate Friday-end weekly cal
        rebaldates = wd.date_range(begweek, endweek)
        r = wd.date_tuples(rebaldates)
        batchsize = 40
        batches = [r[i:(i+batchsize)] for i in range(0, len(r), batchsize)]
        for batch in batches:
            crsp.cache_ret(batch, replace=True)

    
    # load Compustat
    def update_pstat():
        downloads = '/home/terence/Downloads/stocks2021/'
        dir = os.path.join(downloads, 'PSTAT') + '/'
        df = pstat.load_csv('links',
                            csvfile=dir + 'links.txt.gz',
                            sep='\t',    # rows=33036
                            drop={'lpermno': ['0', 0], 'linkprim': ['N', 'J']},
                            replace={'linkdt': (['C', 'E', 'B'], 0),
                                     'linkenddt': (['C', 'E', 'B'], 0)})
        lag = df.shift()
        f = (lag.gvkey == df.gvkey) & (lag.lpermno != df.lpermno)
        print('permnos in links changed in ', sum(f), 'of', len(df)) # 1063

        downloads = '/home/terence/Downloads/stocks2020/'
        dir = os.path.join(downloads, 'PSTAT') + '/'
        pstat.load_csv('annual', dir + 'pstat.csv.gz') #rows = 464753

        downloads = '/home/terence/Downloads/stocks2020/'
        dir = os.path.join(downloads, 'PSTAT') + '/'
        pstat.load_csv('quarterly', dir +  'quarterly.csv.gz') # 1637274
        pstat.load_csv('customer', dir + 'supplychain.csv.gz') #107114

        downloads = '/home/terence/Downloads/stocks2020/'
        dir = os.path.join(downloads, 'PSTAT') + '/'
        for s in glob.glob(dir +  'keydev*.txt.gz'):
            tic = time.time()   # 12256909
            df = pstat.load_csv('keydev',
                                csvfile=s,
                                sep='\t',
                                drop={'gvkey': [0, '0'],
                                      'announcedate': [0, '0'],
                                      'keydevid': [0, '0']})
            print(s, time.time() - tic)    

    # load IBES
    def update_ibes():
        downloads = '/home/terence/Downloads/stocks2021/'
        dir = os.path.join(downloads, 'IBES') + '/'    
        ibes.load_csv('ident', dir + 'ident.txt.gz', sep='\t')  # 85550
        ibes.write_links()  #  (missing, count) = 14642  85550
        ibes.load_csv('summary', dir + 'summary.txt.gz', sep='\t') # 8470688
        #ibes.load_csv('adjust', downloads + 'adjustment.csv') #rows=24777
        #ibes.load_csv('surprise', downloads + 'surprise.csv')  #rows=528933

    def test_rets():
        find = StocksBuffer(crsp, 20210101, 20211231)
        df = find.get_ret(20210101, 20210131)
        print(df) 
        m = crsp.get_ret(20210101, 20210131)
        print(m)

