"""Base class for structured data sets

- CRSP (daily, monthly, names, delistings, distributions, shares outstanding)
- S&P/CapitalIQ Compustat (Annual, Quarterly, Key Development, customers)
- IBES Summary

Notes:

- Optionally cache SQL query results to Redis store

Copyright 2022-2024, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_list_like, is_integer_dtype
from sqlalchemy import Table, Column, Index
from sqlalchemy import Integer, String, Float, SmallInteger, Boolean, BigInteger
from datetime import datetime
from typing import Dict, List, Tuple, Any
from finds.database.sql import SQL, as_dtypes
from finds.database.redisdb import RedisDB
from .busday import BusDay

_VERBOSE = 0

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
      identifier: Field name of identifier key by this dataset group
      name: Display name for this dataset group
    """

    def __init__(self,
                 sql: SQL, 
                 bd: BusDay, 
                 tables: Dict[str, Table], 
                 identifier: str,
                 name: str, 
                 rdb: RedisDB | None = None, 
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

    def drop_all(self):
        """Drop all associated tables from SQL database"""
        if self.tables_:
            for table in self.tables_.values():
                table.drop()

    def load_dataframe(self,
                       df: DataFrame,
                       table: Table,
                       to_replace: Any = None,
                       value: Any = None,
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
        #table.create(checkfirst=True)
        self.sql.create_all()
        self.sql.load_dataframe(table=table.key, df=df, index_label=None)
        self._print("(structured store)", table.key, len(df))
        return len(df)
                
    def read_dataframe(self, table: str, where: str = '') -> DataFrame:
        """Read signal values from sql and return as data frame

        Args:
          table: Table to read from
          where: Where clause str for sql select

        Returns:
          DataFrame of query results
        """
        where = bool(where)*'WHERE ' + where
        return self.sql.read_dataframe(f"SELECT * FROM {table} {where}")
    
    def load_csv(self,
                 dataset: str, 
                 csvfile: str, 
                 drop: Dict[str, List[Any]] = {},
                 keep: Dict[str, List[Any]] = {},                  
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
          keep: {column: values} keep rows whose columns have any of values
          drop: {column: values} drop rows with any given value in column
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

        # drop rows where col has value val        
        for col, vals in drop.items():
            rows = df.index[df[col].isin(vals)]
            self._print('Dropping', len(rows), 'rows with', col, 'in', vals)
            df.drop(index=rows, inplace=True)

        # clean up column dtypes and rows
        df = as_dtypes(
            df=df,
            columns={k.lower(): v.type for k, v in table.columns.items()},
            drop_duplicates=[p.key.lower() for p in table.primary_key],
            replace=replace)
        self._print("(load_csv)", len(df), table)

        # drop again in case dtypes got changed
        for col, vals in drop.items():
            rows = df.index[df[col].isin(vals)]
            df.drop(index=rows, inplace=True)
        
        # keep rows where col has value vals
        rows = df.index
        for col, vals in keep.items():
            rows = rows[df.loc[rows, col].isin(vals).values]
        self._print('Keeping', len(rows), 'of', len(df), 'rows with', keep)
        df = df.loc[rows]
            
        # Create sql table and load from DataFrame
        #table.create(self.sql.engine, checkfirst=True)
        self.sql.create_all()
        self.sql.load_dataframe(table=table.key, df=df, index_label=None)
        return df

    def build_lookup(self, source: str, target: str,
                     date_field: str,  dataset: str, fillna: Any) -> Any:
        """Helper to build lookup of target from source identifiers
        Args:
          source: Name of source identifier key
          target: Name of target identifier key to return
          date_field: Name of date field in database table
          dataset: Internal name of table containing identifier mappings
          fillna: Value to return if not found
        """
        table = self[dataset]    # physical name of table
        assert table is not None
        assert source in table.c
        assert target in table.c
        return Lookup(sql=self.sql,
                      source=source,
                      target=target,
                      date_field=date_field,
                      table=table.key,
                      fillna=fillna)

    def get_permnos(self,
                    keys: List[str], 
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

    def get_linked(self,
                   dataset: str, 
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
        lookups = sql.read_dataframe(f"SELECT {source} as source,"
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

    def __call__(self, label: List | str, date: int = 99999999) -> Any:
        """Return target identifiers matched to source as of date"""
        if is_list_like(label):
            return [self(b) for b in label]
        if label in self.keys:
            a = self.lookups.get_group(label)
            b = a[a['date'] <= date].sort_values('date')  
            return (b.iloc[-1] if len(b) else   # latest before prevailing date
                    a.iloc[0]).at['target']   # else first
        return self.fillna

    def __getitem__(self, labels):
        return self(labels)

