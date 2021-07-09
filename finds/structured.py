"""Implements interface for structured data sets

- CRSP, Compustat, IBES, delistings, distributions, shares outstanding

Author: Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sqlalchemy import Column, Index
from sqlalchemy import Integer, String, Float, SmallInteger, Boolean, BigInteger
from pandas.api import types
from .solve import fractiles
try:
    from settings import ECHO
except:
    ECHO = False

def parse_where(where, prefix):
    """Helper method to parse a dict to SQL where clause query

    Parameters
    ----------
    where : dict (key, value)
        Key may be field name optionally appended with filtering operator:
      name_eq 	  be equal to the value
      name_ne     be equal to the value
      name_lt     less than the value
      name_le     less than or equal to the value
      name_gt     greater than the value
      name_ge     greater than or equal to the value
      name_in     be included in the value
      name_notin  not be included in the value
    prefix : str in {'AND','WHERE'}
        Whether this is continuation of a longer where clause
    """
    def parse(name, value):
        """helper method to process filtering operator, if any"""
        if name.endswith('_eq'):
            return f"{name[:-3]} = '{value}'"
        elif name.endswith('_ne'):
            return f"{name[:-3]} != '{value}'"
        elif name.endswith('_le'):
            return f"{name[:-3]} <= '{value}'"
        elif name.endswith('_lt'):
            return f"{name[:-3]} < '{value}'"
        elif name.endswith('_ge'):
            return f"{name[:-3]} >= '{value}'"
        elif name.endswith('_gt'):
            return f"{name[:-3]} > '{value}'"
        elif name.endswith('_in'):
            value = "','".join(value)
            return f"{name[:-3]} in ('{value}')"
        elif name.endswith('_notin'):
            value = "','".join(value)
            return f"{name[:-6]} not in ('{value}')"
        else:
            return f"{name} = '{value}'"            
    if where:
        if isinstance(where, dict):
            where = " AND ".join(f"{parse(k, v)}'" for k,v in where.items())
        return " " + prefix + " " + where
    return ''

def as_dtypes(df, columns, drop_duplicates=[], sort_values=[], keep='first',
              replace={}):
    """Helper method to convert dtypes of data frame, from sqlalchemy types

    Parameters
    ----------
    df : DataFrame
        Apply new data types from target columns dict
    columns : dict of {label : sqlalchemy type}
        Target (sqlalchemy) column types
    sort_values: str or list of str
        Column names to sort by
    drop_duplicates: list of string
        Subset of fields to drop duplicate
    keep : 'first' or 'last'
        If drop_duplicates, which row to keep
    replace : dict of {column str : tuple of old and new values}
        Each tuple is ([list of old values], new value)

    Notes
    -----
    Blank values in boolean and int fields are set to False/0.
    Invalid/blank values in double field are coerced to NaN.
    Invalid values in int field are coerced to NaN.
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
    """Base class for interface to structured datasets, internally stored in SQL

    Parameters
    ----------
    sql : SQL instance
        Connection to mysql database
    bd : BusDay instance
        Custom business calendar object
    tables : dict of {label: Table}
        Sqlalchemy Table objects, and their labels
    identifier : str
        Field name of unique identifier key
    name : str
        Display name for this instance
    rdb : Redis object, default is None
        Connector to Redis cache, if desired

    Attribute
    ---------
    identifier : str
        Field name of identifier key used by datasets in this instance
    name : str
        Display name for this instance
    """

    def __init__(self, sql, bd, tables, identifier, name, rdb=None, echo=ECHO):
        """Initialize a connection to structured datasets"""
        self.bd = bd
        self.sql = sql
        self.tables_ = tables
        self.rdb = rdb
        self.identifier = identifier
        self.name = name
        self.echo_ = echo
        
    def _print(self, *args, echo=None):
        if echo or self.echo_:
            print(*args)

    def __str__(self):
        return self.name

    def __getitem__(self, dataset):
        """Return the table object corresponding to a dataset label"""
        return self.tables_.get(dataset, None)

    def dtypes(self, dataset):
        """Return empty DataFrame with columns of corresponding dtype"""
        return as_dtypes(None, {k : v.type
                                for k,v in self[dataset].columns.items()})
        
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

    def load_csv(self, dataset, csvfile, drop={}, replace={}, sep=',',
                 encoding='latin-1', header=0, low_memory=False,
                 na_filter=False, **kwargs):
        """Insert ignore into a SQL table from a csvfile

        Parameters
        ----------
        dataset : str
            dataset name
        csvfile : str
            csv file name
        drop : dict of {column : value}
            drop rows where column has value, e.g. {'gvkey' : 0}
        replace : dict of {column: [to_replace, replace_with]}
            values to be replaced in a column
        sep : string, optional
            csv file delimiter

        Notes
        -----
        Create new table, if not exists, using associated schema
        New records with duplicate key are dropped (insert ignore used)
        """
        table = self[dataset]
        df = pd.read_csv(csvfile, sep=sep, encoding=encoding, header=header,
                         low_memory=low_memory, na_filter=na_filter)
        # 'utf-8' codec can't decode byte 0xf6 => encoding='latin-1'
        self._print('(read_csv)', len(df), csvfile)

        df.columns = df.columns.map(str.lower).map(str.rstrip)
        for col, vals in drop.items():  # drop rows where col has value val
            df.drop(index=df.index[df[col].isin(vals)], inplace=True)

        df = as_dtypes(
            df=df,
            columns={k.lower(): v.type for k, v in table.columns.items()},
            drop_duplicates=[p.key.lower() for p in table.primary_key],
            replace=replace)

        for col, vals in drop.items():  # drop rows where col has value in val
            df.drop(index=df.index[df[col].isin(vals)], inplace=True)
        self._print("(load_csv)", len(df), table)
        table.create(checkfirst=True)
        self.sql.load_dataframe(table=table.key, df=df, index_label=None)
        return df

    class _lookup:
        """Builds callable to look-up a record by identifier and prevailing date

        Parameters
        ----------
        source : str
            name of source identifier
        target : str
            name of target identifier to return
        date_field : str
            name of date field in database
        table : str
            physical table name containing identifier records

        Examples
        --------
        """
        def __init__(self, sql, source, target, date_field, table, fillna):
            lookups = sql.read_dataframe(
                f"SELECT {source} as source, {target} AS target, "
                f"{date_field} AS date FROM {table} "
                f"WHERE {source} IS NOT NULL AND {target} IS NOT NULL")
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

        def __call__(self, label, date=99999999):
            if types.is_list_like(label):
                return [self(b) for b in label]
            if label in self.keys:
                a = self.lookups.get_group(label)
                b = a[a['date'] <= date]  # prevailing (if any), else first
                return (b.iloc[-1] if len(b) else a.iloc[0]).at['target']
            return self.fillna

        def __getitem__(self, labels):
            return self(labels)

    def build_lookup(self, source, target, date_field, table, fillna):
        """Builds callable to lookup record by identifiers and prevailing date"""
        if not (source in self[table].c and target in self[table].c):
            raise Exception(f"{source} and {target} must be in {table}")
        return self._lookup(self.sql, source, target, date_field, table, fillna)

        
    def get_links(self, keys, date, link_perm, link_date, permno='permno'):
        """Returns matching permnos as of a prevailing from 'links' table

        Parameters
        ----------
        keys: list
            Input list of identifiers to lookup
        date: int
            Prevailing date of link
        link_perm: str
            Name of permno field in 'links' table
        link_date: str
            Name of link date field in 'links' table

        Returns
        -------
        permnos: list
            Linked permnos, as of prevailing date; 0 if not found
        """
        s = ("SELECT {table}.{key}, {table}.{link_perm} AS {permno}, "
             "{table}.{link_date} FROM"
             "  (SELECT {key}, MAX({link_date}) AS dt FROM {table} "
             "    WHERE {link_date} <= {date} "
             "      AND {link_perm} > 0 "
             "    GROUP BY {key}, {link_date}) AS a"
             "  INNER JOIN {table} "
             "    ON {table}.{key} = a.{key} "
             "      AND {table}.{link_date} = a.dt").format(
                 table=self['links'].key,
                 date=date,
                 link_perm=link_perm,
                 link_date=link_date,
                 permno=permno,
                 key=self.identifier)
        permnos = self.sql.read_dataframe(s).set_index(self.identifier)
        permnos = permnos[~permnos.index.duplicated(keep='last')]
        keys = DataFrame(keys)
        keys.columns = [self.identifier]
        result = keys.join(permnos, on=self.identifier, how='left')[permno]
        return result.fillna(0).astype(int).to_list()

    def get_linked(self, dataset, date_field, fields, link_perm, link_date,
                   where='', limit=None):
        """Query dataset, and join 'links' table to return data with permno

        Parameters
        ----------
        dataset: str
            Name internal Table to query data from
        date_field : str
            Name of date field in data table
        fields: list of str
            Data fields to retrieve
        link_date: str
            Name of link date field in 'links' table
        link_perm: str
            Name of permno field in 'links' table
        where: str or dict, optional
            Where clause (default '')
        limit: int, optional
            Maximum rows to return (default None to return all)

        Returns
        -------
        DataFrame
            result of query
        """
        where = parse_where(where, 'AND')
        limit = " LIMIT " + str(limit) if limit else ''
        fields = ", ".join(
            [f"{self[dataset].key}.{f.lower()}"
             for f in list(set(fields + [self.identifier, date_field]))])
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
                table=self[dataset],
                key=self.identifier,
                date_field=date_field,
                where=where,
                limit=limit)
        self._print("(get_linked)", q)
        return self.sql.read_dataframe(q)

class Stocks(Structured):
    """Provide interface to structured stock price data sets

    Parameters
    ----------
    sql : SQL instance
        Connection to mysql database
    bd: BusDay instance
        Custom business calendar object
    tables: dict of {label: Table}
        Sqlalchemy Table objects, and their label
    identifier: str
        Field name of unique identifier key

    Attribute
    ---------
    identifier: str
        Field name of identifier key used by datasets in this instance
    """

    def __init__(self, sql, bd, tables, identifier, name, rdb=None, echo=ECHO):
        """Initialize a connection to Stocks-type structured datasets"""
        super().__init__(sql, bd, tables, identifier=identifier, name=name,
                         rdb=rdb, echo=echo)

    def get_series(self, permnos, field, start=19000000, end=29001231,
                   dataset='daily'):
        """Return time series of a field for multiple permnos as data frame

        Parameters
        ----------
        permnos: str or list of str
            Identifiers to filter
        field: str
            Name of column to extract
        start: int, optional (default is earliest available)
            Inclusive start date (YYYYMMDD)
        end: int, optional (default is latest available
            Inclusive end date (YYYYMMDD)
        table: str, optional
            Dataset name of table to retrieve ret (default is 'daily' table)

        Returns
        -------
        DataFrame or Series
            Values of desired field, indexed by date, with permnos in columns
        """
        if isinstance(permnos, str):
            q = ("SELECT date, {field} FROM {table}"
                 "  WHERE date >= {start} AND date <= {end} "
                 "    AND {permno} = {permnos}").format(
                     permno=self.identifier,
                     field=field,
                     table=self[dataset].key,
                     start=int(start),
                     end=int(end),
                     permnos=permnos)
            self._print('(get_series)', q)
            return self.sql.read_dataframe(q)[field].rename(permnos)
        else:
            q = ("SELECT date, {permno}, {field} FROM {table}"
                 "  WHERE date >= {start} AND date <= {end} "
                 "    AND {permno} IN ('{permnos}')").format(
                     permno=self.identifier,
                     field=field,
                     table=self[dataset].key,
                     start=int(start),
                     end=int(end),
                     permnos="', '".join([str(p) for p in permnos]))
            self._print('(get_series)', q)
            return self.sql.read_dataframe(q).pivot(
                index='date', columns=self.identifier, values=field)[permnos]

    def get_ret(self, start, end, dataset='daily', field='ret',
                use_cache=True):
        """Compound returns between start and end dates, return as data frame

        Parameters
        ----------
        start: int
            Inclusive start date (YYYYMMDD)
        end: int
            Inclusive end date (YYYYMMDD)
        dataset: str, optional
            Dataset name of table to retrieve ret (default is 'daily' table)
        use_cache: bool, default is True (read, if exists, and write cache)
            If False, then write but not read cache.  None to ignore cache

        Returns
        -------
        DataFrame
            prod(min_count=1) of returns in column 'ret', indexed by permno

        Notes
        -----
        If start and end are first and last business dates of a month, then:
          search range is expanded to include first and last calendar dates
          of respective months and 'monthly' table is queries
        if start and end are first and last business date of a month, then:
          return section from weekly returns table
        """
        if self.use_monthly(start, end):
            dataset = 'monthly'
            start = (start // 100) * 100
            end = (end // 100 * 100) + 99
        rkey = "_".join([field, str(self), str(start), str(end)])
        if use_cache and self.rdb and self.rdb.redis.exists(rkey):
            self._print('(get_ret load)', rkey)
            return self.rdb.load(rkey)

        if dataset == 'monthly' and (start // 100) == (end // 100):
            q = ("SELECT {field}, {identifier} FROM {table} "
                 " WHERE date >= {start} AND date <= {end}").format(
                     table=self[dataset].key,
                     field=field,
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
        df[field] += 1
        df = (df.groupby(self.identifier).prod(min_count=1) - 1).dropna()
        if use_cache is not None and self.rdb and start != end:
            self._print('(get_ret dump)', rkey)
            self.rdb.dump(rkey, df)
        return df

    def get_compounded(self, periods, permnos, use_cache=True):
        """Compound returns within list of periods, for given permnos

        Parameters
        ----------
        periods: list of tuple(beg: int, end: int)
            Each tuple is begin and end date (inclusive) of returns period
        permnos: list
            List of permnos
        use_cache: bool, default is True (read, if exists, and write cache)
            If False, then write but not read cache.  None to ignore cache

        Returns
        -------
        DataFrame
            Time series of compounded returns (in rows), for permnos (in cols)
        """
        # accumulate horizontally, then finally transpose
        r = DataFrame(index=permnos)
        for beg, end in periods:
            r[end] = self.get_ret(beg, end, use_cache=use_cache).reindex(permnos)
        return r.transpose()

    def cache_ret(self, dates, field='ret', dataset='daily', date_field='date',
                  overwrite=False):
        """Pre-generate compounded returns from daily for redis store"""
        q = ("SELECT {field}, {identifier}, {date_field} FROM {table} "
             " WHERE date >= {start} AND date <= {end}").format(
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
            if not overwrite and self.rdb.redis.exists(rkey):
                self._print('(cache_ret exists)', rkey)
                continue
            df = rets[rets['date'].ge(start) & rets['date'].le(end)]\
                 .drop(columns='date')
            df = (df.groupby(self.identifier).prod(min_count=1) - 1).dropna()
            self._print('(cache_ret dump)', rkey, start, end, len(df))
            self.rdb.dump(rkey, df)
    
    
    def get_window(self, dataset, field, permnos, date_field, dates, left, right,
                   avg=False):
        """Retrieve field values for permnos in a window centered around dates

        Parameters
        ----------
        dataset : str
            Dataset label
        field : str
            Name of field to retrieve
        permnos : list
            List of identifiers to retrieve
        date_field : str
            Name of date field in database
        dates : list of int
            List of corresponding dates of center of event window
        left : int
            Relative (inclusive) offset of start of event window
        right : int
            Relative (inclusive) offset of end of event window

        Returns:
        --------
        DataFrame
            Columns [0:(right-left)] contain field values in event window
        """
        dates = list(dates)
        permnos = list(permnos)
        if avg:
            # Generate and save dates to sql temp
            df = DataFrame({'a': self.bd.offset(dates, left),
                            'b': self.bd.offset(dates, right),
                            self.identifier: permnos},
                           index=np.arange(len(dates)))
            self.sql.load_dataframe(table=self.sql.temp_, df=df,
                                    index_label='n', if_exists='replace')
            if types.is_integer_dtype(df[self.identifier].dtype):
                q = f"CREATE INDEX a on {self.sql.temp_} ({self.identifier},a,b)"
                self.sql.run(q)
                q = f"CREATE INDEX b on {self.sql.temp_} ({self.identifier},b,a)"
                self.sql.run(q)

            # join on (permno, date) and retrieve from target table
            q = ("SELECT {temp}.n, "
                 " AVG({field}) as {field} FROM {temp} LEFT JOIN {table}"
                 " ON {temp}.{identifier} = {table}.{identifier} "
                 " WHERE {table}.{date_field} >= {temp}.a "
                 " AND {table}.{date_field} <= {temp}.b"
                 " GROUP BY {temp}.n").format(
                     temp=self.sql.temp_,
                     identifier=self.identifier,
                     field=field,
                     date_field=date_field,
                     table=self[dataset].key)
            df = self.sql.read_dataframe(q).drop_duplicates(subset=['n'])\
                                           .set_index('n')
            result = DataFrame({'permno': permnos, 'date': dates},
                               index=np.arange(len(dates))).join(df, how='left')
        else:
            # Generate and save dates to sql temp
            cols = ["day" + str(i) for i in range(1 + right - left)]
            df = DataFrame(data=self.bd.offset(dates, left, right), columns=cols)
            df[self.identifier] = permnos
            self.sql.load_dataframe(self.sql.temp_, df, if_exists='replace')

            # Loop over each date, and join as columns of result
            result = DataFrame({'permno': permnos, 'date': dates})
            for col in cols:
                # create index on date to speed up join with target table
                if types.is_integer_dtype(df[self.identifier].dtype):
                    q = "CREATE INDEX {col} on {temp} ({ident}, {col})".format(
                        temp=self.sql.temp_, ident=self.identifier, col=col)
                    self.sql.run(q)

                # join on (permno, date) and retrieve from target table
                q = ("SELECT {temp}.{identifier}, {field}"
                     " FROM {temp} LEFT JOIN {table}"
                     " ON {table}.{identifier} = {temp}.{identifier} "
                     "  AND {table}.{date_field} = {temp}.{col}").format(
                         temp=self.sql.temp_,
                         identifier=self.identifier,
                         field=field,
                         date_field=date_field,
                         table=self[dataset].key,
                         col=col)
                df = self.sql.read_dataframe(q)
                # left join, so assume same order
                result[col] = df[field].values
        self.sql.run('drop table if exists ' + self.sql.temp_)
        result.columns = [int(c[3:]) if c.startswith('day') else c
                          for c in result.columns]
        return result.reset_index(drop=True)

    def get_many(self, dataset, permnos, fields, date_field, dates, exact=True):
        """Retrieve multiple fields for lists of permnos and dates

        Parameters
        ----------
        dataset : str
            Dataset label
        permnos : list
            List of identifiers to retrieve
        dates : list of int
            List of corresponding dates of center of event window
        field : list of str
            Names of fields to retrieve
        date_field : str
            Names of date field in database
        exact : bool, default is True
            Whether require exact date match, or allow (most recent) earlier date

        Returns
        -------
        DataFrame
            with permno, date, and retrieved field values across columns
        """
        field = "`, `".join(list(fields))
        self.sql.load_dataframe(table=self.sql.temp_,
                                df=DataFrame({self.identifier: list(permnos),
                                              'date': list(dates)},
                                             index=np.arange(len(permnos))),
                                index_label='_seq',
                                if_exists='replace')
        if exact:
            q = ("SELECT {temp}._seq, {temp}.{identifier}, "
                 "  {temp}.date AS date, `{field}` "
                 "  FROM {temp} LEFT JOIN {table}"
                 "    ON {table}.{identifier} = {temp}.{identifier} "
                 "    AND {table}.{date_field} = {temp}.date").format(
                     temp=self.sql.temp_,
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
                     temp=self.sql.temp_,
                     identifier=self.identifier,
                     field=field,
                     date_field=date_field,
                     table=self[dataset].key)
            df = self.sql.read_dataframe(q)\
                         .sort_values(['_seq', 'date'], na_position='first')\
                         .drop_duplicates(subset=['_seq'], keep='last')\
                         .set_index('_seq').sort_index()
        self.sql.run('drop table if exists ' + self.sql.temp_)
        return df

    def get_section(self, dataset, fields, date_field, date, start=None):
        """Return a cross-section of values of fields as of a single date

        Parameters
        ----------
        dataset : str
            Dataset to extract from
        fields: list of str
            Names of columns to return
        date_field: str
            Name of date column in the table
        date: int
            Desired date in YYYYMMDD format
        start: int or None, default is None
            Inclusive start of date range to return prevailing permno row.
            If None, then as of date exact.

        Returns
        -------
        r : Series (if fields is str) or DataFrame (if fields is list-like)
            indexed by permno

        Note
        ----
        If start is not None, then the latest prevailing record for each
        between (non-inclusive) start and (inclusive) date is returned

        Examples
        --------
        t = crsp.get_section('shares', ['shrenddt','shrout'], 'shrsdt', date)
        u = crsp.get_section('names', ['nameendt','comnam'], 'date', date-10000)
        """
        assert(fields)
        if not types.is_list_like(fields):
            fields = [fields]
        if self.identifier not in fields:
            fields += [self.identifier]
        if start is None:
            q = ("SELECT {fields} FROM {table} "
                 " WHERE {date_field} = {date}").format(
                     fields=", ".join(fields),
                     table=self[dataset],
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
                     table=self[dataset],
                     permno=self.identifier,
                     date_field=date_field,
                     date=date,
                     start=start)
        self._print('(get_section)', q)
        return self.sql.read_dataframe(q).set_index(self.identifier)

    def get_range(self, dataset, fields, date_field, beg, end, use_cache=None):
        """Return field values within a date range

        Parameters
        ----------
        dataset : str
            Dataset to extract from
        fields: list of str, or dict of {field: new str}
            Names of columns to return (and rename as)
        date_field: str
            Name of date column in the table
        beg: int
            Inclusive start date in YYYYMMDD format
        end: int
            Inclusive end date in YYYYMMDD format

        Returns
        -------
        r : DataFrame
            multi-indexed by permno, date

        Examples
        --------
        t = crsp.get_section('shares', ['shrenddt','shrout'], 'shrsdt', date)
        u = crsp.get_section('names', ['nameendt','comnam'], 'date', date-10000)
        """
        assert(fields)
        if types.is_list_like(fields):
            if isinstance(fields, dict):
                rename = fields
                fields = list(fields.keys())
            else:
                rename = {k:k for k in fields}
        else:
            rename = None
            fields = [fields]
        if self.identifier not in fields:
            fields += [self.identifier]
        rkey = f"CRSP_{'_'.join(fields)}_{beg}_{end}"
        if self.rdb and use_cache and self.rdb.redis.exists(rkey):
            self._print('(get_range load)', rkey)
            return self.rdb.load(rkey)
        q = ("SELECT {fields}, {date_field} FROM {table} WHERE "
             " {date_field} >= {beg} AND {date_field} <= {end}").format(
                 fields=", ".join(fields),
                 table=self[dataset],
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

    def use_monthly(self, beg, end):
        """Check if is bus month beg and end, and exists 'monthly' table"""
        if 'monthly' in self.tables_:
            return beg <= self.bd.begmo(beg) and end >= self.bd.endmo(end)
        return False


class Benchmarks(Stocks):
    """Provide structured stocks data interface to benchmark and index returns

    Parameters
    ----------
    sql : SQL instance
        Connection to mysql database
    bd: BusDay instance
        Custom business calendar object

    Attributes
    ----------
    identifier: 'permno'
        Field name of unique identifier key
    """

    def __init__(self, sql, bd, echo=ECHO):
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
                         name='benchmarks', echo=echo)
    
    def load_series(self, df, name='', item=''):
        """Loads a Series containing benchmark returns to sql

        Parameters
        ----------
        df : Series
            Each column is a time-series to load to sql
        name: str
            Primary label for this source to insert into ident table
        item: str
            Secondary label for this source to insert into ident table

        Notes
        -----
        Each column of input data frame is loaded to sql table 'daily',
        with its series name as 'permno' field, values as 'ret' field,
        and series index as 'date' field.
        """
        self['daily'].create(checkfirst=True)
        permno = df.name
        df = df.rename('ret').to_frame()
        df['permno'] = permno
        delete = self['daily'].delete().where(self['daily'].c['permno']==permno)
        self.sql.run(delete)
        self.sql.load_dataframe(self['daily'].key, df=df, index_label='date')
        self['ident'].create(checkfirst=True)
        delete = self['ident'].delete().where(self['ident'].c['permno']==permno)
        self.sql.run(delete)
        ident = DataFrame.from_dict({0: {'permno': permno, 'name': name,
                                         'item':item}}, orient='index')
        self.sql.load_dataframe(self['ident'].key, df=ident)
        return permno


class CRSP(Stocks):
    """Provide interface to CRSP structured stocks data sets

    Parameters
    ----------
    sql : SQL instance
        Connection to mysql database
    dates: BusDates instance
        Business dates object
    rdb : Redis instance, optional (default None)
        Redis store to cache common query results

    Notes
    -----
    Earliest CRSP prc 19251231, FF 19260701 (except STRev daily 19260126)
    """

    def __init__(self, sql, bd, rdb=None, echo=ECHO):
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
                         rdb=rdb, echo=echo)

    def build_lookup(self, source, target, date_field='date',
                     table='names', fillna=None):
        """Builds callable to look-up record by identifier and prevailing date"""
        if not (source in self[table].c and target in self[table].c):
            raise Exception(f"{source} and {target} must be in {table}")
        return self._lookup(self.sql, source, target, date_field,
                            self[table].key, fillna)

    def get_cap(self, date, use_cache=True, use_daily=True, use_permco=True):
        """Compute a cross-section of market capitalization values

        Parameters
        ----------
        date : int
            As of YYYYMMDD date format
        use_cache: bool, default is True (both read, if exists, and write cache)
            If False, then write but not read cache.  None to ignore cache
        use_daily: bool, default is True
            If True, use shrout from 'daily' table, else from 'shares' table
        use_permco: bool, default is True
            If True, sum caps by permco (default), else by original permno
        """
        rkey = f"cap{'co' if use_permco else ''}_{str(self)}_{date}"
        if self.rdb and use_cache and self.rdb.redis.exists(rkey):
            self._print('(get_cap load)', rkey)
            return self.rdb.load(rkey)
        if use_daily:   # where 'daily' table contains 'shrout'
            cap = self.get_section(dataset='daily', fields=['prc', 'shrout'],
                                   date_field='date', date=date)
            df = DataFrame(cap['shrout'] * cap['prc'].abs(), columns=['cap'])
        else:   # else get 'shrout' from 'shares' table
            permnos = list(self.get_section(
                dataset='daily', fields=[self.identifier], date_field='date',
                date=date).index)
            self._print('LENGTH PERMNOS =', len(permnos))

            prc = self.get_section(dataset='daily', fields=['prc'],
                                   date_field='date', date=date).reindex(permnos)
            self._print('NULL PRC =', sum(prc['prc'].isna()))

            shr = self.get_section(dataset='shares', fields=['shrout'],
                                   date_field='shrsdt', date=date,
                                   start=0).reindex(permnos)
            self._print('NULL SHR =', sum(shr['shrout'].isna()))

            df = DataFrame(shr['shrout'] * prc['prc'].abs(), columns=['cap'])
        if use_permco:
            df = df.join(self.get_section(dataset='names', fields=['permco'],
                                          date_field='date', date=date,
                                          start=0).reindex(df.index))
            sumcap = df.groupby(['permco'])[['cap']].sum()
            df = df[['permco']].join(sumcap, on='permco')[['cap']]
        self._print('NULL CAP =', sum(df['cap'].isna()))
        df = df[df > 0].dropna()
        if self.rdb and use_cache is not None:
            self._print('(get_cap dump)', rkey)
            self.rdb.dump(rkey, df)
        return df

    def get_universe(self, date, minprc=0.0, use_cache=True):
        """Return screened standard CRSP universe of US-domiciled common stocks

        Parameters
        ----------
        date: int
            Desired rebalance date (YYYYMMDD)
        minprc: float, default 0.0
            Minimum share price
        use_cache: bool, default is True (read, if exists, and write cache)
            If False, then write but not read cache.  None to ignore cache

        Returns
        -------
        DataFrame
            Valid universe, indexed by permno, with columns: 
            market cap "decile" (1..10), "nyse" bool, and "siccd", "prc", "cap"

        Notes
        -----
        market cap must be available on date, with prc > 0.0
        shrcd isin [10, 11], exchcd isin [1, 2, 3]
        """
        rkey = "_".join(["universe", str(self), str(date)])
        if use_cache and self.rdb and self.rdb.redis.exists(rkey):
            self._print('(get_universe load)', rkey)
            return self.rdb.load(rkey)
        df = self.get_section(dataset='daily', fields=['prc', 'shrout'],
                              date_field='date', date=date)
        df['cap'] = df['shrout'] * df['prc'].abs()
        df = df.join(self.get_section(dataset='names',
                                      fields=['shrcd','exchcd','siccd','naics'],
                                      date_field='date', date=date, start=0),
                     how='left')
        self._print('LENGTH PERMNOS', str(len(df)))
        self._print('PRC NULL:', sum(df['prc'].isna()),
                      'NEG:', sum(df['prc'] <= 0))
        self._print('SHR ZERO:', sum(df.shrout <= 0))
        self._print('CAP NON-POSITIVE:', len(df) - sum(df.cap.gt(0)))

        df = df[df['cap'].notna() & df['cap'].gt(0) & df['prc'].abs().gt(minprc)
                & df['shrcd'].isin([10, 11]) & df['exchcd'].isin([1, 2, 3])]
        df['nyse'] = df['exchcd'].eq(1)                    # nyse indicator
        df['decile'] = fractiles(values=df['cap'],         #  size deciles 
                                 pctiles=np.arange(10, 100, 10),
                                 keys=df.loc[df['nyse'], 'cap'], ascending=False)
        df = df[['cap', 'decile', 'nyse', 'siccd', 'prc', 'naics']]
        if use_cache is not None and self.rdb:
            self._print('(get_universe dump)', rkey)
            self.rdb.dump(rkey, df)
        return df

    def get_divamt(self, start, end):
        """Accmumulates total dollar dividends between start and end dates

        Parameters
        ----------
        start: int
            Inclusive start date (YYYYMMDD)
        end: int
            Inclusive end date (YYYYMMDD)

        Returns
        -------
        DataFrame
            Accumulated divamt = per share divamt * shrout, indexed by permno

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

    def get_dlstret(self, start, end, use_cache=True):
        """Compound delisting returns from start to end dates for all permnos

        Parameters
        ----------
        start: int
            Inclusive start date (YYYYMMDD)
        end: int
            Inclusive end date (YYYYMMDD)
        use_cache: bool, default is True (read, if exists, and write cache)
            If False, then write but not read cache.  None to ignore cache

        Returns
        -------
        DataFrame (possibly empty)
            Compounded returns in column 'ret', indexed by permno
        """
        rkey = "_".join(["dlst", str(self), str(start), str(end)])
        if use_cache and self.rdb and self.rdb.redis.exists(rkey):
            self._print("(get_dlstret load)", rkey, str(self))
            return self.rdb.load(rkey)

        q = ("SELECT (1+dlret) AS ret, {identifier} FROM {table} "
             "  WHERE dlstdt >= {start} AND dlstdt <= {end}").format(
                 table=self['delist'].key,
                 identifier=self.identifier,
                 start=start,
                 end=end)
        self._print('(get_dlst)', q)
        df = self.sql.read_dataframe(q).sort_values(self.identifier)
        df = (df.groupby(self.identifier).prod(min_count=1) - 1).dropna()
        if use_cache is not None and self.rdb:
            self._print("(get_dlstret dump)", rkey, str(self))
            self.rdb.dump(rkey, df)
        return df

    def get_ret(self, start, end, *args, delist=False, **kwargs):
        """Get compounded returns, with option to include delist returns"""
        ret = super().get_ret(start, end, *args, **kwargs)
        if (delist and 'delist' in self.tables_ and
            self.use_monthly(start, end)):  # if using delist and monthly tables
            dlst = self.get_dlstret(start, end)
            permnos = list(set(ret.index).intersection(dlst.index))
            if len(permnos):
                ret.loc[permnos,'ret'] = ((1 + ret.loc[permnos,'ret']) *
                                          (1 + dlst.loc[permnos,'ret'])) - 1
        return ret


class PSTAT(Structured):
    """Provide interface to Compustat structured data sets

    Parameters
    ----------
    sql : SQL instance
        Connection to mysql database
    bd: BusDays instance
        Custom business days object

    Attributes
    ----------
    role_, event_ : Series
        maps role or event id (in index) to description

    Notes
    -----
    Screen on INDFMT= 'INDL', DATAFMT='STD', POPSRC='D', and CONSOL='C' 
    to keep vast majority of records and uniquely identifies GVKEY, DATADATE.
    """
    role_ = Series({   # Key Development role id labels
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
    event_ = Series({   # Key Development event id labels
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
    
    def __init__(self, sql, bd, name='PSTAT', echo=ECHO):
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
                    #'aco','act','ao','ap','at',
                    #'capx','ceq','che','cogs','csho',
                    #'cshrc','dcpstk','dcvt','dlc','dltt',
                    #'dm','dp','drc','drlt',
                    #'dv','dvt','ebit','ebitda','emp',
                    #'fatb','fatl','gdwl','gwo',
                    #'ib','intan','invt','lco','lct',
                    #'lo','lt','ni','nopi','oancf',
                    #'ob','pi','ppegt','ppent','pstk',
                    #'pstkl','pstkrv','rect',
                    #'revt','sale','scstkc','seq','spi',
                    #'txditc','txfed','txfo',
                    #'txp','txt','xad','xint','xrd','xsga'
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
                Column('keydevtoobjectroletypeid', SmallInteger,
                       primary_key=True),
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
                         echo=echo)

    def build_lookup(self, source, target, date_field='linkdt',
                     table='links', fillna=None):
        """Builds callable to look-up record by identifier and prevailing date"""
        if not (source in self[table].c and target in self[table].c):
            raise Exception(f"{source} and {target} must be in {table}")
        return self._lookup(self.sql, source, target, date_field,
                            self[table].key, fillna)

    def get_links(self, gvkeys, date):
        """Return list of permnos mapped to gvkeys as of a prevailing date"""
        return super().get_links(gvkeys, date, 'lpermno', 'linkdt')

    def get_linked(self, dataset, date_field, fields, where='', limit=None):
        """Query a pstat table, and return with linked crsp permno

        Parameters
        ----------
        dataset: str
            Dataset to query
        date_field: str
            Name of date field in pstat table to query
        fields : list of string
            Fields to return
        where : string or dict (optional)
            Sql where clause, as sql string or dict of {field : value}
        limit : int (optional)
            Maximum number of records to return (default None for all)

        Examples
        --------
        df = pstat.get_linked(dataset='annual', date_field='datadate',
                fields=['ceq','pstkrv','pstkl','pstk'],
                where='ceq > 0 and datadate>=19930104 and datadate<=20991231')
        df = keydev.get_linked(dataset='keydev', date_field='announcedate',
                   fields=['companyname', 'keydeveventtypeid',
                   'keydevtoobjectroletypeid'],
                   where='', limit=''):

        Notes
        -----
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
        return super().get_linked(dataset, date_field, fields, 'lpermno',
                                  'linkdt', where=where, limit=limit)

    
class IBES(Structured):
    """Provide interface to IBES analyst estimates structured data sets

    Parameters
    ----------
    sql : SQL instance
        Connection to SQL database
    bd: BusDays instance
        Custom business days object
    """
    def __init__(self, sql, bd, name='IBES', echo=ECHO):
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
                         echo=echo)

    def build_lookup(self, source, target, date_field='sdates',
                     table='ident', fillna=None):
        """Builds callable to look-up by identifier and prevailing date"""
        if not (source in self[table].c and target in self[table].c):
            raise Exception(f"{source} and {target} must be in {table}")
        return self._lookup(self.sql, source, target, date_field,
                            self[table].key, fillna)

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

    def get_links(self, tickers, date):
        """Return list of permnos mapped to tickers as of a prevailing date"""
        return super().get_links(tickers, date, 'permno', 'sdates')

    def get_linked(self, dataset, date_field, fields, where='', limit=None):
        """Query an ibes table, and return with linked crsp permnos

        Parameters
        ----------
        dataset: str
            Dataset to query
        date_field: str
            Name of date field in ibes table to query
        fields : list of string
            Fields to return
        where : string or dict (optional)
            Sql where clause, as sql string or dict of {field : value}
        limit : int (optional)
            Maximum number of records to return (default None for all)

        Examples
        --------
        ibes.get_linked('ident', date_field='statpers', fields=['cname']):

        # where fpi='6'  /* 1 is for annual forecasts, 6 is for quarterly */
        # and statpers < ANNDATS_ACT /* forecasts prior to earnings annoucement
        # and measure='EPS' and not missing(medest)
        # and not missing(fpedats)  and (fpedats-statpers)>=0;
        # (fpedats-statpers)>=0;
        """
        return super().get_linked(dataset, date_field, fields, 'permno',
                                  'sdates', where=where, limit=limit)

class chunk_stocks:
    """Cached daily returns, to provide Stocks-like interface"""
    
    def __init__(self, stocks, beg, end, fields=['ret', 'retx'],
                 identifier='permno'):
        """Create object and load daily returns into its cache

        Parameters
        ----------
        stocks : Stocks structured data object
           To access stock returns data
        beg : int
            Earliest date of daily stock returns to load
        end : int
            Latest date of daily stock returns to load
        fields : str list, default is ['ret', 'retx']
            Column names of returns fields to load
        """
        q = (f"SELECT permno, date, {', '.join(fields)} FROM "
             f" {stocks['daily'].key} WHERE date>={beg} AND date<={end}")
        self.rets = stocks.sql.read_dataframe(q).sort_values(['permno', 'date'])
        self.fields = fields
        self.identifier = identifier
        self.bd = stocks.bd

    def get_ret(self, beg, end, fields=None):
        """Return compounded stock returns between beg and end dates"""
        df = self.rets[self.rets['date'].between(beg,end)].drop(columns=['date'])
        df.loc[:, self.fields] += 1
        df = (df.groupby(self.identifier).prod(min_count=1) - 1).fillna(0)
        return df[fields or self.fields]

class chunk_signal:
    """Cache dataframe of signals values, provide Signals-like interface"""
    def __init__(self, df, identifier='permno'):
        """Initialize instance from input dataframe"""
        self.data = df
        self.identifier = identifier

    def __call__(self, label, date, start, rebaldate='rebaldate'):
        """Select from rebaldates that fall between start and date, keep latest

        Parameters
        ----------
        label : str
            Column to return
        date : int
            End date
        start : int
            Non-inclusive start date. Set to [0] for all, 
            [previous busday] for exact date, [previous endmo] for any in month
        rebaldate: str, optional (default = 'rebaldate')
            Column name of rebaldate
        """        
        df = self.data.loc[self.data[rebaldate].le(date) &
                           self.data[rebaldate].gt(start),
                           [self.identifier, rebaldate, label]]
        df = df.sort_values([self.identifier, rebaldate], na_position='first')\
               .drop_duplicates([self.identifier], keep='last').dropna()
        return df.set_index(self.identifier)

class Signals(Stocks):
    """Provide structured stocks data interface to derived signal values 

    Parameters
    ----------
    sql : SQL instance
        Connection to SQL database
    """
    def __init__(self, sql, echo=ECHO):
        """Initalize a connection to derived Signals values datasets"""
        super().__init__(sql, bd=None, tables=None, identifier='permno',
                         name='signals', echo=echo)

    def __call__(self, label, date, start=None):
        """Return cross-section of signal values available as of a date

        Parameters
        ----------
        label : str
            Name of signal to retrieve
        date : int
            Rebalance date (YYYYMMDD format)
        start : int (YYYYMMDD format), optional
            Non-inclusive start date of range to search for signal value
            (default None means exact date)
        """
        return self.get_section(dataset=label, fields=['rebaldate', label],
                                date_field='rebaldate', date=date, start=start)

    def table_key(self, label):
        """Helper method generates table key name associated with label"""
        return '__' + label     # prefix with "__"

    def __getitem__(self, label):
        """Overrides class method to return Table schema of label"""
        return self.sql.Table(
            self.table_key(label),   # dynamically generate new table name
            Column('permno', Integer, primary_key=True),
            Column('rebaldate', Integer, primary_key=True),
            Column(label, Float))

    def summary(self, label):
        """Perform a 'proc summary' by rebaldate on a signal's values"""
        return self.sql.summary(self.table_key(label), label, key='rebaldate')

    def write(self, data, label, overwrite=True, rebaldate='rebaldate',
              permno='permno'):
        """Saves a new sql table from dataframe of signal values

        Parameters
        ----------
        data : DataFrame
            Signal values, with columns = ['permno', 'rebaldate', label]
        label : string
            Name of signal. becomes name of column and table (prefixed by '__')
        overwrite : bool, optional (default is True)
            If False, append to table ignoring duplicates. Else recreate table
        permno : string (optional), default is 'permno'
            Column name of permno identifiers in input dataframe
        rebaldate : string (optional), default is 'rebaldate'
            Column name of rebalance dates in input dataframe

        Returns
        -------
        n : int
            number of rows saved

        Notes
        -----
        first removes duplicate keys, and drops null rows before saving to table
        """

        df = data[[permno, rebaldate, label]].copy()
        df.index.name = None # 'permno' may be both index level and column label
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

    def read(self, label, where=''):
        """Read signal values from sql and return as data frame

        Parameters
        ----------
        label : str
            Name of signal
        where : dict or str
            Where conditions

        Returns
        -------
        df : DataFrame
            columns = ['permno', 'rebaldate', label]
        """
        where = parse_where(where, 'WHERE')
        table = self.table_key(label)
        q = f"SELECT permno, rebaldate, {label} FROM {table} {where}"
        return self.sql.read_dataframe(q).sort_values(['permno', 'rebaldate'])

class Finder:
    """This class builds a general method to lookup tables by any identifier"""

    def __init__(self, sql, identifier=None, table=None):
        """Initialize lookup method with optional identifier type and table

        Examples
        --------
        find = Find(sql, identifier='comnam', table='names')
        """
        self.sql = sql
        self.identifier = identifier
        self.table = table

    def __call__(self, label, identifier=None, table=None, **kwargs):
        """Lookup a label. Guesses identifier type and table if not specified
        or initialized

        Examples
        --------
        crsp name:   find('GOOG', 'comnam')
        pstat name:  find('GOOG', 'conm')
        ibes name:   find('GOOG', 'cname')
        crsp permno: find(18144)
        pstat gvkey: find(328795, 'gvkey')
        ibes ticker: find('0011', 'ticker', 'ident')
        crsp ticker: find('aapl')
        crsp cusip : find('03783310')
        pstat cusip: find('03783310','cusip','links')
        ibes cusip : find('03783310','cusip','ident')

        find('45483', 'permco', 'names')
        """
        if len(kwargs):
            for k,v in kwargs.items():
                identifier = k
                label = v
        label = str(label).upper()
        if identifier is None:   # guess identifier if not specified
            if self.identifier is not None:
                identifier = self.identifier  # was initialized
            elif len(label) == 5 and label.isnumeric():
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
        if table is None:   # guess table if not specified
            if self.table is not None:
                table = self.table           # was initialized
            elif identifier in ['permno', 'ncusip', 'tsymbol', 'comnam']:
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

def famafrench_sorts(stocks, label, signals, rebalbeg, rebalend, 
                     window=0, pctiles=[30, 70], leverage=1, months=[], 
                     minobs=100, minprc=0, mincap=0, maxdecile=10,
                     rebals='endmo'):
    """Generate time series of holdings by two-way sort procedure

    Parameters
    ----------
    stocks : Structured object
        Stock returns and price data
    label : string
        Signal name to retrieve either from Signal sql table or {data} dataframe
    signals : Signals, or chunk_signal object
        Call to extract cross section of values for the signal
    rebalbeg : int
        First rebalance date (YYYYMMDD)
    rebalend : int
        Last holding date (YYYYMMDD)
    pctiles : tuple of int, default is [30, 70]
        Percentile breakpoints to sort into high, medium and low buckets
    window: int, optional (default 0)
        Number of months to look back for signal values (non-inclusive day),
        0 (default) is exact date
    months: list of int, optional
        Month/s (e.g. 6 = June) to retrieve universe and market cap,
        [] (default) means every month
    maxdecile: int, default is 10
        Include largest stocks from decile 1 through decile (10 is smallest)
    minobs: int, optional
        Minimum required universe size with signal values
    leverage: numerical, default is 1.0
        Leverage multiplier, if any
    rebals: str or list of int (default is 'endmo')
        rebalance freq str, or list of rebaldates

    Notes
    -----
    Independent sort by median (NYSE) mkt cap and 30/70 (NYSE) HML percentiles
    Subportfolios of the intersections are value-weighted; 
       spread portfolios are equal-weighted subportfolios
    Portfolio are resorted every June, and other months' holdings are adjusted 
      by monthly realized retx
    """
    if isinstance(rebals, str):
        rebals = stocks.bd.date_range(rebalbeg, rebalend, freq=rebals)
    else:
        rebals = sorted(set(rebals).union([rebalbeg, rebalend]))
    if months:   # identify rebals in range
        keys = {k: [] for k in set([r//100 for r in rebals])
                if k % 100 in months}
        for r in rebals:
            if (r // 100) in keys:
                keys[r//100].append(r)
        months = [max(v) for v in keys.values()]
    holdings = {label: dict(), 'smb': dict()}  # to return two sets of holdings
    sizes = {h : dict() for h in ['HB','HS','MB','MS','LB','LS']}
    for rebal in rebals:  #[:-1]

        # check if this is a rebalance month
        if not months or rebal in months or not holdings[label]:
            
            # rebalance: get this month's universe of stocks with valid data
            df = stocks.get_universe(rebal)
            
            # get signal values within lagged window
            start = (stocks.bd.endmo(rebal, months=-abs(window)) if window
                     else stocks.bd.offset(rebal, offsets=-1))
            signal = signals(label=label, date=rebal, start=start)
            df[label] = signal[label].reindex(df.index)

            df = df[df['prc'].abs().gt(minprc) &
                    df['cap'].gt(mincap) &
                    df['decile'].le(maxdecile)].dropna()

            if (len(df) < minobs):  # skip if insufficient observations
                continue

            # split signal into desired fractiles, and assign to subportfolios
            df['fractile'] = fractiles(df[label],
                                       pctiles=pctiles,
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
                sizes[subname][rebal] = sum(subs[subname])
            for subname, weight in zip(['HB','HS','MB','MS','LB','LS'],
                                       [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]):
                cap = df.loc[subs[subname], 'cap']
                weights['smb'][subname] = leverage * weight * cap / cap.sum()
                sizes[subname][rebal] = sum(subs[subname])
            #print("(famafrench_sorts)", rebal, len(df))
            
        else:  # else not a rebalance month, so simply adjust holdings by retx
            retx = 1 + stocks.get_ret(stocks.bd.offset(prevdate, 1),
                                      rebal, field='retx')['retx']
            for port, subports in weights.items():
                for subport, old in subports.items():
                    new = old * retx.reindex(old.index, fill_value=1)
                    weights[port][subport] = new / (abs(np.sum(new))
                                                    * len(subports) / 2)

        # combine this month's subportfolios
        for h in holdings:
            holdings[h][rebal] = pd.concat(list(weights[h].values()))
        prevdate = rebal
    return {'holdings': holdings, 'sizes': sizes}
    rebaldates = stocks.bd.date_range(rebalbeg, rebalend, 'endmo')
    holdings = {label: dict(), 'smb': dict()}  # to return two sets of holdings
    sizes = {h : dict() for h in ['HB','HS','MB','MS','LB','LS']}
    for rebaldate in rebaldates:  #[:-1]

        # check if this is a rebalance month
        if not months or (rebaldate//100)%100 in months or not holdings[label]:
            
            # rebalance: get this month's universe of stocks with valid data
            df = stocks.get_universe(rebaldate)
            
            # get signal values within lagged window
            start = (stocks.bd.endmo(rebaldate, months=-abs(window)) if window
                     else stocks.bd.offset(rebaldate, offsets=-1))
            signal = signals(label=label, date=rebaldate, start=start)
            df[label] = signal[label].reindex(df.index)

            df = df[df['prc'].abs().gt(minprc) &
                    df['cap'].gt(mincap) &
                    df['decile'].le(maxdecile)].dropna()

            if (len(df) < minobs):  # skip if insufficient observations
                continue

            # split signal into desired fractiles, and assign to subportfolios
            df['fractile'] = fractiles(df[label],
                                       pctiles=pctiles,
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
                                      rebaldate, field='retx')['retx']
            for port, subports in weights.items():
                for subport, old in subports.items():
                    new = old * retx.reindex(old.index, fill_value=1)
                    weights[port][subport] = new / (abs(np.sum(new))
                                                    * len(subports) / 2)

        # combine this month's subportfolios
        for h in holdings:
            holdings[h][rebaldate] = pd.concat(list(weights[h].values()))
    return {'holdings': holdings, 'sizes': sizes}

if False:
    from settings import settings
    import os
    import glob
    import time
    from pandas import DataFrame, Series
    from finds.database import SQL, Redis
    from finds.busday import BusDay, Weekly
    from finds.structured import PSTAT, CRSP, IBES, Benchmarks, Signals
    from finds.readers import fetch_FamaFrench

if False:   # open all structured datasets
    sql = SQL(**settings['sql'], echo=True)
    user = SQL(**settings['user'], echo=True)
    rdb = Redis(**settings['redis'])
    bd = BusDay(sql)
    bench = Benchmarks(sql, bd)
    crsp = CRSP(sql, bd, rdb)
    pstat = PSTAT(sql, bd)
    ibes = IBES(sql, bd)
    signals = Signals(user)

if False:  # to populate SQL data tables from csv raw input
    downloads = os.path.join(settings['remote'], 'stocks2021')
    sql = SQL(**settings['sql'], echo=True)
    rdb = None
    bd = BusDay(sql)
        
    # load benchmarks (mostly FamaFrench)
    bench = Benchmarks(sql, bd)
    datasets = fetch_FamaFrench()
    print("\n".join(f"[{i}] {d}" for i, d in enumerate(datasets)))
    for name, item, suffix in datasets:
        df = fetch_FamaFrench(name=name, item=item, suffix=suffix,
                              index_formatter=bd.offset)
        for col in df.columns:
            bench.load_series(df[col], name=name, item=item)
    print(DataFrame(**sql.run('select * from ' + bench['ident'].key)))

    dirname = os.path.join(downloads, 'CRSP') + '/'
    df = pd.read_csv(dirname + 'indexes.txt.gz', sep='\t').set_index('DATE')
    for col in df.columns:
        bench.load_series(df[col].dropna(), name='crsp')
    df = pd.read_csv(dirname + 'sbbi.txt.gz', sep='\t').set_index('caldt')
    df.columns = df.columns + "(mo)"
    for col in df.columns:
        bench.load_series(df[col].dropna(), name='sbbi')

    
    # load CRSP: TODO handle missing return codes (< -1, see below)
    crsp = CRSP(sql, bd, rdb)
    dirname = os.path.join(downloads, 'CRSP') + '/'
    crsp.load_csv('names', dirname +  'names.txt.gz', sep='\t')   # 103383
    crsp.load_csv('shares', dirname + 'shares.txt.gz', sep='\t') # 2346131
    crsp.load_csv('dist', dirname + 'dist.txt.gz', sep='\t') # 935880
    crsp.load_csv('delist', dirname + 'delist.txt.gz', sep='\t')  # 33584
    crsp.load_csv('monthly', dirname + 'monthly.txt.gz', sep='\t') #4606907
    for i, s in enumerate(sorted(glob.glob(dirname + 'stocks*.txt.gz'))):
        tic = time.time()
        crsp.load_csv('daily', s, sep='\t')
        print(s, time.time() - tic)
    
    # load IBES
    ibes = IBES(sql, bd)
    dirname = os.path.join(downloads, 'IBES') + '/'    
    ibes.load_csv('ident', dirname + 'ident.txt.gz', sep='\t')  # 85550
    ibes.write_links()  #  (missing, count) = 14642  85550
    ibes.load_csv('summary', dirname + 'summary.txt.gz', sep='\t') # 8470688
    #ibes.load_csv('adjust', downloads + 'adjustment.csv') #rows=24777
    #ibes.load_csv('surprise', downloads + 'surprise.csv')  #rows=528933

    # load Compustat
    pstat = PSTAT(sql, bd)
    dirname = os.path.join(downloads, 'PSTAT') + '/'
    df = pstat.load_csv('links', dirname + 'links.txt.gz', sep='\t', # rows=33036
                        drop={'lpermno': ['0', 0], 'linkprim': ['N', 'J']},
                        replace={'linkdt': [['C', 'E', 'B'], 0],
                                 'linkenddt': [['C', 'E', 'B'], 0]})
    lag = df.shift()
    f = (lag.gvkey == df.gvkey) & (lag.lpermno != df.lpermno)
    print('permnos in links changed in ', sum(f), 'of', len(df)) # 1063
    pstat.load_csv('annual', dirname + 'pstat.csv.gz') #rows = 464753
    pstat.load_csv('quarterly', dirname +  'quarterly.csv.gz') # 1637274
    pstat.load_csv('customer', dirname + 'supplychain.csv.gz') #107114
    for s in glob.glob(dirname +  'keydev*.txt.gz'):
        tic = time.time()   # 12256909
        df = pstat.load_csv('keydev', s, sep='\t',
                            drop={'gvkey': [0, '0'],
                                  'announcedate': [0, '0'],
                                  'keydevid': [0, '0']})
        print(s, time.time() - tic)


            
if False:
    from finds.busday import minibatch
    rebalbeg, rebalend = 19730701, 20210101
    rebalbeg = 19251231
    wd = Weekly(sql, 'Fri', rebalbeg, rebalend)
    beg = bd.offset(wd.weeks['beg'].to_list(), 1)
    dates = [(a,b) for a,b in zip(beg, wd.weeks['end'])]
    #dates = list(wd.weeks[['beg','end']].itertuples(index=False, name=None))
    for datebatch in minibatch(dates, batchsize=40):
        crsp.cache_ret(datebatch, overwrite=True)
