"""SQL class wrapper, with convenience methods for pandas DataFrames

Copyright 2022-2024, Terence Lim

MIT License
"""
from typing import List, Dict, Mapping, Any, Tuple
import random
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sqlalchemy
from sqlalchemy import text, Integer, SmallInteger, Boolean, Float, String
from sqlalchemy.orm import sessionmaker      
from finds.database import Database

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
        try:
            if col in replace:
                df[col] = df[col].replace(*replace[col])
            if isinstance(v, Integer) or isinstance(v, SmallInteger):
                df[col] = df[col].replace("(?<=\d)-","", regex=True) # crsp dates
                df[col] = df[col].replace('', 0).astype(int)
            elif isinstance(v, Boolean):
                df[col] = df[col].replace('', False).astype(bool)
            elif isinstance(v, Float):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            elif isinstance(v, String):
                df[col] = df[col].astype(str).str.encode('ascii', 'ignore')\
                                                 .str.decode('ascii')
            else:
                raise Exception('(as_dtypes) Unknown type for column: ' + col)
        except:
            raise Exception('(as_dtypes) bad data in column: ' + col)
    return df




class SQL(Database):
    """Interface to sqlalchemy, with convenience functions for dataframes"""

    def __init__(self,
                 user: str,
                 password: str,
                 host: str = 'localhost',
                 port: str = '3306',
                 database: str = '',
                 autocommit: str = 'true',
                 charset: str = 'utf8',
                 temp: str = f"temp{random.randint(0, 8192)}",
                 **kwargs):
        super().__init__(**kwargs)
        self.url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"\
            + f"?charset={charset}&local_infile=1&autocommit={autocommit}"
        self._t = temp  # name of temp table for this process
        self.create_engine()

    @staticmethod
    def create_database(user: str, password: str, host: str = 'localhost',
                        port: str = '3306', database: str = '', **kwargs):
        """Create new database using this user's credentials"""
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}"        
        engine = sqlalchemy.create_engine(url)
        with engine.begin() as conn:
            conn.execute(text("COMMIT"))
            conn.execute(text(f"CREATE DATABASE {database}"))

    def create_engine(self):
        """Call and store sqlalchemy.create_engine() and MetaData()"""
        self.engine = sqlalchemy.create_engine(self.url, echo=self._verbose > 0)
        self.metadata = sqlalchemy.MetaData()

    def rollback(self):
        """Call sessionmaker() to rollback current transaction in progress"""
        Session = sessionmaker(self.engine)
        with Session() as session:
            session.rollback()

    def Table(self, key: str, *args, **kwargs) -> sqlalchemy.Table:
        """Wraps sqlalchemy.Table() after removing key from metadata"""
        if key in self.metadata.tables:    # remove from metadata if existed
            self.metadata.remove(self.metadata.tables[key])
        table = sqlalchemy.Table(key, self.metadata, *args, **kwargs)
        #self.metadata.create_all(self.engine)
        return table

    def create_all(self):
        """Create all tables in metadata"""
        self.metadata.create_all(self.engine)        

    @classmethod
    def Index(cls, *args) -> sqlalchemy.Index:
        """Wraps sqlalchemy.Index() with auto-generated index name from args"""
        return sqlalchemy.Index("_".join(args), *args)

    def remove(self, key: str):
        """Remove a table by key name from metadata instance"""
        if key in self.metadata.tables:
            self.metadata.remove(self.metadata.tables[key])

    def run(self, q) -> Dict | None:
        """Execute sql command

        Args:
          q: query string

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

        if isinstance(q, str):
            q = text(q)
        for _ in range(2):
            try:
                with self.engine.begin() as conn:
                    try:
                        r = conn.execute(q)
                        return {'data': r.fetchall(), 'columns': r.keys()}
                    except Exception:
                        return None
                break
            except Exception as e:
                self._print(e)
                self.create_engine()
        raise RuntimeError('(sql.run) ' + q)

    def summary(self,
                table: str,
                val: str,
                key: str = '') -> DataFrame:
        """Return summary statistics for a field, optionally grouped-by key

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

    def load_infile(self,
                    table: str,
                    csvfile: str,
                    options: str =''):
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

    def load_dataframe(self,
                       table: str, 
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

    def pivot(self,
              table: str, 
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

        if where and not where.strip().upper().startswith('WHERE'):
            where = 'WHERE ' + where

        if isinstance(chunksize, int):  # execute in chunks
            rows = self.read_dataframe(
                f"SELECT DISTINCT {index} FROM {table} {where}"
            )
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
                out = out.append(df.pivot(index=index,
                                          columns=columns,
                                          values=values),
                                 sort=True)
            return out
        else:  # execute as single chunk
            where += " LIMIT " + str(limit) if limit else ''
            q = f"SELECT {index}, {columns}, {values} FROM {table} {where}"
            self._print('(pivot)', q)
            return self.read_dataframe(q).pivot(index=index,
                                                columns=columns,
                                                values=values)

if __name__ == "__main__":
    from secret import credentials
    VERBOSE = 1

    # Create new databases
    SQL.create_database(**credentials['sql'])
    SQL.create_database(**credentials['user'])

    # Show data tables
    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    print(sql.run('show tables'))

    # Show user tables
    user = SQL(**credentials['user'], verbose=VERBOSE)
    print(user.run('show tables'))

    # test a transaction
    df = DataFrame(data=[[1, 1.5, 'a'], [2, '2.5', None]],
                   columns=['a', 'b', 'c'],
                   index=['d', 'e'])
    user.run('drop table if exists test')
    user.load_dataframe('test', df)
    s = user.run('select * from test')
    print('test:')
    print(DataFrame(**s))
        
    
