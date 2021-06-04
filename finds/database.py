"""Convenience class and methods to interface with database engines

- SQL, sqlalchemy
- MongoDB, pymongo
- redis

Author: Terence Lim
License: MIT
"""
import redis
import sqlalchemy
import pymongo
from pyarrow import default_serialization_context as pa
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import os
import warnings
ECHO = False

class SQL(object):
    """Provides convenience interface sqlalchemy engine

    Parameters
    ----------
    user, password, host : string
        to connect to mysql server
    port: int
        to connect to mysql server        
    database : string
        database name

    Attributes
    ----------
    engine : sqlalchemy engine instance
        python interface to sqlalchemy commands
    metadata : sqlalchemy metadata instance
        collects table objects and their associated schema constructs

    Notes
    -----
    SQLAlchemy methods:
    metadata.clear
    metadata.create_all
    metadata.remove
    metadata.tables
    table.key
    table.c
    table.columns
    table.create
    table.delete
    table.drop
    table.exists
    table.insert
    table.select
    table.update
    """
    def __init__(self, user='', password='', host='localhost', port='3306',
                 database='', autocommit='true', charset='utf8',
                 temp = "temp" + str(np.random.randint(8192)), echo=ECHO):
        """Initialize a sqlalchemy connection"""
        self.q = (f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?"
                  f"charset={charset}&local_infile=1&autocommit={autocommit}")
        self.echo_ = echo
        self.temp_ = temp  # name of temp table for this process
        self._create_engine()

    def _print(self, *args, echo=None):
        if echo or self.echo_:
            print(*args)

    def _create_engine(self):
        self.engine = sqlalchemy.create_engine(self.q, echo=self.echo_)
        self.metadata = sqlalchemy.MetaData(self.engine)
        
    def Table(self, key, *args, **kwargs):
        """Wraps sqlalchemy.Table(), to first remove table entry from metadata"""
        if key in self.metadata.tables:
            self.metadata.remove(self.metadata.tables[key])
        return sqlalchemy.Table(key, self.metadata, *args, **kwargs)

    @classmethod
    def Index(cls, *args):
        """Wraps sqlalchemy.Index(), to infer an index name from args"""
        return sqlalchemy.Index("_".join(args), *args)

    def remove(self, key):
        """Remove a table, by its key name, from metadata instance"""
        if key in self.metadata.tables:
            self.metadata.remove(self.metadata.tables[key])

    def run(self, q, *args, **kwargs):
        """Execute a sql command

        Returns
        -------
        r : result object
            A result set or None.

        Examples
        --------
        sql.run('select * from testing')
        sql.run('select distinct permno from benchmarks')
        sql.run("show databases")
        sql.run("show tables")
        sql.run("show create table _")
        sql.run("describe _")
        sql.run("truncate table _", fetch=False)
        """
        
        for _ in range(2):
            try:
                with self.engine.begin() as conn:
                    try:
                        r = conn.execute(q, *args, **kwargs)
                        return {'data': r.fetchall(), 'columns': r.keys()}
                    except:
                        return None
                break
            except:
                self._create_engine()
        raise Exception('(sql.run) ' + q)

    def summary(self, table, val, key=None):
        """Return summary statistics of a field, optionally grouped-by

        Parameters
        ----------
        table : str
            Name of table
        val : str
            Field name to summarise
        key : str (optional)
            Field to group by

        Returns
        -------
        DataFrame
            count, average, max, min

        Examples
        --------
        sql.summary('annual','revt','sic')
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
        
    def load_infile(self, table, csvfile, options=''):
        """Load table from csv file, using mysql's load data local infile"""
        q = (f"LOAD DATA LOCAL INFILE '{csvfile}' INTO TABLE {table} "
             f" FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED "
             f" BY '\\n' IGNORE 1 ROWS {options};")
        try:
            print_verbose("(load_infile)", q)
            self.run(q)
        except Exception as e:
            print("(load_infile) Got exception = ", e, " Query = ", q)
            raise e

    def load_dataframe(self, table, df, index_label=None, to_sql=True,
                       if_exists='append'):
        """Load dataframe into sql table, ignoring duplicate keys

        Parameters
        ----------
        table : str 
            Physical name of table to insert into
        df: DataFrame
            The source dataframe
        index_label: string, optional
            Column name to load dataframe index as, None to not load (default)
        to_sql: boolean, default True
            If True, attempt pandas.to_sql(), which may fail if duplicate keys
            If False or pandas.to_sql() fails, put in temp and insert ignore
        if_exists: string, optional
            Action to take if table exists -- 'replace' or 'append' (default)
        
        """
        df.columns = df.columns.map(str.lower).map(str.rstrip)
        chunksize = (int) (1024*1024*32 // len(df.columns))
        try:     # to_sql raises exception if exist duplicate keys
            assert(to_sql)
            df.to_sql(table,
                      self.engine,
                      if_exists = if_exists,
                      chunksize=chunksize,
                      index = (index_label is not None),
                      index_label = index_label)
        except:  # duplicates exists, so to_sql to temp, and insert ignore
            print_verbose("(load_dataframe) Retrying insert ignore", table)
            self.run('drop table if exists ' + self.temp_)
            df.to_sql(self.temp_,
                      self.engine,
                      if_exists='replace',
                      chunksize=chunksize,
                      index=(index_label is not None),
                      index_label=index_label)
            # warnings.filterwarnings("ignore", category=pymysql.Warning)
            columns = ", ".join(df.columns)
            q = (f"INSERT IGNORE INTO {table} ({columns})"
                 f" SELECT {columns} FROM {self.temp_}")
            self.run(q)
            # warnings.filterwarnings("default", category=pymysql.Warning)
            self.run('drop table if exists ' + self.temp_)

    def read_dataframe(self, query):
        """Return sql query result as data frame

        Parameters
        ----------
        query: str, or SQLAlchemy Selectable
            SQL query or a table name
        coerce_float : boolean, default True
            Attempt to convert values of non-string, non-numeric to float
        """
        result = self.run(query)
        if result is None:
            raise Exception('read_dataframe error in database: ', str(query))
        return DataFrame(**result)

    def pivot(self, table, index, columns, values, where='', limit=None,
              chunksize=None):
        """Return sql query result as pivoted data frame

        Parameters
        ----------
        table: str
            Physical name of table to retrieve from
        index: str
            Field to select as dataframe index
        columns: str
            Field to select as column labels
        values: str
            field to select as values
        where: str or dict
            Where clause, default ''
        limit: int, or None (default)
            Maximum number of rows or chunks to return            

        Returns
        -------
        DataFrame
            pivoted data frame
        """
        where = parse_where(where, 'WHERE')
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
                row = slice(chunksize * i, min(n_features, chunksize * (i + 1)))
                print(i, n_splits)
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
            print_verbose('(pivot)', q)
            return self.read_dataframe(q).pivot(
                index=index, columns=columns, values=values)
        
class Redis(object):
    """Provides convenience interface to redis key-value store

    Parameters
    ----------
    host: str
        IP address of host
    post: int
        Port number
    **kwargs:
        Arguments passed on to redis.StrictRedis constructor

    Attributes
    ----------
    redis: redis object
        python interface to all Redis commands

    Notes
    -----
    Methods:
    redis.delete(key)      -- delete an item
    redis.get(key)         -- get an item
    redis.exists(key)      -- does item exist
    redis.set(key, value)  -- set an item
    redis.keys()           -- get keys

    Command Line:
    decode_responses=False to zlib dataframe
    ./redis-5.0.4/src/redis-server
    ./redis-cli --scan --pattern '*CRSP_2020*' | xargs ./redis-cli del
    CLI> keys *
    CLI> flushall
    CLI> info memory"""

    def __init__(self, **kwargs):
        """Initialize a Redis connection"""
        self.redis = redis.StrictRedis(**kwargs)
        
    def load(self, key):
        """Return dataframe, using pyarrow, given its key from redis store"""
        df = pa().deserialize(self.redis.get(key))  # must use pyarrow
        return df.copy()   # return copy lest flag.writable is False

    def dump(self, key, df):
        """Saves dataframe, using pyarrow, by key name to redis store"""
        self.redis.set(key, pa().serialize(df).to_buffer().to_pybytes())


class MongoDB(object):
    """Provides convenience interface to pymongo database

    Parameters
    ----------
    database : str
        Name of database in MongoDB
    host : str
        IP address of server
    port: int
        Port number

    Attributes
    ----------
    client: MongoClient object
       python interface to all MongoDB client commands

    Examples
    --------
    >>> serverStatusResult = client.admin.command("serverStatus")
    >>> pprint(serverStatusResult)
    >>> collections = client['database'].list_collection_names()
    >>> self.client[database][collections[0]].estimated_document_count()

    Notes
    -----
    Pymongo methods for a collection object:
    count_documents(self, filter, session=None, limit=None)
    create_index(self, keys, unique=False)
    create_indexes(self, indexes)
    delete_one(self, filter)
    distinct(self, key, filter=None)
    drop(self)
    drop_index(self, index_or_name)
    drop_indexes(self)
    estimated_document_count(self)
    find(self, filter={}, projection=[], limit=None)
    find_one(self, filter=None)
    insert_many(self, documents, ordered=True)
    insert_one(self, document)
    list_indexes(self)
    replace_one(self, filter, replacement, upsert=False)
    update_many(self, filter, update, upsert=False)
    update_many(self, filter, update, upsert=False)
    update_one(self, filter, update, upsert=False)

    Operators:
    $eq      Matches values that are equal to a specified value.
    $gt      Matches values that are greater than a specified value.
    $gte     Matches values that are greater than or equal to a specified value.
    $in      Matches any of the values specified in an array.
    $lt      Matches values that are less than a specified value.
    $lte     Matches values that are less than or equal to a specified value.
    $ne      Matches all values that are not equal to a specified value.
    $nin     Matches none of the values specified in an array.
    $and     Joins query clauses with a logical AND
    $not     Inverts the effect of a query expression
    $nor     Joins query clauses with a logical NOR returns
    $or      Joins query clauses with a logical OR returns
    $exists  Matches documents that have the specified field.
    $type    Selects documents if a field is of the specified type.

    Unix Installation:
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 
      --recv 9DA31620334BD75D9DCB49F368818C72E52529D4
    echo "deb [ arch=amd64,arm64 ] 
      https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/4.0 multiverse" | 
      sudo tee /etc/apt/sources.list.d/mongodb-org-4.0.list
    sudo apt-get update
    sudo apt-get install mongodb-org
    sudo systemctl start mongod
    sudo systemctl restart mongod
    sudo systemctl stop mongod
    sudo systemctl enable mongod
    sudo service mongod stop
    sudo apt-get purge mongodb-org* 

    /etc/mongod.conf - configuration file for MongoDB
    dbPath -  where the database files stored (/var/lib/mongodb by default)
    systemLog - logging options (/var/log/mongodb/mongod.log by default)
    """
    def __init__(self, host='localhost', port=27017, echo=ECHO):
        """Initialize a a MongoDB connection"""
        self.client = pymongo.MongoClient(host=host, port=port)
        if echo:
            serverStatusResult = self.client.admin.command("serverStatus")
            pprint(serverStatusResult)
        
    def show(self, database=None):
        """Show database or table names"""
        if database is None:
            return self.client.list_database_names()
        return self.client[database].list_collection_names()

    def drop(self, database, collection=None):
        """Drop database or collection by name"""
        if collection is None:
            return self.client.drop_database(database)
        return self.client[database][collection].drop()

if False:
    from settings import settings
    
if False:  # To create new databases
    user = settings['sql']['user']
    password = settings['sql']['password']
    host = settings['sql']['host']
    port = settings['sql']['port']
    query = f"mysql+pymysql://{user}:{password}@{host}:{port}"
    engine = sqlalchemy.create_engine(query)
    engine.execute("CREATE DATABASE db1")
    engine.execute("CREATE DATABASE user1")

if False:  # to open available databases
    mongodb = MongoDB(**settings['mongodb'])
    sql = SQL(**settings['sql'], echo=True)
    user = SQL(**settings['user'], echo=True)
    
if False:   # unit tests
    df_ = DataFrame(data=[[1, 1.5, 'a'], [2, '2.5', None]],
                    columns=['a', 'b', 'c'], index=['d','e'])
    user.run('drop table if exists test')
    user.load_dataframe('test', df_)
    user.run('select * from test')

