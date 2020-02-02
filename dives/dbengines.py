"""The dbengines module defines wrappers for database engines: pymongo, sqlalchemy, redis"""
# The MIT License
#
# Copyright (c) 2020 Terence Lim (https://terence-lim.github.io/)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation he rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import pymongo, sqlalchemy, redis, os
from pymongo import MongoClient
from pyarrow import default_serialization_context as pa_context

from dives.util import DataFrame
from dives.util import print_debug
try:
    import secret
    verbose = secret.value('verbose')
except:
    verbose = 0

class Redis(object):
    """Class to support a redis connection"""

    def __init__(self, **kwargs):
        self.redis = redis.StrictRedis(**kwargs)

    def exists(self, rkey):
        """check if {rkey} exists in redis database"""
        return self.redis.exists(rkey)
        
    def load(self, rkey):
        """loads dataframe by {rkey} from redis"""
        df = pa_context().deserialize(self.redis.get(rkey))   # must use pyarrow for DataFrame object
        return df.copy()   # return a copy else sometimes flag.writable is False ???!!!

    def dump(self, rkey, df):
        """saves dataframe by {rkey} to redis"""
        self.redis.set(rkey, pa_context().serialize(df).to_buffer().to_pybytes())  # must use pyarrow

    help = """redis CheatSheet
----------------
methods:
redis.delete(key)      -- delete an item
redis.get(key)         -- get an item
redis.exists(key)      -- does item exist
redis.set(key, value)  -- set an item
redis.keys()           -- get keys
redis.hmset(k, d)
redis.hgetall(k)

unix:
decode_responses=False to zlib dataframe
./redis-5.0.4/src/redis-server
./redis-cli --scan --pattern 'users:*' | xargs ./redis-cli del
CLI> keys *
CLI> flushall
CLI> info memory
"""        

class MongoDB(object):
    """wrapper class to maintain a pymongo connection

    Parameters
    ----------
    database : string
        name of database in MongoDB
    host, port : string
        connection info

    Examples
    --------
    mongodb = MongoDB(**_mongo_connect_info)
    serverStatusResult = mongodb.client.admin.command("serverStatus")
    pprint(serverStatusResult)
    collections = mongodb.client[mongodb.database].list_collection_names()  # show table
    """
    
    def __init__(self, host='localhost', port=27017, database=None, **kwargs):
        self.client = MongoClient(host=host, port=port)
        self.database = database                         # name of database to create and use

    help = """PyMongo cheatsheet:

methods:
--------
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

operators
---------
$eq      Matches values that are equal to a specified value.
$gt      Matches values that are greater than a specified value.
$gte     Matches values that are greater than or equal to a specified value.
$in      Matches any of the values specified in an array.
$lt      Matches values that are less than a specified value.
$lte     Matches values that are less than or equal to a specified value.
$ne      Matches all values that are not equal to a specified value.
$nin     Matches none of the values specified in an array.
$and     Joins query clauses with a logical AND returns all documents that match the conditions of both.
$not     Inverts the effect of a query expression and returns documents that do not match the  expression.
$nor     Joins query clauses with a logical NOR returns all documents that fail to match both clauses.
$or      Joins query clauses with a logical OR returns all documents that match the conditions of either.
$exists  Matches documents that have the specified field.
$type    Selects documents if a field is of the specified type.

unix
----
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 9DA31620334BD75D9DCB49F368818C72E52529D4
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/4.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.0.list
sudo apt-get update
sudo apt-get install mongodb-org
sudo systemctl start mongod
sudo systemctl restart mongod
sudo systemctl stop mongod
sudo systemctl enable mongod
sudo service mongod stop
sudo apt-get purge mongodb-org* 

The configuration file for MongoDB is located at /etc/mongod.conf
dbPath -  where the database files will be stored (/var/lib/mongodb by default)
systemLog - logging options (/var/log/mongodb/mongod.log by default)"""


# class SQL(object):
class SQL(object):
    """Class to connect to mysql database, plus some convenient wrapper functions

    Parameters
    ----------
    user, password, host : string
        to connect to mysql server
    db : string
        database name

    Notes:
    -----
    autocommit = true
    local_infile = 1 (allow load of local csv files)
    uses sqlalchemy engine
    """
    
    def __init__(self, temp = "temp" + str(os.getpid()), **kwargs):
        """Open new mysql connection."""
        self._connect_info = kwargs.copy()
        q = ('mysql+pymysql://%s:%s@%s/%s?charset=utf8&local_infile=1&autocommit=true' %
             tuple(self._connect_info[s] for s in ['user','password','host','db']))
        self._engine = sqlalchemy.create_engine(q)
        self._temp = temp
        print_debug('Created engine: ' + q)

    def run(self, q, fetch=True):
        """Run a sql command string.
        
        Parameters
        ----------
        q : string
            SQL query string to execute
        fetch: bool, optional
            If true, return the result.

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
        print_debug("(run) " + q)
        result = self._engine.execute(q)
        if fetch:  # Sometimes the connector libraries return the number of created/deleted rows.
            return {'data' : result.fetchall(), 'columns' : result.keys()}
        return result

    def exists_table(self, table):
        """check if table exists."""
        q = "SELECT COUNT(*) as c FROM information_schema.tables" \
            " WHERE table_schema = '{db}' and table_name = '{table}'" \
            "".format(db = self._connect_info['db'], table = table)
        try:
            print_debug("(exists_table) " + q)
            results = self._engine.execute(q).fetchall()
            return results[0]['c'] != 0
        except Exception as e:
            print("(exists_table) Got exception = ", e, " Query = ", q)
            raise e

    def drop_table(self, table):
        """drop if table exists."""
        q = "DROP TABLE IF EXISTS " + table
        try:
            print_debug("(drop_table) " + q)
            return self.run(q, fetch=False).rowcount
        except Exception as e:
            print("(exists_table) Got exception = ", e, " Query = ", q)
            raise e

    def create_index(self, table, columns, cnx=None):
        """create index on a {table} on {columns} in list."""
        q = "CREATE INDEX `{key}` ON {table} ({columns});" \
            "".format(key = "_".join(columns), table=table, columns = ", ".join(columns))
        try:
            print_debug("(create_index) " + q)
            self.run(q, fetch=False)
        except Exception as e:
            print("(create_index) Got exception = ", e, " Query = ", q)
            raise e
                        
    def create_table(self, table, fields, primary=[], temp=False, **kwargs):
        """Create a new table, after dropping old if exist

        Parameters
        ----------
        table : string
            table of new table
        fields : list of tuples
            list of field (name, type), see examples below

        Examples
        --------
        sql.create_table('testing', [['label', 'VARCHAR(8)'],['date',  'INT(11)']],['label'])
        """
        t = 'TEMPORARY' if temp else ''
        q = "DROP {t} TABLE IF EXISTS `{table}`;".format(t=t, table=table)
        try:
            self.run(q, fetch=False)
            if primary:
                p = ", PRIMARY KEY (" + ", ".join(["`" + p + "`" for p in primary]) + ")"
            else:
                p = ''
            fields = ", ".join(['`' + f[0] + '` ' + f[1] for f in fields])
            q = "CREATE {t} TABLE `{table}` ({fields} {p}) ENGINE=InnoDB DEFAULT CHARSET=utf8;" \
                "".format(t=t, table=table, fields=fields, p=p)
            print_debug("(create_table) " + q)
            self.run(q, fetch=False)
            if 'indexes' in kwargs:
                for index in kwargs['indexes']:
                    self.create_index(table, index)
            
        except Exception as e:
            print("(create_table) Got exception = ", e, " Query = ", q)
            raise e

    def count_table(self, table, val, key=None):
        """return summary statistics of variable {val}, optionally groupby {key}"""
        if self.exists_table(table):
            if key:
                q = "SELECT {key}, COUNT(*) as count, AVG({val}) as avg, STD({val}) as std," \
                    " MAX({val}) as max, MIN({val}) as min FROM {table} GROUP BY {key}" \
                    "".format(table=table, key=key, val=val)
                return DataFrame(**self.run(q)).set_index(key).sort_index()
            else:
                q = "SELECT COUNT(*) as count, AVG({val}) as avg, STD({val}) as std," \
                    " MAX({val}) as max, MIN({val}) as min FROM {table}" \
                    "".format(table=table, val=val)
                return DataFrame(index=[val], **self.run(q))
        else:
            return None    
        
        
    def load_infile(self, table, csvfile, options=''):
        """Load table from csv file, using mysql's load data local infile"""
        q = "LOAD DATA LOCAL INFILE '{csvfile}' INTO TABLE {table} FIELDS TERMINATED BY ','"\
            " ENCLOSED BY '\"' LINES TERMINATED BY '\\n' IGNORE 1 ROWS {options};" \
            "".format(csvfile=csvfile, table=table, options=options)
        try:
            print_debug("(load_infile) " + q)
            self.run(q, fetch=False)
        except Exception as e:
            print("(load_infile) Got exception = ", e, " Query = ", q)
            raise e

    def load_dataframe(self, table, df, index_label=None, insert_ignore=False, if_exists='append'):
        '''Load dataframe into sql table, ignoring duplicate keys

        Parameters
        ----------
        table : str 
            physical name of table to insert into
        df: DataFrame
            the source dataframe
        index_label: string, optional
            name of column to load dataframe index as, default=None
        insert_ignore: boolean, optional
            if True, then immediately assume key conflicts and use insert ignore (default=False, 
            and first tries to_sql, which fails on duplicate) before resorting to insert ignore)
        if_exists: string, optional
            action to take if table already exists - 'replace', 'append' (default)

        Notes
        -----
        Insert ignore, i.e. new records with duplicate key are dropped
        If initial DataFrame.to_sql (sqlalchemy) load fails because duplicate key, 
        then use mysql 'insert ignore'

        '''
        df.columns = df.columns.map(str.lower).map(str.rstrip)
        try:     # to_sql raises exception if exist duplicate keys
            assert(insert_ignore == False)
            df.to_sql(table, self._engine, if_exists=if_exists, 
                      index=(index_label is not None), index_label=index_label)
        except:  # duplicates exists, so to_sql to temp, then insert ignore from temp into table
            print_debug("(load_dataframe) Retrying insert ignore for " + table)
            self.run('drop table if exists ' + self._temp, fetch=False)
            df.to_sql(self._temp, self._engine, if_exists='replace',   # must now append
                      index=(index_label is not None), index_label=index_label)
            warnings.filterwarnings("ignore", category=pymysql.Warning)
            q = "INSERT IGNORE INTO {table} ({columns}) SELECT {columns} FROM {temp}" \
                "".format(table=table, columns=", ".join(df.columns), temp=self._temp)
            self.run(q, fetch=False)
            warnings.filterwarnings("default", category=pymysql.Warning)
            self.run('drop table if exists ' + self._temp, fetch=False)

    def insert(self, table, items):
        """insert a new record into table.

        Parameters
        ----------
        table : string
            table of table to insert record into
        items: dict
            dict of column names and values

        Examples
        --------
        sql.insert('testing',{'label' : 'b', 'date' : 3})
        """
        q = "insert ignore into {table} {columns} values ('{values}')" \
            "".format(table=table, columns=" (`" + "`, `".join(items.keys()) + "`) ",
                      values="', '".join(str(v) for v in items.values()))
        try:
            print_debug('(insert) ' + q)
            return self.run(q, fetch=False)
        except Exception as e:
            print("(insert) Got exception = ", e, " Query = ", q)
            return None


    def select(self, table, fields = ["*"], where=''):
        """Select records in table satisfying where cause

        Parameters
        ----------
        table : string
            table of table to find records
        where: dict
            dict of column names and values
        fields: list of string
            list of column names to select

        Examples
        --------
        sql.select('testing', where = {'label' : 'b'})
        """
        if len(where):
            where = " where " + " and ".join("`{k}`='{v}'".format(k=k, v=v) for k,v in where.items())
        q = "select {fields} from `{table}` {where}" \
            "".format(fields= ", ".join(fields), table=table, where=where)
        try:
            print_debug(q)
            return DataFrame(**self.run(q, fetch=True))
        except Exception as e:
            print("(select) Got exception = ", e, " Query = ", q)
            return None

        
    def delete(self, table, where=''):
        """Delete records in table satisfying where cause

        Parameters
        ----------
        table : string
            physical name of table to find records
        where: dict, optional
            dict of column names and values. Default is '' which truncates entires table

        Examples
        --------
        sql.delete('testing',where = {'label' : "'a'"})
        """
        if len(where):
            where = ' where ' + " and ".join("`{k}`='{v}'".format(k=k,v=v) for k,v in where.items())
        q = "delete from {table} {where}".format(table=table, where=where)
        try:
            print_debug(q)
            return self.run(q, fetch=False).rowcount
        except Exception as e:
            print("(delete) Got exception = ", e)
            return None

if __name__ == "__main__":
    def test_csv(csvfile, sep=',', header=0, quoting=0):
        """reads an csv file to diagnose, and suggest field types"""
        df = pd.read_csv(csvfile,sep=sep, header=header, quoting=quoting, #encoding='utf-8', 
                         low_memory=False,na_filter=False)
        #  encoding = "ISO-8859-1"
        df.columns = map(str.lower, df.columns)
        print(csvfile, '   rows =', len(df), '   cols =', len(df.columns),':')
        for i in range(len(df.dtypes)):
            #        print(i, df.dtypes[i], df.dtypes[i] in [int, float])
            if df.dtypes[i] in [int, float]:
                xmin = np.min(df[df.columns[i]])
                xmax = np.max(df[df.columns[i]])
                if df.dtypes[i] in [int]:
                    if max(abs(xmin), abs(xmax)) < (127 // 2):
                        t = 'TINYINT DEFAULT 0'
                    elif max(abs(xmin), abs(xmax)) < (32767 // 2):
                        t = 'SMALLINT DEFAULT 0'
                    else:
                        t = 'INT(11) DEFAULT 0'
                    s = 'd'
                else:
                    t = 'DOUBLE DEFAULT NULL'
                    s = 'g'
                print(("['%s', '%s'],    # '%" + s + "' - '%" + s + "'  (%s)") %
                      (df.columns[i], t, xmin, xmax, df.dtypes[i]))
            else:
                j = df[df.columns[i]].astype(str).map(len).idxmax()
                if (df[df.columns[i]].str.contains('[a-zA-Z]').any()):
                    print("['%s', 'VARCHAR(%d)'],    # 'DOUBLE' '%s' @ %d" %
                          (df.columns[i], len(df.iloc[j][i]), df.iloc[j][i], j))
                else:
                    print("['%s', 'DOUBLE'],    # 'VARCHAR(%d)' '%s' @ %d" %
                          (df.columns[i], len(df.iloc[j][i]), df.iloc[j][i], j))
        print()
        return df


    
