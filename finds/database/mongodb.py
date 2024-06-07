"""MongoDB client wrapper

Copyright 2022, Terence Lim

MIT License
"""
from typing import List, Dict
from pymongo import MongoClient
from .database import Database

def parse_where(where: Dict | str | List) -> Dict:
    """Helper to parse dict or list-like where clause to pymongo command str

    Args:
        where: keyword name may have operator suffix:

        - name_eq:     be equal to the value
        - name_ne:     be equal to the value
        - name_lt:     less than the value
        - name_le:     less than or equal to the value
        - name_gt:     greater than the value
        - name_ge:     greater than or equal to the value
        - name_in:     be included in the value
        - name_notin:  not be included in the value

    Returns:
        Dictionary of conditions in pymongo format
    """
    result = dict()
    if isinstance(where, dict):    # where dict of {key: condition}'s =>
        for k,v in where.items():    # parse condition to test key value:
            if isinstance(v, dict):     # if dict: 
                result[k] = v
            elif isinstance(v, set):    # if set: key isin set
                result[k] = {'$in' : list(v)}
            elif types.is_list_like(v): # if tuple:
                assert(v[0] <= v[1])      # inclusive bounds on key value
                result[k] = {'$gte': v[0], '$lte': v[1]}
            else:                       # if scalar: key value equals
                result[k] = {'$eq': v}
    elif isinstance(where, str):     # where string => test if key name exists
        result = {where : {'$exists' : True}}
    elif where:                      # where list-like => test if all keys exist
        try:
            result = {k : {'$exists' : True} for k in where}
        except:
            raise Exception('[where] must be a dict, array-like or str')
    return result
                        

class MongoDB(Database):
    """Interface to pymongo

    Args:
        database: Name of database in MongoDB
        host: IP address of server
        port: Port number

    Attributes:
        client: MongoClient instance providing MongoDB interface

    Examples:
        >>> mdb = MongoDB()
        >>> serverStatusResult = mdb.client.admin.command("serverStatus")
        >>> pprint(serverStatusResult)
        >>> collections = mdb.client['database'].list_collection_names()
        >>> mdb.client[database][collections[0]].estimated_document_count()

    Methods:
    ::

        count_documents(filter, session=None, limit=None)
        create_index(keys, unique=False)
        create_indexes(indexes)
        delete_one(filter)
        distinct(key, filter=None)
        drop()
        drop_index(index_or_name)
        drop_indexes()
        estimated_document_count()
        find(filter={}, projection=[], limit=None)
        find_one(filter=None)
        insert_many(documents, ordered=True)
        insert_one(document)
        list_indexes()
        replace_one(filter, replacement, upsert=False)
        update_many(filter, update, upsert=False)
        update_many(filter, update, upsert=False)
        update_one(filter, update, upsert=False)

    MongoDB Operators:
    ::

    $eq     Matches values that are equal to a specified value.
    $gt     Matches values that are greater than a specified value.
    $gte    Matches values that are greater than or equal to a specified value.
    $in     Matches any of the values specified in an array.
    $lt     Matches values that are less than a specified value.
    $lte    Matches values that are less than or equal to a specified value.
    $ne     Matches all values that are not equal to a specified value.
    $nin    Matches none of the values specified in an array.
    $and    Joins query clauses with a logical AND
    $not    Inverts the effect of a query expression
    $nor    Joins query clauses with a logical NOR returns
    $or     Joins query clauses with a logical OR returns
    $exists Matches documents that have the specified field.
    $type   Selects documents if a field is of the specified type.

    Unix Installation:

    https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/

    ::

        sudo systemctl start mongod
        sudo systemctl restart mongod
        sudo systemctl stop mongod
        sudo systemctl enable mongod
        sudo service mongod stop
        sudo apt-get purge mongodb-org* 

    /etc/mongod.conf - configuration file for MongoDB:

    - dbPath:  where the database files stored (/var/lib/mongodb)
    - systemLog: logging options (/var/log/mongodb/mongod.log)
    """

    def __init__(self, host: str = 'localhost', port: int = 27017, **kwargs):
        super().__init__(**kwargs)
        self.client = MongoClient(host=host, port=port)
        if self._verbose:
            result = self.client.admin.command("serverStatus")
            print(result)

    def show(self, database: str = ''):
        """List all database or collection (table) names

        Args:
            database: List collections in (blank to list all databases)
        """
        if not database:
            return self.client.list_database_names()
        return self.client[database].list_collection_names()

    def drop(self, database: str, collection: str = ''):
        """Drop a database or collection (table) by name

        Args:
            database: Name of database to drop collection
            collection: Name of collection to drop (blank to drop database)
        """
        if not collection:
            self.client.drop_database(database)
        self.client[database][collection].drop()


if __name__ == "__main__":
    #    from os.path import dirname, abspath
    #    sys.path.insert(0, dirname(dirname(abspath(__file__))))
    from env.conf import credentials
    VERBOSE = 1
    
    mdb = MongoDB(verbose = VERBOSE)
    mdb.show()
    db = mdb.client['database']
    c = db['collection']   # creation is automatic
    
    c.insert_one({'hello': 'goodbye'})
    found = c.find_one({'hello' : {'$exists' : True}})
    print(found)
