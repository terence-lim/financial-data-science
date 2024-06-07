"""Finder class to lookup reference info across multiple datasets

Copyright 2022, Terence Lim

MIT License
"""
import pandas as pd
from pandas import DataFrame, Series
from finds.database.sql import SQL
_VERBOSE = 1

class Finder:
    """Looks-up reference info across multiple datasets by various identifiers"""

    def __init__(self, sql: SQL, identifier: str = '', table: str = ''):
        """Initialize lookup method with optional identifier type and table

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
        >>> find('0011', 'ticker', 'idsum')
        >>> find('aapl')
        >>> find('03783310')
        >>> find('03783310','cusip', 'links')
        >>> find('03783310','cusip', 'idsum')
        >>> find('45483', 'permco', 'names')
        """

        if len(kwargs):
            for k, v in kwargs.items():
                identifier = k
                label = v
        label = str(label).upper()
        assert label
        
        if not identifier:   # guess identifier if not specified
            if 5 <= len(label) <= 6 and label.isnumeric():
                identifier = 'permno'
                label = int(label)
            elif label.isnumeric():
                identifier = 'gvkey'
                label = int(label)
            elif len(label) in [8, 9] and any(c.isdigit() for c in label):
                identifier = 'ncusip'
                label = label[:8]
            elif len(label) < 6:
                identifier = 'tsymbol'
            else:
                identifier = 'comnam'

        if not table:   # guess table if not specified
            if identifier in ['permno', 'ncusip', 'tsymbol', 'comnam', 'permco']:
                table = 'names'
            elif identifier in ['gvkey', 'conm', 'cik']:
                table = 'links'
            else:
                table = 'idsum'
                
        like = '='
        if identifier in ['comnam', 'conm', 'cname']:
            label = '%' + label.upper() + '%'
            like = 'LIKE'  # for identifiers of str type, match with wildcard
        elif identifier in ['permno', 'gvkey', 'cik', 'permco']:
            label = int(label)
        elif identifier in ['ncusip', 'cusip']:
            label = label[:8]
        q = "SELECT * FROM {table} WHERE {identifier} {like} '{label}'".format(
            table=table, identifier=identifier, like=like, label=label)
        result = self.sql.run(q)
        return DataFrame(**result) if result is not None else None

if __name__ == "__main__":
    from secret import credentials
    VERBOSE = 1
    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    find = Finder(sql)
    print(find('GOOG'))

    print(find('META', identifier='tsymbol'))
    
    print(find(comnam='FACEBOOK'))

