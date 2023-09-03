"""IBES summary, ident and history of analysts earnings estimates

Copyright 2022, Terence Lim

MIT License
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import time
from pandas import DataFrame, Series
from pandas.api.types import is_list_like, is_integer_dtype
from sqlalchemy import Table, Column, Index
from sqlalchemy import Integer, String, Float, SmallInteger, Boolean, BigInteger
from finds.database.sql import SQL
from finds.busday import BusDay
from .structured import Structured
_VERBOSE = 1

class IBES(Structured):
    """Provide interface to IBES analyst estimates structured datasets
    
    Args:
        sql: Connection to SQL database
        bd: Custom business day calendar object

    Notes:

    - TICKER: IBES Ticker
    - STATPERS : IBES Statistical Period (monthly)
    - OFTIC: Official Ticker
    - FPEDATS: Forecast Period End Date
    - SDATES: Identification start date
    """

    def __init__(self, sql: SQL, 
                       bd: BusDay, 
                       name='IBES', 
                       verbose=_VERBOSE):
        """Initialize a connection to IBES datasets"""
        tables = {
            'history': sql.Table(
                'history',
                Column('ticker', String(6), primary_key=True),  # IBES Ticker
                Column('oftic', String(8)),  # Official Ticker
                Column('statpers', Integer, primary_key=True), # Stat Period
                Column('measure', String(3), primary_key=True), # forecast type
                Column('fy0a', Float),
                Column('curcode', String(3)),
                Column('fy0edats', Integer),
                Column('price', Float),
                Column('prdays', Integer),
                Column('shout', Float),
                Column('iadiv', Float),
                Column('curr_price', String(3)),                
            ),
            'summary': sql.Table(
                'summary',
                Column('ticker', String(6), primary_key=True), 
                Column('fpedats', Integer, primary_key=True),
                Column('statpers', Integer, primary_key=True),
                Column('measure', String(3), primary_key=True), #new key
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
                Column('sdates', Integer, primary_key=True),
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
        self.sql.create_all()
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


if __name__ == "__main__":
    from pathlib import Path
    from env.conf import credentials
    
    VERBOSE = 0
    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    user = SQL(**credentials['user'], verbose=VERBOSE)
    bd = BusDay(sql)
    ibes = IBES(sql, bd)
    
    # load IBES
    downloads = Path('/home/terence/Downloads/stocks2022/v1/IBES/')
    df = ibes.load_csv('ident', downloads / Path('ident.txt.gz'), sep='\t')
    print(len(df), 85550)
    ibes.write_links()  #  (missing, count) = 15340  88963
    
    df = ibes.load_csv('history', downloads / Path('history.txt.gz'), sep='\t')
    df = ibes.load_csv('summary', downloads / Path('summary.txt.gz'), sep='\t')
    print(len(df), 11776742)
    
    #ibes.load_csv('adjust', downloads + 'adjustment.csv') #rows=24777
    #ibes.load_csv('surprise', downloads + 'surprise.csv')  #rows=528933

