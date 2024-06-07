"""CRSP daily and monthly stock files

Copyright 2022-2024, Terence Lim

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
from finds.database.redisdb import RedisDB
from finds.structured.busday import BusDay
from finds.structured.stocks import Stocks, StocksBuffer
from finds.recipes.filters import fractile_split

_VERBOSE = 1

"""TODO !
(0) verify this all true (keep monthly_delist): merge of last month and delists
(1) when loading monthly, also need to overwrite monthly return 
    (by insert overwrite) when monthly delisting -> retain overwriting table!
    - if special delisting code, then include 0.0
(2) conditionally:
    if daily delist date in or after month of last return: -0.3 or delisting ret
"""

class CRSP(Stocks):
    """Implements an interface to CRSP structured stocks dataset

    Args:
      sql: Connection to mysql database
      bd: Business dates object
      rdb: Optional connection to Redis for caching selected query results
      monthly: Use monthly (True) or daily file; default (None) autoselects

    Notes:

    - Earliest CRSP prc is 19251231
    """

    def __init__(self,
                 sql: SQL, 
                 bd: BusDay, 
                 rdb: RedisDB | None = None,
                 monthly: bool | None = None,
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
                Column('nextdt', Integer, default=0), # 'String(8) '19870612' @0
                Column('dlamt', Float),    # '0' - '2349.5'  (float64)
                Column('dlretx', Float),    # 'Float' '-0.003648' @ 3
                Column('dlprc', Float),    # '-1315' - '2349.5'  (float64)
                Column('dlpdt', Integer, default=0),  #'String(8)''19870612' @0
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
                Column('retx', Float),
                Column('dlstcd', SmallInteger),
                Column('dlret', Float)                
            )
        }
        super().__init__(sql, bd, tables, identifier='permno', name='CRSP',
                         rdb=rdb, verbose=verbose)
        self._monthly = monthly

    def build_lookup(self, source: str, target: str, date_field='date', 
                     dataset: str = 'names', fillna: Any = 0) -> Any:
        """Build lookup function to return target identifier from source"""
        return super().build_lookup(source=source, target=target,
                                    date_field=date_field, dataset=dataset,
                                    fillna=fillna)

    def get_cap(self, date: int,  cache_mode: str = "rw", 
                use_shares: bool = False,  use_permco: bool = False) -> Series:
        """Compute a cross-section of market capitalization values

        Args:
          date: YYYYMMDD int date of market cap
          cache_mode: 'r' to try read from cache first, 'w' to write to cache
          use_shares: If True, use shrout from 'shares' table, else 'daily'
          use_permco: If True, sum caps by permco, else by permno

        Returns:
          Series of market cap indexed by permno
        """
        rkey = f"cap{'co' if use_permco else ''}_{str(self)}_{date}"
        if self.rdb and 'r' in cache_mode and self.rdb.redis.exists(rkey):
            self._print('(get_cap load)', rkey)
            return self.rdb.load(rkey)['cap']
        if use_shares:   # use shares tables
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
        else:   # where 'daily' table contains 'shrout'
            cap = self.get_section(dataset='daily', 
                                   fields=['prc', 'shrout'],
                                   date_field='date', 
                                   date=date)
            df = DataFrame(cap['shrout'] * cap['prc'].abs(), columns=['cap'])
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
        if self.rdb and 'w' in cache_mode:
            self._print('(get_cap dump)', rkey)
            self.rdb.dump(rkey, df)
        return df['cap']

    def get_universe(self, date: int, cache_mode: str = "rw") -> DataFrame:
        """Return standard CRSP universe of US-domiciled common stocks

        Args:
          date: Rebalance date (YYYYMMDD)
          cache_mode: 'r' to try read from cache first, 'w' to write to cache

        Returns:
          DataFrame of screened universe, indexed by permno, with columns: 
          market cap "decile" (1..10), "nyse" bool, "siccd", "prc", "cap"

        Notes:

        - Market cap must be available on date, with non-missing prc
        - shrcd in [10, 11], exchcd in [1, 2, 3]
        """

        assert date == self.bd.offset(date), f"get_universe: {date} not valid date"
        rkey = "_".join(["universe", str(self), str(date)])
        if 'r' in cache_mode and self.rdb and self.rdb.redis.exists(rkey):
            self._print('(get_universe load)', rkey)
            df = self.rdb.load(rkey)
        else: 
            df = self.get_section(dataset='daily',
                                  fields=['prc', 'shrout'],
                                  date_field='date',
                                  date=date)\
                     .fillna(0)
            df['cap'] = df['prc'].abs().mul(df['shrout'])
            df = df.join(self.get_cap(date=date,
                                      cache_mode=cache_mode,
                                      use_shares=True,
                                      use_permco=True),
                         rsuffix='co',
                         how='inner')\
                   .fillna(0)
            df = df.join(self.get_section(dataset='names',
                                          fields=['shrcd', 'exchcd',
                                                  'siccd', 'naics'],
                                          date_field='date',
                                          date=date,
                                          start=0),
                         how='inner')
            self._print('LENGTH PERMNOS', str(len(df)))
            self._print('PRC NULL:', df['prc'].isna().sum(),
                        'NEG:', df['prc'].le(0).sum())
            self._print('CAP NON-POSITIVE:', len(df) - df['cap'].gt(0).sum())
            
            df = df[df['capco'].gt(0) &
                    df['cap'].gt(0) &
                    df['shrcd'].isin([10, 11]) &
                    df['exchcd'].isin([1, 2, 3])]
            df['nyse'] = df['exchcd'].eq(1)                     # nyse indicator
            df['decile'] = fractile_split(values=df['capco'],    # size deciles 
                                          pct=np.arange(10, 100, 10),
                                          keys=df.loc[df['nyse'], 'capco'],
                                          ascending=False)
            df = df[['cap', 'capco', 'decile', 'nyse', 'siccd', 'prc', 'naics']]
            if 'w' in cache_mode and self.rdb:
                self._print('(get_universe dump)', rkey)
                self.rdb.dump(rkey, df)
        return df

    def get_divamt(self, beg: int, end: int) -> DataFrame:
        """Accmumulates total dollar dividends between beg and end dates

        Args:
          beg: Inclusive start date (YYYYMMDD)
          end: Inclusive end date (YYYYMMDD)

        Returns:
          DataFrame with accumulated divamts = per share divamt * shrout
        """
        q = ("SELECT {dist}.{identifier} AS {identifier}, "
             " SUM({table}.shrout * {dist}.divamt) AS divamt "
             "FROM {dist} INNER JOIN {table} "
             " ON {table}.{identifier} = {dist}.{identifier} AND "
             "    {table}.date = {dist}.exdt "
             " WHERE {dist}.divamt > 0 AND {dist}.exdt >= {beg} "
             "   AND {dist}.exdt <= {end} GROUP BY {identifier} ").format(
                 dist=self['dist'].key,
                 identifier=self.identifier,
                 table=self['daily'].key,
                 beg=beg,
                 end=end)
        return self.sql.read_dataframe(q).set_index(self.identifier)

    dlstcodes_ = set([500, 520, 580, 584]).union(list(range(551,575)))

    @classmethod
    def is_dlstcode(self, dlstcd: Series | int) -> Series | int:
        """Delisting returns if missing for these codes should be -0.3"""
        if isinstance(dlstcd, int):
            return dlstcd in CRSP.dlstcodes_
        else:
            return dlstcd.isin(CRSP.dlstcodes_)

    def get_dlret(self, beg: int, end: int, dataset: str = 'delist') -> Series:
        """Compounded delisting returns from beg to end dates for all permnos

        Args:
          beg: Inclusive start date (YYYYMMDD)
          end: Inclusive end date (YYYYMMDD)
          dataset: either 'delist' or 'monthly' containing delisting returns

        Returns:
          Series of delisting returns

        Notes:
          Sets to -0.3 if missing and code in [500, 520, 551...574, 580, 584]
        """
        q = ("SELECT (1+dlret) AS dlret, {identifier}, dlstcd FROM {table} "
             "  WHERE {dlstdt} >= {beg} AND {dlstdt} <= {end}"
             "    AND dlstcd > 0").format(
                 table=self[dataset].key,
                 identifier=self.identifier,
                 dlstdt='dlstdt' if dataset=='delist' else 'date',
                 beg=beg,
                 end=end)
        self._print('(get_dlst)', q)
        df = self.sql.read_dataframe(q).sort_values(self.identifier)
        if len(df):
            df.loc[(df['dlret'].isna() &
                    CRSP.is_dlstcode(df['dlstcd'])).values, 'dlret'] = 0.7
            df = df[[self.identifier, 'dlret']].groupby(self.identifier)\
                                               .prod(min_count=1)\
                                               .dropna() - 1
        return df['dlret']

    def get_ret(self, beg: int, end: int, dataset: str = 'daily',
                field: str = 'ret', **kwargs) -> Series:
        """Get compounded returns, with option to include delist returns

        Args:
          beg: starting returns date
          end: ending returns date
          dataset: name of returns dataset (ignore if initialized as 'monthly')
          field: Name of returns field in dataset, in {'ret', 'retx')
        """

        # select or autoselect monthly data set
        if self._monthly or dataset == 'monthly':
            use_monthly = True
        elif self._monthly is None:
            use_monthly = ('monthly' in self.tables_ and 
                           beg <= self.bd.begmo(beg) and
                           end >= self.bd.endmo(end))
        else:
            use_monthly = False
        if use_monthly:  # set beg and end to envelope calendar month dates
            dataset = 'monthly'
            beg = (beg // 100) * 100
            end = (end // 100 * 100) + 99

        df = super().get_ret(beg, end, dataset=dataset, field=field, **kwargs)

        # if monthly dataset, then adjust for delisting return
        if use_monthly:
            dlst = self.get_dlret(beg, end, dataset='monthly')
            df = DataFrame(df).join(dlst, how='outer')
            df = (1+df[field].fillna(0)) * (1+df['dlret'].fillna(0)) - 1
        return df.rename(field)

    def cache_ret(self,
                  dates: List[Tuple[int, int]], 
                  replace: bool, 
                  dataset: str = 'daily',
                  field: str = 'ret', 
                  date_field: str ='date'):
        """Pre-generate compounded returns from daily for redis store"""
        assert dataset == 'daily', "dataset must be daily"
        super().cache_ret(dates=dates, replace=replace, field=field,
                          date_field=date_field, dataset=dataset)
    

class CRSPBuffer(StocksBuffer):
    """Cache returns into memory, and provide Stocks-like interface"""
    
    def __init__(self,
                 stocks: Stocks, 
                 beg: int, 
                 end: int,
                 fields: List[str],
                 dataset: str):
        """Create object and load returns into its cache

        Args:
          stocks: Stocks structured data object to access stock returns data
          beg: Earliest date of daily stock returns to pre-load
          end: Latest date of daily stock returns to pre-load
          fields: Column names of returns fields, e.g. ['ret', 'retx', 'prc']
          dataset: Name of dataset to extract from, e.g. 'daily', 'monthly'
        """
        if dataset == 'monthly' and 'dlret' not in fields:
            fields += ['dlret']
        super().__init__(stocks=stocks, beg=beg, end=end, dataset=dataset,
                         identifier=stocks.identifier, fields=fields)

    def get_ret(self, beg: int, end: int, field: str = 'ret') -> Series:
        """Return compounded stock returns between beg and end dates

        Args:
          beg: Begin date to compound returns
          end: End date (inclusive) to compound returns
          field: Name of returns field in dataset, in {'ret', 'retx')
        """
        assert field in ['ret', 'retx']
        df = super().get_ret(beg=beg, end=end, field=field)
        if self._dataset == 'monthly':   # CRSP partial month delisting return
            dlst = super().get_ret(beg=beg, end=end, field='dlret')
            df = DataFrame(df).join(dlst, how='outer')
            df = (1+df[field].fillna(0)) * (1+df['dlret'].fillna(0)) - 1
        return df.fillna(0).rename(field)

    def get_universe(self, date: int, cache_mode: str = "rw") -> DataFrame:
        """Simply pass through original method to retrieve universe"""
        return self.stocks.get_universe(date=date, cache_mode=cache_mode)
        
if __name__ == "__main__":
    from pathlib import Path
    from finds.structured import CRSP, CRSPBuffer
    from secret import credentials, paths
    VERBOSE = 1
    
    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    user = SQL(**credentials['user'], verbose=VERBOSE)
    rdb = RedisDB(**credentials['redis'])
    bd = BusDay(sql, endweek=3)  # weekly frequency, 3=Wed-close-to-Wed-close
    crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
    downloads = paths['data'] / 'CRSP'

    # load CRSP: TODO handle missing return codes (< -1, see below)
    df = crsp.load_csv('names', downloads / 'names.txt.gz', sep='\t')
    print(len(df), '~', 103383)
    df = crsp.load_csv('shares', downloads / 'shares.txt.gz', sep='\t')
    print(len(df), '~', 2346131)
    df = crsp.load_csv('dist', downloads / 'dist.txt.gz', sep='\t')
    print(len(df), '~', 935880)
    df = crsp.load_csv('delist', downloads / 'delist.txt.gz', sep='\t')
    print(len(df), '~', 33584)
    df = crsp.load_csv('monthly', downloads / 'monthly.txt.gz', sep='\t')
    print(len(df), '~', 4606907)

    for s in sorted(downloads.glob('daily*.txt.gz'), reverse=True):
        tic = time.time()
        df = crsp.load_csv('daily', csvfile=s, sep='\t',
                           drop={'permno': ['PERMNO', '.'],
                                 'date': ['.'],
                                 'shrout':['.']})
        print(s, round(time.time() - tic, 0), 'secs:', len(df), s)


    # Pre-generate weekly returns and save in Redis cache
    begweek = 19251231
    endweek = 20231229

    rebaldates = bd.date_range(begweek, endweek, freq='weekly')
    r = bd.date_tuples(rebaldates)
    batchsize = 40
    batches = [r[i:(i+batchsize)] for i in range(0, len(r), batchsize)]
    batches.reverse()
    for batch in batches:
        crsp.cache_ret(batch, field='ret', replace=True)
        crsp.cache_ret(batch, field='retx', replace=True)

    # test CRSPBuffer
    stocks = CRSPBuffer(crsp, 20210101, 20211231, dataset='monthly')
    beg, end = 20210101, 20210131
    df = stocks.get_ret(beg, end)
    print(df) 
    m = crsp.get_ret(beg, end, cache_mode="")
    print(m)


if False:
    # changed so now may get from monthly column or compound delist table
    def get_dlstret(self, beg: int, end: int, cache_mode: str = "rw") -> Series:
        """Compounded delisting returns from beg to end dates for all permnos

        Args:
            beg: Inclusive start date (YYYYMMDD)
            end: Inclusive end date (YYYYMMDD)
            cache_mode: 'r' to try read from cache first, 'w' to write to cache

        Returns:
            Series of compounded returns
        """
        rkey = "_".join(["dlst", str(self), str(beg), str(end)])
        if 'r' in cache_mode and self.rdb and self.rdb.redis.exists(rkey):
            self._print("(get_dlstret load)", rkey, str(self))
            return self.rdb.load(rkey)['ret']

        q = ("SELECT (1+dlret) AS ret, {identifier} FROM {table} "
             "  WHERE dlstdt >= {beg} AND dlstdt <= {end}").format(
                 table=self['delist'].key,
                 identifier=self.identifier,
                 beg=beg,
                 end=end)
        self._print('(get_dlst)', q)
        df = self.sql.read_dataframe(q).sort_values(self.identifier).fillna(0)
        if len(df):
            df = (df.groupby(self.identifier).prod(min_count=1)-1).dropna()
        if 'w' in cache_mode and self.rdb:
            self._print("(get_dlstret dump)", rkey, str(self))
            self.rdb.dump(rkey, df)
        return df['ret']
