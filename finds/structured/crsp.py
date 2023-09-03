"""CRSP daily, monthly, names, delistings, distributions, shares outstanding

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
from finds.database.redisdb import RedisDB
from finds.busday import BusDay
from finds.backtesting import fractiles
from finds.structured.stocks import Stocks

_VERBOSE = 1

"""(0) verify this is all true (keep monthly_delist): merge of last month and delists
(1) when loading monthly, also need to overwrite monthly return 
    (by insert overwrite) when monthly delisting -> retain overwriting table!
    - if special delisting code, then include 0.0
(2) conditionally:
    if daily delist date in or after month of last return: -0.3 or delisting return

Monthly: 

The monthly Delisting Return is calculated from the last month ending
price to the last daily trading price if no other delisting
information is available. In this case the delisting payment date is
the same as the delisting date.  If the return is calculated from a
daily price, it is a partial-month return. The partial-month returns
are not truly Delisting Returns since they do not represent values
after delisting, but allow the researcher to make a more accurate
estimate of the Delisting Returns

Daily:

Delisting Return is the return of security after it is delisted. It is
calculated by comparing a value after delisting against the price on
the security’s last trading date. The value after delisting can
include a price on another exchange or the total value of
distributions to shareholders. If there is no opportunity to trade a
stock after delisting before it is declared worthless, the value after
delisting is zero. Delisting Returns are calculated similarly to total
returns except that the value after delisting is used as the current
price.

"""

class CRSP(Stocks):
    """Implements an interface to CRSP structured stocks datasets

    Args:
      sql: Connection to mysql database
      bd: Business dates object
      rdb: Optional connection to Redis for caching selected query results
      monthly: Monthly (True) or daily (False) dataset; default (None) autoselects

    Notes:

    - Earliest CRSP prc is 19251231, FF is 19260701 
      (except STRev daily is 19260126)
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

    def get_cap(self,
                date: int, 
                cache_mode: str ="rw", 
                use_daily: bool = True, 
                use_permco: bool = True) -> Series:
        """Compute a cross-section of market capitalization values

        Args:
            date: YYYYMMDD int date of market cap
            cache_mode: 'r' to try read from cache first, 'w' to write to cache
            use_daily: If True, use shrout from 'daily' table, else 'shares'
            use_permco: If True, sum caps by permco, else by permno

        Returns:
            Series of market cap indexed by permno
        """
        rkey = f"cap{'co' if use_permco else ''}_{str(self)}_{date}"
        if self.rdb and 'r' in cache_mode and self.rdb.redis.exists(rkey):
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
        rkey = "_".join(["universe", str(self), str(date)])
        if 'r' in cache_mode and self.rdb and self.rdb.redis.exists(rkey):
            self._print('(get_universe load)', rkey)
            df = self.rdb.load(rkey)
        else: 
            df = self.get_section(dataset='daily',
                                  fields=['prc', 'shrout'],
                                  date_field='date',
                                  date=date)
            df['cap'] = df['shrout'] * df['prc'].abs()
            # TODO: market cap by permco ?
            
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
            df['nyse'] = df['exchcd'].eq(1)                 # nyse indicator
            df['decile'] = fractiles(values=df['cap'],      # size deciles 
                                     pct=np.arange(10, 100, 10),
                                     keys=df.loc[df['nyse'], 'cap'],
                                     ascending=False)
            df = df[['cap', 'decile', 'nyse', 'siccd', 'prc', 'naics']]
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
        if isinstance(dlstcd, int):
            return dlstcd in CRSP.dlstcodes_
        else:
            return dlstcd.isin(CRSP.dlstcodes_)

    def get_dlret(self, beg: int, end: int, dataset: str = 'delist',
                    cache_mode: str = "rw") -> Series:
        """Compounded delisting returns from beg to end dates for all permnos

        Args:
          beg: Inclusive start date (YYYYMMDD)
          end: Inclusive end date (YYYYMMDD)
          dataset: either 'delist' or 'monthly' containing delisting returns
          cache_mode: 'r' to try read from cache first, 'w' to write to cache

        Returns:
          Series of delisting returns

        Notes:
          Sets to -0.3 if missing and code in [500, 520, 551...574, 580, 584]
        """
        rkey = "_".join(["dlst", dataset, str(beg), str(end)])
        if 'r' in cache_mode and self.rdb and self.rdb.redis.exists(rkey):
            self._print("(get_dlret load)", rkey, dataset)
            return self.rdb.load(rkey)['dlret']
        
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
            df.loc[((df['dlret'].isna())
                    & CRSP.is_dlstcode(df['dlstcd'])).values, 'dlret'] = 0.7
            df = df[[self.identifier, 'dlret']].groupby(self.identifier)\
                                               .prod(min_count=1)\
                                               .dropna() - 1
        if 'w' in cache_mode and self.rdb:
            self._print("(get_dlret dump)", rkey, dataset)
            self.rdb.dump(rkey, df)            
        return df['dlret']

    def get_ret(self, beg: int, end: int, dataset: str = 'daily',
                delist: bool = False, **kwargs) -> Series:
        """Get compounded returns, with option to include delist returns

        Args:
          beg: starting returns date
          end: ending returns date
          dataset: name of returns dataset to use (unless initialized 'monthly')
          delist: whether to adjust by delisting returns
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
        if use_monthly:
            dataset = 'monthly'
            beg = (beg // 100) * 100
            end = (end // 100 * 100) + 99

        df = super().get_ret(beg, end, dataset=dataset, **kwargs)

        # if adjust for delisting returns
        if delist:
            dlst = self.get_dlret(beg, end,
                                  dataset='monthly' if use_monthly else 'delist')
            df = DataFrame(df).join(dlst, how='outer').fillna(0.0)
            df = (1+df['ret']) * (1+df['dlret']) - 1
        return df


if __name__ == "__main__":
    from pathlib import Path
    from finds.structured.stocks import StocksBuffer
    from secret import credentials
    downloads = Path('/home/terence/Downloads/stocks2023/CRSP')
    
    VERBOSE = 1
    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    user = SQL(**credentials['user'], verbose=VERBOSE)
    rdb = RedisDB(**credentials['redis'])
    bd = BusDay(sql)
    crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)

if False:

    # load CRSP: TODO handle missing return codes (< -1, see below)
    df = crsp.load_csv('names', downloads / Path('names.txt.gz'), sep='\t')
    print(len(df), 103383)
    df = crsp.load_csv('shares', downloads / Path('shares.txt.gz'), sep='\t')
    print(len(df), 2346131)
    df = crsp.load_csv('dist', downloads / Path('dist.txt.gz'), sep='\t')
    print(len(df), 935880)
    df = crsp.load_csv('delist', downloads / Path('delist.txt.gz'), sep='\t')
    print(len(df), 33584)
    df = crsp.load_csv('monthly', downloads / Path('monthly.txt.gz'), sep='\t')
    print(len(df), 4606907)
    
    for s in sorted(downloads.glob('daily*.txt.gz'), reverse=True):
        tic = time.time()
        df = crsp.load_csv('daily', csvfile=s, sep='\t',
                           drop={'permno': ['PERMNO', '.'],
                                 'date': ['.'],
                                 'shrout':['.']})
        print(s, round(time.time() - tic, 0), 'secs:', len(df), s)

    # Pre-generate weekly returns and save in Redis cache
    begweek = 19251231
    endweek = 20221230
    rebaldates = bd.date_range(begweek, endweek, freq='weekly')
    r = bd.date_tuples(rebaldates)
    batchsize = 40
    batches = [r[i:(i+batchsize)] for i in range(0, len(r), batchsize)]
    for batch in batches:
        crsp.cache_ret(batch, replace=True)

    # test StocksBuffer
    stocks = StocksBuffer(crsp, 20210101, 20211231, dataset='monthly')
    beg, end = 20210101, 20210131
    df = stocks.get_ret(beg, end)
    print(df) 
    m = crsp.get_ret(beg, end, cache_mode="")
    print(m)


if False:
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
