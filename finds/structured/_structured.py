"""Classes to implement interface for structured data sets

- CRSP (daily, monthly, names, delistings, distributions, shares outstanding)
- S&P/CapitalIQ Compustat (Annual, Quarterly, Key Development, customers)
- IBES Summary

Notes:

- Redis store: some methods optionally cache SQL query results in Redis
- Signals class to create, store and retrieve derived signal values
- Subclasses to mimic parent class database interfaces with in-memory batch
- Lookup identifiers within and across datasets


Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_list_like, is_integer_dtype
from sqlalchemy import Table, Column, Index
from sqlalchemy import Integer, String, Float, SmallInteger, Boolean, BigInteger
from datetime import datetime
from finds.database import SQL, RedisDB
from finds.busday import BusDay
from finds.structured import Lookup

if __name__ == "__main__":
#    from os.path import dirname, abspath
#    sys.path.insert(0, dirname(dirname(abspath(__file__))))
    from conf import credentials, VERBOSE

    import glob
    import time
    from pandas import DataFrame, Series
    from finds.database import SQL, Redis
    from finds.busday import BusDay, WeeklyDay

    VERBOSE = 1

    # open all structured datasets
    if True:
        sql = SQL(**credentials['sql'], verbose=VERBOSE)
        user = SQL(**credentials['user'], verbose=VERBOSE)
        rdb = Redis(**credentials['redis'])
        bd = BusDay(sql)
        bench = Benchmarks(sql, bd)
        find = Finder(sql)
        print(find('GOOG'))

        crsp = CRSP(sql, bd, rdb=rdb)
        pstat = PSTAT(sql, bd)
        ibes = IBES(sql, bd)

    # load benchmarks (mostly FamaFrench)
    def update_FamaFrench():
        print("\n".join(f"[{i}] {d}" 
              for i, d in enumerate(FFReader._datasets)))
        for name, item, suffix in FFReader._datasets:
            date_formatter = (bd.endmo if suffix == '(mo)' else bd.offset)
            df = FFReader.fetch(name=name, 
                                item=item,
                                suffix=suffix,
                                date_formatter=date_formatter)
            for col in df.columns:
                print(bench.load_series(df[col], name=name, item=str(item)))
        print(DataFrame(**sql.run('select * from ' + bench['ident'].key)))

    def test_bench():
        print(bench.get_series('CMA', 'ret'))
        print(bench.get_series(['CMA', 'HML'], 'ret'))


    # load CRSP: TODO handle missing return codes (< -1, see below)
    def update_crsp():
        dir = '/home/terence/Downloads/stocks2022/v1/CRSP/'
        crsp.load_csv('names',
                      os.path.join(dir, 'names.txt.gz'),
                      sep='\t')   # 103383
        crsp.load_csv('shares',
                      os.path.join(dir, 'shares.txt.gz'),
                      sep='\t') # 2346131
        crsp.load_csv('dist',
                      os.path.join(dir, 'dist.txt.gz'),
                      sep='\t') # 935880
        crsp.load_csv('delist',
                      os.path.join(dir, 'delist.txt.gz'),
                      sep='\t')  # 33584
        crsp.load_csv('monthly',
                      os.path.join(dir, 'monthly.txt.gz'),
                      sep='\t') #4606907
#        for s in sorted(glob.glob(os.path.join(dir, 'stocks*.txt.gz')),
#                        reverse=True):
        for s in [os.path.join(dir, 'stocks20202021.txt.gz')]:
            tic = time.time()
            crsp.load_csv('daily',
                          csvfile=s,
                          sep='\t', 
                          drop={'permno': ['PERMNO', '.'],
                                'date': ['.'],
                                'shrout':['.']})
            print(s, round(time.time() - tic, 0), 'secs')


    # Pre-generate weekly returns and save in Redis cache
    begweek = 19251231
    endweek = 20221230
    def update_weekly(day='Thu'):
        wd = WeeklyDay(sql, day)   # Generate weekly cal
        rebaldates = wd.date_range(begweek, endweek)
        r = wd.date_tuples(rebaldates)
        batchsize = 40
        batches = [r[i:(i+batchsize)] for i in range(0, len(r), batchsize)]
        for batch in batches:
            crsp.cache_ret(batch, replace=True)

    
    # load Compustat
    def update_pstat():
        dir = '/home/terence/Downloads/stocks2022/v1/PSTAT/'
        df = pstat.load_csv('links',
                            csvfile=os.path.join(dir, 'links.txt.gz'),
                            sep='\t',    # rows=33036
                            drop={'lpermno': ['0', 0], 'linkprim': ['N', 'J']},
                            replace={'linkdt': (['C', 'E', 'B'], 0),
                                     'linkenddt': (['C', 'E', 'B'], 0)})
        lag = df.shift()
        f = (lag.gvkey == df.gvkey) & (lag.lpermno != df.lpermno)
        print('permnos in links changed in ', sum(f), 'of', len(df)) # 1063

        pstat.load_csv('annual',
                       os.path.join(dir, 'annual.txt.gz'),
                       sep='\t') #rows = 464753
        pstat.load_csv('quarterly',
                       os.path.join(dir, 'quarterly.txt.gz'),
                       sep='\t') #1637274
        pstat.load_csv('customer',
                       os.path.join(dir, 'supplychain.csv.gz'),
                       sep='\t') #107114
        for s in glob.glob(os.path.join(dir, 'keydev*.txt.gz')):
            tic = time.time()   # 12256909
            df = pstat.load_csv('keydev',
                                csvfile=s,
                                sep='\t',
                                drop={'gvkey': [0, '0'],
                                      'announcedate': [0, '0'],
                                      'keydevid': [0, '0']})
            print(s, time.time() - tic)    

    # load IBES
    def update_ibes():
        dir = '/home/terence/Downloads/stocks2022/v1/IBES/'
        ibes.load_csv('ident',
                      os.path.join(dir, 'ident.txt.gz'),
                      sep='\t')  # 85550
        ibes.write_links()  #  (missing, count) = 15340  88963
        ibes.load_csv('history',
                      os.path.join(dir, 'history.txt.gz'),
                      sep='\t')
        ibes.load_csv('summary',
                      os.path.join(dir, 'summary.txt.gz'),
                      sep='\t') # 11776742
        #ibes.load_csv('adjust', downloads + 'adjustment.csv') #rows=24777
        #ibes.load_csv('surprise', downloads + 'surprise.csv')  #rows=528933

    def test_rets():
        stocks = StocksBuffer(crsp, 20210101, 20211231)
        df = stocks.get_ret(20210101, 20210131)
        print(df) 
        m = crsp.get_ret(20210101, 20210131)
        print(m)


    """Update
    CRSP tab-delimited text & gzip (*.txt.gz), YYMMDDn8 date format
    - names, shares, dist, delist, monthly
    - monthly: prc, ret, retx
    - daily: shrout, ...
    update_crsp()

    Compustat/CRSP Merged Linking: 
    - all Linking options: LC, LU ?
    - Link and Identifying Information
    update_pstat()
    


    """

    
