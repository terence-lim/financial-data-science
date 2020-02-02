"""
Construct quant factors
  - monthly portfolio sorts
  - backtest performance, compare to Fama-French returns on Ken French website

References:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

import matplotlib.pyplot as plt
import time
from pprint import pprint
import dives
import dives.util
import dives.dbengines
import dives.structured
import dives.evaluate

import importlib
importlib.reload(dives)
importlib.reload(dives.util)
importlib.reload(dives.structured)
importlib.reload(dives.dbengines)
importlib.reload(dives.evaluate)

from dives.util import fractiles, DataFrame
from dives.dbengines import SQL, Redis
from dives.structured import BusDates, Benchmarks, CRSP, PSTAT
from dives.evaluate import BackTest
import secret
sql = SQL(**secret.value('sql'))       
rdb = Redis(**secret.value('redis'))       
    
bd = BusDates(sql)
bench = Benchmarks(sql, bd)
backtest = BackTest(sql, bench, 'RF')
crsp = CRSP(sql, bd, rdb)
pstat = PSTAT(sql, bd)

if True: # Weekly reversal strategy
    crsp.rdb = None      # do not use redis cache
    tic = time.time()
    holdings = {}
#    rebalbeg, rebalend = 19500601, 20190630
    rebalbeg, rebalend = 20060701, 20080831
    for weeknum in bd.week_range(rebalbeg, rebalend):
        pordate = bd.week_end(weeknum)              # rebalance date
        beg = bd.shift(bd.week_beg(weeknum), -1)    # date prior to period for measuring past return
        start = bd.week_beg(weeknum)                # first day of period to measure past return
        end = bd.week_beg(weeknum)                  # last day of period to measure past return

        df = crsp.get_universe(pordate)             # generate new universe every week: common domestic
        df['beg'] = crsp.get_section('daily', ['prc'], 'date', beg).reindex(df.index)
        df['end'] = crsp.get_section('daily', ['prc'], 'date', end).reindex(df.index)
        df['signal'] = crsp.get_ret(start, end).reindex(df.index)
        df = df[df['beg'].notnull() & df['end'].notnull() & df['signal'].notnull()]
        df['fractile'] = fractiles(df['signal'], [30,70], df['signal'][df.nyse])
        permnos, weights = [],[]            
        subs = [(df.fractile == 3) & (df.deciles > 5),               # big winner subportfolio
                (df.fractile == 3) & (df.deciles <= 5),              # small winner subportfolio
                (df.fractile == 1) & (df.deciles > 5),               # big loser subportfolio
                (df.fractile == 1) & (df.deciles <= 5)]              # small loser subportfolio
        for sub, weight in zip(subs, [0.5, 0.5, -0.5, -0.5]):        # combine subportfolios
            weights += list(weight * df.loc[sub,'cap'] / df.loc[sub,'cap'].sum())
            permnos += list(df.index[sub])
        holdings[pordate] = DataFrame(data=weights, index=permnos, columns=['weights'])

    benchnames = ['ST_Rev']
    backtest.performance(crsp, holdings, label=('weekly momentum'))
    backtest.attribution(benchnames, flip=True)     # flip sign of returns because is reversal strategy
    backtest.plot(benchnames)
    print('Elapsed %.1f secs ' + str(time.time() - tic))
    print(backtest.results.summary())
    pprint(backtest.annualized)
    
if True: # to create HML signal (allow 6 months for data to become available)
    tic = time.time()
    crsp.rdb = rdb   # reaffirm use redis cache
    rebalbeg, rebalend = 19640601, 20190630
    rebalbeg, rebalend = 19500601, 20190630
    beg, end = crsp.dates.begmo(rebalbeg, -6), crsp.dates.endmo(rebalend, -6)

    # retrieve required fields from database
    df = pstat.get_linked(table = 'annual', date_field = 'datadate',
                          fields = ['ceq','pstk'],  # 'pstkrv', 'pstkl',
                          where = ("ceq > 0 and datadate >= {beg} and datadate <= {end}" \
                                   "".format(beg=beg, end=end)))
    # construct book values
    df['hml'] = 0
    df.loc[df['pstk'].gt(0),   'hml'] = -df.loc[df['pstk'].gt(0),   'pstk']
#    df.loc[df['pstkl'].gt(0),  'hml'] = -df.loc[df['pstkl'].gt(0),  'pstkl']
#    df.loc[df['pstkrv'].gt(0), 'hml'] = -df.loc[df['pstkrv'].gt(0), 'pstkrv']
    df['hml'] += df['ceq']
    df['rebaldate'] = 0
    
    # construct b/m ratio
    for datadate in sorted(df['datadate'].unique()):
        f = df['datadate'].eq(datadate)
        rebaldate = crsp.dates.endmo(datadate, 6)  # rebalance date >= 6 months after fiscal data date
        pricedate = bd._months.loc[((rebaldate-10000)//10000, 12), 'endmo']      # december market cap
        permnos = list(df.loc[f, 'permno'])
        df.loc[f, 'rebaldate'] = rebaldate
        df.loc[f, 'cap'] = (crsp.get_cap(pricedate).reindex(permnos)).values
    df['hml'] /= df['cap']

    benchnames = ['HML(mo)']
    signal = 'hml'
    holdings = crsp.portfolio_sorts(signal,
                                    data = df,
                                    beg = rebalbeg,
                                    end = rebalend,
                                    window = 12,     # use latest signal from recent 12 months rebalances
                                    month = 6)       # determine universe of stocks every June
    backtest.performance(crsp, holdings, label = signal)
#    backtest.save()
    backtest.attribution(benchnames, flip=False)     # do not flip returns (not a reversal strategy)
    backtest.plot(benchnames)
#    perf = backtest.load(signal)
    print('Elapsed %.1f secs ' + str(time.time() - tic))
    print(backtest.results.summary())
    pprint(backtest.annualized)

