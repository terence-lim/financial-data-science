"""Fama-French Research Factors

- Fama-French monthly research factors: HML, SMB, Mom, STRev
- Portfolio sorts
- CRSP, Compustat, Ken French Data Library

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4):
    return finds.display.show(df, latex=latex, ndigits=ndigits)
figext = '.jpg'

import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from finds.database import SQL, Redis
from finds.structured import CRSP, Signals, Benchmarks, PSTAT
from finds.busday import BusDay
from finds.recipes import fractiles
from conf import credentials, VERBOSE, paths, CRSP_DATE

LAST_DATE = CRSP_DATE
LAST_DATE= 20201231

VERBOSE = 1
imgdir = paths['images']
sql = SQL(**credentials['sql'], verbose=VERBOSE)
user = SQL(**credentials['user'], verbose=VERBOSE)

rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
pstat = PSTAT(sql, bd, verbose=VERBOSE)
bench = Benchmarks(sql, bd)
signals = Signals(user)

def _print(*args, **kwargs):
    if VERBOSE > 0:
        print(*args, **kwargs)

# Load items from Compustat Annual
# Construct HML as shareholders equity plus investment tax credits, 
#   less preferred stock, divided by December market cap.
# Require 6 month reporting lag and at least two years history in Compustat

label = 'hml'
lag = 6               # number of months to lag fundamental data

# retrieve data fields from compustat, linked by permno
df = pstat.get_linked(dataset = 'annual',
                      date_field = 'datadate',
                      fields = ['seq', 'pstk', 'pstkrv', 'pstkl', 'txditc'],
                      where = ("indfmt = 'INDL'"
                               "  AND datafmt = 'STD'"
                               "  AND curcd = 'USD' "
                               "  AND popsrc = 'D'"
                               "  AND consol = 'C'"
                               "  AND seq > 0 "))

# subtract preferred stock, add back deferred investment tax credit
df[label] = np.where(df['pstkrv'].isna(), df['pstkl'], df['pstkrv'])
df[label] = np.where(df[label].isna(), df['pstk'], df[label])
df[label] = np.where(df[label].isna(), 0, df[label])
df[label] = df['seq'] + df['txditc'].fillna(0) - df[label]
df.dropna(subset = [label], inplace=True)
df = df[df[label] > 0][['permno', 'gvkey', 'datadate', label]]

# count years in Compustat        
df = df.sort_values(by=['gvkey','datadate'])
df['count'] = df.groupby(['gvkey']).cumcount()   

# construct b/m ratio
df['rebaldate'] = 0
for datadate in sorted(df['datadate'].unique()):
    f = df['datadate'].eq(datadate)
    df.loc[f, 'rebaldate'] = crsp.bd.endmo(datadate, abs(lag)) # 6 month lag
    df.loc[f, 'cap'] = crsp.get_cap(crsp.bd.endyr(datadate))\
                           .reindex(df.loc[f, 'permno']).values # Dec mktcap
    _print(datadate, sum(f))
df[label] /= df['cap']
df = df[df[label].gt(0) & df['count'].gt(1)]  # 2+ years in Compustat
signals.write(df, label)

# Helper for plotting to compare portfolio returns
from finds.display import plot_date, plot_scatter
def plot_ff(y, label, num=1, imgdir=None):
    """Helper to plot similarity of portfolio and benchmark returns"""
    y = y.rename(columns={'excess': label})
    corr = np.corrcoef(y, rowvar=False)[0,1]
    fig, (ax1, ax2) = plt.subplots(2, 1, num=num, clear=True, figsize=(9,10))
    plot_date(y, ax=ax1, title=" vs ".join(y.columns))
    plot_scatter(y.iloc[:,0], y.iloc[:,1], ax=ax2, abline=False)
    ax2.set_title(f"corr={corr:.4f}", fontsize=8)
    plt.tight_layout(pad=3)
    if imgdir is not None:
        plt.savefig(os.path.join(imgdir, label  + figext))
        print(f"<Correlation of {label} vs {benchname}"
              f" ({y.index[0]} - {y.index[-1]}): {corr:.4f}")

## Construct HML portfolio returns and compare to benchmark
from finds.structured import famafrench_sorts
label, benchname = 'hml', 'HML(mo)'
rebalend = LAST_DATE
rebalbeg = 19640601
portfolios = famafrench_sorts(crsp,
                              label,
                              signals,
                              rebalbeg,
                              rebalend,
                              window=12,
                              months=[6])
holdings = portfolios['holdings'][label]

from finds.backtesting import BackTest
backtest = BackTest(user, bench, 'RF', LAST_DATE)
result = backtest(crsp, holdings, label)
y = backtest.fit([benchname], 19700101, LAST_DATE)
plot_ff(y, label, num=1, imgdir=imgdir)


## Compare SMB
label, benchname = 'smb', 'SMB(mo)'
holdings = portfolios['holdings'][label]
result = backtest(crsp, holdings, label)
y = backtest.fit([benchname], 19700101, LAST_DATE)
plot_ff(y, label, num=2, imgdir=imgdir)   


## Construct Mom

# Load monthly universe and stock returns from CRSP.
# Signal is stocks' total return from 12 months ago, skipping most recent month
# Construct 2-way portfolio sorts, and backtest returns  

label, benchname, past, leverage = 'mom', 'Mom(mo)', (2,12), 1
rebalbeg, rebalend = 19270101, LAST_DATE

df = DataFrame()      # collect each month's momentum signal values
for rebaldate in bd.date_range(rebalbeg, rebalend, 'endmo'):  
    beg = bd.endmo(rebaldate, -past[1])   # require price at this date
    start = bd.offset(beg, 1)             # start date, inclusive, of signal
    end = bd.endmo(rebaldate, 1-past[0])  # end date of signal
    p = [crsp.get_universe(rebaldate),    # retrieve prices and construct signal
         crsp.get_ret(start, end).rename(label),
         crsp.get_section('monthly', ['prc'], 'date', beg)['prc'].rename('beg'),
         crsp.get_section('monthly', ['prc'], 'date', end)['prc'].rename('end')]
    q = pd.concat(p, axis=1, join='inner').reset_index().dropna()
    q['rebaldate'] = rebaldate
    df = pd.concat([df, q[['permno', 'rebaldate', label]]], axis=0)
    _print(rebaldate, len(df), len(q))
signals.write(df, label, overwrite=True)

portfolios = famafrench_sorts(crsp,
                              label,
                              signals,
                              rebalbeg,
                              rebalend,
                              window=0,
                              months=[],
                              leverage=leverage)
holdings = portfolios['holdings'][label]
result = backtest(crsp, holdings, label)
y = backtest.fit([benchname])
plot_ff(y, label, num=3, imgdir=imgdir)


## Construct STRev

# Signal value is recent month's stock returns, sign flipped
label, benchname, past, leverage = 'strev', 'ST_Rev(mo)', (1,1), -1
rebalbeg, rebalend = 19260101, LAST_DATE   #rebalbeg = 20100101

# loop over each rebalance date to construct and collect signals values 
df = DataFrame()
for rebaldate in bd.date_range(rebalbeg, rebalend, 'endmo'):  
    beg = bd.endmo(rebaldate, -past[1])   # beg price date of signal
    end = bd.endmo(rebaldate, 1-past[0])  # end price date of signal

    # Retrieve universe, require have prices at beg and end dates,
    # and construct signal as returns compounded between start and end dates
    p = [crsp.get_universe(rebaldate),
         crsp.get_section('monthly', ['prc'], 'date', beg)['prc'].rename('beg'),
         crsp.get_section('monthly', ['prc'], 'date', end)['prc'].rename('end'),
         crsp.get_ret(bd.offset(beg, 1), end).rename(label)]
    q = pd.concat(p, axis=1, join='inner').reset_index().dropna()
    q['rebaldate'] = rebaldate
    df = pd.concat((df, q[['permno','rebaldate', label]]), axis=0)
    _print(rebaldate, len(df), len(q))

# Save signals values
signals.write(df, label, overwrite=True)
                  
# Construct holdings with 2-way sort, and evaluate backtest
portfolios = famafrench_sorts(crsp,
                              label,
                              signals,
                              rebalbeg,
                              rebalend,
                              window=0,
                              months=[],
                              leverage=leverage)
holdings = portfolios['holdings'][label]
result = backtest(crsp, holdings, label)
y = backtest.fit([benchname])
plot_ff(y, label, num=4, imgdir=imgdir)
