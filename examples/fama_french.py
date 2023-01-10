"""Fama-French portfolio sorts

- Value and size effect
- Bivariate portfolio sorts
- Fama-French research factors: HML, SMB, Mom, STRev

Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from finds.database import SQL, Redis
from finds.structured import CRSP, Signals, Benchmarks, PSTAT, famafrench_sorts
from finds.backtesting import BackTest
from finds.busday import BusDay
from finds.recipes import fractiles
from finds.display import plot_date, plot_scatter, plot_hist, show
from tqdm import tqdm
from conf import credentials, VERBOSE, paths, CRSP_DATE

LAST_DATE = CRSP_DATE
%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None

imgdir = paths['images']
sql = SQL(**credentials['sql'], verbose=VERBOSE)
user = SQL(**credentials['user'], verbose=VERBOSE)

rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
pstat = PSTAT(sql, bd, verbose=VERBOSE)
bench = Benchmarks(sql, bd)
signals = Signals(user)

## Construct HML and SMB signal
"""
- load items from Compustat Annual
- Construct HML as shareholders equity plus investment tax credits, 
   less preferred stock, divided by December market cap.
- Require 6 month reporting lag and at least two years history in Compustat
"""

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
for datadate in tqdm(sorted(df['datadate'].unique())):
    f = df['datadate'].eq(datadate)
    rebaldate = crsp.bd.endmo(datadate, abs(lag)) # 6 month lag
    capdate = crsp.bd.endyr(datadate)   # Dec mktcap
    if rebaldate >= LAST_DATE or capdate >= LAST_DATE:
        continue
    df.loc[f, 'rebaldate'] = rebaldate
    df.loc[f, 'cap'] = crsp.get_cap(capdate).reindex(df.loc[f, 'permno']).values
df[label] /= df['cap']
df = df[df[label].gt(0) & df['count'].gt(1)]  # 2+ years in Compustat
signals.write(df, label)

### helpers to show histograms and comparisons of portfolio returns
def plot_ff(y, label, savefig=None):
    """helper to scatter plot and compare portfolio returns"""
    y = y.rename(columns={'excess': label})
    corr = np.corrcoef(y, rowvar=False)[0,1]
    fig, (ax1, ax2) = plt.subplots(2, 1, clear=True, figsize=(5, 5))
    plot_date(y, ax=ax1, title=" vs ".join(y.columns), fontsize=7)
    plot_scatter(y.iloc[:,0], y.iloc[:,1], ax=ax2, abline=False, fontsize=7)
    plt.legend([f"corr={corr:.4f}"], fontsize=8)
    plt.tight_layout(pad=0.5)
    if savefig:
        plt.savefig(imgdir / savefig)
    print(f"<Correlation of {label} vs {benchname}"
          f" ({y.index[0]} - {y.index[-1]}): {corr:.4f}")

def plot_summary(y, label, savefig=None):
    """helper to plot histogram and statistics of portfolio returns"""
    y = y[label]
    kurt = kurtosis(y, bias=True, fisher=True)  # excess kurtosis
    skewness = skew(y, bias=True)
    fig, ax = plt.subplots(1, 1, clear=True, figsize=(5, 5))
    ax.hist(y, bins=30)
    ax.set_title(f"Histogram of monthly returns ({y.index[0]}-{y.index[-1]})")
    ax.set_xlabel(f"skewness={skewness:.4f}, excess kurtosis={kurt:.4f}")
    plt.legend([label])
    plt.tight_layout()
    if savefig:
        plt.savefig(imgdir / savefig)
        
## Construct HML bivariate portfolio sorts, and show counts
label, benchname = 'hml', 'HML(mo)'
rebalend = LAST_DATE
rebalbeg = 19640601
portfolios = famafrench_sorts(stocks=crsp,
                              label=label,
                              signals=signals,
                              rebalbeg=rebalbeg,
                              rebalend=rebalend,
                              window=12,
                              months=[6])

sizes = {b: [int(np.mean(list(portfolios['sizes'][b[0]+s[0]].values())))
             for s in "SB"]
         for b in ["High b/m", "Medium b/m", "Low b/m"]}
show(DataFrame(sizes, index=["Small size", "Big size"]),
     caption="Average number of stocks in bivariate-sorted subportfolios",
     **SHOW)

### Plot histogram and comparison of HML returns
holdings = portfolios['holdings'][label]        
backtest = BackTest(user, bench, 'RF', LAST_DATE, verbose=VERBOSE)
result = backtest(crsp, holdings, label)
y = backtest.fit([benchname], 19700101, LAST_DATE)
plot_ff(y, label, savefig=imgdir / (label + '.jpg'))
plot_summary(y, benchname, savefig=imgdir / (label + '_hist.jpg'))
          
### Linear regression on Mkt-Rf and intercept
x = bench.get_series('mkt-rf(mo)', start=y.index[0], end=y.index[-1])
lm = smf.ols(f'Q("{benchname}")~Q("{x.name}")', data=pd.concat([y, x], axis=1))\
        .fit(cov_type='HAC', cov_kwds={'maxlags': 6})
print(f"Period: {rebalbeg}-{rebalend}")
print(lm.summary())

### Plot histogram and comparison of SMB returns
label, benchname = 'smb', 'SMB(mo)'
holdings = portfolios['holdings'][label]
result = backtest(crsp, holdings, label)
y = backtest.fit([benchname], 19700101, LAST_DATE)
plot_ff(y, label, savefig=imgdir / (label + '.jpg'))
plot_summary(y, benchname, savefig=imgdir / (label + '_hist.jpg'))

# Linear regression on Mkt-Rf and intercept
x = bench.get_series('mkt-rf(mo)', start=y.index[0], end=y.index[-1])
lm = smf.ols(f'Q("{benchname}")~Q("{x.name}")', data=pd.concat([y, x], axis=1))\
        .fit(cov_type='HAC', cov_kwds={'maxlags': 6})
print(f"Period: {rebalbeg}-{rebalend}")
print(lm.summary())


## Construct MOM signal
"""
- Load monthly universe and stock returns from CRSP.
- Signal is stocks' total return from 12 months ago, skipping most recent month
- Construct 2-way portfolio sorts, and backtest returns
"""

label, benchname, past, leverage = 'mom', 'Mom(mo)', (2,12), 1
rebalbeg, rebalend = 19270101, LAST_DATE

df = DataFrame()      # collect each month's momentum signal values
for rebaldate in tqdm(bd.date_range(rebalbeg, rebalend, 'endmo')):
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
signals.write(df, label, overwrite=True)

## Construct MOM bivariate portfolio sorts
portfolios = famafrench_sorts(stocks=crsp,
                              label=label,
                              signals=signals,
                              rebalbeg=rebalbeg,
                              rebalend=rebalend,
                              window=0,
                              months=[],
                              leverage=leverage)
holdings = portfolios['holdings'][label]
result = backtest(crsp, holdings, label)
y = backtest.fit([benchname])

# plot histogram and comparison of returns
plot_ff(y, label, savefig=imgdir / (label + '.jpg'))
plot_summary(y, benchname, savefig=imgdir / (label + '_hist.jpg'))

# Linear regression on Mkt-Rf and intercept
lm = smf.ols(f'Q("{benchname}")~Q("{x.name}")', data=pd.concat([y, x], axis=1))\
        .fit(cov_type='HAC', cov_kwds={'maxlags': 6})
print(f"Period: {rebalbeg}-{rebalend}")
print(lm.summary())


## Construct STRev signal

# Signal value is recent month's stock returns, sign flipped
label, benchname, past, leverage = 'strev', 'ST_Rev(mo)', (1,1), -1
rebalbeg, rebalend = 19260101, LAST_DATE   #rebalbeg = 20100101

# loop over each rebalance date to construct and collect signals values 
df = DataFrame()
for rebaldate in tqdm(bd.date_range(rebalbeg, rebalend, 'endmo')):
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

# Save signals values
signals.write(df, label, overwrite=True)
                  
# Construct bivariate portfolios sort
portfolios = famafrench_sorts(stocks=crsp,
                              label=label,
                              signals=signals,
                              rebalbeg=rebalbeg,
                              rebalend=rebalend,
                              window=0,
                              months=[],
                              leverage=leverage)
holdings = portfolios['holdings'][label]
result = backtest(crsp, holdings, label)
y = backtest.fit([benchname])

# plot histogram and comparison of returns
plot_ff(y, label, savefig=imgdir / (label + '.jpg'))
plot_summary(y, benchname, savefig=imgdir / (label + '_hist.jpg'))

# Linear regression on Mkt-Rf and intercept
lm = smf.ols(f'Q("{benchname}")~Q("{x.name}")', data=pd.concat([y, x], axis=1))\
        .fit(cov_type='HAC', cov_kwds={'maxlags': 6})
print(f"Period: {rebalbeg}-{rebalend}")
print(lm.summary())
