"""Survivorship-bias and low-price stocks strategy

- Low-price portfolio spread returns
- Survivorship-bias
- Autocorrelation-consistent standard errors: Newey-West

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

import os
import glob
import time
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from finds.database import SQL, Redis
from finds.busday import BusDay
from finds.structured import CRSP, Finder
from finds.recipes import fractiles
from conf import credentials, VERBOSE, paths

VERBOSE = 0

sql = SQL(**credentials['sql'], verbose=VERBOSE)
rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
imgdir = paths['images']

begyear = 1992
endyear = 2021
retdates = bd.date_range(bd.begyr(begyear), bd.endyr(endyear), 'begmo')
rebaldates = bd.offset(retdates, -1)
percentiles = [20, 80]
maxhold = 12   # hold each monthly portfolio for one year (12 monthss)

# 1. Survivors-only, overlapping returns
out1 = []
num1 = []
ports = []
select = crsp.get_universe(rebaldates[-1])
for rebaldate in rebaldates:
    univ = crsp.get_universe(rebaldate)
    univ = univ.loc[univ.index.intersection(select.index)]  # survivors only
    num1.append(len(univ))
    #print(rebaldate, len(univ), len(select))

    # get price-based assignments using NYSE tritiles on rebalance date
    tritile = fractiles(values=univ['prc'].abs(),
                        pct=percentiles,
                        keys=univ.loc[univ['nyse'], 'prc'].abs())

    # construct cap-wtd tritile spread portfolios
    porthi, portlo = [univ.loc[tritile==t, 'cap'] for t in [1, 3]]
    p =  pd.concat((-porthi/porthi.sum(), portlo/portlo.sum()))
    
    
    # compute and store cap-weighted average returns over maxhold periods
    begret = bd.offset(rebaldate, 1)
    nhold = min(maxhold, len(retdates) - retdates.index(begret))
    endret = bd.endmo(begret, nhold - 1)  # truncate end if maxhold is beyond
    rets = crsp.get_ret(begret, endret, delist=True)
    ret = rets.reindex(p.index).fillna(0.).mul(p, axis=0).sum()

    out1.append(float(ret) / nhold)
    
# 2. All stocks in each monthly universe, overlapping returns
out2 = []
num2 = []
for rebaldate in rebaldates:
    
    # get price-based assignments based on NYSE tritiles on rebalance date
    univ = crsp.get_universe(rebaldate)
    num2.append(len(univ))
    tritile = fractiles(values=univ['prc'].abs(),
                        pct=percentiles,
                        keys=univ.loc[univ['nyse'], 'prc'].abs())

    # construct cap-wtd tritile spread portfolios
    porthi, portlo = [univ.loc[tritile==t, 'cap'] for t in [1, 3]]
    p =  pd.concat((-porthi/porthi.sum(), portlo/portlo.sum()))

    # compute and store cap-weighted average returns over maxhold periods
    begret = bd.offset(rebaldate, 1)
    nhold = min(maxhold, len(retdates) - retdates.index(begret))
    endret = bd.endmo(begret, nhold - 1)  # if maxhold is beyond end date
    rets = crsp.get_ret(begret, endret, delist=True)
    ret = rets.reindex(p.index).fillna(0.).mul(p, axis=0).sum()

    out2.append(float(ret) / nhold)

# 3. All stocks in each monthly universe, non-overlapping returns
num3 = []
out3 = []
ports = []
for rebaldate in rebaldates:
    
    # get price-based assignments based on NYSE tritiles on rebalance date
    univ = crsp.get_universe(rebaldate)
    num3.append(len(univ))
    tritile = fractiles(values=univ['prc'].abs(),
                        pct=percentiles,
                        keys=univ.loc[univ['nyse'], 'prc'].abs())

    # construct cap-wtd tritile spread portfolios
    porthi, portlo = [univ.loc[tritile==t, 'cap'] for t in [1, 3]]
    p =  pd.concat((-porthi/porthi.sum(), portlo/portlo.sum()))

    # keep only last maxhold months' rebalances
    ports.insert(0, p)
    if len(ports) > maxhold:
        ports.pop(-1)

    # compute all portfolios' monthly capwtd returns, and store eqlwtd average
    begret = bd.offset(rebaldate, 1)
    endret = bd.endmo(begret)
    rets = crsp.get_ret(begret, endret, delist=True)
    ret = np.mean([rets.reindex(p.index).fillna(0.).mul(p, axis=0).sum()
                   for p in ports])
    out3.append(ret)

    # adjust stock weights by monthly appreciation
    retx = crsp.get_ret(begret, endret, field='retx')
    ports = [(1+retx.reindex(p.index).fillna(0.)).mul(p, axis=0) for p in ports]

show(DataFrame({'Survivors-only Overlapping': [np.mean(out1), np.std(out1)],
                'Overlapping Returns': [np.mean(out2), np.std(out2)],
                'Non-Overlapping Returns': [np.mean(out3), np.std(out3)]},
               index=['mean','std']),
     caption='"Low Price" Strategy: Monthly Spread Returns')

# Plot Spread Returns

nskip = 5
nyear = endyear - begyear + 1
fig, ax = plt.subplots(num=1, clear=True, figsize=(10, 5))
results = pd.concat([Series({str(hold): np.mean(out[-(hold*12):])
                             for hold in range(nyear, 0, -nskip)}, name=name)
                     for name, out in zip(['biased', 'no bias'], [out1, out2])],
                    axis=1)
results.plot(kind='bar', ax=ax, title='"Low-Price" Portfolio Spread Returns',
             xlabel='Portfolios formed within previous t years before 2021',
             ylabel='Monthly Spread Returns')
bx = ax.twinx()
results = pd.concat([Series({str(hold): num[-(hold*12)]
                             for hold in range(nyear, 0, -nskip)}, name=name)
                     for name, num in zip(['biased', 'no bias'], [num1, num2])],
                    axis=1)
results.plot(ax=bx, ylabel='Number of stocks in universe', marker='o', ls=':')
ax.legend(['Survivors-only spread', 'All-stocks spread'], loc='lower left')
bx.legend(['Survivors-only #Stocks', 'All #Stocks'])
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'survivor_spread' + figext))
plt.show()

# ACF of overlapping and non-overlapping portfolio returns
           
import statsmodels.formula.api as smf
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(out2,
                         lags=36,
                         title='ACF of Overlapping Returns')
plt.savefig(os.path.join(imgdir, 'survivor_overlap' + figext))
sm.graphics.tsa.plot_acf(out3,
                         lags=36,
                         title='ACF of Non-Overlapping Returns')
plt.savefig(os.path.join(imgdir, 'survivor_nonoverlap'  + figext))
plt.show()

# Newey-West adjusted t-stats

res = []
keys = ['Survivors-only overlapping',
        'All stocks overlapping',
        'All stocks non-overlapping']
for out, label in zip([out1, out2, out3], keys):
    data = DataFrame(out, columns=['ret'])
    reg = smf.ols('ret ~ 1',data=data).fit()
    a = Series({attr: round(float(getattr(reg, attr)), 6)
                for attr in ['params','bse','tvalues','pvalues']},
               name='uncorrected')
    #print(reg.summary())
    reg = smf.ols('ret ~ 1',data=data)\
             .fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    # coef, stderr, t-value, P>|z|
    b = Series({attr: round(float(getattr(reg, attr)), 6)
                for attr in ['params','bse','tvalues','pvalues']},
               name='NeweyWest')
    res.append(pd.concat([a, b], axis=1))
show(pd.concat(res, axis=1, keys=keys),
     caption='Uncorrected and Newey-West corrected standard errors')     
