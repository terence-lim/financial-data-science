"""Risk Premiums from Fama-Macbeth Cross-sectional Regression

- Fama-Macbeth cross-sectional regression: risk premiums
- CRSP, Compustat, 
- Ken French Data Library: Fama-French test assets

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas_datareader as pdr
from pandas_datareader.data import DataReader
from pandas_datareader.famafrench import FamaFrenchReader
from finds.database import SQL, Redis
from finds.structured import CRSP, Signals, Benchmarks
from finds.busday import BusDay
from finds.backtesting import RiskPremium
from finds.recipes import winsorize, least_squares
from conf import credentials, VERBOSE, paths, CRSP_DATE

LAST_DATE = 20201231
VERBOSE = 1
sql = SQL(**credentials['sql'], verbose=VERBOSE)
user = SQL(**credentials['user'], verbose=VERBOSE)
rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
signals = Signals(user)
bench = Benchmarks(sql, bd)
imgdir = paths['images']


## Retrieve market and test asset returns
mkt = FamaFrenchReader('F-F_Research_Data_Factors', start=1900, end=2099).read()
mkt = mkt[0].rename(columns={'Mkt-RF': 'BETA'})
#mkt.index = mkt.index.strftime('%Y%m').astype(int)

asset_names = ['25_Portfolios_ME_BETA_5x5',
               '25_Portfolios_ME_VAR_5x5',
               '25_Portfolios_ME_RESVAR_5x5',
               '25_Portfolios_5x5']
test_assets = {asset: FamaFrenchReader(asset, start=1900, end=2099).read()
               for asset in asset_names}
#for a in test_assets.values():
#    for p in range(2):
#        a[p].index = a[p].index.strftime('%Y%m').astype(int)

## Fama-MacBeth regressions with 5x5 test assets and estimated loadings        
for asset in asset_names:
    for p, wt in enumerate(['Value-weighted', 'Equal-weighted']):
#        asset = asset_names[0]
#        p, wt = 0, 'Value-weighted'
        
        f = test_assets[asset][p]
        
        # subtract riskfree, and stack data as thin dataframe
        df = f.sub(mkt['RF'], axis=0).dropna().copy()
        rets = df.stack().reset_index(name='ret')\
                         .rename(columns={'level_1':'port', 'level_0':'Date'})

        # estimate test assets market betas from time-series of returns
        data = df.join(mkt[['BETA']], how='left')
        betas = least_squares(data,
                              y=df.columns,
                              x=['BETA'],
                              stdres=True)[['BETA', 'stdres']]  
        
        # orthogonalize beta^2 and residual-volatility regressors
        betas['BETA2'] = smf.ols("I(BETA**2) ~ BETA", data=betas).fit().resid
        betas['RES'] = smf.ols("stdres ~ BETA", data=betas).fit().resid
        r = rets.join(betas, on='port')\
                .sort_values(['port', 'Date'], ignore_index=True)
        
        # run monthly Fama MacBeth cross-sectional regressions
        print(f"\caption{{{asset.replace('_','-')} "
              f"{r['Date'].iloc[0]}:{r['Date'].iloc[-1]}}}")
        fm = r.groupby(by='Date').apply(least_squares,
                                        y=['ret'],
                                        x=['BETA', 'BETA2', 'RES'])

        # time-series means and standard errors of the FM coefficients
        show(DataFrame({'mean': fm.mean(),
                        'stderr': fm.sem(),
                        'tvalue': fm.mean() / fm.sem()}).T.round(4),
             caption=wt)

## Post-1963 time-period
data = data[data.index >= '1963-07']
betas = least_squares(data,
                      y=df.columns,
                      x=['BETA'],
                      stdres=True)[['BETA', 'stdres']]
betas['BETA2'] = smf.ols("I(BETA**2) ~ BETA", data=betas).fit().resid
betas['RES'] = smf.ols("stdres ~ BETA", data=betas).fit().resid
r = rets[rets['Date'] >= '1963-07']\
    .join(betas, on='port')\
    .sort_values(['port', 'Date'], ignore_index=True)
print(f"\caption{{{asset.replace('_','-')} "
      f"{r['Date'].iloc[0]}:{r['Date'].iloc[-1]}}}")
fm = r.groupby(by='Date').apply(least_squares,
                                y=['ret'],
                                x=['BETA','BETA2','RES'])
show(DataFrame({'mean': fm.mean(),
                'stderr': fm.sem(),
                'tvalue': fm.mean() / fm.sem()}).T.round(4))

## Compare to robust cov
ls = smf.ols("ret ~ BETA + BETA2 + RES", data=r).fit()
print(ls.summary())
print(ls.get_robustcov_results('HC0').summary())
print(ls.get_robustcov_results('HAC', maxlags=3).summary())
print(ls.get_robustcov_results('hac-panel',
                               groups=r['port'],
                               maxlags=3).summary())
print(ls.get_robustcov_results('cluster', groups=r['port']).summary())



## Fama MacBeth with individual stocks and standardized scores

rebalbeg=19640601
rebalend=LAST_DATE
rebaldates = crsp.bd.date_range(rebalbeg, rebalend, 'endmo')
loadings = dict()
for pordate in rebaldates:             # retrieve signal values every month
    date = bd.june_universe(pordate)
    univ = crsp.get_universe(date)
    cap = np.sqrt(crsp.get_cap(date))
    smb = -np.log(cap).rename('size')
    hml = signals('hml', date, bd.endmo(date, -12))['hml'].rename('value')
    beta = signals('beta', pordate, bd.begmo(pordate))['beta']*2/3 + 1/3 #shrink
    mom = signals('mom', pordate)['mom'].rename('momentum')
    df = pd.concat((beta, hml, smb, mom),  # inner join of signals with univ
                   join='inner',
                   axis=1).reindex(univ.index).dropna()
    loadings[pordate] = winsorize(df, quantiles=[0.05, 0.95])

## Compute coefficients from FM cross-sectional regressions
riskpremium = RiskPremium(user, bench, 'RF', LAST_DATE)
riskpremium(stocks=crsp,        # FM regressions on standardized scores
            loadings=loadings,
            standardize=['value' ,'size', 'momentum'])
benchnames = {'beta': 'Mkt-RF(mo)',
              'momentum': 'Mom(mo)',
              'size':'SMB(mo)',
              'value': 'HML(mo)'}
out = riskpremium.fit(benchnames.values())  # to compare portfolio-sorts
riskpremium.plot(benchnames)
plt.savefig(os.path.join(imgdir, 'fm' + figext))

# Summarize time-series means
for caption, df in zip(["Fama-MacBeth Risk Factors",
                        "Fama-French Portfolio-Sorts"],
                       [out[0], out[2]]):
    df['tvalue'] = df['mean']/df['stderr']
    df['sharpe'] = np.sqrt(12) * df['mean']/df['std']
    show(df, caption=caption)

## Display correlation of returns
df = out[1].join(out[4])
show(df, caption='Correlation between Fama-MacBeth Risk Factor and '
     + 'Fama-French Portfolio-Sort Monthly Returns')
