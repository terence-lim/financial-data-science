"""Risk Premiums from Fama-Macbeth Cross-sectional Regression

- pandas datareader, Fama French data library

Terence Lim
License: MIT
"""
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
from finds.solve import winsorized
from settings import settings
LAST_DATE = settings['crsp_date']
sql = SQL(**settings['sql'])
user = SQL(**settings['user'])
rdb = Redis(**settings['redis'])
bd = BusDay(sql)
crsp = CRSP(sql, bd, rdb)
bench = Benchmarks(sql, bd)
signals = Signals(user)
logdir = os.path.join(settings['images'], 'fm')

def least_squares(data=None, y=['y'], x=['x'], stdres=False):
    """Helper to compute least square coefs, supports groupby().apply"""
    X = data[x].to_numpy()
    Y = data[y].to_numpy()
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    x = ['Intercept'] + x
    b = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y)).T
    if stdres:
        b = np.hstack([b, np.std(Y-(X @ b.T), axis=0).reshape(-1,1)])
        x = x + ['stdres']
    return (DataFrame(b, columns=x, index=y) if len(b) > 1 else
            Series(b[0], x))   # return as Series for groupby.apply


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
        f = test_assets[asset][p]
        
        # subtract riskfree and stack data as thin dataframe
        df = f.sub(mkt['RF'], axis=0).dropna().copy()
        rets = df.stack().reset_index(name='ret')\
                         .rename(columns={'level_1':'port', 'level_0':'Date'})

        # estimate test assets market betas from time-series of returns
        data = df.join(mkt[['BETA']], how='left')
        betas = least_squares(data, y=df.columns, x=['BETA'],
                         stdres=True)[['BETA', 'stdres']]  # emits stdres too
        
        # orthogonalize beta^2 and residual-volatility regressors
        betas['BETA2'] = smf.ols("I(BETA**2) ~ BETA", data=betas).fit().resid
        betas['RES'] = smf.ols("stdres ~ BETA", data=betas).fit().resid
        r = rets.join(betas, on='port').sort_values(['port',     # join returns
                                                     'Date'], ignore_index=True)

        # run monthly Fama MacBeth cross-sectional regressions
        print(asset, wt, r['Date'].iloc[0], '-', r['Date'].iloc[-1])
        fm = r.groupby(by='Date')\
              .apply(least_squares, y=['ret'], x=['BETA', 'BETA2', 'RES'])

        # time-series means and standard errors of the FM coefficients
        print(DataFrame({'mean': fm.mean(), 'stderr': fm.sem(),
                         'tvalue': fm.mean() / fm.sem()}).T.round(4))

## Post-1963 time-period
data = data[data.index >= '1963-07']
betas = least_squares(data, y=df.columns, x=['BETA'],
                 stdres=True)[['BETA', 'stdres']]
betas['BETA2'] = smf.ols("I(BETA**2) ~ BETA", data=betas).fit().resid
betas['RES'] = smf.ols("stdres ~ BETA", data=betas).fit().resid
r = rets[rets['Date'] >= '1963-07']\
    .join(betas, on='port').sort_values(['port', 'Date'], ignore_index=True)
print(asset, wt, r['Date'].iloc[0], '-', r['Date'].iloc[-1])
fm = r.groupby(by='Date')\
      .apply(least_squares, y=['ret'], x=['BETA','BETA2','RES'])
print(DataFrame({'mean': fm.mean(), 'stderr': fm.sem(),
                 'tvalue': fm.mean() / fm.sem()}).T.round(4))

## Compare to robust cov
ls = smf.ols("ret ~ BETA + BETA2 + RES", data=rets).fit()
print(ls.summary())
print(ls.get_robustcov_results('HC0').summary())
print(ls.get_robustcov_results('HAC', maxlags=3).summary())
print(ls.get_robustcov_results('hac-panel', groups=rets['port'],
                               maxlags=3).summary())
print(ls.get_robustcov_results('cluster', groups=rets['port']).summary())


## Fama MacBeth with individual stocks and standardized scores as loadings
rebalbeg=19640601
rebalend=LAST_DATE
rebaldates = crsp.bd.date_range(rebalbeg, rebalend, 'endmo')
loadings = dict()
for pordate in rebaldates:             # retrieve signal values every month
    date = bd.june_universe(pordate)
    univ = crsp.get_universe(date)
    cap = np.sqrt(crsp.get_cap(date)['cap'])
    smb = -np.log(cap).rename('size')
    hml = signals('hml', date, bd.endmo(date, -12))['hml'].rename('value')
    beta = (signals('beta', pordate, bd.begmo(pordate))['beta']*2/3)+(1/3)
    mom = signals('mom', pordate)['mom'].rename('momentum')
    df = pd.concat((beta, hml, smb, mom),  # inner join of signals with univ
                   join='inner', axis=1).reindex(univ.index).dropna()
    loadings[pordate] = winsorized(df, quantiles=[0.05, 0.95])

## Compute coefficients from FM cross-sectional regressions
riskpremium = RiskPremium(user, bench, 'RF', LAST_DATE)
riskpremium(crsp, loadings,             # FM regressions on standardized scores
            weights=None, standardize=['value' ,'size', 'momentum'])
benchnames = {'beta': 'Mkt-RF(mo)', 'momentum': 'Mom(mo)',
              'size':'SMB(mo)', 'value': 'HML(mo)'}
out = riskpremium.fit(benchnames.values())  # compare portfolio-sort benchmarks
riskpremium.plot(benchnames)
plt.savefig(os.path.join(logdir, 'fm.jpg'))

# Summarize time-series means
df = out[3].join(out[0])
df.loc['tvalue'] = df.loc['mean']/df.loc['stderr']
df.loc['sharpe'] = np.sqrt(12) * df.loc['mean']/df.loc['std']
print(f'Fama MacBeth Estimated Factor and Benchmark Returns'
      f' {riskpremium.perf.index[0]}')
df
print(df.to_latex())

## Display correlation of returns
print('Correlation of Fama MacBeth Estimated Factor and Benchmark Returns')
df = out[4].join(out[2])
print(df.to_latex())





