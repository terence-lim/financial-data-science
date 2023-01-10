"""Risk Premiums from Fama-Macbeth Cross-sectional Regression

- CAPM tests
- Polynomial regression, feature transformations

Copyright 2023, Terence Lim

MIT License
"""
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas_datareader as pdr
from pandas_datareader.data import DataReader
from pandas_datareader.famafrench import FamaFrenchReader
from tqdm import tqdm
from finds.database import SQL, Redis
from finds.structured import CRSP, Signals, Benchmarks
from finds.busday import BusDay
from finds.backtesting import RiskPremium
from finds.recipes import winsorize, least_squares
from finds.display import show
from conf import credentials, VERBOSE, paths, CRSP_DATE

LAST_DATE = CRSP_DATE
%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None)

sql = SQL(**credentials['sql'], verbose=VERBOSE)
user = SQL(**credentials['user'], verbose=VERBOSE)
rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
signals = Signals(user)
bench = Benchmarks(sql, bd)
imgdir = paths['images']


## Retrieve market and test asset returns
asset_names = ['25_Portfolios_ME_BETA_5x5',
               '25_Portfolios_ME_VAR_5x5',
               '25_Portfolios_ME_RESVAR_5x5',
               '25_Portfolios_5x5']
test_assets = {asset: FamaFrenchReader(asset, start=1900, end=2099).read()
               for asset in asset_names}
#for a in test_assets.values():
#    for p in range(2):
#        a[p].index = a[p].index.strftime('%Y%m').astype(int)

mkt = FamaFrenchReader('F-F_Research_Data_Factors', start=1900, end=2099).read()
mkt = mkt[0].rename(columns={'Mkt-RF': 'BETA'})
#mkt.index = mkt.index.strftime('%Y%m').astype(int)

# Show summary statistics of Mkt-Rf, every 25 years
periods=[(1925, 2024), (1925, 1949), (1950, 1974), (1975, 1999), (2000, 2024)]
df = []
for (beg, end) in periods:
    data = mkt[(mkt.index >= f"{beg}-01") & (mkt.index <= f"{end}-12")]['BETA']
    df.append(DataFrame({'of months': len(data),
                         'annual mean': 12*np.mean(data),
                         'annual stdev': np.std(data)*np.sqrt(12),
                         'sharpe ratio': np.mean(data)*np.sqrt(12)/np.std(data),
                         'skewness': skew(data),
                         'excess kurtosis': kurtosis(data, fisher=True)},
                        index=[f"{str(data.index[0])[:4]}-" + 
                               f"{str(data.index[-1])[:4]}"]))
show(pd.concat(df),
     caption="Summary statistics of Mkt-RF monthly returns", **SHOW)


## Fama-MacBeth regressions with 5x5 test assets and estimated loadings
periods = [mkt.index[0], '1963-07']
out = {period: [] for period in periods}
for period in periods:
    
    for asset in asset_names:
        by_asset = []
    
        for p, wt in enumerate(['Value-weighted', 'Equal-weighted']):
            # subtract riskfree, and stack data as thin dataframe
            f = test_assets[asset][p]
            df = f.sub(mkt['RF'], axis=0).dropna().copy()
            rets = df.stack().reset_index(name='ret')\
                             .rename(columns={'level_1':'port',
                                              'level_0':'Date'})
            data = df.join(mkt[['BETA']], how='left')
            data = data[data.index >= period]

            # estimate test assets market betas from time-series of returns
            betas = least_squares(data,
                                  y=df.columns,
                                  x=['BETA'],
                                  stdres=True)[['BETA', 'stdres']]  
        
            # orthogonalize beta^2 and residual-volatility regressors
            betas['RES'] = smf.ols("stdres ~ BETA", data=betas).fit().resid
            betas['BETA2'] = smf.ols("I(BETA**2) ~ BETA + RES",
                                     data=betas).fit().resid
            r = rets.join(betas, on='port')\
                    .sort_values(['port', 'Date'], ignore_index=True)
        
            # run monthly Fama MacBeth cross-sectional regressions
            fm = r.groupby(by='Date').apply(least_squares,
                                            y=['ret'],
                                            x=['BETA', 'BETA2', 'RES'])

            # time-series means and standard errors of the FM coefficients
            sub = DataFrame({'mean': fm.mean(),
                             'stderr': fm.sem(),
                             'tstat': fm.mean() / fm.sem()}).T
            sub.columns = pd.MultiIndex.from_tuples([(wt, col)
                                                     for col in sub.columns])
            sub.index = pd.MultiIndex.from_tuples([(asset, row)
                                                   for row in sub.index])
            by_asset.append(sub)
        out[period].append(pd.concat(by_asset, axis=1))
p = periods[0]
show(pd.concat(out[p]),
     caption=[None, f"Fama-Macbeth Monthly Cross-sectional Regressions"
              + f" {p} to {df.index[-1]}"], **SHOW)

### Post-1963 time-period
p = periods[1]
show(pd.concat(out[p]),
     caption=(f"Fama-Macbeth Monthly Cross-sectional Regressions"
              + f" {p} to {df.index[-1]}"), **SHOW)


### Compare to robust cov
ls = smf.ols("ret ~ BETA + BETA2 + RES", data=r).fit()
print(ls.summary())
print(ls.get_robustcov_results('HC0').summary())
print(ls.get_robustcov_results('HAC', maxlags=6).summary())
print(ls.get_robustcov_results('hac-panel',
                               groups=r['port'],
                               maxlags=6).summary())
print(ls.get_robustcov_results('cluster', groups=r['port']).summary())



## Fama-MacBeth with individual stock returns and standardized scores
"""
Stock exposures: winsored at 5% tail
- size: -log of market cap, standardized
- hml: book-to-market ratio, standardized 
- mom: 12-month skip past month momentum, standardized
- beta: regression of 3 year's weekly returns (min 1 year), shunk by 1/3 to 1.
"""
rebalbeg=19640601
rebalend=LAST_DATE
rebaldates = crsp.bd.date_range(rebalbeg, rebalend, 'endmo')
loadings = dict()
for pordate in tqdm(rebaldates):           # retrieve signal values every month
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

### Compute coefficients from FM cross-sectional regressions
riskpremium = RiskPremium(user, bench, 'RF', LAST_DATE)
riskpremium(stocks=crsp,        # FM regressions on standardized scores
            loadings=loadings,
            standardize=['value' ,'size', 'momentum'])

### Compare estimated risk premiums to benchmark factors
benchnames = {'beta': 'Mkt-RF(mo)',
              'momentum': 'Mom(mo)',
              'size':'SMB(mo)',
              'value': 'HML(mo)'}
out = riskpremium.fit(benchnames.values())  # to compare portfolio-sorts
riskpremium.plot(benchnames)
plt.savefig(imgdir / 'fm.jpg')

### Summarize time-series means of Fama-Macbeth risk premiums
caption, df = ["Fama-MacBeth Cross-sectional Regression Risk Premiums", out[0]]
df['tvalue'] = df['mean']/df['stderr']
df['sharpe'] = np.sqrt(12) * df['mean']/df['std']
show(df, caption=caption, **SHOW)

### Summarize time-series means of Fama-French portfolio sort returns
caption, df = ["Fama-French Portfolio-Sorts", out[2]]
df['tvalue'] = df['mean']/df['stderr']
df['sharpe'] = np.sqrt(12) * df['mean']/df['std']
show(df, caption=caption, **SHOW)

## Show correlation of returns
df = pd.concat([out[1].join(out[4]), out[4].T.join(out[3])], axis=0)
show(df, caption='Correlation of Fama-MacBeth Risk Premiums'
     + ' and Fama-French Portfolio-Sort Returns', **SHOW)
