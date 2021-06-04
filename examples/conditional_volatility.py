"""Conditional Volatility Models

- VaR, halflife, GARCH, EWMA, Scholes-Williams Beta
- VIX, Bitcoin, St Louis Fed FRED

Terence Lim
License: MIT
"""
import os
import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns 
from finds.alfred import Alfred

from settings import settings
imgdir = os.path.join(settings['images'], 'ts')
alf = Alfred(api_key=settings['fred']['api_key'])

# proportion of failures likelihood test
def kupiecLR(s, n, var):
    """Kupiec LR test (S violations in N trials) of VaR"""
    p = 1 - var        # e.g. var95 is 0.95
    t = n - s          # number of non-violations
    num = np.log(1 - p)*(n - s) + np.log(p)*s
    den = np.log(1 - (s/n))*(n - s) + np.log(s/n)*s
    lr = -2 * (num - den)
    return {'lr': lr, 'pvalue': 1 - scipy.stats.chi2.cdf(lr, df=1)}

def pof(X, pred, var=0.95):
    """Kupiec proportion of failures VaR test"""
    Z = X / pred
    z = scipy.stats.norm.ppf(1 - var)
    r = {'n': len(Z), 's': np.sum(Z < z)}
    r.update(kupiecLR(r['s'], r['n'], var))
    return r

# convert alpha to halflife
from pandas.api import types
def halflife(alpha):
    """Returns halflife from alpha = -ln(2)/ln(lambda), where lambda=1-alpha"""
    if types.is_list_like(alpha):
        return [halflife(a) for a in alpha]
    return -np.log(2)/np.log(1-alpha) if 0<alpha<1 else [np.inf,0][int(alpha>0)]
                                      
# Retrive Bitcoin from FRED and plot EWMA and Daily Returns
z = scipy.stats.norm.ppf(0.05)
alpha = [0.03, 0.06]
series_id = 'CBBTCUSD'
X = alf(series_id, log=1, diff=1)[126:]
X.index = pd.DatetimeIndex(X.index.astype(str), freq='infer')
Y = np.square(X)
ewma = [np.sqrt((Y.ewm(alpha=a).mean()).rename(series_id)) for a in alpha]
fig, ax = plt.subplots(num=1, clear=True, figsize=(10,6))
ax.plot(X.shift(-1), ls='-', lw=.5, c='grey')
ax.plot(z * ewma[0], lw=1, ls='-.', c='b')
ax.plot(z * ewma[1], lw=1, ls='--', c='r')
ax.set_title(alf.header(series_id))
ax.set_ylabel('Daily Returns and EWMA Volatility')
ax.legend([series_id] + [f"$\lambda$ = {1-a:.2f}" for a in alpha])
ax.plot(-z * ewma[0], lw=1, ls='-.', c='b')
ax.plot(-z * ewma[1], lw=1, ls='--', c='r')
plt.savefig(os.path.join(imgdir, 'ewma.jpg'))
plt.show()

# Retrieve SP500 and VIX data, compute EWMA
sp500 = alf('SP500', log=1, diff=1).dropna()
vix = alf('VIXCLS')
ewma = np.sqrt((np.square(sp500).ewm(alpha=0.05).mean()).rename('EWMA(0.94)'))
mkt = pd.concat([sp500, ewma, (vix/100)/np.sqrt(252)], axis=1, join='inner')
mkt.index = pd.DatetimeIndex(mkt.index.astype(str), freq='infer')
mkt

# GARCH(1,1) using rugarch
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from finds.pyR import PyR
rugarch_ro = importr('rugarch')  # to use library rugarch
c_ = ro.r['c']
list_ = ro.r['list']
spec = ro.r['ugarchspec'](mean_model=list_(armaOrder=c_(2, 0),
                                           include_mean=False))
model = ro.r['ugarchfit'](spec, data=PyR(mkt['SP500'].values).ro) 
ro.r['show'](model)
for which in [4, 5, 10, 11]:
    ro.r['plot'](model, which=which)
    PyR.savefig(os.path.join(imgdir, f'ugarch{which}.png'))


# Plot all, but for illustration only: the 3 "forecasts" are not comparable
# VIX is implied from 3-month options, GARCH full in-sample, EWMA is rolling avg
mkt['GARCH(1,1)'] = PyR(ro.r['sigma'](model)).values  # fitted volatility values
var = 0.95   # VaR95
z = scipy.stats.norm.ppf(1 - var)
fig, ax = plt.subplots(num=1, clear=True, figsize=(10,7))
ax.plot(mkt['SP500'], ls='-', lw=.5, c='grey')
ax.plot(z * mkt.iloc[:,1], lw=1, ls='-.', c='blue')
ax.plot(z * mkt.iloc[:,2], lw=1, ls='--', c='green')
ax.plot(z * mkt.iloc[:,3], lw=1, ls=':', c='red')
ax.set_title('SP500')
ax.set_ylabel('Daily Returns and VaR')
ax.legend(mkt.columns)
plt.savefig(os.path.join(imgdir, 'var.jpg'))
plt.show()


# get all daily series of financial price categories from FRED
categories = {}
for category in [32255, 33913, 94]:
    c = alf.get_category(category)
    print(category, c['id'], c['name'])
    series = Series({s['id']: s['title'] for s in c['series']
                     if s['frequency'].startswith('Daily') and
                     'DISCONT' not in s['title']})
    categories.update({c['name']: series})
c = pd.concat(list(categories.values())).to_frame()
pd.set_option("max_colwidth", 80)
pd.set_option("max_rows", 100)
c


# Fit ewma and backtest VaR
alphas = 1 - np.linspace(1, 0.91, 10)
results = {'pof': DataFrame(), 'n': DataFrame(), 'end': DataFrame(),
           's': DataFrame(), 'pvalue': DataFrame()}  # to collect results
for category, series in categories.items():
    for series_id in series.index:
        X = alf(series_id, log=1, diff=1, freq='D').dropna()
        if X is None:
            print(f'*** oops {series_id} ***')
            assert(X)
        Y = np.square(X)
        results[series_id] = {}
        for i, alpha in enumerate(alphas):
            ewma = np.sqrt((Y.ewm(alpha=alpha) if alpha>0 else Y.expanding())\
                           .mean()).rename(i)
            r = pof(X[126:], ewma[126:], var=0.95)
            results['pof'].loc[series_id, alpha] = r['s']/r['n']
            results['pvalue'].loc[series_id, alpha] = r['pvalue']
            results['n'].loc[series_id, alpha] = r['n']
            results['s'].loc[series_id, alpha] = r['s']
            results['end'].loc[series_id, alpha] = alf.header(
                series_id, 'observation_end')
            #print(results['pof'].loc[series_id, [alpha]])
print(Series({k: len(v) for k, v in categories.items()}, name='series'))
pd.concat([Series(index=alphas, data=1-alphas, name='lambda').round(2),
           Series(index=alphas, data=halflife(alphas),name='halflife').round(1),
           results['pof'].median(axis=0).rename('pof').round(6)], axis=1)

# Plot distribution of Proportion of Failures
fig, ax = plt.subplots(num=1, clear=True, figsize=(10,6))
results['pof'].boxplot(ax=ax, grid=False)
ax.set_xticklabels([f"{1-c:.3}" for c in results['pof'].columns])
ax.set_xlabel('$\lambda$ smoothing parameter')
ax.set_title('VaR Proportion of Failures Box-Plot')
ax.set_ylabel('Proportion of Failures')
ax.axhline(1 - var, linestyle=':', color='g')
for x, c in enumerate(results['pof'].columns):
    for arg in [results['pof'][c].argmin(), results['pof'][c].argmax()]:
        
        y = results['pof'][c].iloc[[arg]]
        ax.annotate(y.index[0], xy=(x + 1, y[0]-.003), fontsize=8, c='m',
                    rotation=45)
plt.savefig(os.path.join(imgdir, 'pof.jpg'))
plt.show()
