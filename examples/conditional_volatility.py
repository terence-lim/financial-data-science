"""Conditional Volatility

- Value at Risk, Expected Shortfall
- GARCH, EWMA
- VIX, Bitcoin

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

import os
import numpy as np
from scipy.stats import chi2, norm, t, jarque_bera, kurtosis, skew
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns 
from finds.alfred import Alfred
from finds.recipes import kupiecLR, pof, halflife, \
    value_at_risk_historical, expected_shortfall_historical, \
    value_at_risk_normal, expected_shortfall_normal
from conf import VERBOSE, credentials, paths
imgdir = os.path.join(paths['images'], 'ts')
alf = Alfred(api_key=credentials['fred']['api_key'])

# Histogram, Historical VaR and ES of SP500, REITS, Bitcoin, Dollar
labels = ['SP500', 'WILLREITIND', 'CBBTCUSD', 'DTWEXAFEGS']
alpha = 0.95   # VaR parameter
out = {}
fig, axes = plt.subplots(2, 2, num=1, clear=True, figsize=(10, 5))
for label, ax in zip(labels, list(axes.ravel())):
    rets = alf(label, log=1, diff=1).dropna()
    sigma = rets.std()
    var = value_at_risk_historical(rets, alpha=0.95)
    es = expected_shortfall_historical(rets, alpha=0.95)
    var_normal = value_at_risk_normal(rets, alpha)
    es_normal = expected_shortfall_normal(rets, alpha)
    
    n, bins, _ = ax.hist(rets[rets < var], color='red', bins=30)
    stepsize = (bins[-1] - bins[0])/(len(bins)-1)
    bins = np.arange(var, max(rets) + stepsize, stepsize)
    ax.hist(rets[rets >= var], color='green', bins=bins)

    ax.axvline(var, color='blue', ls='--')
    ax.axvline(es, color='blue', ls=':')
    ax.axvline(var_normal, color='cyan', ls='--')
    ax.axvline(es_normal, color='cyan', ls=':')
    ax.legend([f'VaR actual = {var:.4f}',
               f'ES actual = {es:.4f}',
               f'VaR (normal={var_normal:.4f})',
               f'ES (normal={es_normal:.4f})',
               'tail < VaR',
               label + f' (std={sigma:.4f})'], fontsize='x-small')
    ax.set_title(f"{label} {alf.header(label)[:60]}",
                 {'fontsize' : 'small'})
    ax.set_ylabel(f'{min(rets.index)}-{max(rets.index)}',
                  fontsize='small')
    ax.set_xlabel(f'VaR and ES (alpha={alpha*100:.0f})',
                  fontsize='small')
    out[label] = {'std dev': np.std(rets, ddof=1),
                  'skewness': skew(rets),
                  'excess kurtosis': kurtosis(rets)-3,
                  'jarque-bera pvalue': jarque_bera(rets)[1]}
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'es' + figext))
plt.show()

# Non-normal: Jacque-Bera, excess kurtosis, mixtures
show(DataFrame(out).T)


# GARCH(1,1) using rugarch
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from finds.pyR import PyR
def rugarch(rets: Series, savefig: str = '') -> np.array:
    rugarch_ro = importr('rugarch')  # to use library rugarch
    c_ = ro.r['c']
    list_ = ro.r['list']
    spec = ro.r['ugarchspec'](mean_model=list_(armaOrder=c_(0, 0),
                                               include_mean=False))
    model = ro.r['ugarchfit'](spec, data=PyR(rets.values).ro) 
    ro.r['show'](model)

    if savefig:
        for which in [4, 5, 10, 11]:
            ro.r['plot'](model, which=which)
            PyR.savefig(os.path.join(imgdir, f'{savefig}{which}.png'))
    return PyR(ro.r['sigma'](model)).values  # fitted volatility values


# In-sample EWMA conditional volatilty model
def best_ewma(Y):
    best_stat = np.inf  # conditional nll of Gaussian
    for l in np.arange(0.4, 1, 0.01):
        pred = Y.ewm(alpha=1-l).mean()
        stat = jarque_bera(X.shift(-1).div(np.sqrt(pred)).dropna())[0]
        #stat = abs(kurtosis(X.shift(-1).div(np.sqrt(pred)).dropna() - 3))
        #stat = np.log(pred).sum() + np.square(X.shift(-1)).div(pred).sum() / 2
        #stat = (((X.shift(-1)**2).sub(pred))**2).mean()
        #stat = X.shift(-1).abs().sub(np.sqrt(pred)).abs().mean()
        #print(stat, l)
        if stat < best_stat:
            lam = l
            best_stat = stat
    print(lam, best_stat) # np.log(2 * np.pi) * len(pred) / 2)
    return lam

# Retrive Bitcoin from FRED and plot EWMA and Daily Returns
series_id = 'CBBTCUSD'
X = alf(series_id, log=1, diff=1).dropna()
X.index = pd.DatetimeIndex(X.index.astype(str), freq='infer')
Y = np.square(X)

lam = best_ewma(Y)
asset = np.sqrt((Y.ewm(alpha=1 - lam).mean()).rename('EWMA')).to_frame()
asset['EWMA(0.94)'] = np.sqrt((Y.ewm(alpha=1 - 0.94).mean()))
asset['GARCH'] = rugarch(X)

# Conditional Volatility Models EWMA(94), GARCH(1, 1): Bitcoin
fig, ax = plt.subplots(num=1, clear=True, figsize=(10, 5))
ax.plot(X.shift(-1), ls='', marker='.', markersize=2)
ax.plot(norm.ppf(1 - alpha) * asset['EWMA'], lw=1, ls='-', c='r')
ax.plot(norm.ppf(1 - alpha) * asset['GARCH'], lw=1, ls='--', c='m')
ax.set_title(alf.header(series_id)
             + f" ({max(Y.index).strftime('%Y-%m-%d')}:"
             + f"{min(Y.index).strftime('%Y-%m-%d')})")
ax.set_ylabel('Daily Returns and Conditional Volatility')
ax.legend([series_id] + [f"EWMA($\lambda$={lam:.2f})"] + ['GARCH(1,1)'])
ax.plot(-norm.ppf(1 - alpha) * asset['EWMA'], lw=1, ls='-', c='r')
ax.plot(-norm.ppf(1 - alpha) * asset['GARCH'], lw=1, ls='--', c='m')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'ewma' + figext))
plt.show()

y = X.shift(-1)
results = {}
for label, x in zip([series_id, 'EWMA(0.94)', f'EWMA*({lam:.2f})','GARCH(1,1)'],
                    [1.0, asset['EWMA(0.94)'], asset['EWMA'], asset['GARCH']]):
    if isinstance(x, (int, float)):
        pof_ = pof((y / np.std(y))[126:], x)  # skip first six months
    else: 
        pof_ = pof(y[126:], x[126:])  # skip first six months
    results[label] = {'std dev': np.std(y.div(x).dropna()),
                      'skewness': skew(y.div(x).dropna()),
                      'excess kurtosis': kurtosis(y.div(x).dropna()) - 3,
                      f'pof({int(100*alpha)})': pof_['s']/pof_['n'],
                      'pof p-value': pof_['pvalue']}
show(DataFrame.from_dict(results, orient='index'), caption=series_id)



# Conditional Volatility Models SP500: EWMA, GARCH(1, 1), VIX
series_id = 'SP500'
X = alf(series_id, log=1, diff=1).dropna()
Y = np.square(X)
asset = X.to_frame()
vix = alf('VIXCLS')
asset = asset.join(vix/(100 * np.sqrt(252)))

lam = best_ewma(Y)
asset['EWMA'] = np.sqrt(Y.ewm(alpha=1 - lam).mean())
asset['EWMA(0.94)'] = np.sqrt((Y.ewm(alpha=1 - 0.94).mean()))
asset['GARCH'] = rugarch(X)

asset.index = pd.DatetimeIndex(asset.index.astype(str), freq='infer')
X = asset[series_id]

asset

# Plot VaR of SP500: VIX, GARCH
fig, ax = plt.subplots(num=1, clear=True, figsize=(10, 5))
ax.plot(asset['SP500'], ls='', marker='.', markersize=2)
for c, col in zip(['r', 'g'],['VIXCLS', 'GARCH']):
    ax.plot(norm.ppf(1-alpha) * asset.loc[:, col], lw=1, ls='-', c=c)
ax.set_title('SP500 Daily Returns, VIX, and GARCH VaR')
ax.legend(['SP500', 'VIXCLS', 'GARCH'])
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'var' + figext))
plt.show()

y = X.shift(-1)
results = {}
for label, x in zip([series_id, 'EWMA(0.94)', f'EWMA*({lam:.2f})',
                     'GARCH(1,1)', 'VIXCLS'],
                    [1.0, asset['EWMA(0.94)'], asset['EWMA'],
                     asset['GARCH'], asset['VIXCLS']]):
    if isinstance(x, (int, float)):
        pof_ = pof((y / np.std(y))[126:], x)  # skip first six months
    else: 
        pof_ = pof(y[126:], x[126:])  # skip first six months
    results[label] = {'std dev': np.std(y.div(x).dropna()),
                      'skewness': skew(y.div(x).dropna()),
                      'excess kurtosis': kurtosis(y.div(x).dropna()) - 3,
                      f'pof({int(100*alpha)})': pof_['s']/pof_['n'],
                      'pof p-value': pof_['pvalue']}
show(DataFrame.from_dict(results, orient='index'), caption=series_id)


r = asset['SP500'][1:].values / asset.iloc[:-1, -1].values
print('excess kurtosis', kurtosis(r)-3,
      'jarque-bera pvalue', jarque_bera(r)[1])




# pof tests for EWMA(0.94 and 0.97) for long stocks series 

## get all daily series of financial price categories from FRED
categories = {}
for category in [32255, 33913, 94]:
    c = alf.get_category(category)
    print(category, c['id'], c['name'])
    series = Series({s['id']: s['title'] for s in c['series']
                     if s['frequency'].startswith('Daily')
                     and 'DISCONT' not in s['title']
                     and 'Price' not in s['title']})
    categories.update({c['name']: series})
c = pd.concat(list(categories.values())).to_frame()
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 100)
c

wilshires = ['WILL5000IND', 'WILLLRGCAP', 'WILLLRGCAPGR', 'WILLLRGCAPVAL',
             'WILLMIDCAP', 'WILLMIDCAPGR', 'WILLMIDCAPVAL', 
             'WILLSMLCAP', 'WILLSMLCAPGR', 'WILLSMLCAPVAL']
r = pd.concat([alf(series_id,
                   log=1,
                   diff=1,
                   freq='D') for series_id in wilshires],
              join='inner', axis=1).dropna()

alphas = np.array([0.15, 0.12, 0.09, 0.06, 0.03, 0.02, 0.01])
results = {'pof': DataFrame(), 'n': DataFrame(), 'end': DataFrame(),
           's': DataFrame(), 'pvalue': DataFrame()}  # to collect results
for series_id in r.columns:
    X = r[series_id]
    Y = np.square(X)
    results[series_id] = {}
    for i, alpha in enumerate(alphas):
        ewma = np.sqrt((Y.ewm(alpha=alpha)).mean()).rename(i)
        x = pof(X[126:], ewma[125:-1], var=0.95)
        results['pof'].loc[series_id, alpha] = x['s']/x['n']
        results['pvalue'].loc[series_id, alpha] = x['pvalue']
        results['n'].loc[series_id, alpha] = x['n']
        results['s'].loc[series_id, alpha] = x['s']
        results['end'].loc[series_id, alpha] = alf.header(series_id,
                                                          'observation_end')
            #print(results['pof'].loc[series_id, [alpha]])

print(Series({k: len(v) for k, v in categories.items()}, name='series'))
stat = 'pof'
pd.concat([Series(index=alphas,
                  data=1 - alphas,
                  name='lambda').round(2),
           Series(index=alphas,
                  data=halflife(alphas),
                  name='halflife').round(1),
           results[stat].median(axis=0).rename(stat).round(6)], axis=1)

round(results['pof'], 4)  # resist pvalues, show pof
