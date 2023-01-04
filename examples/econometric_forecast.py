"""Econometrics and Forecasting

- Trends: seasonality
- Autocorrelation Function: AR, MA, SARIMAX
- Unit root: integration order
- Forecasting: single-step, multi-step
- Granger casuality, impulse response function

Copyright 2023, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns 
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error        
from finds.alfred import Alfred
from finds.busday import to_datetime
from finds.recipes import integration_order
from finds.display import show
from conf import paths, VERBOSE, credentials

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None

imgdir = paths['images'] / 'ts'
alf = Alfred(api_key=credentials['fred']['api_key'])

series_id, freq, start = 'CPIAUCNS', 'M', 0 #19620101  # not seasonally adjusted

df = alf(series_id, log=1, freq=freq, start=start).dropna()
df.index = to_datetime(df.index)

# Seasonality

## Seasonality Decomposition Plot
result = seasonal_decompose(df, model = 'add')
fig, ax = plt.subplots(nrows=4, ncols=1, clear=True, figsize=(10, 12))
result.observed.plot(ax=ax[0], title=alf.header(result.observed.name),
                     ylabel=result.observed.name, xlabel='', c='b')
result.trend.plot(ax=ax[1], ylabel='Trend', xlabel='', c='r')
result.seasonal.plot(ax=ax[2], ylabel='Seasonal', xlabel='', c='g')
result.resid.plot(ax=ax[3], ls='', ms=3, marker='.', c='m',
                  ylabel='Residual', xlabel='')
plt.tight_layout()
plt.savefig(imgdir / 'seasonal.jpg')

# Autocorrelation Function - Plot ACF and PACF

values = df.diff().dropna().values.squeeze()
fig, axes = plt.subplots(1, 2, clear=True, figsize=(10,5))
plot_acf(values, lags=35, ax=axes[0])
plot_pacf(values, lags=35, ax=axes[1], method='ywm')
plt.tight_layout(pad=2)
plt.savefig(imgdir / 'acf.jpg')

# Stationarity and Unit Root

## Integration Order: Log CPI (Seasonally Adjusted)

series_id = 'CPIAUCSL' #19620101  # seasonally adjusted
#series_id = 'GDPC1'
df = alf(series_id, log=1, start=0).dropna()
df.index = to_datetime(df.index)
p = integration_order(df, noprint=False, pvalue=0.05)
Series({series_id: p}, name='I(p)').to_frame()

## Histogram Plot and Kernel Density Estimate

fig, axes = plt.subplots(1, 2, clear=True, figsize=(10,5))
sns.histplot(df.dropna(),
             bins=30,
             lw=0,
             kde=True,
             #line_kws={"color": "r"},
             ax=axes[0])
axes[0].set_title(f"Density Plot of log({series_id})")
sns.histplot(df.diff().dropna().rename(f"diff log {series_id}"),
             bins=30,
             lw=0,
             kde=True,
             #line_kws={"color": "r"},
             ax=axes[1])
axes[1].set_title(f"Density Plot of diff log({series_id})")
plt.savefig(imgdir / 'order.jpg')
plt.tight_layout(pad=3)


# AR and SARIMAX
## AR(p) is simplest time-model, can nest in SARIMAX(p,d,q,s) with
## integration order I(d), moving average MA(q), seasonality S(s), exogenous X

series_id, freq, start = 'CPIAUCNS', 'M', 0  # not seasonally adjust
log_df = alf(series_id, log=1)
log_df.index = to_datetime(log_df.index)
log_df = log_df.loc[:split_date].dropna()

pdq = (1, 1, 3)   #(12, 1, 0)
seasonal_pdqs = (0, 0, 0, 12)
arima = SARIMAX(log_df,
                order=pdq,
                seasonal_order=seasonal_pdqs,
                trend='c').fit()
fig = arima.plot_diagnostics(figsize=(10,6), lags=36)
plt.tight_layout(pad=2)
plt.savefig(imgdir / 'ar.jpg')
arima.summary()


# Forecasting

series_id, start = 'CPIAUCSL', 0
df = alf(series_id, log=1, diff=1, start=start).dropna()
df.index = to_datetime(df.index)
split_date = '2021-06-30'
df_train = df[df.index <= split_date]
df_test = df[df.index > split_date]

## Select AR lag order
"""ARMA select order is too slow and unstable in statsmodels
# Select ARMA lag order
from statsmodels.tsa.stattools import arma_order_select_ic
series_id = 'CPIAUCSL'
df = alf(series_id, log=1, diff=1).dropna()
df.index = to_datetime(df.index)
split_date = '2021-06-30'
df_train = df[df.index <= split_date]
df_test = df[df.index > split_date]
res = arma_order_select_ic(df_train,
                           max_ar=36,
                           max_ma=12,
                           ic='aic',
                           trend='n')
print('(p, q) = ', res.aic_min_order)
"""

lags = ar_select_order(df_train,
                       maxlag=36,
                       ic='bic',
                       old_names=False).ar_lags
print('(BIC) lags= ', len(lags), ':', lags)



## One-step ahead predictions    
model = AutoReg(df_train, lags=lags, old_names=False).fit()  
print(model.summary())

### Observations to predict are from the test split
all_dates = AutoReg(df, lags=lags, old_names=False)

### Use model params from train split, start predictions from last train row    
df_pred = all_dates.predict(model.params,
                            start=df_train.index[-1]).shift(1).iloc[1:]
mse = mean_squared_error(df_test, df_pred)
var = np.mean(np.square(df_test - df_train.mean()))
print(f"ST Forecast({len(df_pred)}): rmse={np.sqrt(mse)} r2={1-mse/var}")

fig, ax = plt.subplots(clear=True, num=1, figsize=(5, 5))
df_test.plot(ax=ax, c='C1', ls='', marker='*')
df_pred.plot(ax=ax, c='C0', ls='', marker='o')
ax.legend(['Predicted', 'Actual'])
ax.set_title(series_id + " (one-step forecasts)")
ax.set_xlabel('')
plt.tight_layout(pad=2)
plt.savefig(imgdir / 'short.jpg')

## Multi-step ahead predictions
df_pred = all_dates.predict(model.params,
                            start=df_train.index[-1],
                            end=df_test.index[-1],
                            dynamic=0).shift(1).iloc[1:]
mse = mean_squared_error(df_test, df_pred)
var = np.mean(np.square(df_test - df_train.mean()))
print(f"Long-term Forecasts:  rmse={np.sqrt(mse):.6f} r2={1-mse/var:.4f}")
fig, ax = plt.subplots(clear=True, num=2, figsize=(5, 5))
df_test.plot(ax=ax, c='C1', ls='', marker='*')
df_pred.plot(ax=ax, c='C0', ls='', marker='o')
ax.legend(['Predicted', 'Actual'])
ax.set_title(series_id + " (multi-step forecasts)")
ax.set_xlabel('')
plt.tight_layout(pad=2)
plt.savefig(imgdir / 'long.jpg')


# Granger Causality: INDPRO vs CPI

start = 19620101
for series_id, exog_id in [['CPIAUCSL', 'INDPRO'], ['INDPRO', 'CPIAUCSL']]:
    df = pd.concat([alf(s, start=start, log=1)
                    for s in [series_id, exog_id]], axis=1)
    df.index = pd.DatetimeIndex(df.index.astype(str))
    data = df.diff().dropna()

    print(f"Null Hypothesis: {exog_id} granger-causes {series_id}")
    res = grangercausalitytests(data, 3)
    print()

    dmf = (f'{series_id} ~ {series_id}.shift(1) '
           f' + {exog_id}.shift(1) '
           f' + {exog_id}.shift(2) '
           f' + {exog_id}.shift(3) ')
#           f' + {exog_id}.shift(4) ')
    model = smf.ols(formula=dmf, data=data).fit()
    robust = model.get_robustcov_results(cov_type='HAC', use_t=None, maxlags=0)
    print(robust.summary())

# Vector Autoregression: Impulse Response Function
model = VAR(data)
results = model.fit(3)
print(results.summary())
irf = results.irf(12)
#irf.plot(orth=False)
irf.plot_cum_effects(orth=False, figsize=(10, 6))
plt.savefig(imgdir / 'impulse.jpg')


