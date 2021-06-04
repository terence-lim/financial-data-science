"""Forecasting and Econometrics

- seasonality, spectral density, unit root, stationarity
- autocorrelation functions, AR, MA, SARIMAX
- scipy, statsmodels, seaborn, St Louis Fed FRED

Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns 
from finds.alfred import Alfred

from settings import settings
imgdir = os.path.join(settings['images'], 'ts')
alf = Alfred(api_key=settings['fred']['api_key'],
             savefile=settings['scratch'] + 'fred.md')

# Seasonality
import scipy.signal
series_id = 'ND000334Q'  # real gdp
df = alf(series_id, log=1, diff=1, freq='Q').dropna()
x = df.values.flatten()  #y.iloc[:-240].copy()
fig, axes = plt.subplots(1, 2, num=1, clear=True, figsize=(10,5))
axes[0].plot(pd.DatetimeIndex(df.index.astype(str), freq='infer'),
             x.cumsum(), marker=None)
axes[0].set_title(" ".join(alf.header(
    series_id, ['id', 'title', 'seasonal_adjustment']).to_list()), fontsize=10)
    
freq, power = scipy.signal.welch(x - x.mean(), nperseg=4*(len(x)//4))
axes[1].semilogy(freq, power)
argmax = np.argmax(power[:-1])
axes[1].set_xlabel(f"Max spectral density at frequency={freq[argmax]:.2f}, "
                   f"periodicity={1/freq[argmax]:.0f} quarters",fontsize=10)
axes[1].axvline(freq[argmax], ls=':', c='r')
plt.tight_layout(pad=3)
plt.savefig(os.path.join(imgdir, 'welch.jpg'))
plt.show()

# Stationarity and Unit Root
## helper test methods
from statsmodels.tsa.stattools import adfuller
def unit_root(x, pvalue=0.05, noprint=False):
    """test if input series has unit root using augmented dickey fuller"""
    dftest = adfuller(x, autolag='AIC')
    if not noprint:
        results = Series(dftest[0:4],
                         index=['Test Statistic','p-value', 'Lags Used',
                                'Obs Used'])
        for k,v in dftest[4].items():
            results[f"Critical Value ({k})"] = v
        print(results.to_frame().T.to_string(index=False))
    return dftest[1] > pvalue

def integration_order(df, noprint=True, max_order=5, pvalue=0.05):
    """returns order of integration by iteratively testing for unit root"""
    for i in range(max_order):
        if not noprint:
            print(f"Augmented Dickey-Fuller unit root test of I({i}):")
        if not unit_root(df, pvalue=pvalue, noprint=noprint):
            return i
        df = df.diff().dropna()

## Integration Order: Real GDP (Seasonally Adjusted)
s = 'GDPC1'
df = alf(s, log=1, freq='Q')
df.index = pd.DatetimeIndex(df.index.astype(str))
p = integration_order(df, noprint=False, pvalue=0.05)
Series({s: p}, name='I(p)').to_frame()

## Histogram Plot and Kernel Density Estimate
import seaborn as sns
fig, axes = plt.subplots(1, 2, num=1, clear=True, figsize=(10,5))
sns.distplot(df.dropna().rename(f"diff log {s}"), hist=True, kde=True,
             rug=True, ax=axes[0], rug_kws={"color": "C0"},
             kde_kws={"color": "C1"}, hist_kws={"color": "C2"})
axes[0].set_title(f"Density Plot of log({s})")
sns.distplot(df.diff().dropna().rename(f"diff log {s}"), hist=True, kde=True,
             rug=True, ax=axes[1], rug_kws={"color": "C0"},
             kde_kws={"color": "C1"}, hist_kws={"color": "C2"})
axes[1].set_title(f"Density Plot of diff log({s})")
plt.savefig(os.path.join(imgdir, 'order.jpg'))
plt.tight_layout(pad=3)
plt.show()

# Autocorrelation Function
## Plot ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
df = alf('GDPC1', start=19590101, end=20191231, log=1, diff=1, freq='Q')[1:]
df.index = pd.DatetimeIndex(df.index.astype(str), freq='infer')
fig, axes = plt.subplots(1, 2, clear=True, figsize=(10,5))
#df.plot(ax=axes[0], title="$\Delta$ log(GDPC1)")
plot_acf(df.values.squeeze(), lags=20, ax=axes[0])
plot_pacf(df, lags=20, ax=axes[1])
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'acf.jpg'))
plt.show()

# Select AR lag order with BIC
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
s = 'GDPC1'  # real gdp, seasonally adjusted
df = alf(s, log=1, diff=1, start=19591201, freq='Q').loc[:20191231].dropna()
df.index = pd.DatetimeIndex(df.index.astype(str), freq='infer')
df_train = df[df.index <= '2017-12-31']
df_test = df[df.index > '2017-12-31']
lags = ar_select_order(df_train, maxlag=13, ic='bic',old_names=False).ar_lags
print('(BIC) lags= ', len(lags), ':', lags)

# AR and SARIMAX
## AR(p) is simplest time-model, can nest in SARIMAX(p,d,q,s) with
## moving average MA(q), integration order I(d), seasonality S(s), exogenous X
from statsmodels.tsa.statespace.sarimax import SARIMAX
adf = alf(s, log=1, freq='Q').loc[19591201:20171231]
adf.index = pd.DatetimeIndex(adf.index.astype(str), freq='infer')
arima = SARIMAX(adf, order=(2, 1, 0), trend='c').fit()
fig = arima.plot_diagnostics(figsize=(10,6))
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'ar.jpg'))
plt.show()
arima.summary()

# Forecasting
## One-step ahead predictions    
model = AutoReg(df_train, lags=lags, old_names=False).fit()  
print(model.summary())
    
# Observations to predict are from the test split
from sklearn.metrics import mean_squared_error        
all_dates = AutoReg(df, lags=lags, old_names=False)
df_pred = all_dates.predict(model.params,
                            start=df_train.index[-1]).shift(1).iloc[1:]
mse = mean_squared_error(df_test, df_pred)
var = np.mean(np.square(df_test - df_train.mean()))
print(f"Short-term Forecasts:  rmse={np.sqrt(mse):.6f} r2={1-mse/var:.4f}")
fig, ax = plt.subplots(clear=True, num=1, figsize=(4,6))
df_pred.plot(ax=ax, c='C0')
df_test.plot(ax=ax, c='C1')
ax.legend(['Predicted', 'Actual'])
ax.set_title(s + " (one-step forecasts)")
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'short.jpg'))
plt.show()

# Multi-step ahead predictions
df_pred = all_dates.predict(model.params,
                            start=df_train.index[-1],
                            end=df_test.index[-1],
                            dynamic=0).shift(1).iloc[1:]
mse = mean_squared_error(df_test, df_pred)
var = np.mean(np.square(df_test - df_train.mean()))
print(f"Long-term Forecasts:  rmse={np.sqrt(mse):.6f} r2={1-mse/var:.4f}")
fig, ax = plt.subplots(clear=True, num=2, figsize=(4,6))
df_pred.plot(ax=ax, c='C0')
df_test.plot(ax=ax, c='C1')
ax.legend(['Predicted', 'Actual'])
ax.set_title(s + " (multi-step forecasts)")
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'long.jpg'))
plt.show()

# One-step ahead expanding re-estimation window predictions
df_pred = DataFrame(index=df_test.index)
for date in df_pred.index:
    expand = AutoReg(df[df.index < date], lags=lags, old_names=False).fit()
    df_pred.loc[date, 'pred'] = float(expand.predict(start=-1, end=-1))
mse = mean_squared_error(df_test, df_pred)
var = np.mean(np.square(df_test - df_train.mean()))
print(f"Expanding Forecasts:  rmse={np.sqrt(mse):.6f} r2={1-mse/var:.4f}")
fig, ax = plt.subplots(clear=True, num=3, figsize=(4,6))
df_pred.plot(ax=ax, c='C0')
df_test.plot(ax=ax, c='C1')
ax.legend(['Predicted', 'Actual'])
ax.set_title(s + " (expanding re-estimation window forecasts)")
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'recursive.jpg'))
plt.show()
