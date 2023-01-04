"""Linear Regression Diagonostics and Residual Plots

- Linear regression diagnostics, HAC robust standard errors
- Residual analysis: outliers, leverage, influential points
- Multicollinearity, variance inflation factor

Copyright 2023, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns 
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from finds.alfred import Alfred
from finds.display import plot_fitted, plot_leverage, plot_scale, plot_qq, show
from conf import credentials, VERBOSE, paths

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None

imgdir = paths['images'] / 'ts'
alf = Alfred(api_key=credentials['fred']['api_key'],
             savefile=paths['scratch'] / 'alfred.pkl')


# CPI monthly GDP quarterly series from FRED
series_id, freq, start = 'CPIAUCSL', 'M', 19620101
#series_id, freq, start = 'GDPC1', 'Q', 0
series_id, freq, start = 'CPIAUCSL', 'M', 0 #19740101
exog_id = 'WPSFD4131'
df = pd.concat([alf(s, start=start, log=1)
                for s in [series_id, exog_id]], axis=1)
df.index = pd.DatetimeIndex(df.index.astype(str))
data = df.diff().dropna()


## Run Linear Regression (with one lag)
dmf = (f'{series_id} '
       f' ~ {series_id}.shift(1) ')
model = smf.ols(formula=dmf, data=data).fit()
model.summary()

## Run Linear Regression (with 2 lags)
dmf = (f'{series_id} '
       f' ~ {series_id}.shift(1) '
       f' + {series_id}.shift(2) ')
model = smf.ols(formula=dmf, data=data).fit()
model.summary()

## Run Linear Regression (with exog and lag)
dmf = (f'{series_id} '
       f' ~ {series_id}.shift(1) '
       f' + {exog_id}.shift(1) ')
model = smf.ols(formula=dmf, data=data).fit()
model.summary()

# Heteroskedasity and HAC robust errors
robust = model.get_robustcov_results(cov_type='HAC', use_t=None, maxlags=0)
robust.summary()

Y, X = patsy.dmatrices(dmf + ' - 1', data=data)  # exclude intercept term  
show(Series({X.design_info.column_names[i]: variance_inflation_factor(X, i)
             for i in range(X.shape[1])}, name='VIF').to_frame(),
     caption="Variance Inflation Factors")

## Plot residuals and identify outliers
fig, ax = plt.subplots(clear=True, figsize=(8,7))
z = plot_fitted(fitted=model.fittedvalues,
                resid=model.resid,
                ax=ax)
plt.savefig(imgdir / 'outliers.jpg')
show(z.to_frame().T, caption="Residual Outliers")
#plt.show()


## QQ Plot of residuals and identify outliers
fig, ax = plt.subplots(clear=True, figsize=(8,7))
z = plot_qq(model.resid, ax=ax)
plt.savefig(imgdir / 'qq.jpg')
#plt.show()
z

## Plot scale of residuals with outliers
fig, ax = plt.subplots(clear=True, figsize=(8,7))
plot_scale(model.fittedvalues, model.resid, ax=ax)
plt.savefig(imgdir / 'scale.jpg')
    

## Plot leverage and identify influential points                 
fig, ax = plt.subplots(clear=True, figsize=(8,7))
z = plot_leverage(model.resid, model.get_influence().hat_matrix_diag,
                  model.get_influence().cooks_distance[0],
                  ddof=len(model.params), ax=ax)
plt.savefig(imgdir / 'leverage.jpg')
z
