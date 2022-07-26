"""Linear Regression Diagonostics and Residual Plots

- Linear regression diagnostics: HAC robust standard errors
- Outliers: leverage, influential points, residual plots
- Multicollinearity: variance inflation factor
- Interactions and Polynomial Regression

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

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
from finds.display import plot_fitted, plot_leverage, plot_scale, plot_qq
from conf import credentials, VERBOSE, paths

imgdir = os.path.join(paths['images'], 'ts')
alf = Alfred(api_key=credentials['fred']['api_key'],
             savefile=os.path.join(paths['scratch'], 'alfred.pkl'))


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
plt.savefig(os.path.join(imgdir, 'outliers' + figext))
show(z.to_frame().T, caption="Residual Outliers")
#plt.show()


## QQ Plot of residuals and identify outliers
fig, ax = plt.subplots(clear=True, figsize=(8,7))
z = plot_qq(model.resid, ax=ax)
plt.savefig(os.path.join(imgdir, 'qq' + figext))
#plt.show()
z

## Plot scale of residuals with outliers
fig, ax = plt.subplots(clear=True, figsize=(8,7))
plot_scale(model.fittedvalues, model.resid, ax=ax)
plt.savefig(os.path.join(imgdir, 'scale' + figext))
    

## Plot leverage and identify influential points                 
fig, ax = plt.subplots(clear=True, figsize=(8,7))
z = plot_leverage(model.resid, model.get_influence().hat_matrix_diag,
                  model.get_influence().cooks_distance[0],
                  ddof=len(model.params), ax=ax)
plt.savefig(os.path.join(imgdir, 'leverage' + figext))
z


# Polynomials and Interactions
dmf2 = (f'{series_id} '
        f' ~ {series_id}.shift(1) '
        f' + {exog_id}.shift(1) '
        f' + I({series_id}.shift(1)**2)'
        f' + I({exog_id}.shift(1)**2)'
        f' + {series_id}.shift(1):{exog_id}.shift(1)')
model2 = smf.ols(formula=(dmf2), data=data).fit()
robust2 = model2.get_robustcov_results(cov_type='HAC', use_t=None, maxlags=2)
print(robust2.summary())

"""
## Plot diagnostics
fig, ax = plt.subplots(clear=True, figsize=(8,7))
z = plot_fitted(fitted=model2.fittedvalues, resid=model2.resid, ax=ax)
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'fitted2' + figext))
#plt.show()

fig, ax = plt.subplots(clear=True, figsize=(8,7))
plot_scale(model2.fittedvalues, model2.resid, ax=ax)
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'scale2' + figext))
#plt.show()

fig, ax = plt.subplots(clear=True, figsize=(8,7))
z = plot_qq(model2.resid, ax=ax, title=series_id)
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'qq2' + figext))
#plt.show()
"""

fig, ax = plt.subplots(clear=True, figsize=(8,7))
z = plot_leverage(model2.resid, model2.get_influence().hat_matrix_diag,
                  model2.get_influence().cooks_distance[0],
                  ddof=len(model2.params), ax=ax)
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'leverage2' + figext))
#plt.show()
show(z, caption="Influential Points (Model II)")


Y, X = patsy.dmatrices(dmf2 + ' - 1', data=data)  # exclude intercept term
show(Series({X.design_info.column_names[i]: variance_inflation_factor(X, i)
             for i in range(X.shape[1])}, name='VIF'),
     caption="Variance Inflation Factors (Model II)")
