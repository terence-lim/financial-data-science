"""Principal Components of Bond Returns

- Principal components analysis, bond returns

Copyright 2022, Terence Lim

MIT License
"""
import re
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
from finds.alfred import Alfred
from finds.display import show
from conf import VERBOSE, credentials, paths

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None
imgdir = paths['images'] / 'ts'
alf = Alfred(api_key=credentials['fred']['api_key'])

# Bond Return Components and Interest Rate Risk Indicators
## get Merrill Lynch bond indexes
c = alf.get_category(32413)
print(c['id'], c['name'])
t = Series({s['id']: s['title'] for s in c['series']})
t

b = []   # accumulate bond returns
for s in t.index:
    b.append(alf(s, start=19961231) )
bonds = pd.concat(b, axis=1)

## Show blocks of data available
v = bonds.notna().sum(axis=1).rename('count')
v = pd.concat([v, (v != v.shift()).cumsum().rename('notna')], axis=1)
g = v.reset_index().groupby(['notna', 'count'])['date'].agg(['first','last'])
g

start_date = 19981231
rets = bonds.loc[bonds.index >= start_date,
                 bonds.loc[start_date].notna().values]
rets = pd.concat([alf.transform(rets[col], log=1, diff=1)
                  for col in rets.columns], axis=1)
show(Series(alf.header(rets.columns),
            index=rets.columns,
            name='title').to_frame().rename_axis('series'),
     max_colwidth=88,
     caption="Bond Index Total Returns", **SHOW)


# Marginal Variance Explained
x = np.array(rets.iloc[1:].replace(np.nan, 0))
d = rets.iloc[1:].index.rename(None)
c = rets.columns
r = 3
"""
from finds.alfred import marginalRsq
mR2 = marginalRsq(x, standardize=True)
print(f"Explained by {r} factors: {np.sum(np.mean(mR2[:r,:], axis=1)):.3f}"
      f" ({len(x)} obs)")
df = DataFrame({'explained': np.mean(mR2, axis=1)},
               index=np.arange(1, len(mR2) + 1))
df.iloc[4]
"""

## Same calculation with sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
pipe.fit(x)
print(pipe.named_steps['pca'].explained_variance_ratio_)  # sanity check
scree = Series(pipe.named_steps['pca'].explained_variance_ratio_,
               index=np.arange(1, x.shape[1]+1))

## Scree plot
fig, ax = plt.subplots(num=1, clear=True, figsize=(10, 5))
scree[:10].plot(kind='bar', rot=0, width=.8, ax=ax)
ax.set_title('Scree Plot: PCA of FRED BofA Bond Return Indexes', fontsize=16)
ax.xaxis.set_tick_params(labelsize=12)
ax.set_ylabel("Percent Variance Explained", fontsize=14)
ax.set_xlabel("Principal Component", fontsize=14)
plt.savefig(imgdir / 'scree.jpg')

# Rsquare of components explained by Interest Rate Indicators
rates = ['BAA10Y', 'T10Y2Y', 'DGS10']  # compare rate levels
rates = pd.concat([alf(s) for s in rates], axis=1).reindex(d).pad()
rates.index = pd.DatetimeIndex(rates.index.astype(str), freq='infer')

factors = DataFrame(pipe.transform(x)[:, :3],
                    columns=np.arange(1, 4),
                    index=pd.DatetimeIndex(d.astype(str), freq='infer'))

from finds.display import plot_date
import statsmodels.api as sm
rsq = [sm.OLS(factors[col].cumsum(), rates).fit().rsquared
       for col in factors.columns]
res = DataFrame({'Rsq of rate levels': rsq}, index=factors.columns)
res['Rsq of rate changes'] = [sm.OLS(factors[col],
                                      rates.diff().fillna(0))\
                               .fit().rsquared
                               for col in factors.columns]
show(res, max_colwidth=75, 
     caption="R-squared of each PC explained by Interest Rate Indicators", **SHOW)

# Plot of cumulative components compared to levels of selected interest rates   
fig = plt.figure(figsize=(10, 12), num=1, clear=True)
for isub, col in enumerate(factors.columns):
    ax = fig.add_subplot(3, 2, (col * 2) - 1)
    flip = -1 if col == 3 else 1
    (flip*factors[col]).cumsum().plot(ax=ax, color='r')
    ax.legend([f"PC{col}  (r-square={rsq[isub]:.3})"], fontsize=8)
    ax.xaxis.set_tick_params(rotation=0)
    if not isub:
        ax.set_title('Bond Returns Components (cumulated)')
        
    ax = fig.add_subplot(3, 2, (isub + 1) * 2)
    rates.iloc[:, isub].plot(ax=ax, color='b')
    ax.legend([f"{rates.columns[isub]}"], fontsize=8)
    ax.xaxis.set_tick_params(rotation=0)
    if not isub:
        ax.set_title('Selected Interest Rates')
    ax.set_xlabel(alf.header(rates.columns[isub])[:80])
plt.savefig(imgdir / 'components.jpg')
plt.tight_layout()


