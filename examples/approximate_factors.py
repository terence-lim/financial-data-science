"""Approximate Factor Models, VAR and TCN

- PCA, EM
- Approximate factors and selection: Bai and Ng (2002), McCracken and Ng (2016)
- vector autoregression, temporal convolutional networks

Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import re
import time
from datetime import datetime
import random
import statsmodels.api as sm
from finds.alfred import Alfred, fred_md, fred_qd, factors_em, remove_outliers,\
    mrsq, select_bai_ng
from finds.display import plot_bands, show
from finds.recipes import integration_order, not_outlier
from finds.unstructured import Store
from conf import VERBOSE, credentials, paths

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None

store = Store(paths['scratch'])
imgdir = paths['images'] / 'ts'
alf = Alfred(api_key=credentials['fred']['api_key'], verbose=-1)

# FRED-MD, Transformation Codes and Stationarity

## Transformation Codes, and Stationarity
qd_df, qd_codes = fred_qd() # 202004
md_df, md_codes = fred_md() # 201505

print('Number of time series and suggested transformations, by tcode:')
tcodes = pd.concat([Series(alf._tcode[i], name=i).to_frame().T
                    for i in range(1, 8)], axis=0).fillna(False)
tcodes = tcodes.join(qd_codes['transform']\
                     .value_counts()\
                     .rename('fred-qd'))\
               .join(md_codes['transform']\
                     .value_counts()\
                     .rename('fred-md'))\
               .fillna(0).astype({'fred-qd': int, 'fred-md': int})\
               .rename_axis(index='tcode')
tcodes


## Estimate and Compare Integration Order
out = {}
for label, df, transforms in [['md', md_df, md_codes['transform']],
                              ['qd', qd_df, qd_codes['transform']]]:
    stationary = dict()
    for series_id, tcode in transforms.items():
        if tcode in range(1, 8):
            s = np.log(df[series_id]) if tcode in [4, 5, 6] else df[series_id]
            order = integration_order(s.dropna(), pvalue=0.05)
            expected_order = 2 if tcode == 7 else ((tcode - 1) % 3)
            stationary[series_id] = {
                'tcode': tcode,
                'I(p)': order,
                'different': order - expected_order,
                'title': alf.header(series_id)}
            #print(series_id, order, tcode)
    stationary = DataFrame.from_dict(stationary, orient='index')
    stationary = stationary.sort_values(stationary.columns.to_list())
    c = stationary.groupby(['tcode','I(p)'])['title'].count().reset_index()
    out[label] = c.pivot(index='tcode', columns='I(p)',
                         values='title').fillna(0).astype(int)
    out[label].columns=[f"I({p})" for p in out[label].columns]
print('Series by tcode, transformations and estimated order of integration:')
results = pd.concat([tcodes.drop(columns='fred-md'),
                     out['qd'],
                     tcodes['fred-md'],
                     out['md']], axis=1).fillna(0).astype(int)
show(results,
     caption='FRED-MD order of integration, transformations and frequency',
     **SHOW)
show(stationary[stationary['different'] > 0],
     max_colwidth=60,
     caption='FRED-MD series with unit root after transformations',
     **SHOW)


# Verify BaiNg implemention on published FRED-MD and FRED-QD reports
qd_df, qd_codes = fred_qd(202004)
md_df, md_codes = fred_md(201505)
for freq, df, transforms in [['monthly', md_df, md_codes['transform']],
                             ['quarterly', qd_df, qd_codes['transform']]]:    
    # Apply tcode transformations
    transformed = []
    for col in df.columns:
        transformed.append(alf.transform(df[col],
                                         tcode=transforms[col],
                                         freq=freq[0]))
    data = pd.concat(transformed, axis=1).iloc[2:]
    cols = list(data.columns)
    sample = data.index[((np.count_nonzero(np.isnan(data), axis=1)==0)
                         | (data.index <= 20141231))
                        & (data.index >= 19600301)]

    # set missing and outliers in X to NaN
    X = data.loc[sample]
    X = remove_outliers(X)

    # compute factors EM and auto select number of components, r
    Z = factors_em(X, p=2, verbose=1)
    r = select_bai_ng(Z, p=2)
    
    # show marginal R2's of series to each component
    mR2 = mrsq(Z, r).to_numpy()
    show(DataFrame({'selected': r,
                    'variance explained': np.sum(np.mean(mR2[:, :r], axis=0)),
                    'start': min(sample),
                    'end': max(sample),
                    'obs': Z.shape[0],
                    'series': Z.shape[1]},
                   index=[f'factors']),
         caption=f"FRED-{freq[0].upper()}D {freq} series:", **SHOW)

    for k in range(r):
        args = np.argsort(-mR2[:, k])
        show(DataFrame.from_dict({mR2[arg, k].round(4):
                                  {'series': cols[arg],
                                   'description': alf.header(cols[arg])}
                                  for arg in args[:10]},
                                 orient='index'),
             caption=f"Factor:{1+k} Variance Explained={np.mean(mR2[:,k]):.4f}",
             **SHOW)

## Sanity check Extract factors: SVD == PCA
# pipe.fit through 20141231, pipe.transform through 20201231
df, t = fred_md()  #202104 # 201505
transforms = t['transform']

sample_date = 20141231
data = []
for col in df.columns:
    data.append(alf.transform(df[col], tcode=transforms[col], freq='m'))
data = pd.concat(data, axis=1).iloc[2:]
cols = list(data.columns)
sample = data.index[((np.count_nonzero(np.isnan(data), axis=1)==0) |
                     (data.index <= sample_date)) & (data.index >= 19600301)]
train_sample = sample[sample <= sample_date]
test_sample = sample[sample <= 20191231]

# replace missing and outliers with PCA EM and fixed number of components r=8
r = 8
X = data.loc[train_sample] # X = np.array(data.loc[train_sample])
X[~not_outlier(X, method='iq10')] = np.nan
x = factors_em(X, kmax=r, p=0, verbose=0).to_numpy()

# Extract factors with SVD
y = ((x-x.mean(axis=0).reshape(1,-1))/x.std(axis=0,ddof=0).reshape(1,-1))
u, s, vT = np.linalg.svd(y, full_matrices=False)
#factors = DataFrame(u[:, :r], columns=np.arange(1, 1+r),
#                    index=pd.DatetimeIndex(train_sample.astype(str), freq='M'))
Series(s[:r]**2 / np.sum(s**2), index=np.arange(1, r+1), name='R2').to_frame().T

# Equivalent to sklearn PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(r))])
pipe.fit(x)              # fit model on training data
X = data.loc[sample]     # to transform on full sample data
X = factors_em(X, kmax=8, p=0, verbose=0)  # replace missing (not outlier)
factors = DataFrame(StandardScaler().fit_transform(pipe.transform(X)),
                    index=pd.DatetimeIndex(sample.astype(str), freq='infer'),
                    columns=np.arange(1, 1+r))

# Store approximate factors in local folder
store['approximate'] = dict(factors=factors)

Series(pipe.named_steps['pca'].explained_variance_ratio_,
       index=np.arange(1,r+1), name='R2').to_frame().T   # sanity check



## Retrieve recession periods from FRED
vspans = alf.date_spans('USREC')
DataFrame(vspans, columns=['Start', 'End'])

## Plot extracted factors
fig = plt.figure(figsize=(9, 10), num=1, clear=True)
for col in factors.columns:
    ax = fig.add_subplot(4, 2, col)
    flip = -np.sign(max(factors[col]) + min(factors[col])) # try match sign
    (flip*factors[col]).plot(ax=ax, color=f"C{col}")
    for a,b in vspans:
        if b >= min(factors.index):
            ax.axvspan(max(a, min(factors.index)), min(b, max(factors.index)),
                       alpha=0.3, color='grey')
    ax.legend([f"Factor {col} Estimate", 'NBER Recession'], fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"Factor Estimates {factors.index[0]:%b-%Y}:"
             f"{factors.index[-1]:%b-%Y}", fontsize=12)
plt.savefig(imgdir / 'approximate.jpg')




