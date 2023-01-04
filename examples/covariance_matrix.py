"""Covariance Matrix Estimates of Industry Returns

- Covariance Matrix: PCA, SVD, Shrinkage
- TODO: Risk Decomposition, Black-Litterman, Risk Parity

Copyright 2023, Terence Lim

MIT License
"""
import numpy as np
from numpy.linalg import inv, svd
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import pandas as pd
from pandas import DataFrame, Series
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
from finds.database import SQL, Redis
from finds.structured import CRSP, FFReader
from finds.busday import BusDay
from finds.recipes import halflife
from finds.display import plot_bar, show
from conf import credentials, VERBOSE, paths
from typing import Tuple

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None
imgdir = paths['images']
sql = SQL(**credentials['sql'], verbose=VERBOSE)
user = SQL(**credentials['user'], verbose=VERBOSE)

rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)

## Retrieve industry returns from Ken French Data Library website
name, item, suffix = ('49_Industry_Portfolios', 0, '49vw')
df = FFReader.fetch(name=name,
                    item=item,
                    suffix=suffix,
                    date_formatter=bd.endmo)

# show missing dates
nans = df.isnull()
missing = {col: {'first': min(nans[nans[col]].index),
                 'last': max(nans[nans[col]].index),
                  'missing': np.sum(nans[col])}
           for col in df if nans[col].sum()}
df = df[df.index > 19690630]
Y = df.to_numpy()
show(DataFrame.from_dict(missing, orient='index'),
     latex=False,
     caption="Missing Values in FF-49 Industry Monthly Returns")

## PCA and scree plot

# PCA of returns covariances by SVD
# SVD: u S vT = x (T samples x N stocks)
means = Y.mean(axis=0, keepdims=True)
X = Y.copy()

x = np.array(X - means) # pre-process: demean by column
u, s, vT = np.linalg.svd(x, full_matrices=False)
v = vT.T
k = 10
print(np.cumsum(s[:k]**2/np.sum(np.diag(s**2))))
print('u:', u.shape, 's:', s.shape, 'vT:', vT.shape, 'v', v.shape)

# sklearn PCA: X (T samples x N features/stocks), sanity check same results
pca = PCA()               # note: PCA first demeans input X by column mean_
y = pca.fit_transform(X)  # project X (stock returns) onto the components
print(pca.explained_variance_ratio_[:k])
print('y:', y.shape, 'x:', x.shape, 'components_:', pca.components_.shape)

# assert: s == singular_values_ (aka 2-norm of the projection on components)
print('singular values:', np.allclose(pca.singular_values_, s))

# assert: s**2 / len(y) == explained_variance_
print('singular values:', np.allclose(pca.singular_values_, s))

# assert: x @ v == transform(x) (aka projection on components) (aka returns)
print('projections:', [np.allclose((x @ v)[:,i], -y[:,i]) or
                       np.allclose((x @ v)[:,i], y[:,i]) for i in range(k)])

# assert: u @ s == transform(x) (aka factor returns)
print('projections:', [np.allclose(u[:,i]*s[i], -y[:,i]) or
                       np.allclose(u[:,i]*s[i], y[:,i]) for i in range(k)])

# assert: cols of v == rows of components_ (right SVD eigenvectors) (weights)
print('components:', [np.allclose(pca.components_[i,:], -v[:,i]) or
                      np.allclose(pca.components_[i,:], v[:,i])
                      for i in range(k)])


# assert: covariance matrix == loadings.T @ loadings
loadings = np.diag(pca.singular_values_) @ pca.components_
print('covariance matrix:', np.allclose(x.T @ x, loadings.T @ loadings))


### Projection on first component, and average "market" factor
t = pca.components_
top_k = 4
DataFrame({'frac weights +ve': np.mean(t[:top_k, :] >= 0, axis=1),
           'sum weights': np.sum(t[:top_k, :], axis=1),
           'sum abs weights': np.sum(np.abs(t[:top_k, :]), axis=1),
           'corr with eql-wtd market returns':
           [np.corrcoef(x.mean(axis=1), y[:,i])[0,1]
                        for i in range(top_k)]},
          index=[f"PC{i+1}" for i in range(top_k)])


### Plot components/portfolio weights distribution 
fig, axes = plt.subplots(nrows=1, ncols=3, num=1, clear=True, figsize=(10, 4))
for i, ax in enumerate(np.ravel(axes)):
    ax.bar(np.arange(t.shape[1]),
           np.sort(t[i, :]),
           color=f"C{i}")
    ax.legend([f"Weights (Loadings) on PC{i+1}"], fontsize=6)
plt.tight_layout(pad=0)
plt.savefig(imgdir / 'weights.jpg')

### Scree Plot
fig, ax = plt.subplots(1, 1, num=1, clear=True, figsize=(5,3))
k=10
plot_bar(Series(pca.explained_variance_ratio_[:k],
                index=np.arange(1, k+1)),
         ylabel='Explained Variance Ratio',
         xlabel="Component",
         legend=None,
         title='Scree Plot',
         ax=ax,
         fontsize=8,
         labels=[f"{i:.3f}" for i in pca.explained_variance_ratio_[:k]])
plt.tight_layout(pad=1)
plt.savefig(imgdir / 'explained.jpg')


## Evaluate rolling GMV portfolio volatility: shrinkage, PCA, EWMA

# Helper to compute EWMA covariance matrix estimate
def ewma(X, alpha=0.03, demean=False):
    weights = (1 - alpha)**np.arange(len(X))[::-1]
    if demean:
        X = X - X.mean(axis=0, keepdims=True)
    return (weights.reshape((1, -1)) * X.T) @ X / weights.sum()


# Helper method to compute Minimum Variance Portfolio and realized volatility
def gmv(cov, ret):
    """Compute minimum variance portfolio and realized volatility"""
    w = np.linalg.inv(cov) @ np.ones((cov.shape[1], 1))
    return ret @ w/sum(w)


# Rolling monthly evaluation
r = {}     # collect results of covariance matrix models
start_eval = 20000101
for retdate in tqdm(df.index[(df.index >= start_eval)]):
    x_train = Y[df.index < retdate, :]
    x_test = Y[df.index == retdate, :]
    keep = 5 * 12   # keep five years
    N = x_train.shape[1]
    r[retdate] = {}

    cov = EmpiricalCovariance().fit(x_train[-keep:, :]).covariance_
    
    r[retdate]['Full Covariance'] = float(gmv(cov, x_test))

    for alpha in [0.1, 0.06, 0.03, 0.01, 0.003]:
        r[retdate][f'EWMA({halflife(alpha=alpha):.0f}mo)'] = \
            float(gmv(ewma(x_train, alpha=alpha), x_test))

    r[retdate]['Eye'] = float(gmv(np.identity(x_train.shape[1]), x_test))

    r[retdate]['Diagonal'] = float(gmv(np.diagflat(np.diag(cov)), x_test))

    for k in [2, 5, 10, 15, 20]:
        r[retdate][f"PC 1-{k}"] = float(gmv(PCA(k).fit(x_train[-keep:, :])\
                                            .get_covariance(), x_test))

    r[retdate]['LW'] = float(gmv(LedoitWolf().fit(x_train[-keep:, :])\
                                 .covariance_, x_test))
    
    r[retdate]['OAS'] = float(gmv(OAS().fit(x_train[-keep:, :])\
                                  .covariance_, x_test))
    
ts = DataFrame.from_dict(r, orient='index')
vol = np.std(ts, axis=0)
show(vol, caption='Realized volatility of minimum variance portfolios', **SHOW)

## Plot evaluation period realized volatility of minimum variance portfolios

fig, ax = plt.subplots(1, 1, num=1, clear=True, figsize=(10, 5))
plot_bar(vol,
         ylabel='Volatility',
         xlabel='Risk Model',
         title='Test Period Volatility of Minimum Variance Portfolios',
         labels=[f"{v:.4f}" for v in vol],
         legend='',
         fontsize=6,
         ax=ax)
plt.tight_layout()
plt.savefig(imgdir / 'gmv.jpg')

            
# 4. Newey-west, Scholes Williams Beta
# 1/T \sum_t e_t^2 + 2/T \sum_L \sum_T w_l e_t e_t-l: w_l = 1-(l/(L+1))
#
# 5. Risk Decomposition: MCR, BL equilibrium, Risk Parity
# u.T @ x = beta, since u is standardized (orthogonal) factor returns
# u @ beta = stock's return due to orthogonal component
# (u @ beta)*2 = variation of stock returns due to orthogonal component
