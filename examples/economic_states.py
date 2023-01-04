"""Hidden State Models and Economic Time Series

- Hidden Markov Models and Gaussian Mixture Model

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
import statsmodels.api as sm
import random
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from finds.alfred import fred_md, fred_qd, Alfred
from finds.display import show
from conf import VERBOSE, paths, credentials

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None

imgdir = paths['images'] / 'states'

# Load and pre-process time series from FRED
alf = Alfred(api_key=credentials['fred']['api_key'])
vspans = alf.date_spans('USREC')  # to indicate recession periods in the plots
beg = 19600301
end = 20200131
#
# TODO: end = train up to MINUS two quarters, forecast through current
#

# Retrieve FRED-MD series and apply tcode transformations
df, t = fred_md(202004)        # from vintage April 2020?
data = []
for col in df.columns:
    data.append(alf.transform(df[col], tcode=t['transform'][col], freq='m'))
mdf = pd.concat(data, axis=1).iloc[2:]
mdata = mdf[(mdf.index >= beg) & (mdf.index <= end)].dropna(axis=1)
mdata = (mdata - mdata.mean(axis=0)) / mdata.std(axis=0, ddof=0)
mdata.index = pd.DatetimeIndex(mdata.index.astype(str), freq='m')
mdata

# Hidden Markov Model
from typing import List, Dict
def hmm_summary(markov: hmm.GaussianHMM, X: DataFrame,
                lengths: List[int], matrix: bool = False) -> Dict:
    """Helper to return summary statistics from fitting Hidden Markov Model

    Args:
        markov: Fitted GaussianHMM 
        X: Input data of shape (nsamples, nfeatures)
        lengths: Lengths of the individual sequences in X, sum is nsamples
        matrix: Whether to return the transition and stationary matrices

    Returns:
        Dictionary of results in {'aic', 'bic', 'parameters', 'NLL'}
    """
    logL = markov.score(X, lengths)
    T = np.sum(lengths)      # n_samples

    n = markov.n_features    # number of features ~ dim of covariance matrix
    m = markov.n_components  # number of states
    k = markov.n_features + {"diag": m * n,    # parms in mean and cov matrix
                             "full": m * n * (n-1) / 2.0,
                             "tied": n * (n-1) / 2.0,
                             "spherical": m}[markov.covariance_type]
    p = m**2 + (k * m) - 1   # number of indepedent parameters of the model
    
    results = {'aic': -2 * logL + (2 * p),
               'bic': -2 * logL + (p * np.log(T)),
               'parameters': p,
               'NLL' : -logL}
    if matrix:   # whether to return the transition and stationary matrix
        matrix = DataFrame(markov.transmat_)\
            .rename_axis(columns='Transition Matrix:')
        matrix['Stationary'] = markov.get_stationary_distribution()
        results.update({'matrix': matrix})   # return matrix as DataFrame
    return results


## Compare covariance types in Gaussian HMM models 
out = []
for covariance_type in ["full", "diag", "tied", "spherical"]:
    for n_components in range(1,16):
        markov = hmm.GaussianHMM(n_components=n_components,
                                 covariance_type=covariance_type,
                                 verbose=False,
                                 tol=1e-6,
                                 random_state=42,
                                 n_iter=100)\
                    .fit(mdata.values, [len(mdata)])
        result = hmm_summary(markov, mdata, [len(mdata)])
        #print(n_components, Series(results, name=covariance_type).to_frame().T)
        result.update({'covariance_type': covariance_type,
                       'n_components': n_components})
        out.append(Series(result))
results = pd.concat(out, axis=1).T.convert_dtypes()

## Find best bic's
best_bic = []
for covariance_type in ["full", "diag", "tied", "spherical"]:
    result = results[results['covariance_type'] == covariance_type]
    argmin = np.argmin(result['bic'])
    best_bic.append(result.iloc[[argmin]])
best_bic = pd.concat(best_bic, axis=0)
show(best_bic, caption="HMM best bic by covariance type:", **SHOW)

## display estimated transition and stationary distributions of best_bic
n_components = best_bic[best_bic['covariance_type'] == 'diag']['n_components']
n_components = int(n_components)
markov = hmm.GaussianHMM(n_components=n_components,
                         covariance_type='diag',
                         verbose=False,
                         tol=1e-6,
                         random_state=42,
                         n_iter=100)\
            .fit(mdata.values, [len(mdata)])
pred = DataFrame(markov.predict(mdata), columns=['state'], index=mdata.index)
matrix = hmm_summary(markov, mdata, [len(mdata)], matrix=True)['matrix']
show(matrix, caption="HMM stationary and transition probabilities", **SHOW)

## Plot predicted states by selected economic series

def plot_states(modelname: str, labels: np.ndarray, beg: int, end: int,
                series_ids = ['IPMANSICS', 'SPASTT01USM661N']):
    """helper to plot predicted states"""

    # n_components markers
    n_components = len(np.unique(labels))
    markers = ["o", "s", "d", "X", "P", "8", "H", "*", "x", "+"][:n_components] 
    
    fig, axes = plt.subplots(len(series_ids),
                             ncols=1,
                             figsize=(10, 3 * len(series_ids)),
                             num=1,
                             clear=True)
    axes[0].set_title(f"{modelname.upper()} Predicted States", {'fontsize':12})

    # plot each selected series, with states colored
    for series_id, ax in zip(series_ids, axes.ravel()):
        df = alf(series_id)
        df.index = pd.DatetimeIndex(df.index.astype(str), freq='infer')
        df = df[(df.index >= beg) & (df.index <= end)]
        for i, marker in zip(range(n_components), markers):
            df.loc[labels==i].plot(ax=ax,
                                   style=marker,
                                   markersize=2,
                                   color=f"C{i}",
                                   rot=0)
            ax.set_xlabel(f"{series_id}: {alf.header(series_id)}",
                          {'fontsize':8})
        for a,b in vspans:   # shade economic recession periods
            if (b > min(df.index)) & (a < max(df.index)):
                ax.axvspan(max(a, min(df.index)),
                           min(b, max(df.index)),
                           alpha=0.3,
                           color='grey')
        ax.legend([f"state {i}" for i in range(n_components)], fontsize=8)
    plt.tight_layout()
    plt.savefig(imgdir / f"{modelname.lower()}.jpg")

plot_states('HMM', pred.values.flatten(), min(pred.index), max(pred.index))


# Gaussian Mixture Model
gmm = GaussianMixture(n_components=n_components,
                      covariance_type='diag')\
                      .fit(mdata)
labels = gmm.predict(mdata)
plot_states('GMM', labels, min(mdata.index), max(mdata.index))


## Compare persistance of HMM and GMM
dist = DataFrame({'Hidden Markov':
                  sorted(matrix.iloc[:,-1])
                  + [np.mean(pred[:-1].values == pred[1:].values)],
                  'Gaussian Mixture':
                  sorted(Series(labels).value_counts().sort_index()/len(labels))
                  + [np.mean(labels[:-1] == labels[1:])]},
                 index=[f'Stationary probability of state {n_components-s-1}'
                        for s in range(n_components)]
                 + ['Average persistance of states'])
show(dist, caption="Compare HMM with GMM:", **SHOW)
