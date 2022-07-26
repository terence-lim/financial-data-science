"""Covariance matrix of FamaFrench industries

Notes:

Author:
  Terence Lim

License: 
  MIT
"""
import os
import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import pandas as pd
from pandas import DataFrame, Series
from functools import reduce
import matplotlib.pyplot as plt
from finds.database import SQL, Redis
from finds.structured import CRSP, FFReader
from finds.busday import BusDay
from conf import credentials, VERBOSE, paths
from typing import Tuple

VERBOSE = 1
imgdir = paths['images']
sql = SQL(**credentials['sql'], verbose=VERBOSE)
user = SQL(**credentials['user'], verbose=VERBOSE)

rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)

print(FFReader.datasets)
name, item, suffix = ('49_Industry_Portfolios', 0, '49vw')
name, item, suffix = ('49_Industry_Portfolios_daily', 0, '49vw')
df = FFReader.fetch(name=name,
                    item=item,
                    suffix=suffix,
                    date_formatter=bd.endmo)
#monthly = df

def imputeEM(X: np.ndarray, add_intercept: bool = True,
              tol: float = 1e-12, maxiter: int = 200,
              verbose: int = VERBOSE) -> Tuple[np.ndarray, DataFrame]:
    """Fill missing data with EM Normal distribution"""
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    missing = np.isnan(X)   # identify missing entries
    assert(not np.any(np.all(missing, axis=1)))    # no row all missing
    assert(not np.any(np.all(missing, axis=0)))    # no column all missing

    # Initially replace with column means
    cols = np.flatnonzero(np.any(missing, axis=0)) # columns with missing 
    for col in cols: 
        X[missing[:, col], col] = np.nanmean(X[:, col])

    results = []
    for niter in range(maxiter):
        old = X.copy()
        XX = X.T @ X
        for col in cols:
            
            # "M" step: estimate covariance matrix
            xx = np.delete(np.delete(XX, (col), axis=0), (col), axis=1)
            y = X[:, col]
            x = np.delete(X, (col), axis=1)
            M = inv(xx) @ x.T @ y 

            # "E" step: update expected missing values
            y = x @ M
            X[missing[:, col], col] = y[missing[:, col]]

        # check nll and error tolerance
        delta = np.sum((X - old)**2)/np.sum(X**2)  # diff**2/prev**2
        x = X[intercept:, :]
        nll = -multivariate_normal.logpdf(x,
                                          mean=np.mean(x, axis=0),
                                          cov=np.cov(x.T, bias=True),
                                          allow_singular=True)
        results.append({'nll': sum(nll), 'delta': delta})
        if verbose:
            print(f"{niter} {delta:8.3g} {sum(nll):.6f}")
        if delta < tol:
            break
    return X, results

Y, result = imputeEM(df.values)

# 1. Show missing dates, plot EM results

# 2. PCA scree plot, explain

# 3. Estimation methods and GMV evaluation: shrinkage, PCA, EWMA, newey-west
## rolling every year.
## 1/T \sum_t e_t^2 + 2/T \sum_L \sum_T w_l e_t e_t-l: w_l = 1-(l/(L+1))

# 4. Risk Decomposition: MCR, BL equilibrium, Risk Parity
