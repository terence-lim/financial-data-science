"""Economics tools

- Bai and Ng (2002), McCracken and Ng (2015, 2020) factors-EM algorithm

Copyright 2022, Terence Lim

MIT License
"""
import re
from datetime import datetime
import numpy as np
from numpy.ma import masked_invalid as valid
import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from typing import Iterable, Mapping, List, Any, NamedTuple, Dict, Tuple
_VERBOSE = 1

def mrsq(X: DataFrame, kmax: int) -> DataFrame:
    """Return marginal R2 of each variable from incrementally adding factors

    Args:
        X: T observations/samples in rows, N variables/features in columns
        kmax: maximum number of factors.  If 0, set to rank from SVD

    Returns:
        DataFrame with marginal R2 with component in each column

    Notes:

    From matlab code, Bai and Ng (2002) and McCracken at
      https://research.stlouisfed.org/econ/mccracken/fred-databases/
    """
    # pca.components_[i,:] is vT[i, :]
    # pca.explained_variance_ is s**2/(T-1)
    # y = pca.transform(x)    # y = s * u: T x n "projection"
    # beta = np.diag(pca.singular_values_) @ pca.components_  # "loadings"
    # x.T @ x = beta.T @ beta is covariance matrix
    
    Z = (X - X.mean()) / X.std(ddof=0)
    u, s, vT = np.linalg.svd(Z, full_matrices=False)

    mrsq_ = pd.concat([np.mean((u[:,k-1:k] @ u[:,k-1:k].T @ Z)**2, axis=0)
                       for k in np.arange(1, (kmax or len(s)) + 1)],
                      axis=1)
    return mrsq_.div(np.mean((u @ u.T @ Z)**2, axis=0), axis=0)


def select_bai_ng(X: DataFrame, kmax: int = 0, p: int = 2) -> int:
    """Determine number of factors based on Bai & Ng (2002) info criterion

    Args:

        X: T observations/samples in rows, N variables/features in columns
        p: int in [1, 2, 3] to use PCp1 or PCp2 or PCp3 penalty
        kmax: Maximum number of factors.  If 0, set to rank from SVD

    Returns:
        best number of factors based on ICp{p} criterion, or 0 if not determined

    Notes:

    - Simplified the calculation of residual variance from adding components:
      is just the eigenvalues, no need to compute projections
    - The IC curve appears to have multiple minimums: the first "local"
      minimum is selected -- may also be related to why authors suggest a
      prior bound on number of factors.
    """
    assert p > 0
    Z = ((X - X.mean()) / X.std(ddof=0)).to_numpy()
    T, N = Z.shape
    NT = N * T
    NT1 = N + T
    GCT = min(N, T)    
    CT = [np.log(NT/NT1) * (NT1/NT),
          (NT1/NT) * np.log(GCT),
          np.log(GCT) / GCT]
    CT = [i * CT[p-1] for i in range(GCT)]

    u, s, vT = np.linalg.svd(Z, full_matrices=False)
    eigval = s**2
    residual_variance = np.roll(np.sum(eigval) - eigval.cumsum(), 1)
    residual_variance[0] = sum(eigval)
    sigma = residual_variance / sum(eigval)
    ic = (np.log(sigma + 1e-12) + CT)[:(kmax or GCT)]
    return np.where((ic[:-1] - ic[1:]) < 0)[0][0]


def factors_em(X: DataFrame, kmax: int = 0, p: int = 2, max_iter: int = 50,
           tol: float = 1e-12, verbose: int = _VERBOSE) -> DataFrame:
    """Fill in missing values with factor model EM algorithm Bai and Ng (2002)

    Args:
        X: T observations/samples in rows, N variables/features in columns
        kmax: Maximum number of factors.  If 0, set to rank from SVD minus 1
        p: If 0, number of factors is fixed as kmax.  Else picks one of three
           information criterion methods in Bai & Ng (2002) to auto-select

    Returns:
        DataFrame with missing values imputed with factor EM algorithm
    """
    Z = X.copy()          # passed by reference
    Y = np.isnan(Z)       # missing entries
    assert(not np.any(np.all(Y, axis=1)))  # no row can be all missing
    assert(not np.any(np.all(Y, axis=0)))  # no column can be all missing

    # identify cols with missing values, and initially fill with column mean
    missing_cols = Z.isnull().sum().to_numpy().nonzero()[0]
    for col in missing_cols:
        Z.iloc[Y.iloc[:, col], col] = Z.iloc[:, col].mean()

    for n_iter in range(max_iter):
        old_Z = Z.copy()
        mean = Z.mean()
        std = Z.std()
        Z = (Z - mean) / std             # standardize the data

        # "M" step: estimate factors
        u, s, vT = np.linalg.svd(Z)

        # auto-select number of factors if p>0 else fix number of factors
        if p:
            r = select_bai_ng(Z, p=p, kmax=kmax or len(s) - 1)
        else:
            r = kmax or len(s) - 1

        # "E" step: update missing entries
        E = u[:, :r] @ np.diag(s[:r]) @ vT[:r, :]
        for col in missing_cols:
            Z.iloc[Y.iloc[:, col], col] = E[Y.iloc[:, col], col]

        Z = (Z * std) + mean  # undo standardization

        delta = (np.linalg.norm(Z - old_Z) / np.linalg.norm(Z))**2
        if verbose:
            print(f"{n_iter:4d} {delta:8.3g} {r}")
        if delta < tol:       # diff**2/prev**2
            break
    return Z



def impute_em(X: np.ndarray, add_intercept: bool = True,
              tol: float = 1e-12, maxiter: int = 200,
              verbose: int = 1) -> Tuple[np.ndarray, DataFrame]:
    """Fill missing data with EM Normal distribution"""
    if add_intercept:
        X = np.hstack((np.ones((X.shape[0], 1)), X))
    missing = np.isnan(X)   # identify missing entries
    assert(not np.any(np.all(missing, axis=1)))    # no row all missing
    assert(not np.any(np.all(missing, axis=0)))    # no column all missing
    cols = np.flatnonzero(np.any(missing, axis=0)) # columns with missing 

    results = []
    for niter in range(maxiter+1):
        if not niter:
            # Initially, just replace with column means
            for col in cols: 
                X[missing[:, col], col] = np.nanmean(X[:, col])
        else:
            XX = X.T @ X
            inv_XX = inv(XX)
            for col in cols:  # E, M step for each column with missing values
                # "M" step: estimate covariance matrix
                mask = np.ones(X.shape[1], dtype=bool)
                mask[col] = 0
                # x = np.delete(X, (col), axis=1)
                if False:
                    #xx = np.delete(np.delete(XX, (col), axis=0), (col), axis=1)
                    M = inv(XX[:, mask][mask, :]) @ X[:, mask].T @ X[:, col]
                else:
                    M = -inv_XX[mask, col] / inv_XX[col, col]

                # "E" step: update expected missing values
                # y = X[:, mask] @ M
                X[missing[:, col], col] = X[missing[:, col],:][:, mask] @ M
        x = X[:, add_intercept:]
        # record the current NLL
        nll = -sum(multivariate_normal.logpdf(x,
                                              mean=np.mean(x, axis=0),
                                              cov=np.cov(x.T, bias=True),
                                              allow_singular=True))
        if verbose:
            print(f"{niter} {nll:.6f}")
        if niter and prev_nll - nll < tol:
            break
        prev_nll = nll
    return x

######################
#
# Econometrics
#
######################

def integration_order(df: Series, noprint: bool = True, max_order: int = 5,
                      pvalue: float = 0.05, lags: str | int = 'AIC') -> int:
    """Returns order of integration by iteratively testing for unit root

    Args:
        df: Input Series
        noprint: Whether to display results
        max_order: maximum number of orders to test
        pvalue: Required p-value to reject Dickey-Fuller unit root
        lags: Method automatically determine lag length, or maxlag;
              in {"AIC", "BIC", "t-stat"}, int (maxlag), 0 (12*(nobs/100)^{1/4})

    Returns:
        Integration order, or -1 if max_order exceeded
    """
    if not noprint:
        print("Augmented Dickey-Fuller unit root test:")
    for i in range(max_order):
        if not lags:
            dftest = adfuller(df, maxlag=None, autolag=None)
        elif isinstance(lags, str):
            dftest = adfuller(df, autolag=lags)
        else:
            dftest = adfuller(df, autolag=None, maxlag=lags)
        if not noprint:
            results = Series(dftest[0:4],
                             index=['Test Statistic',
                                    'p-value',
                                    'Lags Used',
                                    'Obs Used'],
                             name=f"I({i})")
            for k,v in dftest[4].items():
                results[f"Critical Value ({k})"] = v
            print(results.to_frame().T.to_string())
                
        if dftest[1] < pvalue:  # reject null that is a unit root
            return i
        df = df.diff().dropna()
    return -1

def least_squares(data: DataFrame, y: List[str] = ['y'],
                  x: List[str] = ['x'], add_constant: bool = True,
                  stdres: bool = False) -> Series | DataFrame:
    """To compute least square coefficients, supports groupby().apply

    Args:
        data: DataFrame with x and y series in columns
        x: List of x columns
        y: List of y columns
        add_constant: Whether to add intercept as first column
        stdres: Whether to output residual stdev

    Returns:
        DataFrame (multiple) or Series (simple) of regression coefficients

    """
    X = data[x].to_numpy()
    Y = data[y].to_numpy()
    if add_constant:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        x = ['Intercept'] + x
    b = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y)).T
    if stdres:
        b = np.hstack([b, np.std(Y-(X @ b.T), axis=0).reshape(-1,1)])
        x = x + ['stdres']
    return (DataFrame(b, columns=x, index=y) if len(b) > 1 else
            Series(b[0], x))   # return as Series for groupby.apply

def fstats(x: Series | np.ndarray, tail: float = 0.15) -> np.ndarray:
    """Helper to compute F-stats at all candidate break points
    
    Args:
        x: Input Series
        tail: Tail fractions to skip computations

    Returns:
        Array of f-stats at each candidate break-point
    """
    n = len(x)
    rse = np.array(np.var(x, ddof=0))
    sse = np.ones(n) * rse
    for i in range(int(n * tail), int((1-tail) * n)+1):
        sse[i] = (np.var(x[:i], ddof=0)*i + np.var(x[i:], ddof=0)*(n-i))/n
    return ((n-2)/2) * (rse - sse)/rse


from collections import namedtuple    
def lm(x: np.ndarray | DataFrame | Series, y: np.ndarray | DataFrame | Series,
       add_constant: bool = True, flatten: bool = True) -> NamedTuple:
    """Calculate linear multiple regression model results as namedtuple

    Args:
        x: RHS independent variables
        y: LHS dependent variables
        add_constant: Whether to hstack 'Intercept' column before x variables
        flatten: Whether to flatten fitted and residuals series

    Returns:
        LinearModel named tuple, with key and values

        - coefficients: estimated linear regression coefficients
        - fitted: fitted y values
        - residuals: fitted - actual y values
        - rsq: R-squared
        - rvalue: square root of r-squared
        - stderr: std dev of residuals
    """
    
    def f(a):
        """helper to optionally flatten 1D array"""
        if not flatten or not isinstance(a, np.ndarray):
            return a
        if len(a.shape) == 1 or a.shape[1] == 1:
            return float(a) if a.shape[0] == 1 else a.flatten()
        return a.flatten() if a.shape[0] == 1 else a
    
    X = np.array(x)
    Y = np.array(y)
    if len(X.shape) == 1 or X.shape[0]==1:
        X = X.reshape((-1,1))
    if len(Y.shape) == 1 or Y.shape[0]==1:
        Y = Y.reshape((-1,1))
    if add_constant:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    b = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    out = {'coefficients': f(b)}
    out['fitted'] = f(X @ b)
    out['residuals'] = f(Y) - out['fitted']
    out['rsq'] = f(np.var(out['fitted'], axis=0)) / f(np.var(Y, axis=0))
    out['rvalue'] = f(np.sqrt(out['rsq']))
    out['stderr'] = f(np.std(out['residuals'], axis=0))
    return namedtuple('LinearModel', out.keys())(**out)

