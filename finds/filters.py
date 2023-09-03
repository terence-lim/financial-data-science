"""Filtering functions

Copyright 2022, Terence Lim

MIT License
"""
import re
from datetime import datetime
from typing import Iterable, Mapping, List, Any, NamedTuple, Dict, Tuple
import numpy as np
from numpy.ma import masked_invalid as valid
import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, rfft, irfft



######################
#
#  FFT convolutions and correlations
#
######################
class FFT:
    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        """Demean columns and divide by norm"""
        X = X - np.mean(X, axis=0)
        X = X / np.linalg.norm(X, axis=0)
        return X

    def correlation(X: np.ndarray, Y: np.ndarray | None) -> Series:
        """Compute cross-correlations of two series, using Convolution Theorem

        Args:
            X, Y: series of observations

        Returns:
            Series of cross-correlation values at every displacement lag

        Notes:

        - Cross-correlation[n] = \sum_m^N f[m] g[n + m]
        - equals Convolution (f * g)[n] = \sum_m^N f[m] g[n - m]

        Examples:

        >>> statsmodels.tsa.stattools.acf(X) 
        >>> fft_correlate(X, X)
        """
        N = len(X)
        if Y is None:
            Y = X
        assert len(Y) == len(X)
    
        # normalize and zero pad to length 2N
        X = np.pad(FFT._normalize(X.reshape(-1, 1)), [(0, N), (0,0)])
        Y = np.flipud(np.pad(FFT._normalize(Y.reshape(-1, 1)), [(0, N), (0,0)]))

        # Convolution Theorem
        conv = irfft(rfft(X, axis=0) * rfft(Y, axis=0), axis=0)
        
        shift = (N // 2) + 1     # only first and last N/2 not due to padding
        window = 2*(N // 2) + 1  # make window length odd to center exactly at 0
        return Series(data=np.roll(conv, shift, axis=0)[:window].reshape(-1),
                      index=np.arange(-(window//2), 1+(window//2)))


    @staticmethod
    def align(X: np.ndarray) -> Tuple:
        """Find max cross-correlation, or best lag, of all pairs of columns
    
        Args:
            X: array with series in columns

        Returns:
            Tuple of list of max cross-correlations between each pair of columns,
            and list of corresponding best lags 

        Notes:

        - Apply convolution theorem to compute cross-correlations at lags
        - For each pair of series, the lag with largest correlation is assumed
          to be the displacement which aligns the presentation of the two series

        Examples:

        >>> FFT.align(np.hstack((X[:-1], X[1:])), corr=False)
        """
        N, M = X.shape
        assert M > 1
        shift = (N // 2) + 1
        window = 2 * (N // 2) + 1
        X = np.pad(FFT._normalize(X), [(0, N), (0,0)])
        Y = rfft(np.flipud(X), axis=0)   # FFT of flipped series
        X = rfft(X, axis=0)              # FFT of original series
        corr = []
        disp = []
        for col in range(M-1):
            # inverse FFT of product of Fourier-transformed series
            conv = irfft(X[:, [col]] * Y[:, col+1:], axis=0) 
            corr.extend(np.max(conv, axis=0))
            disp.extend(((np.argmax(conv, axis=0) + shift) % N) - window)
        return corr, disp


    @staticmethod
    def neweywest(X: np.ndarray) -> List:
        """Compute Newey-West weighted cross-correlation of all pairs of columns

        Args:
            X: array with series in columns

        Returns:
            List of Newey-west weighted cross-correlations

        Notes:

        - First apply convolution theorem to compute all cross-autocorrelations,
        - Then for each pair of series, compute Newey-west weighted correlation
        """
        N, M = X.shape
        assert M > 1
        shift = (N // 2) + 1
        window = 2 * (N // 2) + 1
        L = window // 2

        # Newey-West weights, with peak (1.0) at center of window
        NW = np.array([1 - abs(l)/(L+1) for l in range(-L, L+1, 1)])\
               .reshape(1, -1)

        # Convolution Theorem
        X = np.pad(FFT._normalize(X), [(0, N), (0,0)])
        Y = rfft(np.flipud(X), axis=0)
        X = rfft(X, axis=0)

        # Accumulate results for each column against remaining columns 
        result = []
        for col in range(M-1):
            conv = irfft(X[:, [col]] * Y[:, col+1:], axis=0)
            corrs = np.roll(conv, shift, axis=0)[:window]
            np.mean(np.max(corrs, axis=0))
            result.extend()
        return result


######################
#
# Data filters
#
######################


def winsorize(df, quantiles=[0.025, 0.975]):
    """Winsorise dataframe by column quantiles (default=[0.025, 0.975])

    Args:
        df: Input DataFrame
        quantiles: high and low fractions of distribution to truncate
    """
    lower = df.quantile(min(quantiles), interpolation='higher')
    upper = df.quantile(max(quantiles), interpolation='lower')
    if types.is_list_like(lower):
        return df.clip(lower=lower, upper=upper, axis=1)
    else:   # input was Series
        return df.clip(lower=lower, upper=upper)



def is_outlier(x: Any, method: str = 'iq10', bounds: bool = False) -> np.array:
    """Test if elements of x are column-wise outliers

    Args:
        x: Input array to test element-wise
        method: method to filter, in {'iq{D}', 'tukey', 'farout'}
        bounds: If True, return (low, high) values of inlier range

    Returns:
        boolean array if element is not a column-wise outlier

    Notes:
    - 'iq{D}' - screen by iq range [Q2 +/- D times (Q3-Q1)]
    - 'tukey' -  [Q1 - 1.5(Q3-Q1), Q3 + 1.5(Q3-Q1)] 
    - 'farout' - tukey with 3IQ

    """
    def nancmp(f, a, b):
        valid = ~np.isnan(a)
        valid[valid] = f(a[valid] , b)
        return valid

    x = np.array(x)
    if len(x.shape) == 1:
        lb, median, ub = np.nanpercentile(x, [25, 50, 75])
        if method.lower().startswith(('tukey', 'far')):
            w = 1.5 if method[0].lower() == 't' else 3.0
            f = [lb - (w * (ub - lb)), ub + (w * (ub - lb))]
            if not bounds:
                f = (nancmp(np.greater_equal, x, f[0]) &
                     nancmp(np.less_equal, x, f[1]))
        elif method.lower().startswith('iq'):
            w = float(re.sub('\D', '', method))
            f = [median - (w * (ub - lb)), median + (w * (ub - lb))]
            if not bounds:
                f = (nancmp(np.greater_equal, x, f[0]) &
                     nancmp(np.less_equal, x, f[1]))
        else:
            raise Exception("outliers method not in ['iq[D]', 'tukey']")
        return ~np.array(f)
    else:
        return np.hstack([is_outlier(x[:, i], method=method, bounds=bounds)\
                          .reshape(-1, 1)
                          for i in range(x.shape[1])])

from numpy.ma import masked_invalid as valid
def weighted_average(df: DataFrame, weights: str = '') -> Series:
    """Weighted means of data frame

    Args:
        df: DataFrame containing values, and optional weights, in columns
        weights: Column name to use as weights

    Returns:
        Series of weighted means

    Notes:
    - ignores NaN's using numpy.ma 
    """
    if not weights:
        cols = df.columns
    else:
        cols = df.columns.difference([weights])
        weights = df[weights].astype(float)
    return Series(np.ma.average(valid(df[cols].astype(float)),
                                weights=weights,
                                axis=0), index=cols)

def remove_outliers(X: DataFrame, method: str = 'iq10') -> DataFrame:
    """Set column-wise outliers to np.nan

    Args:
        X: Input array to test element-wise
        method: method to filter outliers, in {'iq{D}', 'tukey', 'farout'}

    Returns:
        DataFrame with outliers set to NaN

    Notes:
    - 'iq{D}' -  within [median +/- D times (Q3-Q1)]
    - 'tukey' -  within [Q1 - 1.5(Q3-Q1), Q3 + 1.5(Q3-Q1)] 
    - 'farout' - within [Q1 - 3(Q3-Q1), Q3 + 3(Q3-Q1)] 
    """
    Z = X.copy()
    q1 = Z.quantile(1/4)
    q2 = Z.quantile(1/2)
    q3 = Z.quantile(3/4)
    iq = q3 - q1
    if method.lower().startswith(('tukey', 'far')):
        scalar = 1.5 if method[0].lower() == 't' else 3.0
        outlier = Z.lt(q1 - scalar * iq) | Z.gt(q3 + scalar * iq)
        Z[outlier] = np.nan
    elif method.lower().startswith('iq'):
        scalar = float(method[2:])
        outlier = (Z - Z.median()).abs().gt(scalar * iq)
        Z[outlier] = np.nan
    return Z


