"""Financial functions

Copyright 2022, Terence Lim

MIT License
"""
import re
from datetime import datetime
from typing import Iterable, Mapping, List, Any, NamedTuple, Dict, Tuple
import numpy as np
from numpy.ma import masked_invalid as valid
from scipy.stats import norm, chi2
import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types
import matplotlib.pyplot as plt

######################
#
#  Mean Variance Optimization
#
######################

def gmv_portfolio(sigma: np.ndarray, mu: np.ndarray | None=None):
    """Returns position weights of global minimum variance portfolio
    
    Args:
      sigma : covariance matrix
      mu : vector of expected excess (of the risk-free) returns

    Returns:
      Dict of weights, volatility and mean (if mu provided) of the GMV
    """
    ones = np.ones((sigma.shape[0], 1))
    w = la.inv(sigma).dot(ones) / ones.T.dot(la.inv(sigma)).dot(ones)
    return {'weights': w, 'volatility': np.sqrt(w.T.dot(sigma).dot(w)),
            'mean': None if mu is None else w.T.dot(mu)}

def efficient_portfolio(mu: np.ndarray, sigma: np.ndarray, target: float):
    """Returns weights of minimum variance portfolio that exceeds target return
    
    Args:
      sigma : covariance matrix
      mu : vector of expected excess (of the risk-free) returns
      target : required minimum expected return of portfolio

    Returns:
      Dict of weights, volatility and mean of efficient portfolio that achieves target
    """
    mu = mu.flatten()
    n = len(mu)
    ones = np.ones((n, 1))
    M = np.hstack([mu.reshape(-1, 1), ones])
    B = M.T.dot(la.inv(sigma)).dot(M)
    w = la.inv(sigma).dot(M).dot(la.inv(B)).dot(np.array([[target], [1]]))
    return {'weights': w, 'volatility': np.sqrt(float(w.T.dot(sigma).dot(w))),
            'mean': float(w.T.dot(mu))}

def tangency_portfolio(mu: np.ndarray, sigma: np.ndarray):
    """Returns weights of tangency portfolio with largest slope (sharpe ratio)
    
    Args:
      sigma : covariance matrix
      mu : vector of expected excess (of the risk-free) returns

    Returns:
      Dict of weights, volatility and mean of the tangency portfolio
    """
    mu = mu.flatten()
    n = len(mu)
    ones = np.ones((n, 1))
    w = la.inv(sigma).dot(mu)/ones.T.dot(la.inv(sigma).dot(mu))
    return {'weights': w, 'mean': float(w.T.dot(mu)),
            'volatility': np.sqrt(float(w.T.dot(sigma).dot(w)))}


######################
#
# Bond Math
#
######################

def bootstrap_spot(coupon: float, spots: List[float], m: int,
                   price: float=1) -> float:
    """Compute spot rate to maturity of par bond from yield and sequence of spots

    Args:
      coupon : Annual coupon rate
      spots : Simple annual spot rates each period (excl last period before maturity
      m : Number of compounding periods per year
      price: Present value of bond

    Returns:
      Simple spot interest rate till maturity
    """
    if not spots:           # trivial one-period bond
        return coupon / price
    n = len(spots) + 1      # number of coupons through maturity

    # discount factors from given spot rates
    discount = [(1 + spot/m)**(-(1+t)) for t, spot in enumerate(spots)]

    # nominal amount of last payment
    last_payment = 1 + coupon/m
    
    # infer present value of last coupon and principal
    last_pv = price - np.sum(discount) * coupon/m 

    # compute discount factor and annualize the effective rate
    spot = ((last_payment/last_pv)**(1/n) - 1) * m
    return spot

def bond_price(coupon: float, n: int, m: int, yields: float | List[float],
               par: float = 1, **kwargs) -> float:
    """Compute present value of bond given spot rates or yield-to-maturity

    Args:
      coupon : Annual coupon rate
      n : Number of remaining coupons
      m : Number of compounding periods per year
      yields : Simple annual yield-to-maturity or spot rates each period
      par : face or par value of bond

    Returns:
      Present value of bond
    """
    if not pd.api.types.is_list_like(yields):
        yields = [yields] * n        # same spot rate every period
    assert len(yields) == n, "Number of spot rates must equal number of couponds"
    pv = [(1 + yields[t-1]/m)**(-t) * (coupon/m + par*(t == n))
          for t in range(1, n+1)]    # discount every period's payment, plus last par
    return np.sum(pv)


def forwards_from_spots(spots: List[float], m: int, skip: int=0) -> List[float]:
    """Compute forward rates given spot rates

    Args:
      spots : Sequence of simple annual spot rates
      m : Number of compounding periods per year
      skip: Number of initial periods skipped

    Returns:
      List of forward rates, excluding first period of spot rates input
    """
    result = []
    assert len(spots) >= 2, "Require at least two spot rates as input"
    for t in range(1, len(spots)):
        n = skip + t
        numerator = (1 + spots[n]/m)**n         # discounted value of period n
        denominator = (1 + spots[n-1]/m)**(n-1)   # discounter value of period n-1
        result.append(((numerator / denominator) - 1) * m)
    return result

def macaulay_duration(coupon: float, n: int, m: int, price: float, 
                      yields: float | List[float], par: float = 1, **kwargs) -> float:
    """Compute macaulay duration of a bond given spot rates or yield-to-maturity

    Args:
      coupon : Annual coupon rate
      n : Number of remaining coupons
      m : Number of compounding periods per year
      price : current market price of bond
      yields : Simple annual yield-to-maturity or spot rates each period
      par : face or par value of bond

    Returns:
      Macaulay duration
    """
    if not pd.api.types.is_list_like(yields):
        yields = [yields] * n        # same spot rate every period
    assert len(yields) == n, "Number of spot rates must equal number of couponds"
    pv = [(1 + yields[t-1]/m)**(-t) * (t/m) * (coupon/m + par*(t == n))
          for t in range(1, n+1)]    # discount every period's time-weighted payment
    return np.sum(pv) / price

def modified_duration(coupon: float, n: int, m: int, price: float, 
                      yields: float | List[float], par: float = 1,
                      **kwargs) -> float:
    """Compute modified duration of a bond given spot rates or yield-to-maturity

    Args:
      coupon : Annual coupon rate
      n : Number of remaining coupons
      m : Number of compounding periods per year
      price : current market price of bond
      yields : Simple annual yield-to-maturity or spot rates each period
      par : face or par value of bond

    Returns:
      Modified duration
    """
    assert not pd.api.types.is_list_like(yields), "Not Implemented"
    ytm = yields
    return (macaulay_duration(coupon=coupon, n=n, m=m, price=price,
                              yields=yields, par=par) / (1 + ytm/2))

def modified_convexity(coupon: float, n: int, m: int, price: float, 
                       yields: float | List[float], par: float = 1,
                       **kwargs) -> float:
    """Compute mocified convexity of a bond given spot rates or yield-to-maturity

    Args:
      coupon : Annual coupon rate
      n : Number of remaining coupons
      m : Number of compounding periods per year
      price : current market price of bond
      yields : Simple annual yield-to-maturity or spot rates each period
      par : face or par value of bond

    Returns:
      Modified convexity
    """
    assert not pd.api.types.is_list_like(yields), "Not Implemented"
    ytm = yields
    if not pd.api.types.is_list_like(yields):
        yields = [yields] * n        # same spot rate every period
    assert len(yields) == n, "Number of spot rates must equal number of coupons"
    pv = [(1 + yields[t-1]/m)**(-t) * ((t/m)**2 + t/(2*m)) * (coupon/m + par*(t == n))
          for t in range(1, n+1)]    # discount every period's time-weighted payment
    return np.sum(pv) / (price * (1 + ytm/m)**2)


######################
#
# High Frequency
#
######################

def hl_vol(high: DataFrame,
           low: DataFrame,
           last: DataFrame = None) -> DataFrame:
    """Compute Parkinson volatility from high and low prices
    
    Args:
      high : DataFrame of high prices (observations x stocks)
      low : DataFrame of low prices (observations x stocks)
      last : DataFrame of last prices, for forward filling if high low missing

      Returns:
        Estimated volatility
    """
    if last is not None:
        high = high.where(high.notna(), last.shift())
        low = low.where(high.notna(), last.shift())
    return np.sqrt((np.log(high / low)**2).mean(axis=0, skipna=True)
                   / (4 * np.log(2)))

def ohlc_vol(first: DataFrame, high: DataFrame, low: DataFrame,
             last: DataFrame, ffill: bool = False,
             zero_mean: bool = True) -> DataFrame:
    """Compute Garman-Klass or Rogers-Satchell (non zero mean) OHLC vol
    
    Args:
      first : DataFrame of open prices (observations x stocks)
      high : DataFrame of high prices (observations x stocks)
      low : DataFrame of low prices (observations x stocks)
      last : DataFrame of close prices (observations x stocks)

    Returns:
      Estimated volatility 
    """
    if ffill:
        last = last.ffill()
        high = high.where(high.notna(), last.shift())
        low = low.where(low.notna(), last.shift())
        first = low.where(first.notna(), last.shift())
    if zero_mean:  # Garman-Klass (assuming zero mean drift)
        v = ((np.log(high / low)**2) / 2
             - (2*np.log(2) - 1) * (np.log(last / first)**2))\
             .mean(axis=0, skipna=True)
    else:          # Rogers-Satchell (non zero mean drift)
        v = ((np.log(high / close) * np.log(high / close))
             + (np.log(high / close) * np.log(high / close)))\
             .mean(axis=0, skipna=True)
    return np.sqrt(v)
    

######################
#
# Risk Management
#
######################

    
def maximum_drawdown(x: Series, is_price_level: bool = False) -> Series:
    """Compute max drawdown (max loss from previous high) period and returns

    Args:
        x: Returns or price level series
        is_price_level: Whether input are price index levels, or returns

    Returns:
        Series with start and end levels, keyed by dates, of maximum drawdown

    Notes:
        MDD = (Trough - Peak) / Peak
    """
    if is_price_level:
        cumsum = np.log(x)
    else:
        cumsum = np.log(1 + x).cumsum()
    cummax = cumsum.cummax()
    end = (cummax - cumsum).idxmax()
    beg = cumsum[cumsum.index <= end].idxmax()
    dd = cumsum.loc[[beg, end]]
    return np.exp(dd)

def parametric_risk(sigma: float | Series, alpha: float) -> Dict:
    """Calculate parametric gaussian VaR and ES
    
    Args:
      sigma : predicted volatility, or a Series
      alpha : Var level (e.g. 0.95)
    """
    var = -sigma * stats.norm.ppf(1 - alpha)
    es = sigma * stats.norm.pdf(stats.norm.ppf(1 - alpha)) / (1 - alpha)
    return dict(value_at_risk=var, expected_shortfall=es)

def historical_risk(X: Series, alpha: float):
    """Calculate historical VaR, ES, and sample moments
    
    Args:
      X : Series of observed returns
      alpha : Var level (e.g. 0.95)
    """
    X = X.dropna()
    N = len(X)
    var = -np.percentile(X, 100 * (1 - alpha))
    es = -np.mean(X[X < var])
    vol = np.std(X, ddof=0)
    skew = stats.skew(X)
    kurt = stats.kurtosis(X)
    jb = stats.jarque_bera(X)[0]
    jbp = stats.jarque_bera(X)[1]
    return dict(N=N, value_at_risk=var, expected_shortfall=es, volatility=vol,
                skewness=skew, excess_kurtosis=kurt-3, jb_statistic=jb, jb_pvalue=jbp)

def bootstrap_risk(X: Series, alpha: float, n: int) -> DataFrame:
    """Returned bootstrapped risk statistics

    Args:
      X : Series of observed returns
      alpha : VaR level (e.g. 0.95)
      n : Number of bootstrap samples

    Returns:
      DataFrame of n bootstrapped samples of risk statistics
    """
    X = X.dropna()
    N = len(X)
    bootstraps = []
    for _ in range(n):
        Z = Series(np.random.choice(X, N), index=X.index)
        bootstraps.append(historical_risk(Z, alpha=alpha))
    bootstraps = DataFrame.from_records(bootstraps)
    return bootstraps

def kupiec_LR(alpha: float, s: int, n: int):
    """Compute Kupiec likelihood ratio given s violations in n trials

    Args:
      s : number of violations
      n : number of observations
      alpha : VaR level (e.g. 0.95)
    """
    p = 1 - alpha       # prob of violation
    num = np.log(1 - p)*(n - s) + np.log(p)*s
    den = np.log(1 - (s/n))*(n - s) + np.log(s/n)*s
    lr = -2 * (num - den)
    return {'lr': lr, 'violations': s, 'N': n,
            # '5%_critical': stats.chi2.ppf(0.95, df=1),
            'pvalue': 1 - stats.chi2.cdf(lr, df=1)}

def kupiec(X: Series, VaR: Series, alpha: float) -> Dict:
    """Kupiec proportion of failures likelihood ratio test of VaR

    Args:
      X : Series of observed returns
      VaR : Series of predicted VaR
      alpha : VaR level (e.g. 0.95)

    Returns:
      Dict of likelihood statistic and pvalue
    """
    Z = pd.concat([X, VaR], axis=1).dropna()
    n = len(Z)
    s = np.sum(Z.iloc[:, 0] < -Z.iloc[:, 1])  # number of violations < -VaR
    return kupiec_LR(alpha=alpha, s=s, n=n)

# convert alpha to halflife
def halflife(alpha):
    """Returns halflife from alpha = -ln(2)/ln(lambda), where lambda=1-alpha"""
    if types.is_list_like(alpha):
        return [halflife(a) for a in alpha]
    if 0 < alpha < 1: 
        return -np.log(2)/np.log(1-alpha)
    else:
        return np.inf if (alpha > 0) else 0


from cvxopt import matrix, solvers
def quadprog(sigma):
    """Quadratic solver for portfolio optimization"""
    G = matrix(np.diag([-1.]*sigma.shape[1]))
    A = matrix(np.ones((1, sigma.shape[1])))
    b = matrix(np.ones((1, 1)))
    h = matrix(np.zeros((sigma.shape[1], 1)))
    sol = solvers.qp(P=matrix(sigma), q=h, G=G, h=h, A=A, b=b,
                     options=dict(show_progress=False))
    x = np.array(sol['x']).ravel()
    return x

    
if __name__=="__main__":
    # Verify with Jorion Chapter 5 Solution
    ytm = list(np.arange(0.0525, 0.1025, 0.0025))
    spot = np.array([])
    for y in ytm:
        spot = np.append(spot, Interest.bootstrap_rates(y, nominal=spot, m=2))
    jorion_sol5 = [.0797,.0827,.0859,.0892,.0925,.0961,.0997,.1036,.1077,.112]
    assert np.allclose(jorion_sol5, spot[-len(jorion_sol5):], atol=0.0001)
    print(spot)



