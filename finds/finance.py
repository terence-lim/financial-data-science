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
# Finance
#
######################

class Volatility:
    """Class of static methods to compute intra-day volatility measures"""
    
    def HL(high: DataFrame,
           low: DataFrame,
           last: DataFrame = None) -> DataFrame:
        """Compute Parkinson volatility from high and low prices
    
        Args:
          high: DataFrame of high prices (observations x stocks)
          low: DataFrame of low prices (observations x stocks)
          last: DataFrame of last prices, for forward filling if high low missing

        Returns:
          Estimated volatility
        """
        if last is not None:
            high = high.where(high.notna(), last.shift())
            low = low.where(high.notna(), last.shift())
        return np.sqrt((np.log(high / low)**2).mean(axis=0, skipna=True)
                       / (4 * np.log(2)))

    def OHLC(first: DataFrame, high: DataFrame, low: DataFrame,
             last: DataFrame, ffill: bool = False,
             zero_mean: bool = True) -> DataFrame:
        """Compute Garman-Klass or Rogers-Satchell (non zero mean) OHLC vol
    
        Args:
          first: DataFrame of open prices (observations x stocks)
          high: DataFrame of high prices (observations x stocks)
          low: DataFrame of low prices (observations x stocks)
          last: DataFrame of close prices (observations x stocks)

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

# proportion of failures likelihood test
def kupiecLR(s: int, n: int, var: float = 0.95) -> Dict[str, float]:
    """Kupiec Likelihood Ratio test (S violations in N trials) of VaR

    Args:
        s: number of violations
        n: number of observations
        var: VaR level (e.g. 0.95)

    Returns:
        Dictionary of likelihood statistic and pvalue
    """
    
    p = 1 - var        # e.g. var95 is 0.95
    t = n - s          # number of non-violations
    num = np.log(1 - p)*(n - s) + np.log(p)*s
    den = np.log(1 - (s/n))*(n - s) + np.log(s/n)*s
    lr = -2 * (num - den)
    return {'statistic': lr, 'pvalue': 1 - chi2.cdf(lr, df=1)}


def pof(X: Series, pred: Series | float, var: float = 0.95) -> Dict[str, float]:
    """Kupiec proportion of failures VaR test

    Args:
        X: Observed Series
        pred: Predicted standard deviation
        var: VaR level (e.g. 0.95)

    Returns:
        Dictionary {'statistics', 'pvalue', 's': violations, 'n': observations}
    """

    Z = X / pred
    z = norm.ppf(1 - var)
    r = {'n': len(Z), 's': np.sum(Z < z)}
    r.update(kupiecLR(r['s'], r['n'], var))
    return r

# convert alpha to halflife
def halflife(alpha):
    """Returns halflife from alpha = -ln(2)/ln(lambda), where lambda=1-alpha"""
    if types.is_list_like(alpha):
        return [halflife(a) for a in alpha]
    if 0 < alpha < 1: 
        return -np.log(2)/np.log(1-alpha)
    else:
        return np.inf if (alpha > 0) else 0

class RiskMeasure:
    """Class to compute risk measures for a time series
    Args:
        x: Time series of observations
        alpha: Risk tolerance threshold (default is 0.95 for 5% tail)
    """
    def __init__(self, x: Series, alpha: float = 0.95):
        self.x = x
        self.alpha = alpha
        
    def expected_shortfall(self, normal: bool = False):
        """Return value at risk: empirical or normal assumption"""
        if normal:
            return (-np.std(self.x) * norm.pdf(norm.ppf(1 - self.alpha))
                    / (1 - self.alpha))
        else:
            return np.mean(self.x[self.x < self.value_at_risk()])
            
    def value_at_risk(self, normal: bool = False):
        """Return value at risk: empirical or normal assumption"""
        if normal:
            return np.std(self.x) * norm.ppf(1 - self.alpha)
        else:
            return np.percentile(self.x, 100 * (1 - self.alpha))
        

# helper methods for basic bond math calculations
class Interest:
    
    @staticmethod
    def present_value(flow: float, n: float, spot: float) -> float:
        """Present Value of a cash flow at n period, given spot interest rate

        Args:
           flow: Amount of future cash flow
            n: Number of periods to discount
            spot: Interest rate per period

        Returns:
           PV of cash flow discounted by spot rate compounded over n periods
        """
        return flow / ((1 + spot) ** n)


    @staticmethod
    def weighted_maturity(flows: List[float], spot: float, first: int = 1,
                          returned: bool = False) -> float:
        """Average maturity weighted by PV of flows discounted by spot rate 

        Args:
            flows: List of cash flow amounts at each future period
            spot: Interest rate per period
            first: First period when cash flows begin
            returned: If `True`, the tuple (`average`, `sum_of_weights`)
                is returned, otherwise only the average is returned.

        Returns:
            Weighted average maturity of future cash flows
        """
        v = [Interest.present_value(flow=flow,
                                    n=n + first,
                                    spot=rate)
             for n, (flow, rate) in enumerate(zip(flows, spot))]
        return np.average(np.arange(len(v)) + first, weights=v, returned=returned)


    @staticmethod
    def par_duration(nominal: float, n: int, face: float = 1.,
                     m: int = 1, first: float = 1.) -> float:
        """Macaulay duration of a coupon bond, currently selling at par price

        Args:
            nominal: Nominal annual coupon rate of the bond
            n: Number of years till maturity
            face: Face value of bond to be returned at maturity
            m: Number of intra-year coupon payments
            first: First year when cash flows begin

        Returns:
            Macaulay duration of a par coupon bond

        Notes:
            Assumes bond currently selling at par
        """
        coupon = nominal * face     # assume par bond
        flows = [coupon / m] * (n * m - 1) + [face + coupon / m]  # face in last
        d, v = Interest.weighted_maturity(flows,
                                          spot=[nominal / m] * (n * m),
                                          first=first * m,
                                          returned=True)
        return d / m

    @staticmethod
    def discounted_cash_flow(flows: float | List[float],
                             spot: float | List[float],
                             first: int = 1) -> float:
        """PV of future cash flows, starting at first period

        Args:
            flows: Amounts of future annual cash flows
            spot: Interest rate, or rates, per year
            first: First period when cash flow begins

        Returns:
            Discounter present value of future cash flows
        """

        if not types.is_list_like(flows):    # flows can be different each period
            flows = [flows]                  # else assume same flow every period
            if not types.is_list_like(spot): # spot can be different per flow
                spot = [spot]                # else use same spot each period
        if len(flows) == 1:
            flows = list(flows) * len(spot)  # flows to be same length as spot
        if len(spot) == 1:
            spot = list(spot) * len(flows)   # spot to be same length as flows
        return np.sum([Interest.present_value(flow=flow,
                                              n=first + n,
                                              spot=rate)
                       for n, (flow, rate) in enumerate(zip(flows, spot))])


    @staticmethod
    def forward_rates(spot: List[float], base=0) -> List[float]:
        """Forward rates implied by spot rates starting after base periods

        Args:
            spot: List of current annual spot interest rates at each period
            base: Base periods skipped by initial spot rate in input list

        Returns:
            List of forward curve annual rates
        """
        return [(((1 + num)**(n + 1 + base) / (1 + den)**(n + base)) - 1)
                for n, (num, den) in enumerate(zip(spot, [0] + list(spot[:-1])))]


    @staticmethod
    def bootstrap_rates(ytm: float, nominal: List[float], m: int = 1) -> float:
        """Nominal rate to maturity of par bond from ytm and sequence of nominals

        Args:
            ytm: Annualized yield to maturity of par bond
            nominal: Annualized spot rates each period (excl last maturity period)
            m: Number of periods per year

        Returns:
            Nominal annualized effective interest rate till maturity

        Notes:
            Assumes bond currently selling at par
        """
        n = len(nominal) + 1       # implicit number of coupons through maturity
        spot = [r / m for r in nominal]  # spot rate per period
        pv = (1 - Interest.discounted_cash_flow(flows=ytm/m, spot=spot))
        return (((1 + (ytm / m)) / pv)**(1 / n) - 1) * m


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



