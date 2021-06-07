"""Quant, financial and econometrics helpers

- linear algebra, stationarity, robust covariances
- maturity, bootstrap, annuity, compounding, rate of discount and interest
- value at risk, duration, half-life

Author: Terence Lim
License: MIT
"""
import numpy as np
from pandas import DataFrame, Series
from statsmodels.tsa.stattools import adfuller
from pandas.api import types
try:
    from settings import ECHO
except:
    ECHO = False

def least_squares(data=None, y='y', x='x', add_constant=True, stdres=False):
    """Compute least squares fitted coefficients: helper for groupby apply

    Parameters
    ----------
    data : DataFrame, default is None
        supplies dependent and regressor variables, useful for .groupby
    y : str or DataFrame/Series, default is 'y'
        column names of dependent variables, or DataFrame if data is None
    x : str or DataFrame/Series, default is 'x'
        column names of independent variables, or DataFrame if data is None
    add_constant : bool, default is True
        if True, then hstack 'Intercept' column of ones before x variables
    stdres : bool, default is False
        if True, then also return estimated residual std dev

    Returns
    -------
    coef : Series or DataFrame
        DataFrame (Series) of fitted coefs when y is multi- (one-) dimensional

    Examples
    --------
    fm = data.groupby(by='year').apply(least_squares, y='y', x='x')
    """
    if data is None:
        X = x.to_numpy()
        Y = DataFrame(y).to_numpy()
        x = list(x.columns)
        y = list(y.columns)
    else:
        x = [x] if isinstance(x, str) else list(x)
        y = [y] if isinstance(y, str) else list(y)
        X = data[x].to_numpy()
        Y = data[y].to_numpy()
    if add_constant:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        x = ['Intercept'] + x
    b = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y)).T
    if stdres:
        b = np.hstack([b, np.std(Y-(X @ b.T), axis=0).reshape(-1,1)])
        x = x + ['stdres']
    return (Series(b[0], index=x) if len(b)==1 else
            DataFrame(b, columns=x, index=y))

from numpy.ma import masked_invalid as valid
def weighted_average(df, weights=None, axis=0):
    """Weighted means of data frame, ignores NaN's using numpy.ma"""
    cols = df.columns.difference([weights])
    return Series(np.ma.average(valid(df[cols]),
                                weights=df[weights] if weights else None,
                                axis=axis), index=cols)

def fractiles(values, pctiles, keys = None, ascending = False):
    """Sort and assign values into fractiles

    Parameters
    ----------
    values : array_like
        Input array to assign to fractiles
    pctiles : array_like
        Percentile values to determine fractile breakpoints
    keys : array_like, optional
        To determine percentile breakpoints from (if None, use values)
    ascending: bool
        if True, assign fractiles in ascending order; else descending (default)

    Returns
    -------
    fractiles : list of ints
        list of fractile assignments {1,.., len(pctiles)} 
        s.t. value <= lowest pctile
    """
    keys = values if keys is None else keys
    keys = np.array(keys)[~np.isnan(keys)]
    bp = list(np.percentile(keys, sorted(pctiles))) + [np.inf]
    return (1 + np.searchsorted(bp, values, side='left') if ascending
            else 1 + len(pctiles) - np.searchsorted(bp, values, side='left'))

def winsorized(df, quantiles=[0.025, 0.975]):
    """Winsorise dataframe by column quantiles (default=[0.025, 0.975])"""
    lower = df.quantile(min(quantiles), interpolation='higher')
    upper = df.quantile(max(quantiles), interpolation='lower')
    if types.is_list_like(lower):
        return df.clip(lower=lower, upper=upper, axis=1)
    else:   # input was Series
        return df.clip(lower=lower, upper=upper)


def unit_root(x, pvalue=0.05, noprint=False):
    """Test if input series has unit root using augmented dickey fuller"""
    dftest = adfuller(x, autolag='AIC')
    if not noprint:
        results = Series(dftest[0:4],
                         index=['Test Statistic','p-value', 'Lags Used',
                                'Obs Used'])
        for k,v in dftest[4].items():
            results[f"Critical Value ({k})"] = v
        print(results.to_frame().T.to_string(index=False))
    return dftest[1] > pvalue

def integration_order(df, noprint=True, max_order=5, pvalue=0.05):
    """Returns order of integration by iteratively testing for unit root"""
    for i in range(max_order):
        if not noprint:
            print(f"Augmented Dickey-Fuller unit root test of I({i}):")
        if not unit_root(df, pvalue=pvalue, noprint=noprint):
            return i
        df = df.diff().dropna()
        
# Covariance Matrix and Minimum Variance Portfolio Volatility
def gmv(cov, realized=None):
    """Compute minimum variance portfolio and realized volatility"""
    w = np.linalg.inv(cov) @ np.ones((cov.shape[1], 1))
    return {'cond': np.linalg.cond(cov), 'vol': None if realized is None
            else  np.std(realized@(w/sum(w)))}

"""
Spot rate: interest rate that makes the PV of zero-coupon bond equal to its price
Forward rate: future spot rate that discounts from a future date to a closer.
   future date, which can be locked in by today's spot rates to those two dates.
Yield to maturity: the average interest rate implied by the price of coupon bond.
Interest rates that involve multi-period cash flows may be annualized and 
  expressed in nominal or effective form -- these differ when cash flows are
  paid intra-year rather than all at once at year-end.
Effective rate: more accurate as it is computed to measure effects of compounding
  at the given frequency within the year.
Nominal rate: interest rate stated at an annual scale but ignores the 
  effect of compounding intra-year hence understates the effective rate.
Annualize interest rates from the number, m, of cash flow periods in a year:
    Effective = (1 + i)**m - 1 => i = (1 + Effective)**(1/m) - 1
    Nominal = (i * m) => i = Nominal / m
    => Effective = [1 + (Nominal / m)]**m - 1
Continuous Compounding: if the annual amount of cash flow is divided continously
    Effective = exp(Nominal) - 1, by taking limits as m -> infinity 
    (Euler gave the name, e, for this mathematical constant, but perhaps could 
     have been called b since it was Bernoulli who first noticed the sequence 
     approaches this limit for more and smaller compounding intervals)
    => Nominal = ln(1 + Effective), aka "Force of Interest"
Accumulation Function(t) = (1 + Effective)**mt, where t is number of years
Discount rate = 1/(1 + Interest Rate), easier form for compounding calculations
  that often take the form of discounting future cash flows to today.
"""
def effective_to_nominal(effective, m=np.inf):
    """convert effective to nominal m-compounded (default is continuous) rates"""
    return np.log(1 + effective) if np.isinf(m) else ((1+effective)**(1/m) - 1)*m

def nominal_to_effective(nominal, m=np.inf):
    """convert nominal m-compounded (default is continuous) to effective rates"""
    return np.exp(nominal) - 1 if np.isinf(m) else (1 + nominal/m)**m - 1

def forward(spot, base=0):
    """forward rate implied by spot starting after base periods"""
    return [(((1 + num)**(n + 1 + base) / (1 + den)**(n + base)) - 1)
            for n, (num, den) in enumerate(zip(spot, [0] + list(spot[:-1])))]
 
def pv(flow, n, spot):
    """PV of cash flow after compounding spot rate over n periods"""
    return flow / ((1 + spot) ** n)

def dcf(flows, spot, first=1):
    """PV of flows, starting at period first, by compounding associated spot"""
    if not types.is_list_like(flows):    # flows can be different each period
        flows = [flows]                  #  else assume same flow every period
    if not types.is_list_like(spot):     # spot rates can be different each flow
        spot = [spot]                    #  else use same spot rate every period
    if len(flows) == 1:
        flows = list(flows) * len(spot)  # repeat flows to be same length as spot
    if len(spot) == 1:
        spot = list(spot) * len(flows)   # repeat spot to be same length as flows
    return np.sum([pv(flow=flow, n=first+n, spot=rate)
                   for n, (flow, rate) in enumerate(zip(flows, spot))])

def bootstrap(ytm, nominal, m):
    """Nominal rate till par bond maturity given ytm and sequence of nominals"""
    n = len(nominal) + 1 # implicit number of coupons, including last at maturity
    return (((1 + ytm/m)/(1 - dcf(ytm/m, [r/m for r in nominal])))**(1/n) - 1)*m

# Can do more fancy actuarial math here...
#   annuity with (1) growth (2) perpetual (3) continuous-compounded rate
def annuity(nominal, n, m, due=0):
    """PV of annuity (due) of n annual flows split by m given nominal rates"""
    return np.sum([pv(flow=1/m, n=p+1-due, spot=nominal/m)
                   for p in range((1 - due) + (n * m))])

# Can do more fancy bond math here...
def weighted_maturity(flows, spot, first=1, returned=False):
    """Average maturity weighted by PV of flows given effective spot rates"""
    v = [pv(flow=flow, n=n+first, spot=rate)
         for n, (flow, rate) in enumerate(zip(flows, spot))]
    return np.average(np.arange(len(v))+first, weights=v, returned=returned)

def duration(nominal, n, coupon=True, face=1, m=1, first=1, concept='Macaulay'):
    """Bond duration given annual coupon, n years, and m-compounded nominal rate

    Notes
    -----
    Only first two letters (in any case) to select duration concept to compute:
    'Macaulay' (default): weighted average maturity of cash flows
    'Modified': percentage derivative of price with respect to yield
    'Dollar': dollar change in price per percentage point change in yield

    Examples
    --------
    # https://en.wikipedia.org/wiki/Bond_duration sanity-check
    print(4.53, duration(nominal=0.065, coupon=0.05, n=5, m=1, concept='mac'))

    print(7.99, duration(nominal=0.05, coupon=0.05, m=2, concept='mac'))
    print(7.79, duration(nominal=0.05, coupon=0.05, n=10, m=2, concept='mod'))
    print(7.79, duration(nominal=0.05, coupon=0.05, n=10, m=2, concept='dol'))

    print(10, duration(nominal=0.05, coupon=0, n=10, m=2, concept='mac'))
    print(9.76, duration(nominal=0.05, coupon=0, n=10, m=2, concept='mod'))
    print(5.95, duration(nominal=0.05, coupon=0, n=10, m=2, concept='dol'))
    """
    if coupon is True:
        coupon = nominal * face  # par bond
    if not coupon:
        coupon = 0    # else zero-coupon bond
    flows = [coupon/m]*(n*m - 1) + [face + coupon/m] 
    d,v = weighted_maturity(flows, [nominal/m]*(n*m), first=first,returned=True)
    d = d / m
    if concept[:2].lower() not in ['ma']:
        d = d / (1 + nominal/m)
        if concept[:2].lower() in ['do', 'dv']:
            d = d * v
    return d

# proportion of failures likelihood test
def kupiecLR(s, n, var):
    """Kupiec LR test (S violations in N trials) of VaR"""
    p = 1 - var        # e.g. var95 is 0.95
    t = n - s          # number of non-violations
    num = np.log(1 - p)*(n - s) + np.log(p)*s
    den = np.log(1 - (s/n))*(n - s) + np.log(s/n)*s
    lr = -2 * (num - den)
    return {'lr': lr, 'pvalue': 1 - scipy.stats.chi2.cdf(lr, df=1)}

def pof(X, pred, var=0.95):
    """Kupiec proportion of failures VaR test"""
    Z = X / pred
    z = scipy.stats.norm.ppf(1 - var)
    r = {'n': len(Z), 's': np.sum(Z < z)}
    r.update(kupiecLR(r['s'], r['n'], var))
    return r

# convert alpha to halflife
from pandas.api import types
def halflife(alpha):
    """Returns halflife from alpha = -ln(2)/ln(lambda), where lambda=1-alpha"""
    if types.is_list_like(alpha):
        return [halflife(a) for a in alpha]
    return -np.log(2)/np.log(1-alpha) if 0<alpha<1 else [np.inf,0][int(alpha>0)]
