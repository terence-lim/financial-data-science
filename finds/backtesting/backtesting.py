"""Utilities for backtesting

Copyright 2022, Terence Lim

MIT License
"""
from typing import List, Dict, Tuple, Any, Iterable
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from finds.structured.stocks import Stocks
from finds.structured.signals import Signals

# construct spread portfolio weights
def univariate_sorts(stocks: Stocks,
                     label: str,
                     signals: Signals,
                     rebalbeg: int,
                     rebalend: int, 
                     window: int = 0,
                     months: List[int] = [],
                     pct: Tuple[float, float] = (20., 80.),
                     leverage: float = 1.,
                     minobs: int = 100,
                     minprc: int = 0,
                     mincap: int = 0,
                     maxdecile: int = 10) -> Dict[int, Series]:
    """Monthly series of cap-weighted holdings of univariate spread portfolios

    Args:
        stocks: Stocks object for accessing stock returns and price data
        label: Name of signal to retrieve
        signals: Call to extract cross section of values for the signal
        rebalbeg: First rebalance date (YYYYMMDD)
        rebalend: Last holding date (YYYYMMDD)
        pct: Percentile breakpoints to sort high, medium and low buckets
        window: No. of months to look back for signal values; 0 is exact month
        months: Months (e.g. 6=June) to retrieve univ; empty for all months
        maxdecile: Include largest stocks decile from 1 through maxdecile
        mincap: Minimum market cap
        minobs: Minimum required sample size with non-missing signal values
        leverage: Multiplier for leverage or shorting
    """
    rebaldates = stocks.bd.date_range(rebalbeg, rebalend, 'endmo')
    holdings = dict()
    for rebaldate in rebaldates:

        # check if this is a rebalance month
        if not months or (rebaldate//100)%100 in months or not holdings:

            # rebalance: get this month's universe
            df = stocks.get_universe(rebaldate)

            # get signal values within lagged window
            if window:  # lookback window to get signal values
                start = stocks.bd.endmo(rebaldate, months=-abs(window))
            else:       # no window, so signal value as of exact rebaldate
                start = stocks.bd.offset(rebaldate, offsets=-1)
            signal = signals(label=label, date=rebaldate, start=start)
            df[label] = signal[label].reindex(df.index)
            
            df = df[df['prc'].abs().gt(minprc)
                    & df['cap'].gt(mincap)
                    & df['decile'].le(maxdecile)].dropna()
            if (len(df) < minobs):  # skip if insufficient observations
                continue

            # split signal into desired fractiles            
            df['fractile'] = fractiles(df[label],
                                       pct=pct,
                                       keys=df[label][df['nyse']])
            subs = {'H' : (df['fractile'] == 1),
                    'M' : (df['fractile'] == 2),
                    'L' : (df['fractile'] == 3)}
            weights = dict()
            for subname, weight in zip(['H', 'L'],
                                       [leverage, -leverage]):
                cap = df.loc[subs[subname], 'cap']
                weights[subname] = weight * cap / cap.sum()
            #print("(portfolio_sorts)", rebaldate, len(df))
        else:   # if not rebalance, then adjust previous stock weights by retx
            retx = stocks.get_ret(stocks.bd.begmo(rebaldate),
                                  rebaldate,
                                  field='retx') + 1
            for port, old in weights.items():
                new = old * retx.reindex(old.index, fill_value=1)
                weights[port] = new / (abs(new.sum()) * len(weights) / 2)
        holdings[rebaldate] = pd.concat(list(weights.values()), axis=0)
    return holdings


    
def bivariate_sorts(stocks: Stocks, 
                    label: str, 
                    signals: Signals, 
                    rebalbeg: int, 
                    rebalend: int,
                    window: int = 0, 
                    pct: Tuple[float, float] = (30., 70.), 
                    leverage: float = 1.,
                    months: List[int] = [], 
                    minobs: int = 100, 
                    minprc: float = 0., 
                    mincap: float = 0., 
                    maxdecile: int = 10) -> Tuple[Dict, Dict, Dict]:
    """Generate monthly time series of holdings by two-way sort procedure

    Args:
        stocks: Stocks object for accessing stock returns and price data
        label: Name of signal to retrieve
        signals: Call to extract cross section of values for the signal
        rebalbeg: First rebalance date (YYYYMMDD)
        rebalend: Last holding date (YYYYMMDD)
        pct: Percentile breakpoints to sort high, medium and low buckets
        window: No. of months to look back for signal values; 0 is exact month
        months: Months (e.g. 6=June) to retrieve univ; empty for all months
        maxdecile: Include largest stocks decile from 1 through maxdecile
        mincap: Minimum market cap
        minobs: Minimum required sample size with non-missing signal values
        leverage: Multiplier for leverage or shorting

    Returns
       3-tuple of spread holdings, smb holdings, and subportfolio sizes

    Notes:

    - Independent sort by median (NYSE) mkt cap and 30/70 (NYSE) HML percentiles
    - Subportfolios of the intersections are value-weighted; 
    - Spread portfolios are equal-weighted of subportfolios
    - Portfolio are resorted every June; and other months' holdings are 
      adjusted by monthly realized retx (i.e. dividends not reinvested)
    """
    rebaldates = stocks.bd.date_range(rebalbeg, rebalend, 'endmo')
    holdings = {label: dict(), 'smb': dict()}  # to return two sets of holdings
    sizes = {h : dict() for h in ['HB','HS','MB','MS','LB','LS']}
    for rebaldate in rebaldates:  #[:-1]

        # check if this is a rebalance month
        if not months or (rebaldate//100)%100 in months or not holdings[label]:
            
            # rebalance: get this month's universe of stocks with valid data
            df = stocks.get_universe(rebaldate)
            
            # get signal values within lagged window
            if window:
                start = stocks.bd.endmo(rebaldate, months=-abs(window))
            else:
                start = stocks.bd.offset(rebaldate, offsets=-1)
            signal = signals(label=label,
                             date=rebaldate,
                             start=start)
            df[label] = signal[label].reindex(df.index)

            df = df[df['prc'].abs().gt(minprc)
                    & df['cap'].gt(mincap)
                    & df['decile'].le(maxdecile)].dropna()
            if (len(df) < minobs):  # skip if insufficient observations
                continue

            # split signal into desired fractiles, and assign to subportfolios
            df['fractile'] = fractiles(df[label],
                                       pct=pct,
                                       keys=df[label][df['nyse']],
                                       ascending=False)
            subs = {'HB' : (df['fractile'] == 1) & (df['decile'] <= 5),
                    'MB' : (df['fractile'] == 2) & (df['decile'] <= 5),
                    'LB' : (df['fractile'] == 3) & (df['decile'] <= 5),
                    'HS' : (df['fractile'] == 1) & (df['decile'] > 5),
                    'MS' : (df['fractile'] == 2) & (df['decile'] > 5),
                    'LS' : (df['fractile'] == 3) & (df['decile'] > 5)}
            weights = {label: dict(), 'smb': dict()}
            for subname, weight in zip(['HB','HS','LB','LS'],
                                       [0.5, 0.5, -0.5, -0.5]):
                cap = df.loc[subs[subname], 'cap']
                weights[label][subname] = leverage * weight * cap / cap.sum()
                sizes[subname][rebaldate] = sum(subs[subname])
            for subname, weight in zip(['HB','HS','MB','MS','LB','LS'],
                                       [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]):
                cap = df.loc[subs[subname], 'cap']
                weights['smb'][subname] = leverage * weight * cap / cap.sum()
                sizes[subname][rebaldate] = sum(subs[subname])
            #print("(famafrench_sorts)", rebaldate, len(df))
            
        else:  # else not a rebalance month, so adjust holdings by retx
            retx = 1 + stocks.get_ret(stocks.bd.begmo(rebaldate),
                                      rebaldate,
                                      field='retx')
            for port, subports in weights.items():
                for subport, old in subports.items():
                    new = old * retx.reindex(old.index, fill_value=1)
                    weights[port][subport] = new / (abs(np.sum(new))
                                                    * len(subports) / 2)

        # combine this month's subportfolios
        for h in holdings:
            holdings[h][rebaldate] = pd.concat(list(weights[h].values()))
    return holdings[label], holdings['smb'], sizes

    
def compound_ret(rets: Series, intervals: Tuple | List[Tuple]) -> List[float]:
    """Compounds series of returns between (list of) date tuples (inclusive)"""
    if len(intervals)==0:
        return []
    elif len(intervals)==1:
        return [compound_ret(rets, intervals[0])]
    elif len(intervals)==2 and isinstance(intervals[0], int):  # a single tuple
        d = rets.index
        return np.prod(rets[(d >= intervals[0]) & (d <= intervals[1])] + 1) - 1
    else:      # list of date tuples: recursively evaluate each tuple
        return [compound_ret(rets, interval) for interval in intervals]


def fractiles(values: Iterable, pct: Iterable, keys: Iterable | None = None, 
              ascending: bool = False) -> List[int]:
    """Sort and assign values into fractiles

    Args:
        values: input array to assign to fractiles
        pct: list of percentiles between 0 and 100
        keys: key values to determine breakpoints, use values if None
        ascending: if True, assign to fractiles in ascending order
    
    Returns:
        list of fractile assignments
    """
    if keys is None:
        keys = values
    keys = np.array(keys)[~np.isnan(keys)]  # drop nan
    bp = list(np.percentile(keys, sorted(pct))) + [np.inf]
    if ascending:
        return 1 + np.searchsorted(bp, values, side='left')
    else:
        return 1 + len(pct) - np.searchsorted(bp, values, side='left')
