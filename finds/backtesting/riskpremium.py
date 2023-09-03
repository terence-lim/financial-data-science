"""Evaluate risk premiums from cross-sectional regressions

Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
import scipy
from matplotlib import dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types
from sqlalchemy import Integer, String, Float, Column, Index
from typing import Dict, Any, Tuple, List
from finds.structured.stocks import Stocks
from finds.structured.benchmarks import Benchmarks
from finds.database.sql import SQL
from finds.plots import plot_date
from finds.econs import least_squares
from .backtesting import compound_ret
_VERBOSE = 1

class RiskPremium:
    """Compute and test of factor loading risk premiums

    Args:
      sql: Connection to user database to store results
      bench: Benchmark dataset of market and indexreturns
      rf: Name of riskfree rate from bench database
      end: Last date of price and returns data
    """
    def __init__(self, sql: SQL, bench: Benchmarks, rf: str, end: int):
        """Initialize for testing factor loading premiums"""
        self.sql = sql
        self.bench = bench
        self.rf = bench.get_series([rf], 'ret', end=end)[rf]
        rf = bench.get_series([rf + "(mo)"], 'ret', end=end)  # monthly riskfree
        self.monthly_ = {(bench.bd.begmo(d), bench.bd.endmo(d)):
                         float(rf.loc[d]) for d in rf.index}
        self.end_ = end

    def __call__(self, stocks: Stocks, loadings: Dict[int, DataFrame],
                 weights: str = "", standardize: List[str] = []) -> Series:
        """Estimate factor risk premiums with cross-sectional FM regressions

        Args:
          stocks: Stocks' returns data
          loadings: dict keyed by rebalance date of loadings DataFrames
          standardize: List of columns to demean and rescale (eql-wtd std = 1)
          weights: List of weights for weighted least squares and demean

        Returns:
          Series of means and stderrs of FM cross-sectional regression
        """
        pordates = sorted(list(loadings.keys()))
        self.holdrets = stocks.bd.date_tuples(pordates)
        out = []
        for pordate, holdrets in zip(pordates[:-1], self.holdrets):
            if holdrets in self.monthly_: 
                rf = self.monthly_[holdrets]
            else:
                rf = compound_ret(self.rf, holdrets)
            df = loadings[pordate]
            if not weights:
                w = np.ones(len(df))
            else:
                w = df[weights].to_numpy()
                df = df.drop(columns=[weights])
            x = df.columns
            for col in standardize: # weighted mean <- 0, equal wtd stdev <- 1
                df[col] -= np.average(df[col], weights=w)
                df[col] /= np.std(df[col])
            df = df.join(stocks.get_ret(*holdrets, delist=True)-rf, how='left')
            p = least_squares(df.dropna(),
                              x=x,
                              y=['ret'],
                              add_constant=False)
            p.name = holdrets[1]
            out.append(p)
        self.perf = pd.concat(out, axis=1).T
        self.results = {'mean': self.perf.mean(),
                        'stderr': self.perf.sem(),
                        'std': self.perf.std(),
                        'count': len(self.perf)}
        return DataFrame(self.results)

    def fit(self, benchnames: List[str] = []) -> List[DataFrame]:
        """Compute risk premiums and benchmark correlations"""
        out = []
        corr = self.perf.corr()
        out.append(DataFrame(self.results)\
                   .rename_axis('Factor Returns', axis=1))
        out.append(corr.loc[self.perf.columns, self.perf.columns]\
                   .rename_axis('Correlation of Factor Returns:',
                                axis=1))
        if benchnames:
            df = self.bench.get_series(benchnames, 'ret')
            b = DataFrame({k: compound_ret(df[k], self.holdrets)
                           for k in benchnames},
                          index=self.perf.index)
            out.append(DataFrame({'mean': b.mean(),
                                  'stderr': b.sem(),
                                  'std': b.std(),
                                  'count': len(b)})\
                       .rename_axis('Benchmarks', axis=1))
            corr = b.join(self.perf).corr()
            out.append(corr.loc[benchnames, benchnames]\
                       .rename_axis('Correlation of Benchmarks', axis=1))
            out.append(corr.loc[self.perf.columns, benchnames]\
                       .rename_axis('Correlation of Factor Returns'
                                    ' and Benchmarks', axis=1))
        return out
    
    def plot(self, factors: List[str] = [], num: int = 1, fontsize: int = 8,
             figsize: Tuple[float, float] = (10, 5)):
        """Plot computed time series of factor returns"""
        if not factors:
            factors = list(self.perf.columns)
        if isinstance(factors, str):
            factors = [factors]
        b = dict()
        if isinstance(factors, dict):
            df = self.bench.get_series(factors.values(), 'ret')
            for k,v in factors.items():
                b[k] = Series(compound_ret(df[v], self.holdrets),
                              index=self.perf.index,
                              name=v)
            factors = list(factors.keys())
        nrow = int(np.ceil(np.sqrt(len(factors))))
        ncol = int(np.ceil(len(factors) / nrow))
        fig, axes = plt.subplots(nrow, ncol, clear=True, num=num,
                                 squeeze=False, figsize=figsize)
        for i, (ax, col) in enumerate(zip(np.ravel(axes), factors)):
            if len(b):
                plot_date(y1=self.perf[col].cumsum(),
                          legend1=[col],
                          y2=b[col].cumsum(),
                          legend2=[b[col].name],
                          cn=i*2, 
                          loc1='upper left',
                          loc2='lower right',
                          fontsize=fontsize,
                          ax=ax)
            else:
                plot_date(y1=self.perf[col].cumsum(),
                          ax=ax,
                          cn=i,
                          fontsize=fontsize,
                          legend1=[col])
        plt.tight_layout(pad=3)

if __name__=="__main__":
    from env.conf import credentials
    
