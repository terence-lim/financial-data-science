"""Evaluate backtests, event studies and risk premiums

- Walk-forward portfolio rebalances Backtest: Sharpe ratio, appraisal ratio, ...

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
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy.builtins import Q
from sqlalchemy import Integer, String, Float, Boolean, Column, Index
from typing import Dict, Any, Tuple, List
from finds.structured.stocks import Stocks
from finds.structured.benchmarks import Benchmarks
from finds.database.sql import SQL
from finds.finance import maximum_drawdown
from finds.plots import plot_date, plot_bands
from .backtesting import compound_ret

_VERBOSE = 1

def to_type(v, t=str):
    """Convert each element in nested list to target type"""
    return [to_type(u, t) for u in v] if types.is_list_like(v) else t(v)

class BackTest:
    """Base class for computing portfolio backtest returns

    Args:
      sql: Database connection to store results
      bench: Structured dataset to retrieve riskfree and benchmark returns
      rf: Column name of riskfree rate in benchmark dataset
      max_date: Last date of any backtest
      table: Name of table in user database to store results in

    Notes:
      - If backtest dates appears to be monthly 
        frequency, monthly risk free rates will be retrieved and used
        rather than compounding from daily (reduce precision errors).
        Assumes that monthly risk free rates also available through {bench}, 
        with name suffixed by "(mo)". 

    Examples:

    >>> backtest = BackTest(user, bench, 'RF', 20200930)
    """

    # schema of the table to store backtested performance returns
    def __init__(self, sql: SQL, bench: Benchmarks, rf: Series, max_date: int,
                 table: str = 'backtests', verbose: int = _VERBOSE):
        """Initialize class to evaluate backtest performance"""
        table = sql.Table(table,
                          Column('permno', String(32), primary_key=True),
                          Column('begret', Integer),
                          Column('endret', Integer, primary_key=True),
                          Column('longs', Integer),
                          Column('shorts', Integer),
                          Column('buys', Float),
                          Column('sells', Float),
                          Column('long_weight', Float),
                          Column('short_weight', Float),
                          Column('excess', Float),
                          Column('ret', Float))
        self.bd = bench.bd
        self.sql = sql
        self.table_ = table
        self.identifier = 'permno'
        self.name = 'backtest'
        self._verbose = verbose
        self.bench = bench
        self.max_date = max_date
        self.rf = bench.get_series([rf], 'ret', end=max_date)[rf]
        rf = bench.get_series([rf + "(mo)"], 'ret', end=max_date)  # monthly
        self.monthly_ = {(bench.bd.begmo(d), bench.bd.endmo(d)): float(v)
                         for d, v in rf.iloc[:,0].items()}
        
        self.annualized = {} # collect annualized backtest statistics
        self.perf   = None   # raw performance before attribution
        self.excess = None   # with excess returns after attribution
        self.label  = None   # label name

    def __call__(self, stocks: Stocks,
                       holdings: Dict[int, Series],
                       label: str,
                       overlap: int = 0) -> DataFrame:
        """Compute holding returns and rebalance statistics
        
        Args:
          stocks: Structured data set with stocks data
          holdings: Sequence of holdings keyed by rebalance date
            Each Series is indexed by permno, with weights in column
            Last item (can be empty) is dropped for calculating returns
          label: Name to save this backtest
          overlap: Number of months to smooth overlapping holdings

        Returns:
          DataFrame of holdings returns after every rebalance data

        Notes:
          - if CRSP (i.e. 'delist' table exists and using monthly), 
            include dlst returns
        """
        for d, h in holdings.items():
            if not h.index.is_unique:
                raise ValueError(f"duplicate holdings index date={d}")

        pordates = sorted(list(holdings.keys()))
        if self._verbose:
            print(len(pordates), 'dates:', pordates[0], '-', pordates[-1])
        
        perf = {}                    # accum performance each period
        smooth = []                  # to queue rolling holdings
        prev = Series(dtype=float)   # prior holdings, to be adjusted by retx
        
        holding_periods = stocks.bd.date_tuples(pordates)
        for pordate, (begret, endret) in zip(pordates[:-1], holding_periods):

            if (begret, endret) in self.monthly_:
                riskfree = self.monthly_[(begret, endret)]
            else:
                riskfree = compound_ret(self.rf, (begret, endret))
        
            # insert current holdings into smooth
            if len(smooth) > overlap:  # smooth has list of recent holdings
                smooth.pop()
            smooth.insert(0, holdings[pordate].copy())

            # compute rolling weights, after combining union of permnos in curr
            permnos = sorted(set(np.ravel([list(p.index) for p in smooth])))
            curr = Series(index=permnos, data=[0] * len(permnos), dtype=float)
            for weights in smooth:      # assign final smoothed weight
                curr[weights.index] += weights / len(smooth)

            # compute portfolio return this month
            ret = sum(stocks.get_ret(begret, endret, delist=True)\
                      .reindex(curr.index, fill_value=0) * curr)
            
            # compute turnover
            delta = pd.concat((prev, curr), axis=1, join='outer').fillna(0)
            delta = delta.iloc[:, 1] - delta.iloc[:, 0]  # change in holdings

            # update this month's performance
            perf[int(endret)] = {'begret': int(begret),
                                 'endret': int(endret),
                                 'longs': sum(curr > 0),
                                 'shorts': sum(curr < 0),
                                 'long_weight': curr[curr > 0].sum(),
                                 'short_weight': curr[curr < 0].sum(),
                                 'ret': ret, 
                                 'excess': ret - (curr.sum() * riskfree),
                                 'buys': delta[delta>0].abs().sum(),
                                 'sells': delta[delta<0].abs().sum()}

            # adjust stock weights by retx till end of holding period
            retx = stocks.get_ret(begret, endret, field='retx')
            prev = curr * (1 + retx.reindex(curr.index)).fillna(1)
            for i in range(len(smooth)):
                smooth[i] *= (1 + retx.reindex(smooth[i].index)).fillna(1)
                if self._verbose:
                    print(f"(backtest) {pordate} {len(curr)} {ret:.4f}")
        self.perf = DataFrame.from_dict(perf, orient='index')
        self.label = label
        self.excess = None
        return perf

    def write(self, label: str):
        """Save backtest performance returns to database"""
        self.table_.create(self.sql.engine, checkfirst=True)
        delete = self.table_.delete()\
                            .where(self.table_.c['permno'] == label)
        self.sql.run(delete)
        self.perf['permno'] = label
        self.sql.load_dataframe(self.table_.key, self.perf)

    def read(self, label: str = ''):
        """Load backtest performance returns from database"""
        if not label:
            q = (f"SELECT {self.identifier},"
                 f" count(*) as count,"
                 f" min(begret) as begret,"
                 f" max(endret) as endret "
                 f" from {self.table_.key} group by {self.identifier}")
            return self.sql.read_dataframe(q).set_index(self.identifier)
        q = (f"SELECT * from {self.table_.key}"
             f"  where {self.identifier} = '{label}'")
        self.perf = self.sql.read_dataframe(q)\
                            .sort_values(['endret'])\
                            .set_index('endret', drop=False)\
                            .drop(columns=['permno'])
        self.label = label
        self.excess = None
        return self.perf

    def get_series(self, field: str = 'ret', start: int = 19000000,
                   end: int = 0) -> DataFrame:
        """Retrieve saved backtest as a series"""
        return self.sql.pivot(self.table_.key,
                              index='endret',
                              columns='permno',
                              values=field,
                              where=(f"endret >= {start}"
                                     f" AND endret <= {end or self.max_date} "
                                     f" AND permno = {self.label}"))\
                       .rename(columns={'endret': 'date'})

    def fit(self, benchnames: List[str],
                  beg: int = 0,
                  end: int = 0,
                  haclags: int = 1) -> DataFrame:
        """Compute performance attribution against benchmarks 

        Args:
          benchnames: Names of benchmark returns to compute attribution against
          beg, end: date range of returns
          haclags: number of Newey-West lags for robustcov statistics

        Returns:
          DataFrame of excess returns performance following each rebalance date

        Annualized (dict) performance ratios:

        - 'excess': annualized excess (of portfolio weight*riskfree) return
        - 'sharpe': annualized sharpe ratio
        - 'alpha': annualized alpha
        - 'appraisal': annualized appraisal ratio
        - 'welch-t': t-stat for structural break after 2002
        - 'welch-p': p-value for structural break after 2002
        - 'turnover': annualized total turnover rate
        - 'buys': annualized buy rate
        - 'sells': annualized sell rate
        """
        # collect performance between beg and end dates
        end = end or self.max_date
        d = self.perf.loc[beg:end].index
        nyears = len(self.rf.loc[d[0]:d[-1]]) / 252
        p = self.perf.loc[d, 'excess'].rename(self.label).to_frame()

        # collect benchmark returns
        df = self.bench.get_series(benchnames, 'ret', end=self.max_date)
        retdates = to_type(self.perf.loc[d, ['begret','endret']].values, int)
        for b in benchnames:
            p[b] = compound_ret(df[b], retdates)

        # compute time-series regression results
        rhs = ' + '.join([f"Q('{b}')" for b in benchnames])
        r = smf.ols(f"Q('{self.label}') ~ {rhs}", data=p).fit()
        r = r.get_robustcov_results(cov_type='HAC', use_t=None, maxlags=haclags)
        pre2002 = p.loc[p.index < 20020101, self.label]
        post2002 = p.loc[p.index >= 20020101, self.label]
        welch = scipy.stats.ttest_ind(post2002, pre2002, equal_var=False)
        mult = (len(p) - 1) / nyears
        self.annualized = {
            'excess': mult * np.mean(p[self.label]),
            'sharpe': np.sqrt(mult)*p[self.label].mean() / p[self.label].std(),
            'alpha': mult * r.params[0],
            'appraisal': np.sqrt(mult) * r.params[0] / np.std(r.resid),
            'welch-t': welch[0],
            'welch-p': welch[1],
            'turnover': np.mean(self.perf.loc[d, ['buys','sells']]\
                                .abs().values) * mult / 2,
            'longs': self.perf.loc[d, 'longs'].mean(),
            'shorts': self.perf.loc[d, 'shorts'].mean(),
            'buys': mult * self.perf.loc[d, 'buys'].mean() / 2,
            'sells': mult * self.perf.loc[d, 'sells'].mean() / 2}
        self.results = r
        self.excess = p.rename(columns={self.label: 'excess'})
        return self.excess

    def plot(self, label: str = '', num: int = 1, flip: bool | None = False,
             drawdown: bool = False, figsize: Tuple[float, float] = (10, 5),
             marker: str | None = '', fontsize: int = 9):
        """Plot time series of excess vs benchmark returns

        Args:
          num: Figure number to use in plt
          label: legend label
          flip: Whether to flip returns. If None, then auto-detect
          drawdown: To plot peak and trough points of maximum drawdown
        """
        label = label or self.label
        if flip:
            label = 'MINUS ' + label
            m = -1
        else:
            m = 1
        if self.excess is None:   # attribution was not run
            excess = m * self.perf[['excess']].rename(columns={'excess': label})
            perf = self.perf
        else:
            excess = self.excess.rename(columns={'excess': label})
            excess[label] *= m
            perf = self.perf[(self.perf.index >= self.excess.index[0]) &
                             (self.perf.index <= self.excess.index[-1])]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, clear=True,
                                       figsize=figsize, num=num)
        plot_date(y1=excess.cumsum(),
                  ylabel1='cumulative ret',
                  marker=marker,
                  ax=ax1,
                  fontsize=fontsize,
                  points=(maximum_drawdown(self.perf['excess'])
                          if drawdown else None))
        plot_date(y1=perf[['longs','shorts']],
                  y2=(perf['buys'] + perf['sells']) / 4, 
                  ax=ax2,
                  marker=marker,
                  fontsize=fontsize,
                  ls=':',
                  cn=excess.shape[1],
                  ylabel1='number of holdings',
                  ylabel2='turnover',
                  legend2=['turnover'])
        plt.tight_layout(pad=3)


