"""Evaluate backtests, event studies and risk premiums

- Event studies: cumulative abnormal returns
- Risk premiums: Fama-MacBeth regressions
- Walk-forward portfolio rebalances Backtest: Sharpe ratio, appraisal ratio, ...
- DailyPerformance: Daily returns performance of periodic portfolio holdings

Copyright 2022, Terence Lim

MIT License
"""
import sys
from os.path import dirname, abspath
import time
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
from sqlalchemy import Integer, String, Float, SmallInteger, Boolean, \
    BigInteger, Column, Index
from typing import Dict, Any, Tuple, List
from finds.structured import Structured, Stocks, Benchmarks
from finds.database import SQL
from finds.recipes import least_squares, fft_align, maximum_drawdown
from finds.display import plot_date, plot_bands

_VERBOSE = 1

def to_type(v, t=str):
    """Convert each element in nested list to target type"""
    return [to_type(u, t) for u in v] if types.is_list_like(v) else t(v)

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

class EventStudy(Structured):
    """Class to support statistical tests of event studies

    Args:
      sql: connection to user database to store results
      bench: Benchmark dataset of index returns for market adjustment
      stocks: Stocks structured dataset containing stock returns
      max_date: last date to run event study
      table: name of table in user database to append summary results to
    """
    def __init__(self, sql: SQL, bench: Benchmarks, stocks: Stocks,
                 max_date: int, table: str = 'events'):
        """Initialize for event study calculations"""
        tables = {'events':
                  sql.Table(table,
                            Column('event', String(32), primary_key=True),
                            Column('model', String(32), primary_key=True),
                            Column('beg', Integer),
                            Column('end', Integer),
                            Column('rows', Integer),
                            Column('days', Integer),
                            Column('effective', Float),
                            Column('window', Float),
                            Column('window_t', Float),
                            Column('post', Float),
                            Column('post_t', Float))}
        super().__init__(sql=sql,
                         bd=bench.bd,
                         tables=tables,
                         identifier='event',
                         name='EventStudy')
        self.bench = bench
        self.stocks = stocks
        self.max_date = max_date
        self.summary_ = {}   # to hold summary statistics
        self.plot_ = {}      # to hold plottable series

    def __call__(self, event: str, df: DataFrame, left: int,
                 right: int, post: int, date_field: str) -> DataFrame:
        """Retrieve event window market-adjusted returns where valid/available

        Args:
          event: Unique label for this study
          df:  Input DataFrame of stock identifiers and event dates
          left: Start (inclusive) of announcement window around event date
          right: End (inclusive) of announcement window around event date
          post: End (inclusive) post-drift period
          date_field: Name of date column in df

        Notes:
        - populates 'car' attribute with cumulative arithmetic abnormal returns
        - populates 'bhar' attribute with abnormal cumulative product returns
        """
        ret = self.stocks.get_window(dataset='daily',
                                     field='ret',
                                     permnos=df[self.stocks.identifier],
                                     dates=df[date_field],
                                     date_field='date',
                                     left=left,
                                     right=post)\
                         .rename(columns={self.stocks.identifier: 'permno'})
        cols = list(range(post - left + 1))

        # require at least window and one post-event returns available
        rets = ret[ret[cols[:(right-left+2)]].notna().all(axis=1)]
        rets.index = np.arange(len(rets))

        # get market returns for market-model adjustment
        mkt = self.bench.get_window(dataset='daily',
                                    field='ret',
                                    permnos=['Mkt-RF'] * len(rets),
                                    date_field='date',
                                    dates=rets['date'],
                                    left=left,
                                    right=post)\
                        .rename(columns={self.bench.identifier: 'permno'})
        rf = self.bench.get_window(dataset='daily',
                                   field='ret',
                                   permnos=['RF'] * len(rets),
                                   date_field='date',
                                   dates=rets['date'],
                                   left=left,
                                   right=post)\
                       .rename(columns={self.bench.identifier: 'permno'})
        mkt = (mkt[cols] + rf[cols]).reset_index(drop=True)
        ar = (rets[cols] - mkt[cols]).cumsum(axis=1).fillna(0)
        br = ((1 + rets[cols]).cumprod(axis=1)
              - (1 + mkt[cols]).cumprod(axis=1)).fillna(0)
        self.car = rets[['permno', 'date']].join(ar)
        self.bhar = rets[['permno', 'date']].join(br)
        self.left = left
        self.right = right
        self.post = post
        self.rows = rets[['permno', 'date']]
        self.event = str(event)
        return self.rows

    def write(self) -> int:
        """Store daily cumulative returns in new user table"""
        cols = list(range(self.post - self.left + 1))
        df = [getattr(self, ar).melt(id_vars=['permno', 'date'],
                                     value_vars=cols,
                                     var_name='num',
                                     value_name=ar) for ar in ['car','bhar']]
        df = pd.concat([df[0], df[1].iloc[:, -1]], axis=1)
        df['num'] -= (self.right - self.left)
        table = self.sql.Table(self['events'].key + '_' + str(self.event),
                               Column('permno', Integer),
                               Column('date', Integer),
                               Column('num', Integer),
                               Column('car', Float),
                               Column('bhar', Float))
        return self.load_dataframe(df, table)

    def read(self, left: int, right: int, post: int) -> DataFrame:
        """Fetch daily cumulative abnormal returns series from user table

        Args:
            event: name suffix of table to save in
            left: offset of left announcement window
            right: offet of right announcement window
            post: offset of right post-announcement window

        Returns:
            DataFrame of car and bhar daily cumulative returns
        """
        df = self.read_dataframe(self['events'].key + '_' + str(self.event))
        df['num'] += (right - left)
        self.car = df.pivot(index=['permno', 'date'],
                            columns=['num'],
                            values='car').reset_index()
        self.bhar = df.pivot(index=['permno', 'date'],
                             columns=['num'],
                             values='bhar').reset_index()
        self.left = left
        self.right = right
        self.post = post
        return len(df)

    _models = ['bhar', 'car', 'sbhar', 'scar',
               'adj-sbhar', 'adj-scar', 'conv-sbhar', 'conv-scar']
    
    def fit(self, model: str = 'scar', rows: List[int] = [], 
            rho : float | None = None) -> Dict[str, Dict]:
        """Compute summary statistics from cumulative ar or bhar

        Args:
            model: names of predefined model to compute summary statistics
            rows: Subset of rows to evaluate; empty selects all rows
            car: Whether to evaluate CAR (True) or BHAR (False)
            rho: assumed correlation of event returns.  If None, then compute
                 from max convolution of post-announcement returns

        Returns:
            Dict of summary statistics of full and subsamples

            - 'window', 'window-tvalue' are CAR at end of event window
            - 'post', 'post-tvalue' are CAR from event end till post-drift end
            - 'car', 'car-stderr' are daily CAR from beginning of announcement
            - 'rows' is number of input rows
            - 'days' is number of unique dates (same announce dates are grouped)
            - 'effective' is the number of days after correlation effects

        TODO names of models:

          - "car" or "bhar"
          - unstandardized or 's'tandardized or 'adj-s'tandardized (simple rho)
            or 'conv-s'tandardized (with average rho from convolution max)

        Notes:

        - Kolari and Pynnonen (2010) eqn[3] cross-sectionally adjusted 
          Patell or BMP with average correlation: 
          multiply variance by 1 + (p*(n-1))
        - Kolari, Pape, Pynnonen (2018) eqn[15] adjusted by average overlap 
          and average covariance ratio (i.e. correlation)
        """
        #assert model in self._models
        tic = time.time()
        window = self.right - self.left + 1
        cols = ['date'] + list(range(self.post-self.left+1))
        is_car = model.endswith('car') 
        rets = (self.car if is_car else self.bhar)[cols]
        cumret = (rets.iloc[rows] if len(rows) else rets).copy()
        n = int(len(cumret))
        b = int(min(cumret['date']))
        e = int(max(cumret['date']))
        L = self.post - self.left
        D = self.post - self.right

        # if announce date not a trading day, set to (after close of) previous
        cumret['date'] = self.bd.offset(cumret['date'])

        # portfolio method for same announcement date
        cumret = cumret.groupby('date').mean()

        # Average Cumulative AR
        means = cumret.mean()
        
        # 1. compute the average overlap (truncate at 0) of all pairs
        date_idx = self.bd.date_range(min(cumret.index), max(cumret.index))
        date_idx = Series(index=date_idx, data=np.arange(len(date_idx)))
        date_idx = np.sort(date_idx[cumret.index].values)
        overlap = []
        for k, v in enumerate(date_idx[:-1]):
             x = D - (date_idx[k+1:] - v)
             x[x < 0] = 0
             overlap.extend(x.tolist())
        tau = np.mean(overlap) / D
        
        # 2. compute ratio of average covariance to variance as average max corr
        if rho is None:
            rets = np.log(1+cumret.where(cumret > -0.99, -0.99))\
                     .diff(axis=1)\
                     .iloc[:, window:]\
                     .fillna(0)
            corr = fft_align(rets.values.T, corr=True)
            rho = np.mean(corr)

        # 3. apply simplification of eqn(15) of Kolari et al 2018
        effective = len(cumret) / (1 + (rho * tau * (len(cumret) - 1)))

        #_print('Elapsed', time.time() - tic, 'secs')
        # - unscaled for economic, scaled for statistical
        # - ADJ-BMP MKT method is simple xc t-test of SCAR adjusted by avg corr
        # - table show actual means, but t-values of SCAR with and without corr
        # - plot shows SCAR with corr bands

        stderr = cumret.std() / np.sqrt(effective)
        posterr = (cumret.iloc[:, window:]\
                   .sub(cumret.iloc[:, window-1], axis=0))\
                   .std() / np.sqrt(effective)
        #cumret.iloc[:, window:].std() / np.sqrt(effective)
        
        tstat = means[window - 1] / stderr[window - 1]
        post = cumret.iloc[:, -1] - cumret.iloc[:, window - 1]
        post_sem = post.std() / np.sqrt(effective)
        summary = {model : {'window'    : means[window - 1], 
                            'window_t'  : means[window - 1]/stderr[window - 1],
                            'post'      : post.mean(), 
                            'post_t'    : post.mean() / post_sem,
                            'beg'       : b,
                            'end'       : e,
                            'effective' : int(effective),
                            'days'      : len(cumret),
                            'rows'      : n}}
        self.plot_[model] = {'means'   : means.values,
                             'stderr'  : stderr.values,
                             'posterr' : posterr.values,
                             'car'     : is_car}
        self.summary_.update(summary)
        return summary

    def write_summary(self, overwrite=True) -> DataFrame:
        """Save event study summary to database

        Args:
            overwrite: Whether to overwrite, else append rows

        Returns:
            DataFrame of rows written
        """
        table = self['events']     # summary table to write to
        table.create(checkfirst=True)
        if overwrite:
            delete = table.delete().where(table.c['event'] == self.event)
            self.sql.run(delete)
        summ = DataFrame.from_dict(self.summary_, orient='index')\
                        .reset_index()\
                        .rename(columns={'index': 'model'})
        summ['event'] = self.event
        self.sql.load_dataframe(table.key, summ)
        return summ

    def read_summary(self, event: str = '', model: str = ''):
        """Load event study summary from database

        Args:
            event: Name of event to retrieve; if blank, retrieve all events
            model: Name of model to retrieve; if blank, retrieve all models

        Returns:
            DataFrame of rows written
        """
        s = [f"{k}='{v}'" for k, v in [['event', event], ['model', model]] if v]
        where = bool(s) * ('where ' + ' and '.join(s))
        q = f"SELECT * from {self['events'].key} {where}"
        return self.sql.read_dataframe(q)

    def plot(self, model: str, drift: bool = False,
             ax: Any = None, loc: str = 'best', title: str = '',
             c: str = 'C0', vline: List[float] = [],
             hline: List[float] = [], width: float = 1.96):
        """Plot cumulative abnormal returns, drift and confidence bands

        Args:
            model: Name of model to plot computed results
            drift: Whether to start confidence bands at post-event drift start
            ax: Axes to plot
            loc: Legend location
            title: Main title
            c: Color
            vline: List of x-axis points to plot vertical line
            hline: List of y-axis points to plot horizontal line
            width: Number of std errs for confidence bands
        """
        ax = ax or plt.gca()
        window = self.right - self.left + 1
        if not vline:
            vline = [self.right]
        if not hline:
            hline = [self.plot_[model]['means'][window-1] if drift else 0]
        p = self.plot_[model]      # plottable series for model
        s = self.summary_[model]    # summary stats for model
        plot_bands([0] + list(p['means']),
                   ([0]
                    + ([0] * (window if drift else 0))
                    + list(p['posterr' if drift else 'stderr'])),
                   x=list(range(self.left-1, self.post+1)),
                   loc=loc,
                   hline=hline,
                   vline=vline,
                   title=title,
                   c=c, width=width,
                   legend=["CAR" if p['car'] else "BHAR", f"{width} stderrs"],
                   xlabel=(f"{int(s['beg'])}-{int(s['end'])}"
                           + f" (n={int(s['rows'])},"
                           + f" dates={int(s['days'])},"
                           + f" effective={int(s['effective'])})"),
                   ylabel="CAR" if p['car'] else "BHAR",
                   ax=ax)
        plt.tight_layout(pad=3)
            
            
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

           

class BackTest(Structured):
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
        tables = {'backtests':
                  sql.Table(table,
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
                            Column('ret', Float))}

        super().__init__(sql, bench.bd, tables, 'permno', name='backtests')
        self._verbose = verbose
        self.bench = bench
        self.max_date = max_date
        self.rf = bench.get_series([rf], 'ret', end=max_date)[rf]
        rf = bench.get_series([rf + "(mo)"], 'ret', end=max_date)  # monthly
        self.monthly_ = {(bench.bd.begmo(d), bench.bd.endmo(d)):
                         float(rf.loc[d]) for d in rf.index}
        
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
        self._print(len(pordates), 'dates:', pordates[0], '-', pordates[-1])
        
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
            self._print(f"(backtest) {pordate} {len(curr)} {ret:.4f}")
        self.perf = DataFrame.from_dict(perf, orient='index')
        self.label = label
        self.excess = None
        return perf


    def write(self, label: str):
        """Save backtest performance returns to database"""
        self['backtests'].create(checkfirst=True)
        delete = self['backtests']\
            .delete().where(self['backtests'].c['permno'] == label)
        self.sql.run(delete)
        self.perf['permno'] = label
        self.sql.load_dataframe(self['backtests'].key, self.perf)

    def read(self, label: str = ''):
        """Load backtest performance returns from database"""
        if not label:
            q = (f"SELECT {self.identifier},"
                 f" count(*) as count,"
                 f" min(begret) as begret,"
                 f" max(endret) as endret "
                 f" from {self['backtests'].key} group by {self.identifier}")
            return self.sql.read_dataframe(q).set_index(self.identifier)
        q = (f"SELECT * from {self['backtests'].key}"
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
        return self.sql.pivot(self['backtests'].key,
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
        - 'inforatio': annualized information ratio
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
        r = smf.ols(f"{self.label} ~ {rhs}", data=p).fit()
        r = r.get_robustcov_results(cov_type='HAC', use_t=None, maxlags=haclags)
        pre2002 = p.loc[p.index < 20020101, self.label]
        post2002 = p.loc[p.index >= 20020101, self.label]
        welch = scipy.stats.ttest_ind(post2002, pre2002, equal_var=False)
        mult = (len(p) - 1) / nyears
        self.annualized = {
            'excess': mult * np.mean(p[self.label]),
            'sharpe': np.sqrt(mult)*p[self.label].mean() / p[self.label].std(),
            'alpha': mult * r.params[0],
            'inforatio': np.sqrt(mult) * r.params[0] / np.std(r.resid),
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
             drawdown: bool = False, figsize: Tuple[float, float] = (10, 5)):
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
                  label1='cumulative ret',
                  marker=None,
                  ax=ax1,
                  points=(maximum_drawdown(self.perf['excess'])
                          if drawdown else None))
        plot_date(y1=perf[['longs','shorts']],
                  y2=(perf['buys'] + perf['sells']) / 4, 
                  ax=ax2,
                  marker=None,
                  ls1='-',
                  ls2=':',
                  cn=excess.shape[1],
                  label1='number of holdings',
                  label2='turnover',
                  legend2=['turnover'])
        plt.tight_layout(pad=3)


class DailyPerformance:
    """Computing daily realized returns on periodic holdings
    
    Args:
      stocks: Stocks returns dataset
    """
    
    def __init__(self, stocks: Stocks):
        self.stocks = stocks
        
    def __call__(self, holdings: Dict[int, Series], end: int) -> Series:
        """Return series of daily returns through end date

        Args:
          holdings: dict (key is int date) of holdings Series (index is permno)
          end: Last date of daily returns to compute performance for

        Returns:
          Series of daily realized portfolio returns
        """
        rebals = sorted(holdings.keys())   # rebalance dates, including initial
        dates = self.stocks.bd.date_range(rebals[0], end)[1:] # return dates
        curr = holdings[rebals[0]]         # initial portfolio
        perf = dict()                      # collect daily performance
        for date in dates[1:]:   # loop over return dates
            ret = self.stocks.get_section(dataset='daily',
                                          fields=['ret','retx'],
                                          date_field='date',
                                          date=date)
            perf[date] = sum(curr*ret['ret'].reindex(curr.index, fill_value=0))
            if date in rebals:   # update daily portfolio holdings
                curr = holdings[date]
            else:
                curr = curr * (1 + ret['retx'].reindex(curr.index).fillna(0))
        return Series(perf, name='ret')

if __name__=="__main__":
    from conf import credentials, VERBOSE
    
