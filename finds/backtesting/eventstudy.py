"""Evaluate event studies

- Event studies: cumulative abnormal returns

Copyright 2022, Terence Lim

MIT License
"""
import sys
from os.path import dirname, abspath
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
from sqlalchemy import Integer, String, Float, Boolean, Column
from typing import Dict, Any, Tuple, List
from finds.structured.structured import Structured
from finds.structured.stocks import Stocks
from finds.structured.benchmarks import Benchmarks
from finds.database.sql import SQL
from finds.recipes.filters import fft_align
from finds.utils.plots import plot_date, plot_bands

_VERBOSE = 1

class EventStudy(Structured):
    """Class to support statistical tests of event studies

    Args:
      sql: connection to user database to store results
      bench: Benchmark dataset of index returns for market adjustment
      stocks: Stocks structured dataset containing stock returns
      max_date: last date to run event study
      table: physical name of table in user database to append summary results
    """
    def __init__(self, sql: SQL, bench: Benchmarks, stocks: Stocks,
                 max_date: int, table: str = 'events'):
        """Initialize class instance for event study calculations"""
        table = sql.Table(table,
                          Column('label', String(32), primary_key=True),
                          Column('model', String(32), primary_key=True),
                          Column('beg', Integer),
                          Column('end', Integer),
                          Column('rows', Integer),
                          Column('days', Integer),
                          Column('rho', Float),
                          Column('tau', Float),
                          Column('effective', Float),
                          Column('window', Float),
                          Column('window_t', Float),
                          Column('post', Float),
                          Column('post_t', Float),
                          Column('created', Integer))
        self.bd = bench.bd
        self.sql = sql
        self.table_ = table
        self.identifier = 'label'
        self.name = 'eventstudy'
        self.bench = bench
        self.stocks = stocks
        self.max_date = max_date
        self.summary_ = {}   # to hold summary statistics
        self.plot_ = {}      # to hold plottable series

    def __call__(self, label: str, df: DataFrame, left: int,
                 right: int, post: int, date_field: str) -> DataFrame:
        """Construct event window market-adjusted returns where valid/available

        Args:
          label: Unique label (used for graph labels and table name)
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
        self.label = str(label)
        return self.rows

    def _event_key(self, label: str) -> str:
        """Construct name of table to read/write event returns, given label"""
        return self.table_.key + '_' + str(self.label),
    
    def write(self) -> int:
        """Store daily cumulative returns in new user table"""
        cols = list(range(self.post - self.left + 1))
        df = [getattr(self, ar).melt(id_vars=['permno', 'date'],
                                     value_vars=cols,
                                     var_name='num',
                                     value_name=ar) for ar in ['car','bhar']]
        df = pd.concat([df[0], df[1].iloc[:, -1]], axis=1)
        df['num'] -= (self.right - self.left)
        table = self.sql.Table(self._event_key(self.label),
                               Column('permno', Integer),
                               Column('date', Integer),
                               Column('num', Integer),
                               Column('car', Float),
                               Column('bhar', Float))
        return self.load_dataframe(df, table)

    def read(self, left: int, right: int, post: int) -> DataFrame:
        """Fetch daily cumulative abnormal returns series from user table

        Args:
            label: name suffix of table to retrieve from
            left: offset of left announcement window
            right: offet of right announcement window
            post: offset of right post-announcement window

        Returns:
            DataFrame of car and bhar daily cumulative returns
        """
        df = self.read_dataframe(self._event_key(self.label),)
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

    _models = ['sbhar', 'scar']
    # 'bhar', 'car', 'adj-sbhar', 'adj-scar', 'conv-sbhar', 'conv-scar']
    
    def fit(self, model: str = 'scar', rows: List[int] = [], 
            rho : float | None = None) -> Dict[str, Dict]:
        """Compute car, or bhar, and summary statistics of selected rows

        Args:
          model : name of predefined model to compute summary statistics
          rows : Subset of rows to evaluate; empty list selects all rows
          car : Whether to evaluate CAR (True) or BHAR (False)
          rho : Average cross correlation of event returns.  If None, then 
                 compute from max convolution of post-announcement returns

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
          Patell or BMP with avg correlation: multiply variance by 1 + (p*(n-1))
        - Kolari, Pape, Pynnonen (2018) eqn[15] adjusted by average overlap (tau)
          and average covariance ratio (correlation rho)
        """
        #assert model in self._models
        window = self.right - self.left + 1
        cols = ['date'] + list(range(self.post-self.left+1))
        is_car = model.endswith('car')
        rets = (self.car if is_car else self.bhar)[cols]
        cumret = (rets.iloc[rows] if len(rows) else rets).copy()
        n = int(len(cumret))
        b = int(min(cumret['date']))
        e = int(max(cumret['date']))
        L = self.post - self.left    # total period including announcement and post
        D = self.post - self.right   # length of post-announcement period

        # if announce date not a trading day, set to (after close of) previous
        cumret['date'] = self.bd.offset(cumret['date'])

        # portfolio method for same announcement dates
        cumret = cumret.groupby('date').mean()

        # Average Cumulative AR
        means = cumret.mean()

        # 1. compute the average overlap (truncate at 0) of all pairs
        date_idx = self.bd.date_range(min(cumret.index), max(cumret.index))
        date_idx = Series(index=date_idx, data=np.arange(len(date_idx)))
        date_idx = np.sort(date_idx[cumret.index].values)
        overlap = []
        for k, v in enumerate(date_idx[:-1]):
             x = D - (date_idx[k+1:] - v)  # difference in dates less than D
             x[x < 0] = 0                  # truncate "negative" overlaps
             overlap.extend(x.tolist())
        tau = np.mean(overlap) / D   # average of overlap days, divided by max length

        # 2. compute ratio of average covariance to variance as average max corr
        if rho is None:
            rets = np.log(1+cumret.where(cumret > -0.99, -0.99))\
                     .diff(axis=1)\
                     .iloc[:, window:]\
                     .fillna(0)
            corr, disp, cols = fft_align(rets.values.T)
            rho = np.nanmean(corr)

        # 3. apply simplification of eqn(15) of Kolari et al 2018
        effective = len(cumret) / (1 + (rho * tau * (len(cumret) - 1)))

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
                            'rho'       : rho,
                            'tau'       : tau,
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
        self.table_.create(self.sql.engine, checkfirst=True)
        if overwrite:
            delete = self.table_.delete()\
                                .where(self.table_.c['label'] == self.label)
            self.sql.run(delete)
        summ = DataFrame.from_dict(self.summary_, orient='index')\
                        .reset_index()\
                        .rename(columns={'index': 'model'})
        summ['label'] = self.label
        summ['created'] = self.bd.today()
        self.sql.load_dataframe(self.table_.key, summ)
        return summ

    def read_summary(self, label: str = '', model: str = ''):
        """Load event study summary from database

        Args:
            label: Name of event to retrieve; if blank, retrieve all events
            model: Name of model to retrieve; if blank, retrieve all models

        Returns:
            DataFrame of rows written
        """
        s = [f"{k}='{v}'" for k, v in [['label', label], ['model', model]] if v]
        where = bool(s) * ('where ' + ' and '.join(s))
        q = f"SELECT * from {self.table_.key} {where}"
        return self.sql.read_dataframe(q)

    def plot(self, model: str, drift: bool = False,
             ax: Any = None, loc: str = 'best', title: str = '',
             c: str = 'C0', vline: List[float] = [], fontsize: int = 10,
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
            fontsize: Base font size
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
                   c=c, width=width, fontsize=fontsize,
                   legend=["CAR" if p['car'] else "BHAR", f"{width} stderrs"],
                   xlabel=(f"{int(s['beg'])}-{int(s['end'])}"
                           + f" (n={int(s['rows'])},"
                           + f" days={int(s['days'])},"
                           + f" eff={int(s['effective'])})"),
                   ylabel="CAR" if p['car'] else "BHAR",
                   ax=ax)
        plt.tight_layout(pad=3)
            
            
if __name__=="__main__":
    from conf import credentials, VERBOSE
    
