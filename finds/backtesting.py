"""Class and methods to evaluate backtests, event studies and risk premiums

- Event studies, cumulative abnormal returns
- Risk premiums, Fama-MacBeth regressions
- Sharpe ratio, appraisal ratio, walk-forward backtests

Author: Terence Lim
License: MIT
"""
import numpy as np
import scipy
import pandas as pd
import time
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy.builtins import Q
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from matplotlib import dates as mdates
from sqlalchemy import Integer, String, Float, SmallInteger, Boolean, BigInteger
from sqlalchemy import Column, Index
from pandas.api import types
from .structured import Structured
from .display import plot_date, plot_bands
from .solve import least_squares
try:
    from settings import ECHO
except:
    ECHO = False

def _astype(v, t=str):
    """Convert each element in list of nested lists to target type"""
    return [_astype(u, t) for u in v] if types.is_list_like(v) else t(v)

def _as_compound(rets, intervals, dates=None):
    """Compounds series of returns between (list of) date tuples (inclusive)"""
    if len(intervals)==0:
        return []
    elif len(intervals)==1:
        return [_as_compound(rets, intervals[0])]
    elif len(intervals)==2 and isinstance(intervals[0], int):
        d = rets.index if dates is None else dates
        return np.prod(rets[(d >= intervals[0]) & (d <= intervals[1])] + 1) - 1
    else:
        return [_as_compound(rets, interval) for interval in intervals]


class BackTest(Structured):
    """Base class for computing portfolio backtest returns

    Parameters
    ----------
    sql : SQL object
        SQL database connection to store results
    bench : benchmark Structured dataset instance
        Where riskfree and benchmark returns can be retrieved from
    rf : str
        Column name of riskfree rate in benchmark dataset
    max_date : int
        Last date of any backtest (i.e. bench and stock returns available)
    table : str, default is 'backtests'
        name of table in user database to store results in

    Notes
    -----
    assumes that monthly risk free rates also available through {bench}, 
    with name suffixed by "(mo)". if backtest dates appears to be monthly 
    frequency, monthly risk free rates will be retrieved
    and used rather than compounding from daily (reduce precision errors).

    Examples
    -------
    backtest = BackTest(user, bench, 'RF', 20200930)
    """

    # schema of the table to store backtested performance returns
    def __init__(self, sql, bench, rf, max_date, table='backtests', echo=ECHO):
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
        self.echo = echo
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

    def _print(self, *args, echo=None):
        if echo or self.echo_:
            print(*args)

    def __call__(self, stocks, holdings, label, overlap=0):
        """Compute holding returns and rebalance statistics
        
        Parameters
        ----------
        stocks: Structured data set
            Where securities' identifiers, returns and data can be accessed
        holdings: dict {rebaldate : holdings Series}
            Each Series is indexed by permno, with weights in column
            Last item of dict (can be empty) is ignored for calculating returns
        label: string
            Label to set to name this backtest
        overlap: int, default is 0
            Number of months to smooth holdings

        Returns
        -------
        perf : DataFrame
            portfolio holdings returns after each rebalance date

        Notes
        -----
        if CRSP ('delist' and using monthly): apply dlst returns to performance
        """
        for d, h in holdings.items():
            if not h.index.is_unique:
                raise ValueError(f"duplicate holdings index date={d}")

        pordates = sorted(list(holdings.keys()))
        self._print(len(pordates), 'dates:', pordates[0], '-', pordates[-1])

        perf = {}                    # accum performance each period
        smooth = []                  # to queue rolling holdings
        prev = Series(dtype=float)   # prior holdings, adjusted by returns
        holding_periods = stocks.bd.holding_periods(pordates)
        for pordate, (begret, endret) in zip(pordates[:-1], holding_periods):

            if (begret, endret) in self.monthly_:
                riskfree = self.monthly_[(begret, endret)]
            else:
                riskfree = _as_compound(self.rf, (begret, endret))
        
            # insert current holdings into smooth
            if len(smooth) > overlap:  # smooth has list of recent holdings
                smooth.pop()
            smooth.insert(0, holdings[pordate].copy())

            # compute rolling weights, by adding to superset of permnos in curr
            permnos = sorted(set(np.ravel([list(p.index) for p in smooth])))
            curr = Series(index=permnos, data=[0] * len(permnos), dtype=float)
            for p in smooth:             # assign weight in smoothed final
                curr[p.index] += p / len(smooth)

            # get stocks' returns
            ret = stocks.get_ret(begret, endret, delist=True)
            ret = ret['ret'].reindex(curr.index, fill_value=0)
            r = sum(curr * ret)   # portfolio return this month
            
            # compute turnover
            delta = pd.concat((prev, curr), axis=1, join='outer').fillna(0)
            delta = delta.iloc[:, 1] - delta.iloc[:, 0]  # change in holdings

            # collect
            perf.update({int(endret):
                         {'begret': int(begret),
                          'endret': int(endret),
                          'longs': sum(curr > 0),
                          'shorts': sum(curr < 0),
                          'long_weight': curr[curr > 0].sum(),
                          'short_weight': curr[curr < 0].sum(),
                          'ret': r, 
                          'excess': r - (curr.sum() * riskfree),
                          'buys': delta[delta>0].abs().sum(),
                          'sells': delta[delta<0].abs().sum()}})

            # adjust stock weights by retx till end of holding period
            retx = stocks.get_ret(begret, endret, field='retx')['retx']
            prev = curr * (1 + retx.reindex(curr.index)).fillna(1)
            for i in range(len(smooth)):
                smooth[i] *= (1 + retx.reindex(smooth[i].index)).fillna(1)
            self._print(f"(backtest) {pordate} {len(curr)} {r:.4f}")
        self.perf = DataFrame.from_dict(perf, orient='index')
        self.label = label
        self.excess = None
        return perf

    def drawdown(self):
        """Compute max drawdown and period: amount of loss from previous high"""
        cumsum = self.perf['excess'].cumsum()
        cummax = cumsum.cummax()
        end = (cummax - cumsum).idxmax()
        beg = cumsum[cumsum.index <= end].idxmax()
        dd = cumsum.loc[[beg, end]]
        return dd

    def write(self, label):
        """Save backtest performance returns to database"""
        self['backtests'].create(checkfirst=True)
        delete = self['backtests']\
                 .delete().where(self['backtests'].c['permno'] == label)
        self.sql.run(delete)
        self.perf['permno'] = label
        self.sql.load_dataframe(self['backtests'].key, self.perf)

    def read(self, label=None):
        """Load backtest performance returns from database"""
        if label is None:
            q = (f"SELECT {self.identifier}, count(*) as count,"
                 f" min(begret) as begret,  max(endret) as endret "
                 f" from {self['backtests'].key} group by {self.identifier}")
            return self.sql.read_dataframe(q).set_index(self.identifier)
        q = (f"SELECT * from {self['backtests'].key} "
             f"where {self.identifier} = '{label}'")
        self.perf = self.sql.read_dataframe(q)\
                            .sort_values(['endret'])\
                            .set_index('endret', drop=False)\
                            .drop(columns=['permno'])
        self.label = label
        self.excess = None
        return self.perf

    def get_series(self, field='ret', start=19000000, end=None):
        """Retrieve saved backtest as a series"""
        return self.sql.pivot(self['backtests'].key, index='endret',
                              columns='permno', values=field,
                              where=(f"endret >= {start} AND endret <= "
                                     f"{self.max_date if end is None else end} "
                                     f"AND permno={self.label}"))\
                       .rename(columns={'endret': 'date'})

    def fit(self, benchnames, beg=0, end=None, haclags=1):
        """Compute performance attribution against benchmarks 

        Parameters
        ----------
        benchnames: list of str
            Names of benchmark returns to compute attribution against
        haclags: int, optional
            Option for robustcov statistics = number of Newey-West lags

        Returns
        -------
        DataFrame
            Each row is excess returns performance following each rebalance date

        Attributes
        -----
        annualized : dict of performance ratios
            'excess': annualized excess (of portfolio weight*riskfree) return
            'sharpe': annualized sharpe ratio
            'jensen': annualized jensen alpha
            'appraisal': annualized appraisal ratio
            'welch-t': t-stat for structural break after 2002
            'welch-p': p-value for structural break after 2002
            'turnover': annualized total turnover rate
            'buys': annualized buy rate
            'sells': annualized sell rate
        results : statsmodels OLS results
            from fitting statsmodels OLS of excess on benchmark returns
        """
        # collect performance between beg and end dates
        end = end or self.max_date
        #
        # this can be simplified
        #
        d = self.perf.loc[beg:end].index
        nyears = len(self.rf.loc[d[0]:d[-1]]) / 252
        p = self.perf.loc[d, 'excess'].rename(self.label).to_frame()

        # collect benchmark returns
        df = self.bench.get_series(benchnames, 'ret', end=self.max_date)
        retdates = _astype(self.perf.loc[d, ['begret','endret']].values, int)
        for b in benchnames:
            p[b] = _as_compound(df[b], retdates)

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
            'jensen': mult * r.params[0],
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
        self.excess = p
        return self.excess

    def plot(self, num=1, flip=False, drawdown=False, figsize=(10,12)):
        """Plot time series of excess vs benchmark returns

        Parameters
        ----------
        num : int, optional
            Figure number to use in plt
        flip: bool, default False
            If None, auto-detect and flip returns to be positive
            If False, never flip 
            If True, always flip
        drawdown: bool, default False
            If True, plot peak and trough points of max (additive) drawdown
        """
        if flip:
            label = 'MINUS ' + self.label
            m = -1
        else:
            label = self.label
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
        plot_date(y1=excess.cumsum(), label1='cumulative ret',  marker=None,
                  ax=ax1, points=self.drawdown() if drawdown else None)
        plot_date(y1=perf[['longs','shorts']],
                  y2=(perf['buys'] + perf['sells']) / 4, 
                  ax=ax2, marker=None, ls1='-', ls2=':', cn=excess.shape[1],
                  label1='#holdings', label2='turnover', legend2=['turnover'])
        plt.tight_layout(pad=3)

class EventStudy(Structured):
    """Class to support statistical tests of event studies

    Parameters
    ----------
    sql : SQL object
        connection to user database to store results
    bench : Benchmarks structured data object
        to retrieve benchmark market returns
    max_date : int
        last date to run event study (not used)
    table : str, default is 'events'
        name of table in user database to store results in
    """
    def __init__(self, sql, bench, max_date, table='events'):
        """Initialize for event study calculations"""
        tables = {'events':
                  sql.Table(table,
                            Column('permno', String(32), primary_key=True),
                            Column('name', String(32), primary_key=True),
                            Column('beg', Integer),
                            Column('end', Integer),
                            Column('rows', Integer),
                            Column('days', Integer),
                            Column('effective', Float),
                            Column('window', Float),
                            Column('window_t', Float),
                            Column('post', Float),
                            Column('post_t', Float))}
        super().__init__(sql, bench.bd, tables, 'permno', name='events')
        self.bench = bench
        self.max_date = max_date
        self.ar_ = {}
        self.ret_ = {}
        
    def write(self, label, overwrite=True):
        """Save event study summary to database"""
        self['events'].create(checkfirst=True)
        if overwrite:
            delete = self['events'].delete().where(
                self['events'].c['permno'] == label)
            self.sql.run(delete)
        ar = DataFrame.from_dict(self.ar_, orient='index')
        ar['permno'] = label
        self.sql.load_dataframe(self['events'].key, ar)
        return ar
        
    def read(self, label=None, name=None):
        """Load event study summary from database"""
        where = ' and '.join([f"{k} = '{v}'" for k,v in
                              [['permno', label], ['name', name]] if v])
        q = "SELECT * from {table} {where}".format(
            table=self['events'].key,
            where="where " + where if where else '')
        return self.sql.read_dataframe(q)

    def __call__(self, stocks, df, left, right, post, date_field):
        """Retrieve event window market-adjusted returns where valid/available

        Parameters
        ----------
        stocks : Structured object
            Stock returns data
        df : DataFrame
            Input list of stocks of identifiers and event dates
        left, right, post : int
            left and right (inclusive) window and post-drift date offsets
        date_field : str
            Name of date column in df
        """
        ret = stocks.get_window(
            dataset='daily',
            field='ret',
            permnos=df[stocks.identifier],
            dates=df[date_field],
            date_field='date',
            left=left,
            right=post)
        cols = list(range(post-left+1))
        
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
                                    right=post)
        rf = self.bench.get_window(dataset='daily',
                                   field='ret',
                                   permnos=['RF'] * len(rets),
                                   date_field='date',
                                   dates=rets['date'],
                                   left=left,
                                   right=post)
        mkt = (mkt[cols] + rf[cols]).reset_index(drop=True)
        ar = (rets[cols] - mkt[cols]).cumsum(axis=1).fillna(0)
        br = ((1 + rets[cols]).cumprod(axis=1) -
              (1 + mkt[cols]).cumprod(axis=1)).fillna(0)
        self.car = rets[['permno', 'date']].join(ar)
        self.bhar = rets[['permno', 'date']].join(br)
        self.left = left
        self.right = right
        self.post = post
        self.rows = rets[['permno', 'date']]
        return self.rows

    def fit(self, rows=None, car=False, name='event_', rho=0.3):
        """Compute CAR/BHAR statistics from cumulative rets

        Parameters
        ----------
        rows : list of int, default is None
            Subset of rows to evaluate; None selects all rows
        car : bool, default is False
            Whether to evaluate CAR or BHAR
        name : str, default is None
            save results in cache by this label
        rho : float between 0 and 1, default is 0.3
            rule-of-thumb to adjust stderrs for cross-correlated errors

        Returns
        -------
        dict :
            summary statistics of full and subsamples: 
            'window', 'window-tvalue' are CAR at end of event window
            'post', 'post-tvalue' are CAR from end of event till post-drift end
            'car', 'car-stderr' are daily CAR from beginning of event
            'rows', 'days' are number of stocks, and after groupby dates
        """
        window = self.right - self.left + 1
        cols = ['date'] + list(range(self.post-self.left+1))
        rets = (self.car if car else self.bhar)[cols]
        cumret = (rets if rows is None else rets.iloc[rows])
        n = int(len(cumret))
        b = int(min(cumret['date']))
        e = int(max(cumret['date']))
        cumret = cumret.groupby('date').mean()
        means = cumret.mean()
        L = self.post-self.left
        overlap = ((len(cumret) * (L+1))/(len(self.bd.date_range(b,e)) + L)) - 1
        effective = len(cumret) / (1 + (rho * min(max(overlap, 0), L)))
        stderr = cumret.std() / np.sqrt(effective)
        posterr = ((cumret.iloc[:, window:]\
                    .sub(cumret.iloc[:, window-1], axis=0)).std()
                   / np.sqrt(effective))
        cumret.iloc[:, window:].std() / np.sqrt(effective)
        tstat = means[window - 1]/stderr[window - 1]
        post = cumret.iloc[:,-1] - cumret.iloc[:, window-1]
        post_sem = post.std() / np.sqrt(effective)
        ar = Series({'name'      : name,
                     'window'    : means[window - 1], 
                     'window_t'  : means[window - 1]/stderr[window - 1],
                     'post'      : post.mean(), 
                     'post_t'    : post.mean() / post_sem,
                     'beg'       : b,
                     'end'       : e,
                     'effective' : int(effective),
                     'days'      : len(cumret),
                     'rows'      : n})
        self.ret_[name] = {'means'   : means.values,
                           'stderr'  : stderr.values,
                           'posterr' : posterr.values,
                           'car'     : car}
        self.ar_[name] = ar.copy()
        return ar

    def plot(self, name='event_', drift=False, ax=None,
             loc='best', title='', c='C0', vline=None, hline=None, width=1.96):
        """Plot cumulative abnormal returns, drift and confidence bands"""
        ax = ax or plt.gca()
        window = self.right - self.left + 1
        if vline is None:
            vline = self.right
        if hline is None:
            hline = self.ret_['event_']['means'][window-1] if drift else 0
        r = self.ret_[name]
        ar = self.ar_[name]
        plot_bands([0] + list(r['means']),
                   [0] + ([0] * (window if drift else 0))
                   + list(r['posterr' if drift else 'stderr']),
                   x=np.arange(self.left-1, self.post+1), loc=loc,
                   hline=hline, vline=vline, title=title, c=c, width=width,
                   legend=["CAR" if r['car'] else "BHAR", f"{width} stderrs"],
                   xlabel=(f"{int(ar['beg'])}-{int(ar['end'])}"
                           f" (dates={int(ar['days'])}, n={int(ar['rows'])})"),
                           ylabel="CAR" if r['car'] else "BHAR", ax=ax)
        plt.tight_layout(pad=3)
            
            
def wald_test(R, r, theta, avar):
    """helper method to compute wald test of linear hypotheses

    Parameters
    ----------
    R : (Q x P) array of float
        input coefficients to test hypothesis that R theta = r
    r : (Q x 1) vector of float
        input constants to test hypotheses that R theta = r
    theta : (P x 1) vector of float
        estimated model parameters
    avar : (P x P) array of float


    Returns
    -------
    stats : dict
       'wald': wald test statistic, 'p-value': of chi-square with df=Q
    """
    theta = theta.reshape(-1, 1)       # P x 1  parameter estimates
    r = r.reshape(-1, 1)               # Q x 1  hypotheses
    R = R.reshape(len(r), len(theta))  # Q x P  coefficients
    wald = (R @ theta - r).T @ np.linalg.inv(R @ avar @ R.T) @ (R @ theta - r)
    w = wald.reshape(-1)[0]
    return {'wald': w, 'p-value': 1 - scipy.stats.chi2.cdf(w, len(r))}


class RiskPremium:
    """Class to support statistical tests of factor loading premiums

    Parameters
    ----------
    sql : SQL object
        connection to user database to store results
    bench : Benchmarks structured data object
        to retrieve benchmark market returns
    rf : str
        series name of riskfree rate from bench database
    end : int
        last date to run event study (not used)
    """
    def __init__(self, sql, bench, rf, end):
        """Initialize for testing factor loading premiums"""
        self.sql = sql
        self.bench = bench
        self.rf = bench.get_series([rf], 'ret', end=end)[rf]
        rf = bench.get_series([rf + "(mo)"], 'ret', end=end)  # monthly riskfree
        self.monthly_ = {(bench.bd.begmo(d), bench.bd.endmo(d)):
                         float(rf.loc[d]) for d in rf.index}
        self.end_ = end

    def __call__(self, stocks, loadings, weights=None, standardize=[]):
        """Estimate factor risk premiums with cross-sectional FM regressions

        Parameters
        ----------
        stocks : Structured data object
            From which to retrieve stocks' returns data
        loadings : dict, keyed by rebalance date:int, of DataFrame
            DataFrames indexed by stocks permno, with columns of loadings values
        standardize : list of str, default is []
            List of column labels to demean and rescale (eql-wtd stdev = 1)
        weights : str, default is None
            Column for weighted least squares, and weighted demean

        Returns
        -------
        ret : DataFrame
            means and standard errors of FM regression coefficients/premiums
        """
        self.perf = DataFrame()
        pordates = sorted(list(loadings.keys()))
        self.holdrets = stocks.bd.holding_periods(pordates)
        for pordate, holdrets in zip(pordates[:-1], self.holdrets):
            if holdrets in self.monthly_: 
                rf = self.monthly_[holdrets]
            else:
                rf = _as_compound(self.rf, holdrets)
            df = loadings[pordate]
            if weights is None:
                w = np.ones(len(df))
            else:
                w = df[weights].to_numpy()
                df = df.drop(columns=[weights])
            x = df.columns
            for col in standardize: # weighted mean <- 0, equal wtd stdev <- 1
                df[col] -= np.average(df[col], weights=w)
                df[col] /= np.std(df[col])
            df = df.join(stocks.get_ret(*holdrets, delist=True)-rf, how='left')
            p = least_squares(df.dropna(), x=x, y='ret', add_constant=False)
            p.name = holdrets[1]
            self.perf = self.perf.append(p)
        self.results = {'mean': self.perf.mean(), 'stderr': self.perf.sem(),
                        'std': self.perf.std(), 'count': len(self.perf)}
        return DataFrame(self.results).T

    def fit(self, benchnames=None):
        """Compute risk premiums and benchmark correlations"""
        out = []
        if benchnames:
            df = self.bench.get_series(benchnames, 'ret')
            b = DataFrame({k: _as_compound(df[k], self.holdrets)
                           for k in benchnames}, index=self.perf.index)
            out.append(DataFrame({'mean': b.mean(), 'stderr': b.sem(),
                                  'std': b.std(), 'count': len(b)}).T\
                       .rename_axis('Benchmark Returns', axis=1))
            corr = b.join(self.perf).corr()
            out.append(corr.loc[benchnames, benchnames].rename_axis(
                'Correlation of Benchmark Returns', axis=1))
            out.append(corr.loc[self.perf.columns, benchnames].rename_axis(
                'Correlation of Estimated Factor and Benchmark Returns', axis=1))
        else:
            corr = self.perf.corr()
        out.append(DataFrame(self.results).T.rename_axis(
            'Estimated Factor Returns', axis=1))
        out.append(corr.loc[self.perf.columns, self.perf.columns].rename_axis(
            'Correlation of estimated factor returns:', axis=1))
        #R = []
        #for f in factors:
        #    r = [0] * self.perf.shape[1]
        #    r[self.perf.columns.get_loc(f)] = 1
        #    R = R + [r]
        #R = np.array(R)
        #r = np.zeros((1, len(factors)))
        #theta = self.perf.mean().to_numpy()
        #avar = self.perf.cov() / self.perf.shape[0]
        #w = wald_test(R, r, theta, avar)
        #print("Wald Test H0:", ", ".join(f"{f} is 0" for f in factors))
        #print(f"wald-value (p-value) = {w['wald']:.3} ({w['p-value']:.3})")
        return out

    def plot(self, factors=None, num=1, figsize=(10,6)):
        """Plot computed time series of factor returns"""
        if factors is None:
            factors = list(self.perf.columns)
        if isinstance(factors, str):
            factors = [factors]
        b = {}
        if isinstance(factors, dict):
            df = self.bench.get_series(factors.values(), 'ret')
            for k,v in factors.items():
                b[k] = Series(_as_compound(df[v], self.holdrets),
                              index=self.perf.index, name=v)
            factors = list(factors.keys())
        nrow = int(np.ceil(np.sqrt(len(factors))))
        ncol = int(np.ceil(len(factors) / nrow))
        fig, axes = plt.subplots(nrow, ncol, clear=True, num=num,
                                 squeeze=False, figsize=figsize)
        for i, (ax, col) in enumerate(zip(np.ravel(axes), factors)):
            if len(b):
                plot_date(y1=self.perf[col].cumsum(), legend1=[col],
                          y2=b[col].cumsum(), legend2=[b[col].name], cn=i*2, 
                          loc1='upper left', loc2='lower right', ax=ax)
            else:
                plot_date(y1=self.perf[col].cumsum(), ax=ax, cn=i,
                          legend1=[col])
        plt.tight_layout(pad=3)

           
class as_stocks:
    """Helper class caches a DataFrame to mimic a Stocks object"""

    class bday:
        @staticmethod
        def begmo(date):
            return date
        @staticmethod
        def endmo(date):
            return date
        @staticmethod
        def holding_periods(pordates):
            return list(zip(pordates[1:], pordates[1:]))
        
    def __init__(self, df, rsuffix=None, identifier='permno'):
        """Create instance with the given DataFrame"""
        self.data = DataFrame(df)
        if rsuffix is not None:
            self.data = self.data.join(self.data, how='left', rsuffix=rsuffix)
        self.identifier = identifier

    def get_series(self, permnos, *arg, **kwarg):
        """Return the series for target permnos"""
        return self.data[permnos]

    def get_ret(self, start, end, *args, **kwargs):
        """Return data as 'ret' series between start and end dates"""
        df = DataFrame((self.data.loc[(self.data.index>=start)
                                      & (self.data.index<=end)] + 1).prod() - 1)
        df.columns = ['ret']
        df.index.name = self.identifier
        return df

#if __name__ == "__main__":
if False:
    url = 'https://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/'
    data = pd.read_csv(url + 'test_data.txt', sep='\s+',
                       names=['firmid', 'year','x','y'])
    ls = smf.ols("y ~ x", data=data).fit()
    ls.get_robustcov_results('HC0').bse
    ls.get_robustcov_results('hac-panel', groups=data['firmid'], maxlags=2).bse
    ls.get_robustcov_results('cluster', groups=data['firmid']).bse
    ls.get_robustcov_results('cluster', groups=data['year']).bse
    ls.get_robustcov_results('cluster', groups=data[['firmid','year']]).bse
    mixed = smf.mixedlm("y ~ x - 1", data=data, groups=data['firmid']).fit()    
    fm = data.groupby(by='year').apply(least_squares, y='y', x='x')
    print(fm.mean())
    print(fm.sem())

if False:
    import io, requests, zipfile
    from settings import settings
    url = "https://www.kevinsheppard.com/files/teaching/python/notes/"
#    infile = io.BytesIO(requests.get(url + 'FamaFrench.zip').content)
    infile = settings['remote'] + 'FamaFrench.zip'
    with zipfile.ZipFile(infile).open("FamaFrench.csv") as g:
        ret = pd.read_csv(g, index_col=0) / 100
    factor = list(ret.columns[:3])
    riskfree = ret.columns[3]
    port = list(ret.columns[4:])
    
    stocks = as_stocks(ret[port])
    bench = as_stocks(ret[factor + [riskfree]], rsuffix='(mo)')

    ret[port] = ret[port].sub(ret[riskfree], axis=0)
    b = least_squares(ret, y=port, x=factor).drop(columns='Intercept')
    r = ret.iloc[:,-25:].stack().reset_index(name='ret')\
                                .rename(columns={'level_1':'port'})
    r = r.join(b, on='port', how='left').sort_values(['date','port'])
    fm0 = r.groupby(by='date').apply(least_squares, y='ret', x=factor,
                                     add_constant=False)
    print(DataFrame({'mean': fm0.mean(), 'stderr': fm0.sem()}).T)

    loadings = {k: b for k in ret.index}
    loadings[0] = b   # for initial holdings

    rp = RiskPremium(None, bench, riskfree, max(ret.index))
    premiums = rp(stocks, loadings)
    print(premiums)
    print(premiums * 12 * 100)
    
    R = np.array([[1, 0, 0], [0, 1, 0]])
    r = np.array([0, 0])
    theta = rp.perf.mean().to_numpy()
    avar = rp.perf.cov()/len(rp.perf)
    wald_test(R, r, theta, avar)      # Wald test that HML and SMB are zero
    
import pandas_datareader as pdr
from pandas_datareader.data import DataReader
from pandas_datareader.famafrench import FamaFrenchReader as ffReader
if False:
    s = pdr.famafrench.get_available_datasets()
    b = ffReader('Portfolios_Formed_on_BETA', start=1900, end=2099).read()
    sb = ffReader('25_Portfolios_ME_BETA_5x5', start=1900, end=2099).read()
    m = ffReader('F-F_Research_Data_Factors', start=1900, end=2099).read()
    mkt = m[0].copy().rename(columns={'Mkt-RF': 'BETA'})

    h = ffReader('25_Portfolios_5x5', start=1900, end=2099).read()
    df = h[0].sub(mkt['RF'], axis=0).reindex(h[0].index).copy()
    for factors in  [['BETA'], ['BETA', 'SMB' ,'HML']]:
        rets = df.stack().reset_index(name='ret')\
                         .rename(columns={'level_1':'port'})
        data = df.join(mkt[factors], how='left')
        betas = least_squares(data, y=df.columns, x=factors)[factors]
        rets = rets.join(betas, on='port').sort_values(['Date','port'])
        rets['Date'] = [d.year*100 + d.month for d in rets['Date']]
        
        fm = rets.groupby(by='Date').apply(least_squares, y=['ret'], x=factors,
                                           add_constant=True) / 100
        results = DataFrame({'mean': fm.mean(), 'stderr': fm.sem(),
                             'tvalues': fm.mean() / fm.sem()}).T
        results.loc['tvalues'] = results.loc['mean'].div(results.loc['stderr'])
        print('25_Portfolios_5x5', 'Value-weighted')
        print(results)

    ls = smf.ols("ret ~ BETA + SMB + HML", data=rets).fit()
    print(ls.summary())
    print(ls.get_robustcov_results('cluster', groups=rets['port']).summary())
    
