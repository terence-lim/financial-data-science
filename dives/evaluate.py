"""
the dives.evaluate module defines classes for backtesting and evaluating portfolio returns
"""
# The MIT License
#
# Copyright (c) 2020 Terence Lim
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation he rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

from dives.util import DataFrame, print_debug, isnum
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  # for date formatting in plots

def compound_ret(ret, beg, end):
    """helper method to compound returns in a series {ret} between {beg} and {end} dates (inclusive)"""
    if isnum(beg) or isnum(end):
        return (np.prod(1 + ret[(ret.index >= beg) & (ret.index <= end)])-1)
    else:   # {beg} and {end} may be vectors of dates
        return [compound_ret(ret, b, e) for b,e in zip(beg, end)]


def plot_holdings(x=None, y1=None, y2=None, title='Holdings and Turnover',
                  label1='Number of Holdings', label2='Turnover Rate', xskip=2, labels=''):
    """helper method to plot number of holdings and turnover rate
    
    Parameters
    ----------
    x : DatetimeIndex
        x-axis, e.g. constructed by pd.to_datetime(list(yyyymmdd), format='%Y%m%d')
    y1 : DataFrame
        single time series to plot on left y-axis
    y2 : DataFrame
        single time series to plot on right y-axis
    title : string, optional
        title of plot
    label1 : string, optional
        title of y1 axis
    label2 : string, optional
        title of y2 axis
    xskip : int, optional
        every nth tick on axis to make visible (default 2)
    """
#    fig, ax = plt.subplots()
    ax = plt.gca()    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    symbols = ['c:','m-.','g--','r-','b:']
    for i in range(len(y1.columns)):
        ax.plot(x, y1.iloc[:,i], symbols[np.mod(i,len(symbols))])
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % xskip != 0:
            label.set_visible(False)
    ax.set_ylabel(label1, color='b')
    for t in ax.get_yticklabels():
        t.set_color('b')
    if y2 is not None:
        bx = ax.twinx()
        bx.plot(x, y2, 'y8', markersize=2, fillstyle='none')
        bx.set_ylabel(label2, color='y')
        for t in bx.get_yticklabels():
            t.set_color('y')
    ax.legend(labels)            
    plt.title(title)    

def plot_returns(x=None, y1=None, y2=None, labels=[], title='Excess Returns and Risk',
                 label1='Cumulative Log Returns', label2='EWMA Volatility (3-month half-life)',
                 xskip=2, yscale='linear', date='%Y%m%d', cumprod=True,
                 hlines=[], vlines=[]):
    """helper method to plot lines on primary y-axis and markers on secondary y-axis
        
    Parameters
    ----------
    x : DatetimeIndex
        x-axis, e.g. constructed by pd.to_datetime(list(yyyymmdd), format='%Y%m%d')
    y1 : DataFrame
        multiple time series to plot as lines
    y2 : DataFrame
        multiple time series to plot as markers
    labels : list of string, optional
        labels of the series for legend
    title : string, optional
        title of plot
    label1 : string, optional
        title of y1 axis
    label2 : string, optional
        title of y2 axis
    xskip : int, optional
        every nth tick on axis to make visible (default 2)
    yscale : str, optional
        y-axis scale {"linear" (default), "log", "symlog", "logit"}
    date : string, optional (default = '%Y%m%d')
        how to format x-axis as date (None, False, '' to not format as date)
    cumprod : boolean (default = True)
        whether to apply log(cumprod(1 + y)) transformation
    hlines : list of int (default = [])
        y-axis points where to place horizontal lines
    vlines : list of int (default = [])
        x-axis points where to place vertical lines 
    """
#    fig, ax = plt.subplots()
    ax = plt.gca()
    if date:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date))
    symbols = ['b-','r--','g:','m-.','c--']
    markers = ['bo','r*','gs','m+.','cx']
    if cumprod:
        z = np.log(np.cumprod(1 + y1))
    else:
        z = y1
    for i in range(len(z.columns)):
        ax.plot(x, z.iloc[:,i], symbols[np.mod(i,len(symbols))])
        plt.yscale(yscale)
    for hline in hlines:
        plt.axhline(hline, linestyle=':', color='y')
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % xskip != 0:
            label.set_visible(False)
    ax.set_ylabel(label1, fontsize='small')
    if y2 is not None:
        bx = ax.twinx()
        bx.plot(x, y2.iloc[:,0], 'yo', markersize=2, fillstyle='none')    
#    for i in range(len(y2.columns)):
#        bx.plot(x, y2.iloc[:,i], markers[np.mod(i,len(symbols))], markersize=3, fillstyle='none')
        bx.set_ylabel(label2, color='y', fontsize='small')
        for t in bx.get_yticklabels():
            t.set_color('y')
    for vline in vlines:
        plt.axvline(vline, linestyle=':', color='y')
    ax.legend(labels, fontsize='small')
    plt.title(title, fontsize='small')

    
#
# BackTest is base class for calculating and backtesting portfolio returns
#
class BackTest(object):
    """base class BackTest for calculating and backtesting portfolio returns

    Parameters
    ----------
    bench : instance of benchmark Structured dataset
        where riskfree rate can be retrieved from
    rf : string
        column name of riskfree rate in benchmark dataset

    Notes
    -----
    assumes that monthly risk free rates also available through {bench}, with name suffixed by "(mo)".
    if backtest dates appears to be monthly frequency, monthly risk free rates will be retrieved
    and used rather than compounding from daily (reduce precision errors).

    Examples
    -------
    backtest = BackTest(bench, 'RF')
    """

    # schema of the table to store backtested performance returns
    _schema = {'table' : 'backtests',
               'fields': [['permno', 'VARCHAR(32)'],
                          ['begret', 'INT(11)'],
                          ['endret', 'INT(11)'],
                          ['longs', 'INT(11)'],
                          ['shorts', 'INT(11)'],
                          ['buys', 'DOUBLE'],
                          ['sells', 'DOUBLE'],
                          ['long_weight', 'DOUBLE'],                          
                          ['short_weight', 'DOUBLE'],
                          ['excess', 'DOUBLE'],
                          ['ret', 'DOUBLE']],
               'primary': ['permno', 'endret'],
               'indexes': [['endret','permno'], ['begret','permno'], ['permno','begret']]}
    
    def __init__(self, sql, bench, rf):
        """Create instance to support backtest performance evaluation"""
        self.sql = sql
        self.bench = bench
        self.rf = bench.get_series([rf], 'ret')[rf]
        m = bench.get_series([rf + "(mo)"], 'ret') # monthly riskfree rates as dict keyed by (begmo,endmo)
        self._monthly = {(bench.dates.begmo(d), bench.dates.endmo(d)) : float(m.loc[d]) for d in m.index}
        self._str = 'BackTest'

    def performance(self, dataset, holdings, label, overlap=1):
        """compute performance and rebalance statistics of sequence of holdings
        
        Parameters
        ----------
        dataset: Structured data set
            where securities' identifiers, returns and data can be accessed
        holdings: dict (keyed by date of rebalance) of DataFrames
            Each DataFrame is indexed by permno, with column of "weights"; last item has empty weights
        label: string
            label to set to name this backtest, default is None
        overlap: int, optional
            number of months of overlapping holdings, default is 1 (i.e. no overlaps)

        Returns
        -------
        n : int
          number of performance periods computed

        Notes
        -----
        detects if dataset is CRSP: applies delisting returns to performance
        computes self.perf of type DataFrame containing holding period returns and turnover statistics
        """
        pordates = sorted(list(holdings.keys()))
        perf = DataFrame(index=pordates[:-1])
        smooth = []    # for queue of rolling holdings if overlap desired
        prev = DataFrame(columns = ['weights'])
        
        for pordate, (begret, endret) in zip(pordates[:-1], dataset.dates.holding_periods(pordates)):
            if (begret, endret) in self._monthly:
                riskfree = self._monthly[(begret, endret)]
            else:
                riskfree = compound_ret(self.rf, begret, endret)
                
            # insert current holdings, and compute average (for overlapping)
            smooth.insert(0, holdings[pordate].copy())
            if len(smooth) > overlap:    # smooth has list of recent holdings up to overlap lags
                smooth.pop()
            permnos = set()              # require output to have unique permnos
            for p in smooth:
                permnos = permnos.union(set(p.index))
            port = DataFrame(index = sorted(list(permnos)), columns=['weights'], data=[0]*len(permnos))
            for p in smooth:             # average permno's weight over past portfolios in smooth list
                port.loc[p.index, 'weights'] += p['weights'] / len(smooth)

            # get stocks' returns, and compute performance and turnover
            ret = dataset.get_ret(begret, endret)
            if dataset._str == "CRSP":   # if CRSP, apply delisting return
                dlst = dataset.get_dlstret(begret, endret)
                permnos = list(set(ret.index).intersection(dlst.index))
                if len(permnos):
                    ret.loc[permnos,'ret'] = ((1 + ret.loc[permnos,'ret']) *
                                              (1 + dlst.loc[permnos,'ret'])) - 1
            ret[ret.ret.isnull()] = 0
            port['ret'] = ret.reindex(port.index)['ret']
            v = port.ret.ge(-1.0) & port.weights.notnull()
            denom = max(abs(port['weights'].loc[port.weights.gt(0)].sum()),
                        abs(port['weights'].loc[port.weights.lt(0)].sum()))
            perf.loc[pordate, 'begret'] = int(begret)
            perf.loc[pordate, 'endret'] = int(endret)
            perf.loc[pordate, 'longs'] = sum(port.weights.gt(0))
            perf.loc[pordate, 'shorts'] = sum(port.weights.lt(0))
            perf.loc[pordate, 'long_weight'] = port['weights'].loc[port.weights.gt(0)].sum()
            perf.loc[pordate, 'short_weight'] = port['weights'].loc[port.weights.lt(0)].sum()
            perf.loc[pordate, 'ret'] = sum(port.loc[v, 'weights'] * port.loc[v, 'ret']) / denom
            perf.loc[pordate, 'excess'] = perf.loc[pordate,'ret']-(port.weights.sum()*riskfree)
        
            buys, sells = 0, 0
            curr = holdings[pordate]['weights']
            currset = set(curr.index)
            prevset = set(prev.index)

            for p in prevset.intersection(currset):
                diff = curr[p] - prev[p]
                if diff > 0: buys += diff
                else:
                    if diff < 0: sells += diff
            for p in currset - prevset:
                if curr[p] > 0:
                    buys += curr[p]
                else:
                    sells += curr[p]
            for p in prevset - currset:
                if  prev[p] > 0: sells -= prev[p]    # sell to cover
                else:
                    if prev[p] < 0: buys -= prev[p]  # buy to cover
            perf.loc[pordate, 'buys'] = abs(buys)
            perf.loc[pordate, 'sells'] = abs(sells)

            # derive end-of-period stock weights by stocks' returns
            prev = port['weights'] * (1 + port['ret'])
            for i in range(len(smooth)):
                smooth[i]['weights'] = smooth[i]['weights'] * (1 + ret.reindex(smooth[i].index)['ret'])
            print_debug("(performance) %d %d -%d %.4f" %
                        (pordate, perf.loc[pordate,'longs'],
                         perf.loc[pordate,'shorts'], perf.loc[pordate,'ret']))
            
        self.perf = perf
        t = 252/(sum((self.rf.index>=self.perf.index[0])&(self.rf.index <=self.perf.index[-1]))/
                 (len(self.perf)-1))   # number of intervals in a year
        self._str = label
        return len(perf)

    def save(self, label=None):
        """replace backtest performance returns in saved database table."""
        table = self._schema['table']
        if not self.sql.exists_table(table):
            self.sql.create_table(**self._schema)
        self.sql.delete(table, where={'permno' : self._str})
        self.perf['permno'] = label if label else self._str
        self.sql.load_dataframe(table, self.perf)

    def load(self, label):
        """load backtest performance returns from saved database table."""
        d = self.sql.select(self._schema['table'], where={'permno' : label})
        return DataFrame(d).drop(columns=['permno']).sort_values(['endret'])

    def get_series(self, field, start=19000000, end=29001231):
        """retrieve saved backtest returns as a series"""
        q = "SELECT endret as date, permno, {field} FROM backtests" \
            " WHERE endret >= {start} AND endret <= {end}" \
            "".format(field=field, start=int(start), end=int(end))
        print_debug('(get_series) ' + q)
        return DataFrame(**self.sql.run(q, fetch=True)).pivot(
            index='date', columns='permno', values=field)

    
    def attribution(self, benchnames, haclags=1, label=None, flip=None):
        """Compute performance attribution 

        Parameters
        ----------
        benchnames: list of string
            names of benchmark returns to compute attribution against
        haclags: int, optional
            option for robustcov statistics, number of Newey-West lags
        flip: bool, optional
            None (default): auto-detect and flip returns to be positive, False: never flip, True: always

        Returns
        -------
        summary: statsmodels get_robustcov_results.summary()
            display summary of results from fitting statsmodels OLS of excess returns on benchmarks

        Notes
        -----
        self.annualized is dict contain performance ratios:
          'excess' : annualized average return ub excess if the risk-free rate
          'sharpe' : annualized sharpe ratio
          'jensen': annualized jensen alpha
          'appraisal' : annualized appraisal ratio
          'welch-t' : Welch test t-stat for structural break of mean excess returns after 2002
          'welch-p' : Welch test p-value for structural break of mean excess returns after 2002
          'turnover' : annualized total turnover rate
          'buys' : annualized buy rate
          'sells' : annualized sell rate
        self.results is statsmodels OLS results
        """
        if label is not None:
            self._str = label
        if flip is None:
            flip = np.prod(1-self.perf['excess']) > 1
        self.flip = flip
        df = self.bench.get_series(benchnames, 'ret')
        for b in benchnames:
            self.perf[b] = compound_ret(df[b], self.perf['begret'], self.perf['endret'])
        self.perf['intercept'] = 1
        n = len(self.perf)
        if self.flip:
            excess = -self.perf['excess'].rename('MINUS ' + self._str)
        else:
            excess = self.perf['excess'].rename(self._str)
        results = sm.OLS(excess, self.perf[['intercept'] + benchnames]).fit()
        results = results.get_robustcov_results(cov_type='HAC', use_t=None, maxlags=haclags)
        t = 252/(sum((self.rf.index>=self.perf.index[0])&(self.rf.index <=self.perf.index[-1]))/
                 (len(self.perf)-1))
        pre2002 = self.perf.loc[self.perf.index <= 20021231, 'excess']
        post2002 = self.perf.loc[self.perf.index > 20021231, 'excess']
        welch = sp.stats.ttest_ind(pre2002, post2002, equal_var=False)
        self.annualized = {'excess' : t*np.mean(excess),
                           'sharpe' : np.sqrt(t)*np.mean(excess)/np.std(excess),
                           'jensen': t*results.params[0],
                           'appraisal' : np.sqrt(t) * results.params[0] / np.std(results.resid),
                           'welch-t' : welch[0],
                           'welch-p' : welch[1],
                           'turnover' : t*np.mean(abs(np.mean(self.perf[['buys','sells']])))/2,
                           'buys' : t*np.mean(self.perf['buys'])/2,
                           'sells' : t*np.mean(self.perf['sells'])/2}
        self.results = results
        return results.summary()

    def plot(self, benchnames=[], label=None, plotrisk=True, savefig=None):
        """Plot time series of returns, and ewma rolling volatility (3 month half life)

        Parameters
        ----------
        benchnames: list of strings
            names of benchmark returns in columns of self.perf to plot
        label : string, optional
            label of dependent variable
        plotrisk: bool, optional
            to plot ewma risk on second y-axis (default is True)
        savefig: str, optional
            full file name to save figure (default is to plt.show())
        """
        if label is None:
            label = ('MINUS ' if self.flip else '') + self._str
        days = (sum((self.rf.index >= self.perf.index[0]) &
                    (self.rf.index <= self.perf.index[-1])) / (len(self.perf)-1))
        t = 252/days
        perf = self.perf[['excess'] + list(benchnames)].copy()
        perf['excess'] *= (-1 if self.flip else 1)
        plt.figure(figsize=(6.5, 8))
        plt.clf()
        plt.subplot(2, 1, 1)
        plot_returns(x = pd.to_datetime(list(self.perf['endret']), format='%Y%m%d'), y1 = perf,
                     y2 = np.sqrt((perf**2).ewm(alpha=0.989**days).mean()*t) if plotrisk else None,
                     labels = [label] + list(benchnames))
        plt.subplot(2, 1, 2)
        plot_holdings(x=pd.to_datetime(list(self.perf['endret']), format='%Y%m%d'),
                      y1=self.perf[['longs','shorts']], y2=(self.perf['buys']+self.perf['sells'])/4,
                      labels=['#longs','#shorts'])
        if savefig:
            plt.savefig(savefig)


def run_backtest(backtest, dataset, signal, window, benchnames, rebalbeg, rebalend, data,
                 outdir = '',  html = 'index.html',  flip=None):
    """wrapper to run pipeline of backtest methods, and (optionally) save to file and .jpg

    Parameters
    ----------
    backtest : BackTest instance
        class to contain computed back results
    dataset : Structured instance
        where securities returns can be retrieved from (e.g. CRSP)
    signal : string
        label of signal to backtest
    window : int
        number of months to look back for signal value
    benchnames : list of strings
        names of benchmarks to compare portfolio returns to
    rebalbeg : int
        date of first month to start backtest portfolio rebalance
    rebalend : int
        last holding date of backtest
    data : DataFrame or Signals
        where signal values can be retrieved from.
        if DataFrame, then columns are ['permno','rebaldate', label]
    """
    holdings = dataset.portfolio_sorts(signal, data=data, beg=rebalbeg, end=rebalend, window=window)
    backtest.performance(dataset, holdings, label = signal)
    backtest.save()
    s = backtest.attribution(benchnames, flip=flip)
    backtest.plot(benchnames, savefig = outdir + signal + '.jpg' if outdir else None)
    if html:
        with open(outdir + html, 'at') as f:
            f.write("<pre>\n")
            f.write(str(s) + "\n\nAnnualized performance and turnover\n")
            f.writelines("\n".join('%-10s:%8.4f' % (k,v) for k,v in backtest.annualized.items()))
            f.write("\n</pre>\n")
            f.write('<img src="{}"><hr><p>\n'.format(signal + '.jpg'))
    else:
        print(s)
        print(pd.Series(backtest.annualized))
        plt.show()


if __name__ == '__main__':
    #
    # a backtest example of past(2,12) momentum with monthly rebalances
    #
    tic = time.time()
    crsp.rdb = rdb   # reaffirm use redis cache
    rebalbeg, rebalend = 19640601, 20190630    
    past = (2,12)
    holdings = {}
    for pordate in bd.endmo_range(rebalbeg, rebalend):  
        df = crsp.get_universe(pordate)

        beg = bd.endmo(pordate, -past[1])
        start = bd.shift(beg, 1)
        end = bd.endmo(pordate, 1-past[0])
        df['beg'] = crsp.get_section('daily', ['prc'], 'date', beg).reindex(df.index)
        df['end'] = crsp.get_section('daily', ['prc'], 'date', end).reindex(df.index)
        df['ret'] = crsp.get_section('monthly', ['ret'], 'date', end).reindex(df.index)
        df['signal'] = crsp.get_ret(start, end).reindex(df.index)
        df = df[df['beg'].gt(0) & df['end'].gt(0) & df['prc'].gt(0) & df['ret'].notnull() &
                df['signal'].notnull()]
        df['fractile'] = fractiles(df['signal'], [30,70], df['signal'][df.nyse])
        permnos, weights = [],[]            
        subs = [(df.fractile == 3) & (df.deciles > 5),              # big winner subportfolio
                (df.fractile == 3) & (df.deciles <= 5),             # small winner subportfolio
                (df.fractile == 1) & (df.deciles > 5),              # big loser subportfolio
                (df.fractile == 1) & (df.deciles <= 5)]             # small loser subportfolio
        for sub, weight in zip(subs, [0.5, 0.5, -0.5, -0.5]):       # combine subportfolios
            weights += list(weight * df.loc[sub,'cap'] / df.loc[sub,'cap'].sum())
            permnos += list(df.index[sub])
        holdings[pordate] = DataFrame(data=weights, index=permnos, columns=['weights'])

    benchnames = ['Mom(mo)']
    backtest.performance(crsp, {k:holdings[k] for k in holdings if k >= rebalbeg and k <= rebalend},
                         label=("PAST(%d,%d)" % past), overlap=1)
    backtest.attribution(bench, benchnames, flip=False)
    backtest.plot(benchnames)
    print('Elapsed %.1f secs ' + str(time.time() - tic))
    print(backtest.results.summary())
    print(backtest.annualized)
    plt.show()
