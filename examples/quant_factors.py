"""Factor Investing

- return predicting signals (Green, Hand and Zhang, 2013, and others)
- CRSP, Compustat, IBES


Copyright 2022, Terence Lim

MIT License
"""
#
# TODO: check as_rolling
#
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time
from datetime import datetime
from finds.database import SQL, Redis
from finds.structured import Stocks, PSTAT, CRSP, IBES, Benchmarks, \
    Signals, SignalsFrame
from finds.busday import BusDay, WeeklyDay
from finds.structured import SignalsFrame
from finds.backtesting import BackTest
from finds.recipes import fractiles, maximum_drawdown
from conf import credentials, CRSP_DATE, VERBOSE, paths
from typing import List, Tuple, Any, Dict

LAST_DATE = CRSP_DATE

imgdir = paths['images']
sql = SQL(**credentials['sql'], verbose=VERBOSE)
user = SQL(**credentials['user'], verbose=VERBOSE)

rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
pstat = PSTAT(sql, bd, verbose=VERBOSE)
bench = Benchmarks(sql, bd)
signals = Signals(user)
ibes = IBES(sql, bd, verbose=VERBOSE)

backtest = BackTest(user, bench, 'RF', LAST_DATE)
outdir = os.path.join(paths['images'], 'factors')

# signals to flip signs when forming spread portfolios
leverage = {'mom1m':-1, 'mom36m':-1, 'pricedelay':-1, 'absacc':-1, 'acc':-1,
            'agr':-1, 'chcsho':-1, 'egr':-1, 'mve_ia':-1, 'pctacc':-1,
            'aeavol':-1, 'disp':-1, 'stdacc':-1, 'stdcf':-1, 'secured':-1,
            'maxret':-1, 'ill':-1, 'zerotrade':-1, 'cashpr':-1, 'chinv':-1,
            'invest':-1, 'cinvest':-1, 'idiovol':-1, 'retvol':-1}

## Helpers to lag characteristics and roll returns
def as_lags(df, var, key, nlags):
    """Return dataframe with {nlags} of column {var}, same {key} value in row"""
    out = df[[var]].rename(columns={var: 0})      # first col: not shifted
    for i in range(1, nlags):
        prev = df[[key, var]].shift(i, fill_value=0) # next col: shifted i+1
        prev.loc[prev[key] != df[key], :] = np.nan   # require same {key} value
        out.insert(i, i, prev[var])
    return out

def as_rolling(df, other, width=0, dropna=True):
    """enqueue next dataframe to a sliding window of fixed width of columns"""
    df = df.join(other, how='outer', sort=True, lsuffix='l', rsuffix='r')
    if width and len(df.columns) > width:          # if wider than width
        df = df.iloc[:, (len(df.columns)-width):]  #    then drop first cols
    if dropna:                                     # drop empty rows
        df = df[df.count(axis=1) > 0]
    return df

## Helpers to sort into spread portfolios, and run backtests
def portfolio_sorts(stocks: Stocks,
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
    """Generate monthly time series of holdings by standard sort procedure
        
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

def backtest_pipeline(backtest: BackTest,
                      stocks: Stocks,
                      holdings: DataFrame,
                      label: str,
                      benchnames: List[str],
                      suffix: str = '',
                      overlap: int = 0,
                      outdir: str ='',
                      num: int = 1) -> DataFrame:
    """wrapper to run a backtest pipeline, and (optionally) save file and .jpg

    Args:
      backtest: To compute backtest results
      stocks: Where securities returns can be retrieved from (e.g. CRSP)
      holdings: dict (key int date) of Series holdings (key permno)
      label: Label of signal to backtest
      benchnames: Names of benchmarks to attribute portfolio performance
      overlap: Number of overlapping holdings to smooth
      num: Figure num to plot to

    Returns:
      DataFrame of performance returns in rows

    Notes:
      graph and summary statistics are output to jpg and (appended) html
      backtest object updated with performance and attribution data
    """
    summary = backtest(stocks, holdings, label, overlap=overlap)
    excess = backtest.fit(benchnames)
    backtest.write(label)
    backtest.plot(num=num, label=label + suffix)
    print(pd.Series(backtest.annualized,
                    name=label + suffix).to_frame().T.round(3).to_string())
    if outdir:  # output graph and summary statistics
        plt.savefig(os.path.join(outdir, label + '.jpg'))
        
        # performance metrics from backtest to output
        sub = ['alpha', 'excess', 'inforatio', 'sharpe', 'welch-t', 'welch-p']
        with open(os.path.join(outdir, 'index.html'), 'at') as f:
            f.write(f"<p><hr><h2>{label + suffix}</h2>\n<pre>\n")
            f.write("{}-{} {}\n".format(min(backtest.excess.index),
                                        max(backtest.excess.index),
                                        benchnames))
            f.write("{:12s} ".format("Annualized"))
            f.write("".join(f"{k:>10s}" for k in sub) + "\n")
            f.write("{:12s} ".format(label + ":"))
            f.write("".join(f"{backtest.annualized[k]:10.4f}" for k in sub))
            f.write(f"\n</pre>\n<img src='{label}.jpg'><p>{datetime.now()}\n")
    return summary


testable = {'new', 'monthly', 'weekly', 'daily', 'pstann', 'pstqtr',
            'ibesf1', 'ibesltg', 'rdq_daily', 'ibesf1_pstqtr',
            'ibesq1_pstqtr', 'summarize'}  # subset of steps to rerun
regenerate = True  # False to only run backtest (and not regenerate Signals)
plottable = False  # Whether to pause and show after generating plots
def plt_show(plottable=plottable):
    if plottable:
        plt.show()


if 'new' in testable:
    with open(os.path.join(outdir, 'index.html'), 'wt') as f:
        f.write('<h1>Quant Factors</h1><br>')
        f.write(' <br>')
        f.write('<p>\n')

# Momentum and divyld from CRSP monthly
if 'monthly' in testable:
    if regenerate:
        beg, end = 19251231, LAST_DATE
        intervals = {'mom12m': (2,12),
                     'mom36m': (13,36),
                     'mom6m': (2,6),
                     'mom1m': (1,1)}
        for label, past in intervals.items():
            out = []
            rebaldates = bd.date_range(bd.endmo(beg, past[1]), end, 'endmo')
            for rebaldate in rebaldates:
                start = bd.endmo(rebaldate, -past[1])
                beg1 = bd.offset(start, 1)
                end1 = bd.endmo(rebaldate, 1-past[0])
                df = crsp.get_universe(end1)
                df['start'] = crsp.get_section(dataset='monthly',
                                               fields=['ret'],
                                               date_field='date',
                                               date=start).reindex(df.index)
                df[label] = crsp.get_ret(beg1, end1).reindex(df.index)
                df['permno'] = df.index
                df['rebaldate'] = rebaldate
                df = df.dropna(subset=['start'])
                out.append(df[['rebaldate', 'permno', label]]) # append rows
            out = pd.concat(out, axis=0, ignore_index=True)
            n = signals.write(out, label, overwrite=True)

        beg, end = 19270101, LAST_DATE
        columns = ['chmom', 'divyld', 'indmom']
        out = []
        for rebaldate in bd.date_range(beg, end, 'endmo'):
            start = bd.endmo(rebaldate, -12)
            beg1 = bd.offset(start, 1)
            end1 = bd.endmo(rebaldate, -6)
            beg2 = bd.offset(end1, 1)
            end2 = bd.endmo(rebaldate)
            df = crsp.get_universe(end1)
            df['start'] = crsp.get_section(dataset='monthly',
                                           fields=['ret'],
                                           date_field='date',
                                           date=start).reindex(df.index)
            df['end2'] = crsp.get_section(dataset='monthly',
                                          fields=['ret'],
                                          date_field='date',
                                          date=end2).reindex(df.index)
            df['mom2'] = crsp.get_ret(beg2, end2).reindex(df.index)
            df['mom1'] = crsp.get_ret(beg1, end1).reindex(df.index)
            df['divyld'] = crsp.get_divamt(beg1, end2)\
                               .reindex(df.index)['divamt']\
                               .div(df['cap'])\
                               .fillna(0)
            df['chmom'] = df['mom1'] - df['mom2']

            # 6-month two-digit sic industry momentum (group means of 'mom1')
            df['sic2'] = df['siccd'] // 100
            df = df.join(DataFrame(df.groupby(['sic2'])['mom1'].mean()).rename(
                columns={'mom1': 'indmom'}), on='sic2', how='left')
            df['permno'] = df.index
            df['rebaldate'] = rebaldate
            out.append(df.dropna(subset=['start','end2'])\
                       [['rebaldate', 'permno'] + columns])
        out = pd.concat(out, axis=0, ignore_index=True)
        for label in columns:   # save signal values to sql
            n = signals.write(out, label, overwrite=True)

    benchnames = ['Mkt-RF(mo)']
    rebalbeg, rebalend = 19260101, LAST_DATE
    for num, label in enumerate(['mom12m', 'mom6m', 'chmom', 'indmom',
                                 'divyld', 'mom1m', 'mom36m']):
        holdings = portfolio_sorts(crsp,
                                   label,
                                   SignalsFrame(signals.read(label)),
                                   rebalbeg,
                                   rebalend,
                                   window=1,
                                   months=[],
                                   leverage=leverage.get(label, 1))
        excess = backtest_pipeline(backtest,
                                   crsp,
                                   holdings,
                                   label,
                                   benchnames,
                                   overlap=0,
                                   outdir=outdir,
                                   num=num+1,
                                   suffix=(leverage.get(label, 1) < 0)*'(-)')
    plt_show()

## Helper method to calculate beta, idiovol and price delay from weekly returns
def regress(x: np.array, y: np.array) -> Tuple[float, float, float]:
    """helper method to calculate beta, idiovol and price delay

    Args:
      x: equal-weighted market returns (in ascending time order)
      y: stock returns (in ascending time order).  NaN's will be discarded.

    Returns:
      beta: slope from regression on market returns and intercept
      idiovol: mean squared error of residuals
      pricedelay: increase of adjusted Rsq withfour market lags over without
    """
    v = np.logical_not(np.isnan(y))
    y = y[v]
    x = x[v]
    n0 = len(y)
    A0 = np.vstack([x, np.ones(len(y))]).T
    b0 = np.linalg.inv(A0.T.dot(A0)).dot(A0.T.dot(y))   # univariate coeffs
    sse0 = np.mean((y - A0.dot(b0))**2)
    sst0 = np.mean((y - np.mean(y))**2)
    if (sst0>0 and sse0>0):
        R0 = (1 - ((sse0 / (n0 - 2)) / (sst0 / (n0 - 1))))
    else:
        R0 = 0
    y4 = y[4:]
    n4 = len(y4)         
    A4 = np.vstack([x[0:-4], x[1:-3], x[2:-2], x[3:-1], x[4:],
                    np.ones(n4)]).T
    b4 = np.linalg.inv(A4.T.dot(A4)).dot(A4.T.dot(y4))  # four lagged coeffs
    sse4 = np.mean((y4 - A4.dot(b4))**2)
    sst4 = np.mean((y4 - np.mean(y4))**2)
    if sst4 > 0 and sse4 > 0:
        R4 = (1 - ((sse4 / (n4 - 6)) / (sst4 / (n4 - 1))))
    else:
        R4 = 0    
    return [b0[0],
            sse0 or np.nan,
            (1 -(R0 / R4)) if R0>0 and R4>0 else np.nan]

    
## Weekly returns-based price response signals
if 'weekly' in testable:    
    beg, end = 19260101, LAST_DATE
    columns  = ['beta','idiovol','pricedelay']
    wd = WeeklyDay(sql, 'Fri', beg, end)  # custom weekly trading day calendar
        
    width    = 3*52+1           # up to 3 years of weekly returns
    minvalid = 52               # at least 52 weeks required to compute beta
    weekly   = DataFrame()      # rolling window of weekly stock returns
    mkt      = DataFrame()      # to queue equal-weighted market returns
    out      = []               # to accumulate final calculations

    if regenerate:

        """
        # Pre-generate weekly returns and save in cache-store
        batchsize = 40
        r = wd.date_tuples(wd.date_range(beg, LAST_DATE))
        batches = [r[i:(i+batchsize)] for i in range(0, len(r), batchsize)]
        for batch in batches:
            crsp.cache_ret(batch, replace=True)
        """
        
        for date in wd.date_range(beg, end):
            df = crsp.get_ret(wd.begwk(date), date)
            mkt = as_rolling(mkt,       # rolling window of weekly mkt returns
                             DataFrame(data=[df.mean()], columns=[date]),
                             width=width)
            weekly = as_rolling(weekly, # rolling window of weekly stock returns
                                df.rename(date),
                                width=width) 
            valid = weekly.count(axis=1) >= minvalid  # require min number weeks
            if valid.any():
                result = DataFrame([regress(mkt.values[0], y)
                                    for y in weekly.loc[valid].values],
                                   columns=columns)
                result['permno'] = weekly.index[valid].values
                result['rebaldate'] = date
                if wd.ismonthend(date): # signal value from last week of month
                    out.append(result)
        out = pd.concat(out, axis=0, ignore_index=True)
        for label in columns:
            signals.write(out, label, overwrite=True)

    benchnames = ['Mkt-RF(mo)']
    rebalbeg, rebalend = 19290601, LAST_DATE
    for num, label in enumerate(columns):
        holdings = portfolio_sorts(crsp,
                                   label,
                                   SignalsFrame(signals.read(label)),
                                   rebalbeg,
                                   rebalend,
                                   window=1,
                                   months=[],
                                   leverage=leverage.get(label, 1))
        excess = backtest_pipeline(backtest,
                                   crsp,
                                   holdings,
                                   label,
                                   benchnames,
                                   overlap=0,
                                   outdir=outdir,
                                   num=num+1,
                                   suffix=(leverage.get(label, 1) < 0)*'(-)')

    plt_show()

## Liquidity signals from daily stock returns
if 'daily' in testable:
    beg, end = 19830601, LAST_DATE    # nasdaq/volume from after 1982
    columns = ['ill', 'maxret', 'retvol', 'baspread', 'std_dolvol',
               'zerotrade', 'std_turn', 'turn']
    if regenerate:
        tic = time.time()
        out = []
        dolvol = []
        turn = DataFrame()    # to average turn signal over rolling 3-months
        dt = bd.date_range(bd.begmo(beg,-3), end, 'endmo') # monthly rebalances
        chunksize = 12        # each chunk is 12 months (1 year)
        chunks = [dt[i:(i+chunksize)] for i in range(0, len(dt), chunksize)]
        for chunk in chunks:
            q = (f"SELECT permno, date, ret, askhi, bidlo, prc, vol, shrout "
                 f" FROM {crsp['daily'].key}"
                 f" WHERE date>={bd.begmo(chunk[0])}"
                 f"   AND date<={chunk[-1]}")        # retrieve a chunk
            f = crsp.sql.read_dataframe(q).sort_values(['permno', 'date'])
            f['baspread'] = ((f['askhi'] - f['bidlo']) /
                             ((f['askhi'] + f['bidlo']) / 2))
            f['dolvol'] = f['prc'].abs() * f['vol']
            f['turn1'] = f['vol'] / f['shrout']
            f.loc[f['dolvol']>0, 'ldv'] = np.log(f.loc[f['dolvol']>0, 'dolvol'])
            f['ill'] = 1000000 * f['ret'].abs() / f['dolvol']
            print(q, len(f), int(time.time() - tic))

            for rebaldate in chunk:            # for each rebaldate in the chunk
                grouped = f[f['date'].ge(bd.begmo(rebaldate))
                            & f['date'].le(rebaldate)].groupby('permno')
                df = grouped[['ret']].max().rename(columns={'ret': 'maxret'})
                df['retvol'] = grouped['ret'].std()
                df['baspread'] = grouped['baspread'].mean()
                df['std_dolvol'] = grouped['ldv'].std()
                df['ill'] = grouped['ill'].mean()
                dv = grouped['dolvol'].sum()
                df.loc[dv > 0, 'dolvol'] = np.log(dv[dv > 0])
                df['turn1'] = grouped['turn1'].sum()
                df['std_turn'] = grouped['turn1'].std()
                df['countzero'] = grouped['vol'].apply(lambda v: sum(v==0))
                df['ndays'] = grouped['prc'].count()
            
                turn = as_rolling(turn, df[['turn1']], width=3)
                df['turn'] = turn.reindex(df.index).mean(axis=1, skipna=False)
                df.loc[df['turn1'].le(0), 'turn1'] = 0
                df.loc[df['ndays'].le(0), 'ndays'] = 0
                df['zerotrade'] = ((df['countzero'] + ((1/df['turn1'])/480000))
                                   * 21/df['ndays'])
    
                df['rebaldate'] = rebaldate
                df = df.reset_index()
                out.append(df[['permno', 'rebaldate'] + columns])
                if rebaldate < bd.endmo(end):
                    df['rebaldate'] = bd.endmo(rebaldate, 1)
                    dolvol.append(df[['permno','rebaldate','dolvol']])
        out = pd.concat(out, axis=0, ignore_index=True)
        dolvol = pd.concat(dolvol, axis=0, ignore_index=True)            

        for label in columns:
            n = signals.write(out, label, overwrite=True)
        n = signals.write(dolvol, 'dolvol', overwrite=True)

    rebalbeg, rebalend = 19830601, LAST_DATE
    benchnames = ['Mkt-RF(mo)']
    for num, label in enumerate(columns + ['dolvol']):
        holdings = portfolio_sorts(crsp,
                                   label,
                                   SignalsFrame(signals.read(label)),
                                   rebalbeg,
                                   rebalend,
                                   window=1,
                                   months=[],
                                   leverage=leverage.get(label, 1))
        excess = backtest_pipeline(backtest,
                                   crsp,
                                   holdings,
                                   label,
                                   benchnames,
                                   overlap=0,
                                   outdir=outdir,
                                   num=num+1,
                                   suffix=(leverage.get(label, 1) < 0)*'(-)')
    plt_show()

## Fundamental signals from Compustat Annual
if 'pstann' in testable:
    columns = ['absacc', 'acc', 'agr', 'bm', 'cashpr', 'cfp', 'chcsho',
               'chinv', 'depr', 'dy', 'egr', 'ep', 'gma', 'grcapx',
               'grltnoa', 'hire', 'invest', 'lev', 'lgr' ,
               'pchdepr', 'pchgm_pchsale', 'pchquick',
               'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga',
               'pchsaleinv', 'pctacc', 'quick', 'rd_sale', 'rd_mve',
               'realestate', 'salecash', 'salerec', 'saleinv', 'secured',
               'sgr', 'sp', 'tang', 'bm_ia', 'cfp_ia', 'chatoia' , 'chpmia',
               'pchcapx_ia', 'chempia', 'mve_ia']
    numlag = 6       # number of months to lag data for rebalance
    end = LAST_DATE   # last data date

    if regenerate:
        # retrieve annual, keep [permno, datadate] with non null prccq if any
        fields = ['sic', 'fyear', 'ib', 'oancf', 'at', 'act', 'che', 'lct',
                  'dlc', 'dltt', 'prcc_f', 'csho', 'invt', 'dp', 'ppent',
                  'dvt', 'ceq', 'txp', 'revt', 'cogs', 'rect', 'aco', 'intan',
                  'ao', 'ap', 'lco', 'lo', 'capx', 'emp', 'ppegt', 'lt',
                  'sale', 'xsga', 'xrd', 'fatb', 'fatl', 'dm']
        df = pstat.get_linked(
            dataset='annual',
            fields=fields,
            date_field='datadate',
            where=(f"indfmt = 'INDL' "
                   f"  AND datafmt = 'STD'"
                   f"  AND curcd = 'USD' "
                   f"  AND popsrc = 'D'"
                   f"  AND consol = 'C'"
                   f"  AND datadate <= {end//100}31"))
        fund = df.sort_values(['permno', 'datadate', 'ib'])\
                 .drop_duplicates(['permno', 'datadate'])\
                 .dropna(subset=['ib'])
        fund.index = list(zip(fund['permno'], fund['datadate']))  # multi-index
        fund['rebaldate'] = bd.endmo(fund.datadate, numlag)

        # precompute, and lag common metrics: mve_f avg_at sic2
        fund['sic2'] = np.where(fund['sic'].notna(),
                                fund['sic'] // 100, 0)
        fund['fyear'] = fund['datadate'] // 10000      # can delete this
        fund['mve_f'] = fund['prcc_f'] * fund['csho']
    
        lag = fund.shift(1, fill_value=0)
        lag.loc[lag['permno'] != fund['permno'], fields] = np.nan
        fund['avg_at'] = (fund['at'] + lag['at']) / 2
    
        lag2 = fund.shift(2, fill_value=0)
        lag2.loc[lag2['permno'] != fund['permno'], fields] = np.nan
        lag['avg_at'] = (lag['at'] + lag2['at']) / 2
    
        fund['bm'] = fund['ceq'] / fund['mve_f']
        fund['cashpr'] = (fund['mve_f'] + fund['dltt'] - fund['at'])/fund['che']
        fund['depr'] = fund['dp'] / fund['ppent']
        fund['dy'] = fund['dvt'] / fund['mve_f']
        fund['ep'] = fund['ib'] / fund['mve_f']
        fund['lev'] = fund['lt'] / fund['mve_f']
        fund['quick'] = (fund['act'] - fund['invt']) / fund['lct']
        fund['rd_sale'] = fund['xrd'] / fund['sale']
        fund['rd_mve'] = fund['xrd'] / fund['mve_f']
        fund['realestate'] = ((fund['fatb'] + fund['fatl']) /
                              np.where(fund['ppegt'].notna(),
                                       fund['ppegt'], fund['ppent']))
        fund['salecash'] = fund['sale'] / fund['che']
        fund['salerec'] = fund['sale'] / fund['rect']
        fund['saleinv'] = fund['sale'] / fund['invt']
        fund['secured'] = fund['dm'] / fund['dltt']
        fund['sp'] = fund['sale'] / fund['mve_f']
        fund['tang'] = (fund['che']
                        + fund['rect'] * 0.715
                        + fund['invt'] * 0.547
                        + fund['ppent'] * 0.535) / fund['at']

        # changes: agr chcsho chinv egr gma egr grcapx grltnoa emp invest lgr
        fund['agr'] = (fund['at'] / lag['at'])
        fund['chcsho'] = (fund['csho'] / lag['csho'])
        fund['chinv'] = ((fund['invt'] - lag['invt']) / fund['avg_at'])
        fund['egr'] = (fund['ceq'] / lag['ceq'])
        fund['gma'] = ((fund['revt'] - fund['cogs']) / lag['at'])
        fund['grcapx'] = (fund['capx'] / lag2['capx'])
        fund['grltnoa'] =  (
            ((fund['rect']
              + fund['invt']
              + fund['ppent']
              + fund['aco']
              + fund['intan']
              + fund['ao']
              - fund['ap']
              - fund['lco']
              - fund['lo'])
             / (lag['rect']
                + lag['invt']
                + lag['ppent']
                + lag['aco']
                + lag['intan']
                + lag['ao']
                - lag['ap']
                - lag['lco']
                - lag['lo']))
            - ((fund['rect']
                + fund['invt']
                + fund['aco']
                - fund['ap']
                - fund['lco'])
               - (lag['rect']
                  + lag['invt']
                  + lag['aco']
                  - lag['ap']
                  - lag['lco']))) / fund['avg_at']
        fund['hire'] = ((fund['emp'] / lag['emp']) - 1).fillna(0)
        fund['invest'] = (((fund['ppegt'] - lag['ppegt'])
                           + (fund['invt'] - lag['invt'])) / lag['at'])
        fund['invest'] = fund['invest']\
            .where(fund['invest'].notna(),
                   ((fund['ppent'] - lag['ppent'])
                    + (fund['invt'] - lag['invt'])) / lag['at'])
        fund['lgr'] = (fund['lt'] / lag['lt'])
        fund['pchdepr'] = ((fund['dp'] / fund['ppent'])
                           / (lag['dp'] / lag['ppent']))
        fund['pchgm_pchsale'] = (((fund['sale'] - fund['cogs'])
                                  / (lag['sale'] - lag['cogs']))
                                 - (fund['sale'] / lag['sale']))
        fund['pchquick'] = (((fund['act'] - fund['invt']) / fund['lct'])
                            / ((lag['act'] - lag['invt']) / lag['lct']))
        fund['pchsale_pchinvt'] = ((fund['sale'] / lag['sale'])
                                   - (fund['invt'] / lag['invt']))
        fund['pchsale_pchrect'] = ((fund['sale'] / lag['sale'])
                                   - (fund['rect'] / lag['rect']))
        fund['pchsale_pchxsga'] = ((fund['sale'] / lag['sale'])
                                   - (fund['xsga'] / lag['xsga']))
        fund['pchsaleinv'] = ((fund['sale'] / fund['invt'])
                              / (lag['sale'] / lag['invt']))
        fund['sgr'] = (fund['sale'] / lag['sale'])

        fund['chato'] = ((fund['sale'] / fund['avg_at'])
                         - (lag['sale'] / lag['avg_at']))
        fund['chpm'] = (fund['ib'] / fund['sale']) - (lag['ib'] / lag['sale'])
        fund['pchcapx'] = fund['capx'] / lag['capx']
    
        # compute signals with alternative definitions: acc absacc cfp
        fund['_acc'] = (((fund['act'] - lag['act'])
                         - (fund['che'] - lag['che']))
                        - ((fund['lct'] - lag['lct'])
                           - (fund['dlc'] - lag['dlc'])
                           - (fund['txp'] - lag['txp'])
                           - fund['dp']))
        fund['cfp'] = ((fund['ib'] - (((fund['act'] - lag['act'])
                                       - (fund['che'] - lag['che']))
                                      - ((fund['lct'] - lag['lct'])
                                         - (fund['dlc'] - lag['dlc'])
                                         - (fund['txp'] - lag['txp'])
                                         - fund['dp'])))
                       / fund['mve_f'])
        g = fund['oancf'].notnull()
        fund.loc[g, 'cfp'] = fund.loc[g, 'oancf'] / fund.loc[g, 'mve_f']
        fund.loc[g, '_acc'] = fund.loc[g, 'ib'] - fund.loc[g, 'oancf']
        fund['acc'] = fund['_acc'] / fund['avg_at']
        fund['absacc'] = abs(fund['_acc']) / fund['avg_at']
        fund['pctacc'] = fund['_acc'] / abs(fund['ib'])
        h = (fund['ib'].abs() <= 0.01)
        fund.loc[h, 'pctacc'] = fund.loc[h, '_acc'] / 0.01

        # industry-adjusted
        cols = {'bm_ia': 'bm', 'cfp_ia': 'cfp', 'chatoia': 'chato',
                'chpmia': 'chpm', 'pchcapx_ia': 'pchcapx',
                'chempia': 'hire', 'mve_ia': 'mve_f'}
        group = fund.groupby(['sic2', 'fyear'])
        for k,v in cols.items():
            fund[k] = fund[v] - group[v].transform('mean')

        for label in columns:
            signals.write(fund, label, overwrite=True)

    rebalbeg, rebalend = 19700101, LAST_DATE
    benchnames = ['Mkt-RF(mo)'] #['Mom']  #['ST_Rev(mo)']   #
    for num, label in enumerate(columns):
        holdings = portfolio_sorts(crsp,
                                   label,
                                   SignalsFrame(signals.read(label)),
                                   rebalbeg,
                                   rebalend,
                                   window=12,
                                   months=[6],
                                   leverage=leverage.get(label, 1))
        excess = backtest_pipeline(backtest,
                                   crsp,
                                   holdings,
                                   label,
                                   benchnames,
                                   overlap=0,
                                   outdir=outdir,
                                   num=num+1,
                                   suffix=(leverage.get(label, 1) < 0)*'(-)')
    plt_show()

## Fundamental signals from Compustat Quarterly
if 'pstqtr' in testable:
    columns = ['stdacc', 'stdcf', 'roavol', 'sgrvol', 'cinvest', 'chtx',
               'rsup', 'roaq', 'cash', 'nincr']
    numlag = 4       # require 4 month lag of fiscal data
    end = LAST_DATE

    if regenerate:
        # retrieve quarterly, keep [permno, datadate] key with non null prccq
        fields = ['ibq', 'actq', 'cheq', 'lctq', 'dlcq', 'saleq', 'prccq',
                  'cshoq', 'atq', 'txtq', 'ppentq']
        df = pstat.get_linked(dataset='quarterly',
                              fields=fields,
                              date_field='datadate',
                              where=(f"datadate > 0 "
                                     f"and datadate <= {end//100}31"))
        fund = df.sort_values(['permno','datadate', 'ibq'])\
                 .drop_duplicates(['permno', 'datadate'])\
                 .dropna(subset=['ibq'])
        fund.index = list(zip(fund['permno'], fund['datadate']))
        rebaldate = bd.endmo(fund.datadate, numlag)

        # compute current and lagged: scf sacc roaq nincr cinvest cash rsup chtx
        lag = fund.shift(1, fill_value=0)
        lag.loc[lag['permno'] != fund['permno'], fields] = np.nan
        fund['_saleq'] = fund['saleq']
        fund.loc[fund['_saleq'].lt(0.01), '_saleq'] = 0.01
        
        fund['sacc'] = (((fund['actq'] - lag['actq'])
                         - (fund['cheq'] - lag['cheq']))
                        - ((fund['lctq'] - lag['lctq'])
                           - (fund['dlcq'] - lag['dlcq']))) / fund['_saleq']
        fund['cinvest'] = (fund['ppentq'] - lag['ppentq']) / fund['_saleq']
        fund['nincr'] = (fund['ibq'] > lag['ibq']).astype(int)
        fund['scf']  = (fund['ibq'] / fund['_saleq']) - fund['sacc']
        fund['roaq'] = (fund['ibq'] / lag['atq'])
        fund['cash'] = (fund['cheq'] / fund['atq'])
        
        lag4 = fund.shift(4, fill_value=0)
        lag4.loc[lag4['permno'] != fund['permno'], fields] = np.nan
        fund['rsup'] = ((fund['saleq'] - lag4['saleq'])
                        / (fund['prccq'].abs() * fund['cshoq'].abs()))
        fund['chtx'] = (fund['txtq'] - lag4['txtq']) / lag4['atq']

        # for each var: make dataframe of 15 lags (column names=[0,...,15])
        lags = {col : as_lags(fund, var=col, key='permno', nlags=16)
                for col in ['sacc', 'scf', 'roaq', 'rsup', 'cinvest', 'nincr']}
        for i in range(1, 16):                      # lags[ninrc][i]=1 iff ibq
            lags['nincr'][i] *= lags['nincr'][i-1]  # increasing all prior qtrs

        # compute signals from the 15 lags
        fund['rebaldate'] = rebaldate
        fund['stdacc'] = lags['sacc'].std(axis=1, skipna=False)
        fund['stdcf'] = lags['scf'].std(axis=1, skipna=False)
        fund['roavol'] = lags['roaq'].std(axis=1, skipna=False)
        fund['sgrvol'] = lags['rsup'].std(axis=1, skipna=False)
        fund['cinvest'] = (fund['cinvest'] -
                           lags['cinvest'][[1, 2, 3, 4]].mean(axis=1,
                                                              skipna=False))

        # count number of consecutive increasing quarters
        fund['nincr'] = lags['nincr'][np.arange(8)].sum(axis=1)

        for label in columns:
            signals.write(fund, label, overwrite=True)

    rebalbeg, rebalend = 19700101, LAST_DATE
    benchnames = ['Mkt-RF(mo)']
    for num, label in enumerate(columns):
        holdings = portfolio_sorts(crsp,
                                   label,
                                   SignalsFrame(signals.read(label)),
                                   rebalbeg,
                                   rebalend,
                                   window=3,
                                   months=[],
                                   leverage=leverage.get(label, 1))
        excess = backtest_pipeline(backtest,
                                   crsp,
                                   holdings,
                                   label,
                                   benchnames,
                                   overlap=0,
                                   outdir=outdir,
                                   num=num+1,
                                   suffix='(-)'*(leverage.get(label, 1) < 0))
    plt_show()

## IBES Fiscal Year 1 signals: chfeps, chnanalyst, disp
if 'ibesf1' in testable:
    columns = ['chfeps', 'chnanalyst', 'disp']

    if regenerate:
        df = ibes.get_linked(dataset='summary',
                             fields=['fpedats', 'meanest', 'medest',
                                     'stdev', 'numest'],
                             date_field = 'statpers',
                             where=("meanest IS NOT NULL "
                                    "  AND fpedats IS NOT NULL "
                                    "  AND statpers IS NOT NULL"
                                    "  AND fpi = '1'"))
        out = df.sort_values(['permno', 'statpers', 'fpedats', 'meanest'])\
                .drop_duplicates(['permno', 'statpers', 'fpedats'])
        out['rebaldate'] = bd.endmo(out['statpers'])

        out['disp'] = out['stdev'] / abs(out['meanest'])
        out.loc[abs(out['meanest']) < 0, 'disp'] = out['stdev'] / 0.01
    
        lag1 = out.shift(1, fill_value=0)
        f1 = (lag1['permno'] == out['permno'])        
        out.loc[f1, 'chfeps'] = out.loc[f1, 'meanest'] - lag1.loc[f1, 'meanest']

        lag3 = out.shift(3, fill_value=0)
        f3 = (lag3['permno'] == out['permno'])
        out.loc[f3, 'chnanalyst'] = out.loc[f3, 'numest']-lag3.loc[f3, 'numest']

        for label in columns:
            signals.write(out, label, overwrite=True)

    rebalbeg, rebalend = 19760101, LAST_DATE
    benchnames = ['Mkt-RF(mo)']
    for num, label in enumerate(columns):
        holdings = portfolio_sorts(crsp,
                                   label,
                                   SignalsFrame(signals.read(label)),
                                   rebalbeg,
                                   rebalend,
                                   window=3,
                                   months=[],
                                   leverage=leverage.get(label, 1))
        excess = backtest_pipeline(backtest,
                                   crsp,
                                   holdings,
                                   label,
                                   benchnames,
                                   overlap=0,
                                   outdir=outdir,
                                   num=num+1,
                                   suffix=(leverage.get(label, 1) < 0)*'(-)')
    plt_show()


## IBES Long-term Growth signals: fgr5yr
if 'ibesltg' in testable:
    columns = ['fgr5yr']

    if regenerate:
        df = ibes.get_linked(dataset='summary',
                             fields = ['meanest'],
                             date_field = 'statpers',
                             where=("meanest IS NOT NULL "
                                    "  AND fpi = '0'"
                                    "  AND statpers IS NOT NULL"))
        out = df.sort_values(['permno', 'statpers', 'meanest'])\
                .drop_duplicates(['permno', 'statpers'])\
                .dropna()
        out['rebaldate'] = bd.endmo(out['statpers'])
        out['fgr5yr'] = out['meanest']
        signals.write(out, 'fgr5yr', overwrite=True)

    rebalbeg, rebalend = 19760101, LAST_DATE
    benchnames = ['Mkt-RF(mo)']
    for num, label in enumerate(columns):
        holdings = portfolio_sorts(crsp,
                                   label,
                                   SignalsFrame(signals.read(label)),
                                   rebalbeg,
                                   rebalend,
                                   window=3,
                                   months=[],
                                   leverage=leverage.get(label, 1))
        excess = backtest_pipeline(backtest,
                                   crsp,
                                   holdings,
                                   label,
                                   benchnames,
                                   overlap=0,
                                   outdir=outdir,
                                   num=num+1,
                                   suffix=(leverage.get(label, 1) < 0)*'(-)')
    plt_show()
    
## Announcement date (rdq) in Quarterly, linked to CRSP daily: ear, aeavol
if 'rdq_daily' in testable:
    columns = ['ear', 'aeavol']

    if regenerate:
        # retrieve rdq, and set rebalance date to at least one month delay
        df = pstat.get_linked(dataset='quarterly',
                              fields=['rdq'],
                              date_field='datadate',
                              where=('rdq > 0'))
        fund = df.sort_values(['permno', 'rdq', 'datadate'])\
                 .drop_duplicates(['permno', 'rdq'])\
                 .dropna()
        fund['rebaldate'] = bd.offset(fund['rdq'], 2)

        # ear is compounded return around 3-day window
        out = crsp.get_window(dataset='daily',
                              field='ret',
                              date_field='date',
                              permnos=fund['permno'],
                              dates=fund['rdq'],
                              left=-1,
                              right=1)
        fund['ear'] = (1 + out).prod(axis = 1).values

                              
        # aeavol is avg volume in 3-day window to 20-day average ten-days prior
        actual = crsp.get_window(dataset='daily',
                                 field='vol',
                                 date_field='date',
                                 permnos=fund['permno'],
                                 dates=fund['rdq'],
                                 left=-1,
                                 right=1)
        normal = crsp.get_window(dataset='daily',
                                 field='vol',
                                 date_field='date',
                                 permnos=fund['permno'],
                                 dates=fund['rdq'],
                                 left=-30,
                                 right=-11,
                                 avg=True)
        fund['aeavol'] = normal['vol'].values
    
        signals.write(fund, 'ear', overwrite=True)
        signals.write(fund, 'aeavol', overwrite=True)

    rebalbeg, rebalend = 19700101, LAST_DATE
    benchnames = ['Mkt-RF(mo)']
    for num, label in enumerate(columns):
        holdings = portfolio_sorts(crsp,
                                   label,
                                   SignalsFrame(signals.read(label)),
                                   rebalbeg,
                                   rebalend,
                                   window=3,
                                   months=[],
                                   leverage=leverage.get(label, 1))
        excess = backtest_pipeline(backtest,
                                   crsp,
                                   holdings,
                                   label,
                                   benchnames,
                                   overlap=0,
                                   outdir=outdir,
                                   num=num+1,
                                   suffix=(leverage.get(label, 1) < 0)*'(-)')
    plt_show()

## IBES Fiscal Year 1 linked to Quarterly PSTAT: sfe
if 'ibesf1_pstqtr' in testable:
    beg, end = 19760101, LAST_DATE
    monthnum = lambda d: ((d//10000)-1900)*12 + ((d//100)%100) - 1
    if regenerate:
        df = pstat.get_linked(dataset='quarterly',
                              fields=['prccq'],
                              date_field='datadate')
        df = df.dropna()\
               .sort_values(['permno', 'datadate'])\
               .drop_duplicates(['permno', 'datadate'])
        
        out = ibes.get_linked(dataset='summary',
                              fields=['fpedats', 'meanest'],
                              date_field='statpers',
                              where="fpi='1'")
        out = out.dropna()\
                 .sort_values(['permno', 'statpers', 'fpedats'])\
                 .drop_duplicates(['permno', 'statpers'])
        out['monthnum'] = monthnum(out['statpers'])
        out = out.set_index(['permno', 'monthnum'], drop=False)
        out['sfe'] = np.nan

        for num in range(4): # match ibes statpers to any datadate in last 4 mos
            df['monthnum'] = monthnum(df['datadate']) - num
            df = df.set_index(['permno', 'monthnum'], drop=False)
            out = out.join(df[['prccq']], how='left')
            out['sfe'] = out['sfe'].where(out['sfe'].notna(),
                                          out['meanest'] / out['prccq'].abs())
            out = out.drop(columns=['prccq'])

        out['rebaldate'] = bd.endmo(out['statpers'])
        n = signals.write(out.reset_index(drop=True), 'sfe', overwrite=True)
        
    rebalbeg, rebalend = 19760101, LAST_DATE
    benchnames = ['Mkt-RF(mo)']
    label = 'sfe'
    holdings = portfolio_sorts(crsp,
                               label,
                               SignalsFrame(signals.read(label)),
                               rebalbeg,
                               rebalend,
                               window=3,
                               months=[],
                               leverage=leverage.get(label, 1))
    excess = backtest_pipeline(backtest,
                               crsp,
                               holdings,
                               label,
                               benchnames,
                               overlap=0,
                               outdir=outdir, 
                               suffix=(leverage.get(label, 1) < 0)*'(-)')
    plt_show()
    
## IBES Fiscal Quarter 1, linked to Quarterly: sue
#
# TODO: 'sfe' and 'sue' (where meanest) should use ibes price (prccq restated?)
#
if 'ibesq1_pstqtr' in testable:
    columns = ['sue']
    numlag = 4
    end = LAST_DATE

    if regenerate:    
        # retrieve quarterly, keep [permno, datadate] key with non null prccq
        df = pstat.get_linked(dataset='quarterly',
                              fields=['prccq', 'cshoq', 'ibq'],
                              date_field='datadate',
                              where=f"datadate <= {end//100}31")
        fund = df.dropna(subset=['prccq'])\
                 .sort_values(['permno', 'datadate', 'cshoq'])\
                 .drop_duplicates(['permno', 'datadate'])
        fund['rebaldate'] = bd.endmo(fund['datadate'], numlag)
        fund = fund.set_index(['permno', 'rebaldate'], drop=False)

        # retrieve ibes Q1 where forecast period <= fiscal date, keep latest
        df = ibes.get_linked(dataset='summary',
                             fields=['fpedats', 'medest', 'actual'],
                             date_field='statpers',
                             where=" fpi = '6' AND statpers <= fpedats")
        out = df.dropna()\
                .sort_values(['permno', 'fpedats', 'statpers'])\
                .drop_duplicates(['permno', 'fpedats'], keep='last')
        out['rebaldate'] = bd.endmo(out['fpedats'], numlag)
        out = out.set_index(['permno', 'rebaldate']).reindex(fund.index)

        # initial sue with ibes surprise, scaled by compustat quarterly price
        fund['sue'] = (out['actual'] - out['medest']) / fund['prccq'].abs()

        # lag(4) difference in compustat quarterly for missing surprise
        lag = fund.shift(4, fill_value=0)

        fund['sue'] = fund['sue']\
            .where(fund['sue'].notna() | (lag['permno'] != fund['permno']),
                   ((fund['ibq'] - lag['ibq']) /
                    (fund['prccq'] * fund['cshoq'])).abs())
        signals.write(fund.reset_index(drop=True), 'sue', overwrite=True)

    rebalbeg, rebalend = 19760101, LAST_DATE
    benchnames = ['Mkt-RF(mo)']
    for num, label in enumerate(columns):
        holdings = portfolio_sorts(crsp,
                                   label,
                                   SignalsFrame(signals.read(label)),
                                   rebalbeg,
                                   rebalend,
                                   window=3,
                                   months=[],
                                   leverage=leverage.get(label, 1))
        excess = backtest_pipeline(backtest,
                                   crsp,
                                   holdings,
                                   label,
                                   benchnames,
                                   overlap=0,
                                   outdir=outdir,
                                   num=num+1,
                                   suffix=(leverage.get(label, 1) < 0)*'(-)')
    plt_show()

## Summarize all results
if 'summarize' in testable:
    zoo = backtest.read().sort_values(['begret', 'permno'])
    r = []
    for label in zoo.index:
        perf = backtest.read(label)
        excess = {'ret': backtest.fit(['Mkt-RF(mo)'])}
        excess['annualized'] = backtest.annualized
        excess['dd'] = maximum_drawdown(backtest.perf['excess'])
        post = {'ret': backtest.fit(['Mkt-RF(mo)'],
                                    beg=20020101).copy()}
        post['annualized'] = backtest.annualized.copy()
        s = label + ('(-)' if leverage.get(label, 1) < 0 else '')
        r.append(DataFrame({
            'Start': excess['ret'].index[0],
            'Sharpe Ratio': excess['annualized']['sharpe'],
            'Alpha': excess['annualized']['alpha'],
            'Info Ratio': excess['annualized']['inforatio'],
            'Avg Ret': excess['ret']['excess'].mean(),
            'Vol': excess['ret']['excess'].std(ddof=0),
            'Welch-t': excess['annualized']['welch-t'],
            'InfoRatio2002': post['annualized']['inforatio'],
            'Ret2002': post['ret']['excess'].mean(),
            'Vol2002': post['ret']['excess'].std(ddof=0),            
            'Best' : excess['ret']['excess'].idxmax(),
            'BestRet' : excess['ret']['excess'].max(),
            'Worst' : excess['ret']['excess'].idxmin(),
            'WorstRet' : excess['ret']['excess'].min(),
            'Drawdown': (excess['dd'].iloc[1]/excess['dd'].iloc[0]) - 1,
            'Beg': excess['dd'].index[0],
            'End': excess['dd'].index[1],
            'Turn2002+': post['annualized']['sells']/12,
            'Long2002+': int(post['annualized']['longs']),
            'Short2002+': int(post['annualized']['shorts'])}, index=[s]))
    df = pd.concat(r, axis=0).round(4).sort_values('Welch-t')
    with open(os.path.join(outdir, 'summary.tex'), 'wt') as f:
        f.write(show(df.loc[:, :'Vol2002'],
                     latex=True,
                     caption="Factor Performance Summary Part I"))
        f.write(show(df.loc[:, 'Best':],
                     latex=True,
                     caption="Factor Performance Summary Part II"))
