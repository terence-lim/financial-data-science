"""
Quant Factors

- Fama-French, HML
- weekly reversal, quant quake
- return predicting signals, alpha, information ratio

References:

- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

- Khandani and Lo (2008), “What Happened to the Quants in August 2007?"

- Jeremiah Green, John Hand and Frank Zhang (December 2017), "Characteristics that Provide Independent Information about Average U.S. Monthly Stock Returns”, Review of Financial Studies 30:12, 4389-4436

Glossary:

- The Sharpe ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk.

- Jensen's alpha is the amount of excess return the portfolio has earned over a benchmark -- such as a beta-weighted average of index returns (e.g. the market index in the CAPM).

- Appraisal ratio compares the portfolio's alpha to the portfolio's unsystematic risk or residual standard deviation.

- The information ratio (IR) is a measurement of portfolio returns beyond the returns of a benchmark compared to the volatility of those returns.  In the context of "market-neutral" portfolios, IR is theoretically the Sharpe ratio. Otherwise, it is better measured by the appraisal ratio.
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
import dives
import matplotlib.pyplot as plt
import numpy as np
import time

import dives.util
import dives.dbengines
import dives.structured
import dives.evaluate
import importlib
importlib.reload(dives)
importlib.reload(dives.util)
importlib.reload(dives.structured)
importlib.reload(dives.dbengines)
importlib.reload(dives.evaluate)

from dives.util import fractiles, DataFrame
from dives.dbengines import SQL, Redis
from dives.structured import BusDates, Benchmarks, CRSP, PSTAT, IBES, Signals
from dives.evaluate import BackTest, run_backtest

import secret
sql = SQL(**secret.value('sql'))       
rdb = Redis(**secret.value('redis'))       
    
bd = BusDates(sql)
bench = Benchmarks(sql, bd)
crsp = CRSP(sql, bd, rdb)
pstat = PSTAT(sql, bd)
ibes = IBES(sql, bd)
signals = Signals(sql)
backtest = BackTest(sql, bench, 'RF')
outdir = secret.value('rps')['dir']
html = 'index.html'
outdir, html = '',''

if False:
    #
    # Weekly reversal strategy
    #
    crsp.rdb = None      # do not use redis cache
    tic = time.time()
    holdings = {}
    rebalbeg, rebalend = 20060701, 20080831
    for weeknum in bd.week_range(rebalbeg, rebalend):
        pordate = bd.week_end(weeknum)              # rebalance date
        beg = bd.shift(bd.week_beg(weeknum), -1)    # date prior to period for measuring past return
        start = bd.week_beg(weeknum)                # first day of period to measure past return
        end = bd.week_beg(weeknum)                  # last day of period to measure past return

        df = crsp.get_universe(pordate)             # generate new universe every week: common domestic
        df['beg'] = crsp.get_section('daily', ['prc'], 'date', beg).reindex(df.index)
        df['end'] = crsp.get_section('daily', ['prc'], 'date', end).reindex(df.index)
        df['signal'] = crsp.get_ret(start, end).reindex(df.index)
        df = df[df['beg'].notnull() & df['end'].notnull() & df['signal'].notnull()]
        df['fractile'] = fractiles(df['signal'], [30,70], df['signal'][df.nyse])
        permnos, weights = [],[]            
        subs = [(df.fractile == 3) & (df.deciles > 5),               # big winner subportfolio
                (df.fractile == 3) & (df.deciles <= 5),              # small winner subportfolio
                (df.fractile == 1) & (df.deciles > 5),               # big loser subportfolio
                (df.fractile == 1) & (df.deciles <= 5)]              # small loser subportfolio
        for sub, weight in zip(subs, [-0.5, -0.5, 0.5, 0.5]):        # combine subportfolios for reversal
            weights += list(weight * df.loc[sub,'cap'] / df.loc[sub,'cap'].sum())
            permnos += list(df.index[sub])
        holdings[pordate] = DataFrame(data=weights, index=permnos, columns=['weights'])
    print('Elapsed: %.0f secs' % (time.time() - tic))
    
    benchnames = ['ST_Rev']
    backtest.performance(crsp, holdings, label=('weekly reversal'))
    backtest.attribution(benchnames, flip=False)
    backtest.plot(benchnames)
    pprint(backtest.annualized)
    backtest.results.summary()
    
if False:
    #
    # Create HML signal (allow 6 months for data to become available)
    # - compare to French web site -- highly correlated but some differences remain
    # - welch test is of difference of mean returns before and after Dec 2002
    #
    tic = time.time()
    crsp.rdb = rdb      # reset to use redis cache
    rebalbeg, rebalend = 19500601, 20190630
    beg, end = crsp.dates.begmo(rebalbeg, -6), crsp.dates.endmo(rebalend, -6)

    # retrieve required fields from database
    df = pstat.get_linked(table = 'annual', date_field = 'datadate',
                          fields = ['ceq','pstk'],  # 'pstkrv', 'pstkl',
                          where = ("ceq > 0 and datadate >= {beg} and datadate <= {end}" \
                                   "".format(beg=beg, end=end)))
    # construct book values
    df['hml'] = 0
    df.loc[df['pstk'].gt(0),   'hml'] = -df.loc[df['pstk'].gt(0),   'pstk']
#    df.loc[df['pstkl'].gt(0),  'hml'] = -df.loc[df['pstkl'].gt(0),  'pstkl']
#    df.loc[df['pstkrv'].gt(0), 'hml'] = -df.loc[df['pstkrv'].gt(0), 'pstkrv']
    df['hml'] += df['ceq']
    df['rebaldate'] = 0
    
    # construct b/m ratio
    for datadate in sorted(df['datadate'].unique()):
        f = df['datadate'].eq(datadate)
        rebaldate = crsp.dates.endmo(datadate, 6)        # rebalance date >= 6 months after fiscal end
        pricedate = bd._months.loc[((rebaldate-10000)//10000, 12), 'endmo']      # december market cap
        permnos = list(df.loc[f, 'permno'])
        df.loc[f, 'rebaldate'] = rebaldate
        df.loc[f, 'cap'] = (crsp.get_cap(pricedate).reindex(permnos)).values
    df['hml'] /= df['cap']
    print('{} dates {}-{}: {} records'.format(
        len(df['rebaldate'].unique()), min(df['rebaldate']), max(df['rebaldate']), len(df)))
    print('Elapsed: %.0f secs' % (time.time() - tic))

    benchnames = ['HML(mo)']
    signal = 'hml'
    holdings = crsp.portfolio_sorts(signal, data = df, beg = rebalbeg, end = rebalend,
                                    window = 12,     # use latest signal from recent 12 months rebalances
                                    month = 6)       # determine universe of stocks every June
    backtest.performance(crsp, holdings, label = signal)
    backtest.attribution(benchnames, flip=False)
    backtest.plot(benchnames)
    pprint(backtest.annualized)
    backtest.results.summary()

if False:
    #
    # Momentum and divyld: mom12m, mom1m, mom36m, mom6m, chmom, indmom, divyld
    #
    tic = time.time()
    beg, end = 19270101, 20190630
    intervals = {'mom12m':  (2,12),     # momentum measured over different past periods
                 'mom36m': (13,36),
                 'mom6m' :  (2,6),
                 'mom1m' :  (1,1)}
    for signal, past in intervals.items():
        out = DataFrame()
        for pordate in bd.endmo_range(bd.endmo(beg, past[1]), end):  # loop over each rebalance month
            start = bd.endmo(pordate, -past[1])           # require that  prices available at each point
            beg1 = bd.shift(start, 1)
            end1 = bd.endmo(pordate, 1-past[0])
            df = crsp.get_section('daily', ['prc'], 'date', start)
            df['end1'] = crsp.get_section('daily', ['prc'], 'date', end1).reindex(df.index)
            df[signal] = crsp.get_ret(beg1, end1).reindex(df.index)   # compute momentum from past returns
            df['permno'] = df.index
            df['rebaldate'] = pordate
            df = df.loc[df['prc'].notnull() & df['end1'].notnull()]
            out = out.append(df[['rebaldate', 'permno', signal]], ignore_index=True)    # append rows
        n = signals.save(out, signal, append=False)

    columns = ['chmom', 'divyld','indmom']
    out = DataFrame()
    for pordate in bd.endmo_range(bd.endmo(beg, 12), end):
        start = bd.endmo(pordate, -12)     # require that prices available at each point
        beg1 = bd.shift(start, 1)
        end1 = bd.endmo(pordate, -6)
        beg2 = bd.shift(end1, 1)
        end2 = bd.endmo(pordate)
        df = crsp.get_section('daily', ['prc'], 'date', start)
        df['end1'] = crsp.get_section('daily', ['prc'], 'date', end1).reindex(df.index)
        df['end2'] = crsp.get_section('daily', ['prc'], 'date', end2).reindex(df.index)
        df['shrout'] = crsp.get_section('daily', ['shrout'], 'date', end1).reindex(df.index)
        df['mom2'] = crsp.get_ret(beg2, end2).reindex(df.index)
        df['mom1'] = crsp.get_ret(beg1, end1).reindex(df.index)
        df['divyld'] = crsp.get_divamt(beg1, end2).reindex(df.index)  # compute 12-month divyld
        df['divyld'] /= df['shrout'] * df['end1'].abs()
        df.loc[df['divyld'].isnull(), 'divyld'] = 0                   # if no dividends, divyld is 0
        df['chmom'] = df['mom1'] - df['mom2']              # compute chmom

        # 6-month two-digit sic industry momentum for stocks with exchcd=1,2,3 shrcd=10,11 siccd>0
        df = df.join(crsp.get_section('names', ['siccd', 'exchcd', 'shrcd'],  'date', pordate,
                                      start = 0), how='left')
        f = (df['shrcd'].isin([10, 11])       # only include common stocks domiciled in the US
             & df['exchcd'].isin([1, 2, 3])
             & df['siccd'].gt(0))
        df.loc[f, 'sic2'] = df.loc[f, 'siccd'] // 100
        group = df.groupby(['sic2'])                       # 'indmom' is sic2 group-means of 'mom1'
        df = df.join(DataFrame(group['mom1'].mean()).rename(columns={'mom1' : 'indmom'}),
                     on='sic2', how='left')
        
        df['permno'] = df.index
        df['rebaldate'] = pordate
        out = out.append(df.loc[df['prc'].notnull()
                                & df['end1'].notnull()
                                & df['end2'].notnull()
                                & df['shrout'].notnull(),
                                ['rebaldate', 'permno'] + columns],
                         ignore_index=True)
    for signal in columns:   # save signal values to sql
        n = signals.save(out, signal, append=False)
    print('Elapsed: %.0f secs' % (time.time() - tic))

    benchnames = ['ST_Rev(mo)','Mom(mo)','Mkt-RF(mo)']    # run backtests and compare to these benchmarks
    rebalbeg, rebalend = 19301231, 20190630
    for signal in ['mom12m', 'mom6m', 'mom1m']:
        run_backtest(backtest, crsp, signal, 1,   # only use signal as of current rebalance month
                     benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)
    for signal in ['chmom','indmom']:
        run_backtest(backtest, crsp, signal, 1,   # only use signal as of current rebalance month
                     benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)
    run_backtest(backtest, crsp, 'mom36m', 1,     # only use signal as of current rebalance month
                 ['LT_Rev(mo)','Mom(mo)','Mkt-RF(mo)'],
                 rebalbeg, rebalend, signals, outdir=outdir, html=html)
    run_backtest(backtest, crsp, 'divyld', 1,     # only use signal as of current rebalance month
                 ['HML(mo)', 'Mkt-RF(mo)'], rebalbeg, rebalend, signals, outdir=outdir, html=html)

if False:
    #
    # Liquidity signals from daily stock returns
    #
    beg, end = 19301231, 20190630
    tic = time.time()
    columns = ['ill','maxret','retvol','baspread','std_dolvol','zerotrade','std_turn','turn']
    
    out = DataFrame()
    dolvol = DataFrame()
    turn = DataFrame()        # to average turn signal over rolling 3-month window
    for pordate in bd.endmo_range(beg, end):
        q = "SELECT permno, max(ret) as maxret, " \
            " std(ret) as retvol," \
            " avg((askhi-bidlo)/((askhi+bidlo)/2)) as baspread," \
            " std(log(abs(prc*vol))) as std_dolvol," \
            " 1000000*avg(abs(ret)/(abs(prc)*vol)) as ill," \
            " log(sum(abs(prc)*vol)) as dolvol," \
            " std(vol/shrout) as std_turn," \
            " sum((prc <= 0) or (prc is null)) as countzero," \
            " count(prc > 0) as ndays," \
            " sum(vol/shrout) as turn1" \
            " FROM daily WHERE date >= {beg} AND date <= {end} GROUP BY permno" \
            "".format(beg = bd.begmo(pordate), end = pordate)
        df = DataFrame(**crsp.sql.run(q)).to_numeric()
        turn = turn.hqueue(df[['turn1']], width = 3)     # queue rolling three months of turn
        df['turn'] = turn.reindex(df.index).mean(axis=1, skipna = False)
        df.loc[df['turn1'].le(0), 'turn1'] = np.nan
        df.loc[df['ndays'].le(0), 'ndays'] = np.nan
        df['zerotrade'] = ((df['countzero'] + ((1/df['turn1'])/480000)) * 21/df['ndays'])
        df['rebaldate'] = pordate
        out = out.append(df[['permno', 'rebaldate'] + columns], ignore_index = True)
        if pordate < bd.endmo(end):
            df['rebaldate'] = bd.endmo(pordate, 1)
            dolvol = dolvol.append(df[['permno', 'rebaldate','dolvol']], ignore_index = True)
    print('Elapsed: %.0f secs' % (time.time() - tic))
            
    for signal in columns:
        n = signals.save(out, signal, append = False)
    n = signals.save(dolvol, 'dolvol', append = False)

    rebalbeg, rebalend = 19301231, 20190630
    benchnames = ['ST_Rev(mo)','Mom(mo)','Mkt-RF(mo)']
    for signal in columns + ['dolvol']:
        run_backtest(backtest, crsp, signal, 1,    # only use signal as of current rebalance month
                     benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)


if False:
    #
    # Weekly stock returns-based price response signals
    #    
    def regress(x, y):
        """helper method to calculate beta, idiovol and price delay

        Parameters
        ----------
        x : np.array
          equal-weighted market returns (in ascending time order)
        y : np.array
          stock returns (in ascending time order).  NaN's will be discarded.

        Returns
        -------
        beta: float
          slope from regression on market returns and intercept
        idiovol : float
          mean squared error of residuals
        pricedelay: float
          ratio of adjusted Rsq from adding four market lags, to adjusted Rsq without lags
        """
        v = np.logical_not(np.isnan(y))
        y = y[v]
        x = x[v]
        n0 = len(y)
        A0 = np.vstack([x, np.ones(len(y))]).T
        b0 = np.linalg.inv(A0.T.dot(A0)).dot(A0.T.dot(y))   # compute univariate regression coefficients
        sse0 = np.mean((y - A0.dot(b0))**2)
        sst0 = np.mean((y - np.mean(y))**2)
        R0 = 1 - ((sse0 / (n0 - 1 - 1)) / (sst0 / (n0 - 1)))

        y4 = y[4:]
        n4 = len(y4)         
        A4 = np.vstack([x[0:-4], x[1:-3], x[2:-2], x[3:-1], x[4:], np.ones(n4)]).T
        b4 = np.linalg.inv(A4.T.dot(A4)).dot(A4.T.dot(y4))  # compute four lagged regression coefficients
        sse4 = np.mean((y4 - A4.dot(b4))**2)
        sst4 = np.mean((y4 - np.mean(y4))**2)
        R4 = 1 - ((sse4 / (n4 - 5 - 1)) / (sst4 / (n4 - 1)))
        return [b0[0], sse0, 1 - (R0/R4)]

    beg, end = 19301231, 20190630
    tic = time.time()
    
    columns  = ['beta','idiovol','pricedelay']
    width    = 3*52+1           # up to 3 years of weekly returns
    minvalid = 52               # at least 52 weeks required to compute beta
    weekly   = DataFrame()      # to queue rolling window of weekly stock returns
    mkt      = DataFrame()      # to queue equal-weighted market returns
    out      = DataFrame()      # accumulate all rows from computing signal values
    for num in crsp.dates.week_range(beg, end):
        date = crsp.dates.week_end(num)                   # get and insert next column and market
        df = crsp.get_ret(crsp.dates.week_beg(num), date, nocache=True)

        mkt = mkt.hqueue(          # queue rolling window of weekly market returns
            DataFrame(data = df.mean().astype(float), columns=[num]), width=width)
        weekly = weekly.hqueue(    # queue rolling window of weekly stock returns
            df.rename(columns = {'ret':num}), width=width) 

        valid = np.sum(weekly.notnull(), 1) >= minvalid   # compute only for rows with sufficient notnulls
        if np.sum(valid):
            result = DataFrame(
                data = [regress(mkt.values[0], y.astype(float))   # regress weekly returns
                        for y in weekly.loc[valid].values],       # for each stock with sufficient
                columns = columns)
            result['permno'] = list(weekly.index[valid])
            result['rebaldate'] = date
            if crsp.dates._weeks.loc[num, 'ismonthend']:   # append rows only if last week of the month
                out = out.append(result, ignore_index=True)
    print('Elapsed: %.0f secs' % (time.time() - tic))
    for signal in columns:
        signals.save(out, signal, append=False)

    benchnames = ['ST_Rev(mo)','Mom(mo)','Mkt-RF(mo)']
    rebalbeg, rebalend = 19301231, 20190630
    for signal in columns:
        run_backtest(backtest, crsp, signal, 1,   # only use signal from current rebalance month
                     benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)

if False:
    #
    # Fundamental signals from Compustat Annual
    #
    tic = time.time()
    columns = ['absacc','acc', 'agr', 'bm', 'cashpr', 'cfp', 'chcsho', 'chinv', 'depr', 'dy', 'egr',
               'ep', 'gma', 'grcapx', 'grltnoa', 'hire', 'invest', 'lev', 'lgr' ,'pchdepr',
               'pchgm_pchsale', 'pchquick', 'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga',
               'pchsaleinv', 'pctacc', 'quick', 'rd_sale', 'rd_mve', 'realestate',
               'salecash', 'salerec', 'saleinv', 'secured', 'sgr', 'sp', 'tang',
               'bm_ia', 'cfp_ia', 'chatoia' , 'chpmia', 'pchcapx_ia', 'chempia', 'mve_ia']
    numlag = 6       # number of months to lag data for rebalance
    end = 20181231   # last data date

    # retrieve all annual, keep [permno, datadate] key with non null price (prccq) if any
    fields = ['sic', 'fyear', 'ib', 'oancf', 'at', 'act', 'che', 'lct', 'dlc', 'dltt',
              'prcc_f', 'csho', 'invt', 'dp', 'ppent', 'dvt', 'ceq',
              'revt', 'cogs', 'rect', 'aco', 'intan', 'ao', 'ap', 'lco', 'lo', 'capx',
              'emp', 'ppegt', 'lt', 'sale', 'xsga', 'xrd', 'fatb', 'fatl', 'dm']
    dg = pstat.get_linked(table = 'annual',
                          date_field = 'datadate',
                          fields = fields,
                          where = ('datadate <= %d' % end)
                          ).sort_values(['permno', 'datadate', 'ib'])
    fund = dg.drop_duplicates(['permno','datadate'], keep='first').copy()
    fund.index = list(zip(fund['permno'], fund['datadate']))  # initialize dataframe with multi-index
    fund['rebaldate'] = [bd.endmo(x[1], numlag) for x in fund.index]

    # precompute, and lag common metrics: mve_f avg_at sic2
    fund['sic2'] = fund['sic'] // 100
    fund.loc[fund['sic'].isnull(), 'sic2'] = 0
    fund['fyear'] = fund['datadate'] // 10000    # can delete this
    fund['mve_f'] = fund['prcc_f'] * fund['csho']                      # precompute mve_f
    
    lag = fund.shift(1, fill_value=0)                                  # lag is shift(1)
    lag.loc[lag['permno'] != fund['permno'], fields] = np.nan          # require same permno on row
    fund['avg_at'] = (fund['at'] + lag['at']) / 2                      # precompute avg_at
    
    lag2 = fund.shift(2, fill_value=0)                                 # lag2 is shift(2)
    lag2.loc[lag2['permno'] != fund['permno'], fields] = np.nan        # require same permno on roa
    lag['avg_at'] = (lag['at'] + lag2['at']) / 2                       # lag of avg_at
    
    # compute: cashpr depr dy ep lev quick
    fund['bm'] = fund['ceq'] / fund['mve_f']                           # bm = ceq / mve_f
    fund['cashpr'] = (fund['mve_f'] + fund['dltt'] - fund['at']) / fund['che'] # cashpr =
    fund['depr'] = fund['dp'] / fund['ppent']                          # depr = dp / ppent
    fund['dy'] = fund['dvt'] / fund['mve_f']                           # dy = dvt / mve_f
    fund['ep'] = fund['ib'] / fund['mve_f']                            # ep = ib / mve_f
    fund['lev'] = fund['lt'] / fund['mve_f']                           # lev = lt / mve_f
    fund['quick'] = (fund['act'] - fund['invt']) / fund['lct']         # quick = (act - invt) / lct
    fund['rd_sale'] = fund['xrd'] / fund['sale']                       # rd_sale = xrd / sale
    fund['rd_mve'] = fund['xrd'] / fund['mve_f']                       # rd_mve = xrd / mve_f
    fund['realestate'] = (fund['fatb'] + fund['fatl']) / fund['ppegt'] # realestate = fatb+fatl / ppegt
    h = fund['ppegt'].isnull()                                         #           OR fatb+fatl / ppent
    fund.loc[h, 'realestate'] = ((fund.loc[h, 'fatb'] + fund.loc[h, 'fatl']) / fund.loc[h, 'ppent'])
    fund['salecash'] = fund['sale'] / fund['che']                      # salecash = sale / che
    fund['salerec'] = fund['sale'] / fund['rect']                      # salerec = sale / rect
    fund['saleinv'] = fund['sale'] / fund['invt']                      # saleinv = sale / invt
    fund['secured'] = fund['dm'] / fund['dltt']                        # secured = dm / dltt
    fund['sp'] = fund['sale'] / fund['mve_f']                          # sp = sale / mve_f
    fund['tang'] = (fund['che'] + fund['rect']*0.715 +                 # tang ~ che,rect,invt,ppent/at
                    fund['invt']*0.547 + fund['ppent'] *0.535) / fund['at']

    # compute changes: agr chcsho chinv egr gma egr grcapx grltnoa emp invest lgr
    fund['agr'] = (fund['at'] / lag['at'])                             # agr = chg at
    fund['chcsho'] = (fund['csho'] / lag['csho'])                      # chcsho = chg csho
    fund['chinv'] = ((fund['invt'] - lag['invt']) / fund['avg_at'])    # chinv =
    fund['egr'] = (fund['ceq'] / lag['ceq'])                           # egr = chg ceq
    fund['gma'] = ((fund['revt'] - fund['cogs']) / lag['at'])          # gma = revt-cogs/at
    fund['grcapx'] = (fund['capx'] / lag2['capx'])                     # grcapx = chg2 capx
    fund['grltnoa'] =  ((fund['rect'] + fund['invt'] + fund['ppent'] + # grltnoa = GrNOA - TACC =
                         fund['aco'] + fund['intan'] + fund['ao'] -    #   (chg (rect+invt+ppent+
                         fund['ap'] - fund['lco'] - fund['lo']) /      #        aco+intan+ao
                        (lag['rect'] + lag['invt'] + lag['ppent'] +    #        -ap-lco-lo) -
                         lag['aco'] + lag['intan'] + lag['ao'] -       #   chg (rect+invt+aco-ap-lco)
                         lag['ap'] - lag['lco'] - lag['lo']) -         #   - dp) / avg_at
                        ((fund['rect'] + fund['invt'] + fund['aco'] -
                          fund['ap'] - fund['lco']) -
                         (lag['rect'] + lag['invt'] + lag['aco'] -
                          lag['ap'] - lag['lco']))) / fund['avg_at']
    fund['hire'] = (fund['emp'] / lag['emp']) - 1                      # hire = chg emp
    fund.loc[fund['hire'].isnull(), 'hire'] = 0
    fund['invest'] = (((fund['ppegt'] - lag['ppegt']) +                # invest = 
                       (fund['invt'] - lag['invt'])) / lag['at'])      #   chg ppegt + chg invt / at
    h = fund['invest'].isnull()                                        # if missing ppegt then ppent
    fund.loc[h,'invest'] = (((fund.loc[h,'ppent'] - lag.loc[h,'ppent']) + 
                             (fund.loc[h,'invt'] - lag.loc[h,'invt'])) / lag.loc[h,'at'])
    fund['lgr'] = (fund['lt'] / lag['lt'])                             # lgr = chg lt
    fund['pchdepr'] = ((fund['dp'] / fund['ppent']) /
                       (lag['dp'] / lag['ppent']))                     # pchdepr = chg (dp/ppent)
    fund['pchgm_pchsale'] = (((fund['sale'] - fund['cogs']) /          # pchgm_pchsale = 
                              (lag['sale'] - lag['cogs'])) -           #   chg (sale-cogs) -
                             (fund['sale'] / lag['sale']))             #   chg sale
    fund['pchquick'] = (((fund['act'] - fund['invt']) / fund['lct']) / # pchquick =
                        ((lag['act'] - lag['invt']) / lag['lct']))     #   chg (act-invt/lct)
    fund['pchsale_pchinvt'] = ((fund['sale'] / lag['sale']) -          # pchgm_pchinv =
                               (fund['invt'] / lag['invt']))           #   chg sale - chg invt
    fund['pchsale_pchrect'] = ((fund['sale'] / lag['sale']) -          # pchgm_pchrect =
                               (fund['rect'] / lag['rect']))           #   chg sale - chg rect
    fund['pchsale_pchxsga'] = ((fund['sale'] / lag['sale']) -          # pchgm_pchxsga =
                               (fund['xsga'] / lag['xsga']))           #   chg sale - chg xsga
    fund['pchsaleinv'] = ((fund['sale'] / fund['invt']) /              # pchsaleinv =
                          (lag['sale'] / lag['invt']))                 #   chg (sale/inv)
    fund['sgr'] = (fund['sale'] / lag['sale'])                         # sgr = chg sale

    fund['chato'] = (fund['sale'] / fund['avg_at']) - (lag['sale'] / lag['avg_at'])
    fund['chpm'] = (fund['ib'] / fund['sale']) - (lag['ib'] / lag['sale'])
    fund['pchcapx'] = fund['capx'] / lag['capx']
    
    # compute signals with alternative definitions: acc absacc cfp
    fund['_acc'] = (((fund['act'] - lag['act']) -             # _acc = (chg act - chg che)
                     (fund['che'] - lag['che'])) -            #       - (chg lct - chg dlc - dp) 
                    ((fund['lct'] - lag['lct']) -             # 
                     #((fund['txp'] - lag['txp']) -            # chg txp?
                     (fund['dlc'] - lag['dlc']) - fund['dp']))
    fund['cfp'] = (fund['ib'] -
                   (((fund['act'] - lag['act']) -             # cfp = (chg act - chg che)
                     (fund['che'] - lag['che'])) -            #       - (chg lct - chg dlc -dp) 
                    ((fund['lct'] - lag['lct']) -             #     / (avg at)
                     #((fund['txp'] - lag['txp']) -            # chg txp?
                     (fund['dlc'] - lag['dlc']) - fund['dp']))) / fund['mve_f']
    g = ~fund['oancf'].isnull()
    fund.loc[g, 'cfp'] = fund.loc[g, 'oancf'] / fund.loc[g, 'mve_f']
    fund.loc[g, '_acc'] = fund.loc[g, 'ib'] - fund.loc[g, 'oancf']
    fund['acc'] = fund['_acc'] / fund['avg_at']
    fund['absacc'] = abs(fund['_acc']) / fund['avg_at']
    fund['pctacc'] = fund['_acc'] / abs(fund['ib'])
    h = (fund['ib'] <= 0.01)
    fund.loc[h, 'pctacc'] = fund.loc[h, '_acc'] / 0.01

    # compute industry-adjusted: bm_ia cfp_ia chatoia chpmia pchcapx_ia chempia mve_ia
    cols = {'bm_ia':'bm', 'cfp_ia':'cfp', 'chatoia':'chato', 'chpmia':'chpm', 
            'pchcapx_ia':'pchcapx', 'chempia':'hire', 'mve_ia':'mve_f'}
    group = fund.groupby(['sic2', 'fyear'])
    for k,v in cols.items():
        fund[k] = fund[v] - group[v].transform('mean')
    print('Elapsed: %.0f secs' % (time.time() - tic))

    for signal in columns:
        signals.save(fund, signal)

    rebalbeg, rebalend = 19500101, 20190630
    benchnames = ['HML(mo)', 'Mkt-RF(mo)'] #['Mom']  #['ST_Rev(mo)']   # 
    for signal in columns:
        run_backtest(backtest, crsp, signal, 12,  # use latest signal from recent twelve months' rebalances
                     benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)


if False:
    #
    # Fundamental signals from Compustat Quarterly
    #
    tic = time.time()
    columns = ['stdacc', 'stdcf', 'roavol', 'sgrvol', 'cinvest', 'chtx', 'rsup', 'roaq', 'cash', 'nincr']
    numlag = 4       # require 4 month lag of fiscal data
    end = 20190228   # last data date
    
    # retrieve all quarterly, keep [permno, datadate] key with non null price (prccq) if any
    fields = ['ibq', 'actq', 'cheq', 'lctq', 'dlcq', 'saleq', 'prccq', 'cshoq',
              'atq', 'txtq', 'ppentq']
    dg = pstat.get_linked(table = 'quarterly',
                          date_field = 'datadate',
                          fields = fields,
                          where = 'datadate > 0 and datadate <= {}31'.format(end//100)
                          ).sort_values(['permno','datadate','ibq'])
    fund = dg.drop_duplicates(['permno','datadate'], keep='first').copy()
    fund.index = list(zip(fund['permno'], fund['datadate']))     # set dataframes index and rebaldates
    rebaldate = [bd.endmo(x[1], numlag) for x in fund.index]     # rebalance 4 month delay

    # compute from current and lagged period: scf sacc roaq nincr cinvest cash rsup chtx
    lag = fund.shift(1, fill_value=0)
    lag.loc[lag['permno'] != fund['permno'], fields] = np.nan           # require same permno on row
    fund['_saleq'] = fund['saleq']
    fund.loc[fund['_saleq'].lt(0.01), '_saleq'] = 0.01  # replace negative sales denominator with 0.01
    
    fund['sacc'] = (((fund['actq'] - lag['actq']) -                       # sacc = 
                     (fund['cheq'] - lag['cheq'])) -                      #   ((chg actq - chg cheq)
                    ((fund['lctq'] - lag['lctq']) -                       #    - (chg lctq - chg dlcq))
                     (fund['dlcq'] - lag['dlcq']))) / fund['_saleq']      #   / saleq
    fund['cinvest'] = (fund['ppentq'] - lag['ppentq']) / fund['_saleq']   # cinvest = chg ppentq/sale
    fund['nincr'] = (fund['ibq'] > lag['ibq']).astype(int)                # temp nincr = ibq > lag(ibq)
    fund['scf']  = (fund['ibq'] / fund['_saleq']) - fund['sacc']          # scf = ibq/saleq - sacc
    fund['roaq'] = (fund['ibq'] / lag['atq'])                             # roaq = ibq/lag(atq)
    fund['cash'] = (fund['cheq'] / fund['atq'])                           # roaq = ibq/atq

    lag4 = fund.shift(4, fill_value=0)                                # lag4 is shift(4 quarters)
    lag4.loc[lag4['permno'] != fund['permno'], fields] = np.nan       # require same permno on row
    fund['rsup'] = ((fund['saleq'] - lag4['saleq']) /                 # rsup = chg saleq / mveq
                    (fund['prccq'].abs() * fund['cshoq'].abs()))
    fund['chtx'] = (fund['txtq'] - lag4['txtq']) / lag4['atq']        # chtx = txtq-lag4(txtq)/lag4(at)

    # for each var: make dataframe of 15 lags (column names=[0,...,15])
    lags = {col : fund.lags(col, 'permno', 16)
            for col in ['sacc', 'scf', 'roaq', 'rsup', 'cinvest', 'nincr']}  # variables to collect lags of
    for i in range(1, 16):
        lags['nincr'][i] *= lags['nincr'][i-1]  # lags[ninrc][i]=1 iff ibq increasing in all prior quarters

    # compute signals from the 15 lags
    fund['rebaldate'] = rebaldate
    fund['stdacc'] = lags['sacc'].std(axis=1, skipna=False)
    fund['stdcf'] = lags['scf'].std(axis=1, skipna=False)
    fund['roavol'] = lags['roaq'].std(axis=1, skipna=False)
    fund['sgrvol'] = lags['rsup'].std(axis=1, skipna=False)
    fund['cinvest'] = fund['cinvest'] - lags['cinvest'][[1,2,3,4]].mean(axis=1, skipna=False)
    fund['nincr'] = lags['nincr'][list(range(8))].sum(axis=1)   # count consecutive increasing ibq
    print('Elapsed: %.0f secs' % (time.time() - tic))

    for signal in columns:
        signals.save(fund, signal)

    rebalbeg, rebalend = 19500101, 20190630
    benchnames = ['HML(mo)', 'Mkt-RF(mo)']
    for signal in columns:
        run_backtest(backtest, crsp, signal, 3,   # use latest signal from recent three months' rebalances
                     benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)


if False:
    #
    # IBES FY1 (FPI=1) Fiscal Year 1 signals: chfeps, chnanalyst, disp, nanalyst
    #
    tic = time.time()
    columns = ['chfeps', 'chnanalyst', 'disp']

    # retrieve FPI=1 required fields from database
    df = ibes.get_linked(table = 'summary',         # query ibes summary table, link to CRSP permno
                         date_field = 'statpers',
                         fields = ['fpedats', 'meanest', 'medest', 'stdev', 'numest'],
                         where = ("meanest IS NOT NULL " \
                                  " AND fpedats IS NOT NULL " \
                                  " AND statpers IS NOT NULL " \
                                  " AND summary.fpi='1'")
                         ).sort_values(['permno', 'statpers', 'fpedats'])
    out = df.drop_duplicates(['permno','statpers'], keep='first').copy()
    
    out['rebaldate'] = [bd.endmo(x) for x in out['statpers']]  # rebalance date is statpers month-end

    out['disp'] = out['stdev'] / abs(out['meanest'])      # compute disp = stdev/abs(meanest)
    out.loc[abs(out['meanest']) < 0, 'disp'] = out['stdev'] / 0.01
    
    lag1 = out.shift(1, fill_value=0)                     # compute chfeps = meanest - lag1(meanest)
    f1 = (lag1['permno'] == out['permno'])        
    out.loc[f1, 'chfeps'] = out.loc[f1, 'meanest'] - lag1.loc[f1, 'meanest']

    lag3 = out.shift(3, fill_value=0)                      # compute chnanalyst = numest - lag3(numest)
    f3 = (lag3['permno'] == out['permno'])
    out.loc[f3, 'chnanalyst'] = out.loc[f3, 'numest'] - lag3.loc[f3, 'numest']
    print('Elapsed: %.0f secs' % (time.time() - tic))

    for signal in columns:
        signals.save(out, signal)

    rebalbeg, rebalend = 19760101, 20190630
    benchnames = ['Mom(mo)','Mkt-RF(mo)']
    for signal in columns:
        run_backtest(backtest, crsp, signal, 3,   # use latest signal from recent three months' rebalances
                     benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)


if False:
    #
    # IBES Long-term Growth (FPI=0) signals: fgr5yr
    #
    tic = time.time()
    columns = ['fgr5yr']
    df = ibes.get_linked(table = 'summary',        # query ibes summary, link to CRSP permno
                         date_field = 'statpers',
                         fields = ['meanest'],
                         where = ("meanest IS NOT NULL " \
                                  " AND statpers IS NOT NULL" +
                                  " AND summary.fpi = '0'")).sort_values(['permno','statpers'])
    out = df.drop_duplicates(['permno','statpers'], keep='first').copy()
    out['rebaldate'] = [bd.endmo(x) for x in out['statpers']]   # rebalance is at month-end of statpers
    out['fgr5yr'] = out['meanest']
    print('Elapsed: %.0f secs' % (time.time() - tic))
    signals.save(out, 'fgr5yr')

    rebalbeg, rebalend = 19760101, 20190630
    benchnames = ['HML(mo)','Mkt-RF(mo)']
    for signal in columns:
        run_backtest(backtest, crsp, signal, 3,  # use latest signal from recent three months' rebalances
                     benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)


if False:
    #
    #  Earnings announcement date (rdq) in Compustat Quarterly, linked to CRSP daily: ear, aevol
    #
    tic = time.time()
    columns = ['ear', 'aeavol']
    
    # retrieve rdq, and set rebalance date to at least one month delay
    dg = pstat.get_linked(table='quarterly',      # query pstat quarterly, link to CRSP permno
                          date_field='datadate',
                          fields=['rdq'],
                          where=('rdq > 0')
                          ).sort_values(['permno','rdq','datadate'])
    fund = dg.drop_duplicates(['permno','rdq'], keep='first').copy()
    fund['rebaldate'] = [bd.endmo(d)                         # rebalance date is end of month
                         for d in bd.shift(fund['rdq'], 1)]  #   one day after rdq date

    # ear is compounded return around 3-day window
    out = crsp.get_window('daily', 'ret', fund['permno'], fund['rdq'], -1, 1)
    fund['ear'] = list((1 + out).prod(axis = 1))   # compound the three days's returns

    # aeavol is average volume in 3-day window relative to 20-day average ten-days prior
    actual = crsp.get_window('daily', 'vol', fund['permno'], fund['rdq'], -1, 1)
    normal = crsp.get_window('daily', 'vol', fund['permno'], fund['rdq'], -30, -11)
    fund['aeavol'] = list(actual.mean(axis=1)/normal.mean(axis=1))
    print('Elapsed: %.0f secs' % (time.time() - tic))
    
    signals.save(fund, 'ear')
    signals.save(fund, 'aeavol')
    
    rebalbeg, rebalend = 19500101, 20190630
    benchnames = ['Mom(mo)','Mkt-RF(mo)']
    for signal in columns:
        run_backtest(backtest, crsp, signal, 4,  # use latest signal from recent four months' rebalances
                     benchnames, rebalbeg, rebalend, signals,  outdir=outdir, html=html)


if False:
    #
    #  IBES FY1 signals linked to Quarterly PSTAT: sfe
    #
    tic = time.time()
    beg, end = 19760101, 20190531        
    out = DataFrame()     # accumulate all rows of computed signal values
    for pordate in bd.endmo_range(beg, end):
        df = ibes.get_linked(table = 'summary',       # query ibes, link to CRSP permno
                             date_field = 'statpers',
                             fields = ['fpedats', 'meanest', 'medest', 'actual'],
                             limit = '',
                             where = (" fpedats IS NOT NULL " \
                                      " AND meanest IS NOT NULL " \
                                      " AND statpers >= {date}00 " \
                                      " AND statpers <= {date}99" \
                                      " AND summary.fpi = '1'" \
                                      "".format(date=pordate//100))    # and fpedats <= statpers
                             ).sort_values(['permno', 'statpers', 'fpedats'])
        estim = df.drop_duplicates(['permno'], keep='first').copy().set_index('permno', drop=False)
        fund = pstat.get_linked(table='quarterly',    # query pstat quarterly, link to CRSP permno
                                date_field='datadate',
                                fields=['prccq'],
                                where=('prccq > 0 and datadate > %d and datadate < %d99' %
                                       (bd.begmo(pordate, -4), pordate//100))  # any pstat in past 4 months
                                ).sort_values(['permno','datadate'], ascending=False)
        fund = fund.drop_duplicates(['permno'], keep='first').set_index('permno')
        estim['sfe'] = estim['meanest'] / fund['prccq'].abs().reindex(estim.index)
        estim['rebaldate'] = pordate
        out = out.append(estim[['permno','rebaldate', 'sfe']], ignore_index=True)
    print('Elapsed: %.0f secs' % (time.time() - tic))
    n = signals.save(out, 'sfe', append=False)

    rebalbeg, rebalend = 19760101, 20190630
    benchnames = ['HML(mo)','Mkt-RF(mo)']
    run_backtest(backtest, crsp, 'sfe', 3,   # use latest signal from recent three months' rebalances
                 benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)


if False:
    #
    # IBES Q1 (FPI='6') Fiscal Quarter 1, linked to Compustat Quarterly: sue
    #
    tic = time.time()
    columns = ['sue']
    numlag = 4
    end = 20190228

    # retrieve all quarterly, keep [permno, datadate] key with non null price (prccq) if any
    dg = pstat.get_linked(table = 'quarterly',          # query quarterly pstat, link to CRSP permno
                          date_field = 'datadate',
                          fields = ['prccq', 'cshoq', 'ibq'],
                          where = ("datadate <= {}31".format(end//100))
                          ).sort_values(['permno', 'datadate', 'prccq'])
    fund = dg.drop_duplicates(['permno','datadate'], keep='first').copy()
    fund['rebaldate'] = [bd.endmo(d, numlag) for d in fund['datadate']]  # rebalance numlag months later
    fund.index = list(zip(fund['permno'], fund['rebaldate']))  # construct multi-index for {fund}

    # retrieve all ibes Q1 where forecast date (statpers) <= fiscal date (fpedats), keep latest
    df = ibes.get_linked(table='summary',              # query ibes summary, link to CRSP permno
                         date_field='statpers',
                         fields=['fpedats', 'medest', 'actual'],
                         limit='',
                         where = (" medest IS NOT NULL " \
                                  " AND actual IS NOT NULL " \
                                  " AND summary.fpi = '6'" \
                                  " AND statpers <= fpedats")
                         ).sort_values(['permno', 'fpedats', 'statpers'], ascending = False)
    out = df.drop_duplicates(['permno','fpedats'], keep='first').copy()
    out.index = list(zip(out['permno'], [bd.endmo(d, numlag) for d in out['fpedats']])) 
    out = out.reindex(fund.index)     # construct multi-index for {out} to re-align with {fund}

    # Compute initial sue with with ibes surprise, scaled by compustat quarterly price
    fund['sue'] = (out['actual'] - out['medest']) / fund['prccq'].abs()

    # compute lag(4) difference in compustat quarterly for missing initial sue
    lag = fund.shift(4, fill_value=0)
    f = ((lag['permno'] == fund['permno']) & fund['sue'].isnull())
    fund.loc[f, 'sue'] = ((fund.loc[f, 'ibq'] - lag.loc[f, 'ibq']) /
                          (fund.loc[f, 'prccq'].abs() * fund.loc[f, 'cshoq'].abs()))   
    print('Elapsed: %.0f secs' % (time.time() - tic))
    signals.save(fund, 'sue')
    
    rebalbeg, rebalend = 19760101, 20190630
    benchnames = ['Mom(mo)','Mkt-RF(mo)']
    for signal in columns:
        run_backtest(backtest, crsp, signal, 3,   # use latest signal from recent three months' rebalances
                     benchnames, rebalbeg, rebalend, signals, outdir=outdir, html=html)
# %matplotlib agg
