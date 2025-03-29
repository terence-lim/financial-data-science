"""Update with current market price information

Copyright 2022-2024, Terence Lim

MIT License
"""
import os
import time
import sys
from datetime import datetime
import numpy as np
import random
import pandas as pd
from pandas import DataFrame, Series
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import yfinance as yf
from secret import credentials, paths, CRSP_DATE
VERBOSE = 1

PSTAT_DATE = 20181201  # date to start collecting PSTAT data
REBAL_DATE = 20230630  # at least 4 years after PSTAT data
TODAY_DATE = int(datetime.today().strftime('%Y%m%d'))

downloads = paths['yahoo']
DEBUG = True

def datestr(dt: int) -> str:
    dt = int(dt)
    return f'{dt//10000}-{(dt//100)%100:02d}-{dt%100:02d}'
beg_history = datestr(CRSP_DATE - 200)
end_history = datestr(TODAY_DATE)

ticker_changes = []  # [('FB','META')]

#if __name__ == "__main__":  # hack to splice in latest yahoo finance prices

    # time.sleep(3600*24)

if True:  # download from yahoo
    symbols_ = pdr.nasdaq_trader.get_nasdaq_symbols()
    symbols = symbols_.loc[(symbols_['Financial Status'].astype(str).str.lower().str[0] == 'n').astype(bool) &  
                        ~symbols_['Test Issue'].astype(bool) &
                        ~symbols_['Security Name'].str.lower().str.contains('- warrant') &
                        ~symbols_.index.astype(str).str.contains('\$') & 
                        ~symbols_.index.astype(str).str.contains('\^')]
    symbols.index = symbols.index.astype(str).str.replace('.', '-')
    
    print(TODAY_DATE, CRSP_DATE, beg_history, end_history, downloads, len(symbols), DEBUG)
    tic = time.time()
    
    prices = {}
    ignored = []

    restart = 0  #21
    end = len(symbols)  #restart+2
    for i, symbol in enumerate(symbols.index[restart:end]):
        time.sleep(1.0 + random.random())
        dat = yf.Ticker(symbol)
        df = dat.history(start=beg_history, end=end_history)
        if not len(df):
            print(symbol, ' ignored (len==0)')
            ignored += [symbol]
            continue        
        df.index = df.index.strftime('%Y%m%d').astype(int)
        print(i, symbol, len(df), df.index[0], df.index[-1], max(df['Close']),
              int(time.time() - tic), len(symbols))

        df['ticker'] = symbol
        last_prc = df['Close'].iloc[-1]
        split = df['Stock Splits'].where(df['Stock Splits'] != 0.0, 1)\
                                  .shift(-1).fillna(1).iloc[::-1].cumprod().iloc[::-1]
        df['vol'] = df['Volume'].div(split)
        df['divamt'] = df['Dividends'].abs().mul(split).fillna(0)
        df['ret'] = ((df['Close'].abs() / df['Close'].abs().shift() - 1))
        df['prc'] = df['Close'].abs().mul(split)
        for _ in range(2):   # denominator 'prc' is not exactly right for formula, so iterate a few times
            df['divret'] = df['divamt'].div(df['prc'].abs().shift()).fillna(0)
            df['retx'] = (df['ret'] - df['divret'])
            df['prc'] = (last_prc / (df['retx'].shift(-1).fillna(0) + 1).iloc[::-1].cumprod().iloc[::-1].values) * split

        shr = dat.get_shares_full(start=beg_history, end=end_history)
        if shr is not None and len(shr):
            shr = shr.dropna()
            shr.index = shr.index.strftime('%Y%m%d').astype(int)
            shr = shr[~shr.index.duplicated(keep='first')]
            df['shrout'] = shr
            df['shrout'] = df['shrout'].ffill()
        else:
            df['shrout'] = np.nan
        prices[symbol] = df.reset_index().rename(columns={'index': 'date'})
    print('IGNORED:', len(ignored))
    outdir = downloads / str(TODAY_DATE)
    outdir.mkdir(exist_ok=True)
    pd.concat(prices.values(), ignore_index=True).to_csv(outdir / 'prices.csv.gz', sep='|', index=False)
    symbols.to_csv(outdir / 'symbols.csv.gz', sep='|', index=True)


from finds.database import SQL, RedisDB, as_dtypes
from finds.structured import BusDay, CRSP, Benchmarks, PSTAT, SignalsFrame, CRSPBuffer
from finds.utils import Finder, plot_date
from finds.recipes import fractile_split
from finds.readers import FFReader
from finds.backtesting import DailyPerformance, BackTest, bivariate_sorts    
    
if True:
    # Merge daily prices from all downloaded weekly files to form year-to-date
    files = sorted(downloads.glob('2*'), reverse=True)
    prices = list()
    dists = list()
    old_tickers = set()
    for ifile, pathname in enumerate(files):
        df = pd.read_csv(pathname / 'prices.csv.gz', sep='|')
        df['shrout'] = df['shrout'] / 1000
        df['vol'] = df['vol'].astype(int)
        
        # Only consider new tickers
        new_tickers = set(df['ticker'].unique()).difference(old_tickers)
        df = df.loc[df['ticker'].isin(new_tickers)].rename(columns={'Date': 'date'})
        old_tickers != new_tickers
        
        # Extract dist: divamt and facpr
        dist = pd.concat([df.loc[df['divamt'].gt(0), ['ticker', 'date', 'divamt']],
                          df.loc[df['Stock Splits'].ne(0), ['ticker', 'date', 'Stock Splits']]])\
                 .fillna(0)\
                 .rename(columns={'date': 'exdt', 'Stock Splits': 'facpr'})
        dist['facpr'] = dist['facpr'].where(dist['facpr'] == 0.0, dist['facpr'] - 1).fillna(0.0)
        dist['distcd'] = np.where(dist['divamt'].gt(0), 1232, 5523)
        dists.append(dist)

        df['bidlo'] = df['prc'] * df['Low'] / df['Close']
        df['askhi'] = df['prc'] * df['High'] / df['Close']
        df['openprc'] = df['prc'] * df['Open'] / df['Close']
        prices.append(df[['ticker', 'date', 'bidlo', 'askhi', 'prc', 'vol', 'ret', 'retx', 'openprc', 'shrout']])
        
        print(pathname, 'prices:', len(df), 'dists:', len(dist))

    # Combines dists and prices of all days
    dists = pd.concat(dists, axis=0)
    dists['dclrdt'] = 0
    dists['rcrddt'] = 0
    dists['paydt'] = 0
    dists['acperm'] = 0
    dists['accomp'] = 0
    dists

    prices = pd.concat(prices, axis=0)
    prices['bid'] = np.nan
    prices['ask'] = np.nan
    prices
    
    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    bd = BusDay(sql)
    crsp = CRSP(sql, bd, rdb=None)
    crsp_date = bd.offset(CRSP_DATE)      # last date of CRSP data
    prices = prices.loc[prices['date'] > crsp_date].reset_index(drop=True)
    dists = dists.loc[dists['exdt'] > crsp_date].reset_index(drop=True)

    # get shrout as of last date
    shrout = crsp.get_section(dataset='daily',
                              fields=['shrout'],
                              date_field='date',
                              date=crsp_date,
                              start=-1)

    # get tickers to lookup permno
    tickers = crsp.get_section(dataset='names',
                               fields=['tsymbol', 'date'],
                               date_field='date',
                               date=crsp_date,
                               start=0)\
                  .reindex(shrout.index)\
                  .sort_values(['tsymbol', 'date'])\
                  .drop_duplicates(subset=['tsymbol'], keep='last')

    # kludge in ticker changes
    for t_, ticker_ in ticker_changes:
        tickers.loc[tickers['tsymbol'].eq(t_), 'tsymbol'] = ticker_    
    tickers = tickers.reset_index().set_index('tsymbol')['permno']
    
    # Yahoo has '-' but CRSP has '' between symbol and share class
    prices['ticker'] = prices['ticker'].str.replace('-','')
    dists['ticker'] = dists['ticker'].str.replace('-','') 
    
    # merge permnos into prices and dists table
    prices = prices.join(tickers, on='ticker', how='inner').drop('ticker', axis=1)
    dists = dists.join(tickers, on='ticker', how='inner').drop('ticker', axis=1)

    # fill in prices['shrout'] where indexes in shrout dataframe
    price = prices.groupby('permno').first()[['date', 'shrout']]    # get first record by permno
    price.loc[price['shrout'].isna(), 'shrout'] = shrout['shrout']  # replace missing shrout
    price = price.fillna(0)
    price = price.reset_index().set_index(['permno', 'date'])  # index by (permno, date)
    index = Series(prices[['permno', 'date']].itertuples(index=False, name=None))  # indexes in prices
    prices['shrout'] = index.map(price['shrout'])\
                            .fillna(prices['shrout'].reset_index(drop=True))\
                            .ffill().values

    # create monthly data
    prices['month'] = prices['date'] // 100
    groupby = prices.groupby(['permno', 'month'])
    monthly = (groupby[['ret', 'retx']].apply(lambda x: (1+x).prod()) - 1).join(groupby['prc'].last()).reset_index()
    monthly['month'] = bd.endmo(monthly['month'])
    monthly.rename(columns={'month': 'date'}, inplace=True)
    monthly = monthly[monthly['date'] > crsp_date]
    monthly['dlstcd'] = 0
    monthly['dlret'] = np.nan
    print('monthly:', len(monthly), len(np.unique(monthly['permno'])))
    
    prices = prices[prices['date'] > crsp_date].drop(columns=['month'])
    prices.index = np.arange(len(prices))
    print('daily:', prices['date'].value_counts())

    # save daily file
    prices.to_csv(downloads / 'daily.csv.gz', sep='|')

    # update monthly sql table
    print('monthly', sql.run('select count(*) from monthly'))
    table = crsp['monthly']
    delete = table.delete().where(table.c['date'] > crsp_date)
    sql.run(delete)
    print('monthly', sql.run('select count(*) from monthly'))
    sql.load_dataframe(table.key, monthly, index_label=None, to_sql=True,
                       replace=False)
    print('monthly', sql.run('select count(*) from monthly'))

    # update dist table
    print(len(dists), len(np.unique(dists.permno)), dists['exdt'].min(),
          dists['exdt'].max(), np.unique(dists.distcd))
    dists.to_csv(downloads / 'dists.csv.gz', sep='|', index=False)

    print('dists new:', sql.run('select count(*) from dist'))
    table = crsp['dist']
    delete = table.delete().where(table.c['exdt'] > crsp_date)
    sql.run(delete)
    print('dists before:', sql.run('select count(*) from dist'))
    sql.load_dataframe(table.key, dists, index_label=None, to_sql=True,
                       replace=False)
    print('dists after:', sql.run('select count(*) from dist'))

    # update daily table
    print('daily', sql.run('select count(*) from daily'))
    table = crsp['daily']
    delete = table.delete().where(table.c['date'] > crsp_date)
    sql.run(delete)
    print('daily before:', sql.run('select count(*) from daily'))   
    sql.load_dataframe(table.key, prices, index_label=None, to_sql=True,
                       replace=False)
    print('update', sql.run(f"select count(*) from daily where date > {crsp_date}"))
    print('daily after', sql.run('select count(*) from daily'))

    #
    # (1) remove post-CRSP cache (2) change to "r"ead-only, after pre-CRSP cache saved
    #
    print(f"redis-cli --scan --pattern '*CRSP_{str(TODAY_DATE)[:4]}*' | xargs redis-cli del")


if True:
    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    user = SQL(**credentials['user'], verbose=VERBOSE)
    rdb = RedisDB(**credentials['redis'])   #### don't use redis ???
    bd = BusDay(sql, new=False)
    bench = Benchmarks(sql, bd, verbose=VERBOSE)
    crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
    pstat = PSTAT(sql, bd)
    imgdir = paths['scratch']

    bd = BusDay(sql, new=True)    # new=True to update busdays

if True:  # Compare monthly market-weighted returns to FF research factor
    rebalbeg = bd.offset(REBAL_DATE)  # first rebalance date to compute HML, MOM

    ## Compute monthend universe holdings
    holdings = {}
    for rebal in bd.date_range(rebalbeg, bd.endmo(TODAY_DATE, -1), freq='endmo'):
        univ = crsp.get_universe(rebal, cache_mode="r")
        holdings[rebal] = univ['cap'] / univ['cap'].sum()

    ## Run back-test (auto monthly), compare to market factor benchmark
    label = 'mkt-rf'    
    backtest = BackTest(user, bench, 'RF', bd.offset(TODAY_DATE))    
    backtest(crsp, holdings, label)
    excess = backtest.fit(['Mkt-RF'])

    ## Plot
    fig, ax = plt.subplots(num=1, figsize=(10, 6), clear=True)
    plot_date(y1=excess.iloc[:-1].cumsum(), marker='',
              ax=ax, legend1=['self-computed', 'Ken French Library'],
              loc1='lower left', vlines=[CRSP_DATE],
              title=f"Monthly Returns to {label} factor, through {TODAY_DATE}")
    ax.set_xlabel(f"published {datetime.now().strftime('%Y-%m-%d')}")
    plt.tight_layout()
    plt.savefig(imgdir / f"mkt.jpg")

if True:  # to update FamaFrench benchmarks
    for name, item, suffix in (FFReader.monthly + FFReader.daily):
        date_formatter = (bd.endmo if suffix == '(mo)' else bd.offset)
        df = FFReader.fetch(name=name, 
                            item=item,
                            suffix=suffix,
                            date_formatter=date_formatter)
        for col in df.columns:
            print(bench.load_series(df[col], name=name, item=str(item)))
        print(DataFrame(**sql.run('select * from ' + bench['ident'].key)))

if True:  # to calibrate Fama-French daily benchmarks
    # Compute DailyPerformance backtest, using StocksBuffer
    rebalbeg = bd.offset(REBAL_DATE)  # first rebalance date to compute HML, MOM
    rebalend = bd.endmo(TODAY_DATE, -1)
    perf = DailyPerformance(stocks=CRSPBuffer(stocks=crsp,
                                              beg=rebalbeg,
                                              end=bd.offset(TODAY_DATE),
                                              fields=['ret', 'retx'],
                                              dataset='daily'))
    holdings = {}
    
    # Compute HML factor
    label = 'hml'
    lag = 6               # number of months to lag fundamental data
    df = pstat.get_linked(  # retrieve required fields from compustat
        dataset = 'annual', date_field = 'datadate',
        fields = ['seq', 'pstk', 'pstkrv', 'pstkl', 'txditc'],
        where = (f"indfmt = 'INDL' AND datafmt = 'STD' AND curcd = 'USD' "
                 f"  AND popsrc = 'D' AND consol = 'C' "
                 f"  AND seq > 0 AND datadate >= {PSTAT_DATE}"))

    ## subtract preferred stock, add back deferred investment tax credit
    df[label] = np.where(df['pstkrv'].isna(), df['pstkl'], df['pstkrv'])
    df[label] = np.where(df[label].isna(), df['pstk'], df[label])
    df[label] = np.where(df[label].isna(), 0, df[label])
    df[label] = df['seq'] + df['txditc'].fillna(0) - df[label]
    df.dropna(subset = [label], inplace=True)
    df = df[df[label] > 0][['permno', 'gvkey','datadate',label]]

    ## years in Compustat        
    df = df.sort_values(by=['gvkey','datadate'])
    df['count'] = df.groupby(['gvkey']).cumcount()   

    ## construct b/m ratio
    df['rebaldate'] = 0
    for datadate in sorted(df['datadate'].unique()):
        if (datadate // 10000) < (TODAY_DATE // 10000):
            capdate = min(crsp.bd.endyr(datadate), TODAY_DATE) # Dec year-end cap
            f = df['datadate'].eq(datadate)
            df.loc[f, 'rebaldate'] = crsp.bd.endmo(datadate, abs(lag))
            df.loc[f, 'cap'] = crsp.get_cap(capdate)\
                                   .reindex(df.loc[f, 'permno']).values  
            print(datadate, sum(f))
    df[label] /= df['cap']
    df = df[df[label].gt(0) & df['count'].ge(2)]     # 2+ years in Compustat

    ## compute HML portfolio holdings
    signals = SignalsFrame(df)
    holdings[label], smb = bivariate_sorts(stocks=crsp,
                                           label='hml',
                                           signals=signals,
                                           rebalbeg=rebalbeg,
                                           rebalend=rebalend,
                                           window=12,
                                           months=[6])

    # Compute MOM momentum factor
    label = 'mom'
    past = (2, 12)
    df = []      # collect each month's momentum signal values
    for rebaldate in bd.date_range(rebalbeg, rebalend, 'endmo'):  
        beg = bd.endmo(rebaldate, -past[1])   # require price at this date
        start = bd.offset(beg, 1)             # start date, inclusive, of signal
        end = bd.endmo(rebaldate, 1-past[0])  # end date of signal
        p = [crsp.get_universe(rebaldate, cache_mode="r"),    # retrieve prices for signal
             crsp.get_ret(start, end, cache_mode="r").rename(label),
             crsp.get_section(dataset='monthly',
                              fields=['prc'],
                              date_field='date',
                              date=beg)['prc'].rename('beg'),
             crsp.get_section(dataset='monthly',
                              fields=['prc'],
                              date_field='date',
                              date=end)['prc'].rename('end')]
        q = pd.concat(p, axis=1, join='inner').reset_index().dropna()
        q['rebaldate'] = rebaldate
        df.append(q[['permno', 'rebaldate', label]])
        print(rebaldate, len(df), len(q))
    df = pd.concat(df)
    signals = SignalsFrame(df)
    holdings[label], smb = bivariate_sorts(stocks=crsp,
                                           label=label,
                                           signals=signals,
                                           rebalbeg=rebalbeg,
                                           rebalend=rebalend,
                                           window=0,
                                           months=[])

    # Compute and display daily returns for all factors
    ret = {label: perf(holdings[label], bd.offset(TODAY_DATE))
           for label in holdings.keys()}

    ## display returns vs bench
    bench = Benchmarks(sql, bd)
    port_bench = {'mom': 'Mom', 'hml':'HML'}
    for num, (label, benchname) in enumerate(port_bench.items()):
        benchret = bench.get_series([benchname], 'ret',
                                    beg=min(ret[label].index),
                                    end=max(ret[label].index))    
        fig, ax = plt.subplots(num=1+num, figsize=(9, 5), clear=True)
        plot_date(y1=ret[label].cumsum().to_frame().join(benchret.cumsum()),
                  ax=ax,
                  legend1=['computed', 'Ken French Library'],
                  loc1='lower left',
                  vlines=[CRSP_DATE, max(benchret.index)],
                  marker=' ',
                  fontsize=10,
                  title=f"Daily Returns to {label} factor, through {TODAY_DATE}")
        ax.set_xlabel(f"published {datetime.now().strftime('%Y-%m-%d')}")
        plt.tight_layout()
        plt.savefig(imgdir / f"{label}.jpg")
    plt.show()
