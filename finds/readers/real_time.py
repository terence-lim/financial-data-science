"""Update with current market price information

Copyright 2022-2024, Terence Lim

MIT License
"""
import os
import time
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from finds.database import SQL, RedisDB, as_dtypes
from finds.structured import (BusDay, CRSP, Benchmarks, PSTAT,
                              SignalsFrame, CRSPBuffer)
from finds.utils import Finder, plot_date
from finds.recipes import fractile_split
from finds.readers import FFReader
from finds.backtesting import DailyPerformance, BackTest, bivariate_sorts

from yahoo import get_price
from secret import credentials, paths, CRSP_DATE
VERBOSE = 1

START_DATE = CRSP_DATE - 200  # start collect yahoo
PSTAT_DATE = 20171201  # date to start collecting PSTAT data
REBAL_DATE = 20220630  # at least 4 years after PSTAT data
LAST_DATE = BusDay.today()   

downloads = paths['yahoo']
DEBUG = True
today = datetime.today().strftime('%Y%m%d')

ticker_changes = []  # [('FB','META')]

#if __name__ == "__main__":  # hack to splice in latest yahoo finance prices

    # time.sleep(3600*24)

if True:  # download from yahoo
    symbols = pdr.nasdaq_trader.get_nasdaq_symbols()
    print(today, CRSP_DATE, downloads, len(symbols), DEBUG)
    tic = time.time()
    
    prices = {}
    ignored = []
    for i, symbol in enumerate(symbols.index):
        if (not isinstance(symbol, str) or ('$' in symbol) or ('^' in symbol) or
            symbols.loc[symbol, 'Test Issue']):
            print(symbol, ' ignored (bad symbol)')
            ignored += [symbol]
            continue
        ticker = symbol.replace('.','-')  # yahoo share class follows '-'
        df = get_price(ticker=ticker, start_date=str(START_DATE))
        if df is None:
            print(symbol, ' ignored (get_price error)')
            ignored += [symbol]
            continue            
        if not len(df):
            print(symbol, ' ignored (len==0)')
            ignored += [symbol]
            continue

        print(i, ticker, max(df['close']), int(time.time() - tic), len(symbols))

        df = df.reset_index().rename(columns={'index': 'date'})
        df['ticker'] = ticker
        prices[ticker] = df

    outdir = downloads / today
    outdir.mkdir(exist_ok=True)
    pd.concat(prices.values(), ignore_index=True)\
      .to_csv(outdir / 'prices.csv.gz', sep='|', index=False)
    symbols.to_csv(outdir / 'symbols.csv.gz', sep='|', index=False)

if True:
    # Merge daily prices from all downloaded weekly files to form year-to-date
    files = sorted(downloads.glob('2*'), reverse=True)
    prices = DataFrame(columns=['ticker'])
    divs = DataFrame(columns=['ticker'])
    for ifile, pathname in enumerate(files):
        df = pd.read_csv(pathname / 'prices.csv.gz', sep='|')

        # compute returns
        lag = df[['ticker' ,'close', 'adjClose']].shift()
        df['ret'] = np.where(df['ticker'] == lag['ticker'],
                             abs(df['adjClose'] / lag['adjClose']),
                             np.nan)
        df['retx'] = np.where(df['ticker'] == lag['ticker'],
                              abs(df['close'] / lag['close']),
                              np.nan)

        # estimate dividends
        div = df[['ticker', 'date', 'ret', 'retx', 'close']]\
            .shift(-1, fill_value=0)
        div = div.loc[abs(div['ret'] - div['retx']) > 1e-5]
        div['div'] = (div['ret'] - div['retx']) * div['close'] / div['retx']
        div = div[['div', 'date', 'ticker']].reset_index(drop=True)
        
        if ifile:
            new = set(df['ticker']).difference(set(prices['ticker']))
            prices = pd.concat([prices, df[df['ticker'].isin(new)]],
                               axis=0, sort=False)
            print(pathname, 'added prices', new)

            new = set(div['ticker']).difference(set(divs['ticker']))
            divs = pd.concat([divs, div[div['ticker'].isin(new)]], sort=False)
            print(pathname, 'added divs', new)
        else:
            prices = df
            divs = div
            print(pathname, 'prices has', len(prices))
            print(pathname, 'added divs', len(divs))

    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    bd = BusDay(sql)
    crsp = CRSP(sql, bd, rdb=None)
    date = bd.offset(CRSP_DATE)      # last date of CRSP data

    # get price and shrout as of last date
    price = crsp.get_section(dataset='daily',
                             fields=['prc', 'shrout'],
                             date_field='date',
                             date=date,
                             start=-1)

    # get tickers to lookup permno
    tickers = crsp.get_section(dataset='names',
                               fields=['tsymbol', 'date'],
                               date_field='date',
                               date=date,
                               start=0).reindex(price.index)
    tickers = tickers.sort_values(['tsymbol', 'date'])\
                     .drop_duplicates(keep='last')
    
    # kludge in ticker changes
    for t_, ticker_ in ticker_changes:
        tickers.loc[tickers['tsymbol'].eq(t_), 'tsymbol'] = ticker_    
    tickers = tickers.reset_index().set_index('tsymbol')['permno']
    
    # Yahoo has '-' but CRSP has '' between symbol and share class
    prices['ticker'] = prices['ticker'].str.replace('-','')  # divs
    
    # merge permnos into big prices table
    prices = prices.join(tickers, on='ticker', how='inner')

    # estimate facpr, for internal consistency, based on crsp_date price ratio
    # TODO:
    #   where prices substantially different (more than 0.005 and outside range)
    #     include facpr
    #   else: adjust the next day's ret and retx by the facpr
    
    facpr = prices[prices.date == date][['permno', 'close']]
    facpr = facpr.join(price,
                       on='permno',
                       how='inner')
    facpr['facpr'] = np.round((facpr['prc'].abs() / facpr['close']) - 1, 3)
    facpr = facpr[facpr['facpr'].abs().gt(0.005)
                  & abs(facpr['close'] - abs(facpr['prc'])).ge(0.005)
                  & (facpr['prc'].gt(0) |
                     abs(facpr['close'] - abs(facpr['prc'])).gt(0.25))]
    

    # adjust shrout if necessary, then merge into big df of prices
    facpr.loc[:, 'shrout'] *= (1 + facpr['facpr'])
    facpr = facpr.set_index('permno')
    price.loc[facpr.index, 'shrout'] = facpr.loc[facpr.index, 'shrout']
    prices = prices.join(price[['shrout']], on='permno', how='inner')

    # create monthly data df
    prices['month'] = prices['date'] // 100
    groupby = prices.groupby(['permno', 'month'])
    monthly = (groupby[['ret', 'retx']].prod() - 1)\
              .join(groupby['close'].last()).reset_index()
    monthly['month'] = bd.endmo(monthly['month'])
    monthly.rename(columns={'month': 'date', 'close':'prc'}, inplace=True)
    monthly = monthly[monthly['date'] > date]
    monthly['dlstcd'] = 0
    monthly['dlret'] = np.nan
    print('monthly:', len(monthly), len(np.unique(monthly['permno'])))
    
    # clean up prices table to mimic daily table
    prices['ret'] -= 1
    prices['retx'] -= 1
    prices.rename(columns={'open':'openprc', 'high':'askhi', 'low':'bidlo',
                           'close':'prc', 'volume':'vol'}, inplace=True)
    prices.drop(columns=['month', 'ticker','adjClose'], inplace=True)
    prices['bid'] = np.nan
    prices['ask'] = np.nan
    prices = as_dtypes(prices,
                       {k: v.type for k,v in crsp['daily'].columns.items()})
    prices = prices[prices['date'] > date]
    prices.index = np.arange(len(prices))
    print(prices['date'].value_counts())

    # clean up facpr table and merge into dist dataframe
    #
    # TODO: why not as_dtypes on facpr at the end?
    #
    dist = as_dtypes(None, {k : v.type for k,v in crsp['dist'].columns.items()})
    facpr = facpr.reset_index().drop(columns=['close','prc','shrout'])
    facpr['exdt'] = int(bd.offset(date, 1))
    facpr['distcd'] = 5523
    facpr['facshr'] = facpr['facpr']
    dist = pd.concat([dist, facpr], axis=0, sort=False)
    print(np.isnan(dist).mean(axis=0))

    # clean up divs table and merge into dist dataframe
    divs = divs[divs['date'] > date]\
                .join(tickers, on='ticker', how='inner')
    divs = divs.rename(columns={'div':'divamt', 'date':'exdt'})\
                         .drop(columns=['ticker'])
    divs['distcd'] = 1232
    divs['divamt'] = (divs['divamt'] * 2).round(2)/2 # round half cent
    dist = pd.concat([dist, divs], axis=0, sort=False)  # $ divs
    print(np.isnan(dist).mean(axis=0))

    # save current intermediate files
    tickers.to_csv(downloads / 'tickers.csv.gz', sep='|')
    divs.to_csv(downloads / 'dividends.csv.gz', sep='|')
    facpr.to_csv(downloads / 'facpr.csv.gz', sep='|')
    prices.to_csv(downloads / 'daily.csv.gz', sep='|')

    # update monthly sql table
    print('monthly', sql.run('select count(*) from monthly'))
    table = crsp['monthly']
    delete = table.delete().where(table.c['date'] > date)
    sql.run(delete)
    print('monthly', sql.run('select count(*) from monthly'))
    sql.load_dataframe(table.key, monthly, index_label=None, to_sql=True,
                       replace=False)
    print('monthly', sql.run('select count(*) from monthly'))

    # update dist table
    print(len(dist), len(np.unique(dist.permno)), dist['exdt'].min(),
          dist['exdt'].max(), np.unique(dist.distcd))
    dist[dist.isna()] = 0
    dist = as_dtypes(dist, {k : v.type for k,v in crsp['dist'].columns.items()})
    dist.to_csv(downloads / 'dist.csv.gz', sep='|', index=False)

    print('dist', sql.run('select count(*) from dist'))
    table = crsp['dist']
    delete = table.delete().where(table.c['exdt'] > date)
    sql.run(delete)
    print('dist', sql.run('select count(*) from dist'))
    sql.load_dataframe(table.key, dist, index_label=None, to_sql=True,
                       replace=False)
    print('dist', sql.run('select count(*) from dist'))

    # update daily table
    print('daily', sql.run('select count(*) from daily'))
    table = crsp['daily']
    delete = table.delete().where(table.c['date'] > date)
    sql.run(delete)
    print(sql.run('select count(*) from daily'))
    
    sql.load_dataframe(table.key, prices, index_label=None, to_sql=True,
                       replace=False)
    print('update', sql.run(f"select count(*) from daily where date > {date}"))
    print('daily', sql.run('select count(*) from daily'))

    print(f"redis-cli --scan --pattern '*CRSP_{CRSP_DATE // 10000}*' "
          + f"| xargs redis-cli del")

if True:
    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    user = SQL(**credentials['user'], verbose=VERBOSE)
    #rdb = Redis(**credentials['redis'])   # don't use redis
    bd = BusDay(sql, new=False)
    bench = Benchmarks(sql, bd, verbose=VERBOSE)
    crsp = CRSP(sql, bd, rdb=None, verbose=VERBOSE)
    pstat = PSTAT(sql, bd)
    imgdir = paths['scratch']

    bd = BusDay(sql, new=True)    # new=True to update busdays

if True:  # Compare monthly market-weighted returns to FF research factor
    rebalbeg = bd.offset(REBAL_DATE)  # first rebalance date to compute HML, MOM

    ## Compute monthend universe holdings
    holdings = {}
    for rebal in bd.date_range(rebalbeg, bd.endmo(LAST_DATE, -1), freq='endmo'):
        univ = crsp.get_universe(rebal)
        holdings[rebal] = univ['cap'] / univ['cap'].sum()

    ## Run back-test (auto monthly), compare to market factor benchmark
    label = 'mkt-rf'    
    backtest = BackTest(user, bench, 'RF', bd.offset(LAST_DATE))    
    backtest(crsp, holdings, label)
    excess = backtest.fit(['Mkt-RF'])

    ## Plot
    fig, ax = plt.subplots(num=1, figsize=(10, 6), clear=True)
    plot_date(y1=excess.iloc[:-1].cumsum(), marker='',
              ax=ax, legend1=['self-computed', 'Ken French Library'],
              loc1='lower left', vlines=[CRSP_DATE],
              title=f"Monthly Returns to {label} factor, through {LAST_DATE}")
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
    rebalend = bd.endmo(LAST_DATE, -1)
    perf = DailyPerformance(stocks=CRSPBuffer(stocks=crsp,
                                              beg=rebalbeg,
                                              end=bd.offset(LAST_DATE),
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
        if (datadate // 10000) < (LAST_DATE // 10000):
            capdate = min(crsp.bd.endyr(datadate), LAST_DATE) # Dec year-end cap
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
        p = [crsp.get_universe(rebaldate),    # retrieve prices for signal
             crsp.get_ret(start, end).rename(label),
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
    ret = {label: perf(holdings[label], bd.offset(LAST_DATE))
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
                  title=f"Daily Returns to {label} factor, through {LAST_DATE}")
        ax.set_xlabel(f"published {datetime.now().strftime('%Y-%m-%d')}")
        plt.tight_layout()
        plt.savefig(imgdir / f"{label}.jpg")
    plt.show()
