"""Market Microstructure

- Tick data: NYSE Daily TAQ 
- Spreads: quoted, effective, price impact, realized
- Volatility: variance ratio, Parkinsons, Klass-Garman

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

import numpy as np
import pandas as pd
import time
import os
from pandas import DataFrame, Series
from matplotlib import colors
import matplotlib.pyplot as plt
from finds.database import SQL, Redis
from finds.structured import CRSP
from finds.busday import BusDay
from finds.taq import opentaq, itertaq, bin_trades, bin_quotes, TAQ
from finds.display import plot_time, row_formatted
from finds.recipes import weighted_average
from conf import credentials, paths, VERBOSE

VERBOSE = 0
sql = SQL(**credentials['sql'], verbose=VERBOSE)
user = SQL(**credentials['user'], verbose=VERBOSE)
bday = BusDay(sql)
rdb = Redis(**credentials['redis'])
crsp = CRSP(sql, bday, rdb=rdb, verbose=VERBOSE)
imgdir = paths['images']
taqdir = paths['taq']

open_t = pd.to_datetime('1900-01-01T9:30')    # exclude <= 
close_t = pd.to_datetime('1900-01-01T16:00')  # exclude >
EPSILON = 1e-15


def volatility_HL(high: DataFrame, low: DataFrame,
                  last: DataFrame = None) -> DataFrame:
    """Compute Parkinson volatility from high and low prices
    
    Args:
        high: DataFrame of high prices (observations x stocks)
        low: DataFrame of low prices (observations x stocks)
        last: DataFrame of last prices, for forward filling if high low missing

    Returns:
        Dataframe of Parkinson volatility = 
    """
    if last is not None:
        high = high.where(high.notna(), last.shift())
        low = low.where(high.notna(), last.shift())
    return np.sqrt((np.log(high / low)**2).mean(axis=0, skipna=True)
                   / (4 * np.log(2)))

    
def volatility_OHLC(first: DataFrame, high: DataFrame, low: DataFrame,
                    last: DataFrame, ffill: bool = False,
                    zero_mean: bool = True) -> DataFrame:
    """Compute Garman-Klass or Rogers-Satchell (non zero mean) OHLC volatility
    
    Args:
        first: DataFrame of open prices (observations x stocks)
        high: DataFrame of high prices (observations x stocks)
        low: DataFrame of low prices (observations x stocks)
        last: DataFrame of close prices (observations x stocks)

    Returns:
        Dataframe of Parkinson volatility = 
    """
    if ffill:
        last = last.ffill()
        high = high.where(high.notna(), last.shift())
        low = low.where(low.notna(), last.shift())
        first = low.where(first.notna(), last.shift())
    if zero_mean:  # Garman-Klass (assuming zero mean drift)
        v = ((np.log(high / low)**2) / 2
             - (2*np.log(2) - 1) * (np.log(last / first)**2))\
             .mean(axis=0, skipna=True)
    else:          # Rogers-Satchell (non zero mean drift)
        v = ((np.log(high / close) * np.log(high / close))
             + (np.log(high / close) * np.log(high / close)))\
             .mean(axis=0, skipna=True)
    return np.sqrt(v)


    

# Loop through the sample TAQ data dates available from NYSE and collect info
shareclass = []
daily_all = []

bins = {k: [] for k in ['effective', 'realized', 'impact',
                        'quoted', 'volume', 'offersize', 'bidsize',
                        'ret', 'retq', 'counts']}
tic = time.time()
intervals = ([(v, 's') for v in [1, 2, 5, 15, 30]]
             + [(v, 'm') for v in [1, 2, 5]])
dates = [20191007, 20191008, 20180305, 20180306]
for d, date in enumerate(dates):
    master, trades, quotes = opentaq(date, taqdir)

    # screen on CRSP universe
    univ = crsp.get_universe(date)\
               .join(crsp.get_section(dataset='names',
                                      fields=['ncusip', 'permco', 'exchcd'],
                                      date_field='date',
                                      date=date,
                                      start=0), how='inner')\
               .sort_values(['permco', 'ncusip'])

    # drop duplicate share classes (same permco): keep largest cap
    dups = master['CUSIP'].str\
                          .slice(0, 8)\
                          .isin(univ.loc[univ.duplicated(['permco'],keep=False),
                                         'ncusip'])
    shareclass.extend(master[dups].to_dict(orient='index').values())    
    univ = univ.sort_values(['permco', 'cap'], na_position='first')\
               .drop_duplicates(['permco'], keep='last')\
               .reset_index()\
               .set_index('ncusip', drop=False)

    # Iterate by symbol over Daily Taq trades, nbbo and master files
    for ct, cq, mast in itertaq(trades,
                                quotes,
                                master,
                                cusips=univ['ncusip'],
                                open_t=open_t,
                                close_t=None,
                                verbose=VERBOSE):
        header = {'date':date}
        header.update(univ.loc[mast['CUSIP'][:8],
                               ['permno','decile','exchcd','siccd']])
        header.update(mast[['Symbol', 'Round_Lot']])

        # Compute and collect daily and bin statistics at all intervals
        daily = header.copy()   # to collect current stock's daily stats
        v, u = intervals[-1]
        for (v, u) in intervals:
            bt = bin_trades(ct, v, u, open_t=open_t, close_t=close_t)
            bq = bin_quotes(cq, v, u, open_t=open_t, close_t=close_t)
            daily[f"tvar{v}{u}"] = bt['ret'].var(ddof=0) * len(bt)
            daily[f"tvarHL{v}{u}"] = ((volatility_HL(bt['maxtrade'],
                                                    bt['mintrade'])**2)
                                      * len(bt))
            daily[f"tvarOHLC{v}{u}"] = ((volatility_OHLC(bt['first'],
                                                         bt['maxtrade'],
                                                         bt['mintrade'],
                                                         bt['last'])**2)
                                        * len(bt))
            daily[f"qvar{v}{u}"] = bq['retq'].var(ddof=0) * len(bq)
            daily[f"qvarHL{v}{u}"] = ((volatility_HL(bq['maxmid'],
                                                     bq['minmid'])**2)
                                      * len(bq))
            daily[f"qvarOHLC{v}{u}"] = ((volatility_OHLC(bq['firstmid'],
                                                         bq['maxmid'],
                                                         bq['minmid'],
                                                         bq['mid'])**2)
                                        * len(bq))
            daily[f"tunch{v}{u}"] = np.mean(np.abs(bt['ret']) < EPSILON)
            daily[f"qunch{v}{u}"] = np.mean(np.abs(bq['retq']) < EPSILON)
            daily[f"tzero{v}{u}"] = np.mean(bt['counts'] == 0)


        # Collect final (i.e. 5 minute bins) bt and bq intraday series
        df = bq.join(bt, how='left')
        for s in ['effective', 'realized', 'impact', 'quoted']:
            bins[s].append({**header,
                            **(df[s]/df['mid']).to_dict()})
        for s in ['volume', 'offersize', 'bidsize', 'ret', 'retq', 'counts']:
            bins[s].append({**header,
                            **df[s].to_dict()})
        #print(date, d, len(daily), int(time.time()-tic), 'secs')

        # Collect daily means
        daily.update(df[['bidsize', 'offersize', 'quoted', 'mid']].mean())
        daily.update(df[['volume', 'counts']].sum())
        daily.update(weighted_average(df[['effective', 'impact', 'realized',
                                            'vwap', 'volume']],
                                        weights='volume'))
        daily_all.append(daily)
        #break
    quotes.close()
    trades.close()
    print(d, date, time.time() - tic)
    #break

daily_df = DataFrame(daily_all)
bins_df = {k: DataFrame(bins[k]) for k in bins.keys()}

raise Exception('stop')

if True:
    import pickle
    with open(os.path.join(paths['scratch'], 'tick.daily'), 'wb') as outfile:
        pickle.dump(daily_df, outfile)
    with open(os.path.join(paths['scratch'], 'tick.bins'), 'wb') as outfile:
        pickle.dump(bins_df, outfile)
    with open(os.path.join(paths['scratch'], 'tick.shrcls'), 'wb') as outfile:
        pickle.dump(shareclass, outfile)
if False:    
    import pickle
    with open(os.path.join(paths['scratch'], 'tick.daily'), 'rb') as f:
        daily_df = pickle.load(f)
    with open(os.path.join(paths['scratch'], 'tick.bins'), 'rb') as f:
        bins_df = pickle.load(f)


# Daily average of liquidity metrics in means, by size

## Group by market cap (NYSE deciles 1-3, 4-6, 7-9, 10) and exchange listed
daily_df['Size'] = pd.cut(daily_df['decile'],
                          [0, 3.5, 6.5, 9.5, 11],
                          labels=['large', 'medium', 'small', 'tiny'])
daily_df['Exchange'] = pd.cut(daily_df['exchcd'],
                              [0, 2.5, 4],
                              labels=['NYSE','NASDAQ'])
groupby = daily_df.groupby(['Size', 'Exchange'])

## Collect results for each metric
results = {}    # to collect results as dict of {column: Series}
formats = {}    # and associated row formatter string
results.update(groupby['mid']\
               .count()\
               .rename('Number of Stock/Days').to_frame())
formats.update({'Number of Stock/Days': '{:.0f}'})

result = groupby[['mid', 'vwap']].median()   # .quantile(), and range
result.columns = ['Midquote Price', "VWAP"]
formats.update({k: '{:.2f}' for k in result.columns})
results.update(result)

result = groupby[['counts', 'volume']].median()
result.columns = ['Number of trades', "Volume (shares)"]
formats.update({k: '{:.0f}' for k in result.columns})
results.update(result)

result = np.sqrt(groupby[['tvar15m', 'qvar15m', 'tvarHL15m', 'qvarHL15m',
                          'tvarOHLC15m', 'qvarOHLC15m']].median())
result.columns = ['Volatility(trade price)', "Volatility(midquote)",
                  'Volatility(HL trade price)', "Volatility(HL midquote)",
                  'Volatility(OHLC trade price)', "Volatility(OHLC midquote)"]
formats.update({k: '{:.4f}' for k in result.columns})
results.update(result)

result = groupby[['offersize', 'bidsize']].median()
result.columns = [s.capitalize() + ' (lots)' for s in result.columns]
formats.update({k: '{:.1f}' for k in result.columns})
results.update(result)

spr = ['quoted', 'effective', 'impact', 'realized']
result = groupby[spr].median()
result.columns = [s.capitalize() + ' $ spread' for s in spr]
formats.update({k: '{:.4f}' for k in result.columns})
results.update(result)

rel = [s.capitalize() + ' (% price)' for s in spr]
daily_df[rel] = daily_df[spr].div(daily_df['mid'], axis=0)  # scale spreads
result = 100*groupby[rel].median()
formats.update({k: '{:.4f}' for k in result.columns})
results.update(result)

## display table of results
show(row_formatted(DataFrame(results).T, formats),
     caption="Average Liquidity by Market Cap/Exchange Listed")



# Summarize unchanged midquote and last trade price, and zero-volume bins
def plot_helper(result, xticks, keys, legend, xlabel, title, ylim=[],
                figsize=(5,3), num=1, fontsize=8):
    """helper to plot bar graphs"""
    fig, ax = plt.subplots(num=num, clear=True, figsize=figsize)
    result.plot(kind='bar',
                fontsize=fontsize,
                rot=0,
                width=0.8,
                xlabel='',
                ax=ax)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticklabels(xticks, fontsize=fontsize)
    ax.legend(keys, loc='upper left', bbox_to_anchor=(1.0, 1.0), 
              fontsize=fontsize, title=legend, title_fontsize=8)
    ax.set_xlabel(xlabel, fontsize=fontsize + 2)
    ax.set_title(title, fontsize=fontsize + 2)
    plt.subplots_adjust(right=0.8, bottom=0.15)
    plt.tight_layout()
    return ax

xticks = [f"{v}{u}" for v, u in intervals]                # x-axis: bin lengths
keys = [a + '\n' + b for a, b in groupby.indices.keys()]  # legend labels

labels = [f"tunch{v}{u}" for v, u in intervals]
result = groupby[labels].median()*100
ax = plot_helper(result.T,
                 title="% Unchanged Bins (Last Trade Price)",
                 xticks=xticks,
                 xlabel="Bin Length",
                 keys=keys,
                 legend='Size/Exch',
                 num=1)
plt.savefig(os.path.join(imgdir, 'tunch' + figext))

labels = [f"qunch{v}{u}" for v, u in intervals]
result = groupby[labels].median()*100
ax = plot_helper(result.T,
                 title="% Unchanged Bins (Last MidQuote)",
                 xticks=xticks,
                 xlabel="Bin Length",
                 keys=keys,
                 legend='Size/Exch',
                 num=2)
plt.savefig(os.path.join(imgdir, 'qunch' + figext))

labels = [f"tzero{v}{u}" for v, u in intervals]
result = groupby[labels].median()*100
ax = plot_helper(result.T,
                 title="% Zero-Volume Bins",
                 xticks=xticks,
                 xlabel="Bin Length",
                 keys=keys,
                 legend='Size/Exch',
                 num=3)
plt.savefig(os.path.join(imgdir, 'tzero' + figext))

labels = [f"tvar{v}{u}" for v, u in intervals]
result = groupby[labels].median()
result = result.div(result['tvar1s'].values, axis=0)
ax = plot_helper(result.T,
                 title="Variance Ratio (Last Trade Price)",
                 xticks=xticks,
                 xlabel="Bin Length",
                 keys=keys,
                 legend='Size/Exch',
                 num=4)
plt.savefig(os.path.join(imgdir, 'tvratio' + figext))
    
labels = [f"tvar{v}{u}" for v, u in intervals]
result = np.sqrt(groupby[labels].median())
ax = plot_helper(result.T,
                 title="Daily Ret StdDev (last trade price)",
                 xticks=xticks,
                 xlabel="Bin Length",
                 keys=keys,
                 legend='Size/Exch',
                 num=5)
plt.savefig(os.path.join(imgdir, 'tstd' + figext))

labels = [f"qvar{v}{u}" for v, u in intervals]
result = np.sqrt(groupby[labels].median())
ax = plot_helper(result.T,
                 title="Daily Ret StdDev (midquote)",
                 xticks=xticks,
                 xlabel="Bin Length",
                 keys=keys,
                 legend='Size/Exch',
                 num=6)
plt.savefig(os.path.join(imgdir, 'qstd' + figext))
plt.show()


# Compare methods of volatility estimates, by interval and market cap/exchange
for ifig, (split_label, split_df) in enumerate(groupby):
    vol_df = np.sqrt(split_df[[c for c in daily_df.columns if "qvar" in c]])
    result = []
    for col in [c for c in vol_df.columns if "qvar" in c]:
        if col[4] == 'H':
            m = 'HL'
        elif col[4] == 'O':
            m = 'OHLC'
        else:
            m = 'Close'
        result.append({'method': m + ' ' + {'t': 'trade', 'q': 'quote'}[col[0]],
                       'interval': (int("".join(filter(str.isdigit, col)))
                                    * (60 if col[-1] == 'm' else 1)),
                       'val': vol_df[col].median()})
    result = DataFrame.from_dict(result, orient='columns')\
        .pivot(index='interval', columns='method', values='val')
    ax = plot_helper(result,
                     title="Median Volatility in " + " ".join(split_label),
                     xticks=xticks,
                     xlabel="Bin Length",
                     keys=result.columns,
                     num=1+ifig,
                     ylim=[0.0 ,0.05],
                     legend='method')
    plt.savefig(os.path.join(imgdir, 'tick_' + "_".join(split_label) + figext))


# Intraday spreads, depths and volumes
keys = ['effective', 'realized', 'impact', 'quoted', 
        'volume', 'counts', 'offersize', 'bidsize']
for num, key in enumerate(keys):
    df = bins_df[key].drop(columns=['Round_Lot', 'Symbol'])
    df.index = list(zip(df['permno'], df['date']))
        
    # Group by market cap and exchange
    df['Size'] = pd.cut(df['decile'],
                        [0, 3.5, 6.5, 9.5, 11],
                        labels=['large', 'medium', 'small', 'tiny'])
    df['Exchange'] = pd.cut(df['exchcd'],
                            [0, 2.5, 4],
                            labels=['NYSE','NASDAQ'])
    df = df.drop(columns=['date', 'permno', 'decile', 'exchcd', 'siccd'])\
           .dropna()\
           .groupby(['Size', 'Exchange'])\
           .median().T
    fig, ax = plt.subplots(1, 1, num=num+1, clear=True, figsize=(5, 3))
    plot_time(df.iloc[1:],
              title='Median ' + key.capitalize(),
              ax=ax, 
              fontsize=8,
              loc='upper center',
              legend1=None)
    ax.legend([a + '\n' + b for a,b in df.columns], 
              loc='upper left', bbox_to_anchor=(1.0, 1.0), 
              fontsize=8)
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, key + '' + figext))
plt.show()
