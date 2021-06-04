"""Market Microstructure

- intraday liquidity, variance ratio, effective spreads, tick sign test
- tick data, NYSE Daily TAQ 

Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
import time
import os
from pandas import DataFrame, Series
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from finds.database import SQL, Redis
from finds.structured import CRSP
from finds.busday import BusDay
from finds.taq import opentaq, itertaq, open_t, close_t, bin_trades, bin_quotes
from finds.display import plot_time, row_formatted
from finds.solve import weighted_average
from settings import settings
sql = SQL(**settings['sql'])
user = SQL(**settings['user'])
bday = BusDay(sql)
rdb = Redis(**settings['redis'])
crsp = CRSP(sql, bday, rdb=rdb)
logdir = os.path.join(settings['images'], 'micro')  # None
taqdir = os.path.join(settings['remote'], 'TAQ')
_open = pd.to_datetime('1900-01-01T9:30')    # exclude <= 
_close = pd.to_datetime('1900-01-01T16:00')  # exclude >

# Loop through the sample TAQ data dates available from NYSE and collect info
shareclass = []
daily = []

bins = {k:{} for k in ['effective', 'realized', 'impact', 'quoted', 'volume',
                       'offersize', 'bidsize', 'ret', 'retq', 'counts']}
tic = time.time()
intervals = [(v,'s') for v in [1,2,5,15,30]] + [(v,'m') for v in [1,2,5]]
dates = [20191007, 20191008, 20180305, 20180306]
for d, date in enumerate(dates):
    master, trades, quotes = opentaq(date, taqdir)

    # screen on CRSP universe, and drop duplicate share classes (same permco)
    univ = crsp.get_universe(date)\
               .join(crsp.get_section(dataset='names',
                                      fields=['ncusip', 'permco', 'exchcd'],
                                      date_field='date',
                                      date=date,
                                      start=0), how='inner')\
               .sort_values(['permco', 'ncusip'])
    dups = master['CUSIP'].str.slice(0, 8).isin(
        univ.loc[univ.duplicated(['permco'], keep=False), 'ncusip'])
    shareclass.extend(master[dups].to_dict(orient='index').values())
    univ = univ.sort_values(['permco','cap'], na_position='first')\
               .drop_duplicates(['permco'], keep='last')\
               .reset_index().set_index('ncusip', drop=False)

    # Iterate by symbol over Daily Taq trades, nbbo and master files
    for ct, cq, m in itertaq(trades, quotes, master, cusips=univ['ncusip'],
                             open_t=_open, close_t=None):
        h = {'date':date}
        h.update(univ.loc[m['CUSIP'][:8], ['permno','decile','exchcd','siccd']])
        h.update(m[['Symbol', 'Round_Lot']])

        # Compute and collect daily bin statistics at all intervals
        collect = h.copy()
        v, u = intervals[-1]
        for (v, u) in intervals:
            bt = bin_trades(ct, v, u, open_t=_open, close_t=_close)
            bq = bin_quotes(cq, v, u, open_t=_open, close_t=_close)
            collect[f"tvar{v}{u}"] = bt['ret'].var(ddof=0) * len(bt)
            collect[f"qvar{v}{u}"] = bq['retq'].var(ddof=0) * len(bq)
            collect[f"tunch{v}{u}"] = np.mean(np.abs(bt['ret']) < 1e-15)
            collect[f"qunch{v}{u}"] = np.mean(np.abs(bq['retq']) < 1e-15)
            collect[f"tzero{v}{u}"] = np.mean(bt['counts'] == 0)

        # Collect final set of bt and bq intradaily series
        df = bq.join(bt, how='left')
        for s in ['effective', 'realized', 'impact', 'quoted']:
            bins[s].update({**h, **(df[s]/df['mid']).to_dict()})
        for s in ['volume', 'offersize', 'bidsize', 'ret', 'retq', 'counts']:
            bins[s].update({**h, **df[s].to_dict()})
        #print(date, d, len(daily), int(time.time()-tic), 'secs')

        # Collect daily means
        collect.update(df[['bidsize', 'offersize', 'quoted', 'mid']].mean())
        collect.update(df[['volume', 'counts']].sum())
        collect.update(weighted_average(df[['effective', 'impact', 'realized',
                                            'vwap', 'volume']],
                                        weights='volume', axis=0))
        daily.append(collect)
    quotes.close()
    trades.close()
    #print(d, date, time.time() - tic)

daily_df = DataFrame(daily)
bins_df = {k: DataFrame(bins[k]) for k in bins.keys()}
"""
from settings import pickle_dump
pickle_dump(daily_df, 'tick.daily')
pickle_dump(bins_df, 'tick.bins')
pickle_dump(shareclass, 'tick.shareclass')

from settings import pickle_load
daily_df = pickle_load('tick.daily')
"""

# Daily average of liquidity metrics in means, by size

# Group by market cap and exchange
daily_df['Size'] = pd.cut(daily_df['decile'], [0, 3.5, 6.5, 9.5, 11],
                          labels=['large', 'medium', 'small', 'tiny'])
daily_df['Exchange'] = pd.cut(daily_df['exchcd'], [0, 2.5, 4],
                              labels=['NYSE','NASDAQ'])
g = daily_df.groupby(['Size', 'Exchange'])

results = {}    # to collect results as dict of {column: Series}
formatter = {}  # and associated format string
results.update(g['mid'].count().rename('Number of Stock/Days').to_frame())
formatter.update({'Number of Stock/Days': '{:.0f}'})

result = g[['mid', 'vwap']].median()   # .quantile(), and range
result.columns = ['Midquote Price', "VWAP"]
formatter.update({k: '{:.2f}' for k in result.columns})
results.update(result)

result = g[['counts', 'volume']].median()
result.columns = ['Number of trades', "Volume (shares)"]
formatter.update({k: '{:.0f}' for k in result.columns})
results.update(result)

result = g[['offersize', 'bidsize']].median()
result.columns = [s.capitalize() + ' (lots)' for s in result.columns]
formatter.update({k: '{:.1f}' for k in result.columns})
results.update(result)

spr = ['quoted', 'effective', 'impact', 'realized']
result = g[spr].median()
result.columns = [s.capitalize() + ' $ spread' for s in spr]
formatter.update({k: '{:.4f}' for k in result.columns})
results.update(result)

rel = [s.capitalize() + ' (% price)' for s in spr]
daily_df[rel] = daily_df[spr].div(daily_df['mid'], axis=0)  # scale spreads
result = 100*g[rel].median()
formatter.update({k: '{:.4f}' for k in result.columns})
results.update(result)

## display table of results
row_formatted(DataFrame(results).T, formatter)
#print(to_rowformat(DataFrame(results).T, formatter)\
#      .to_latex(column_format='r'*(len(result.columns)+1), longtable=True))

# helper to plot result summary comparisons
def plot_helper(result, xticks, keys, legend, xlabel, title, num=1):
    fig, ax = plt.subplots(num=num, clear=True, figsize=(5,3))
    result.plot(kind='bar', fontsize=12, rot=0, width=0.8, xlabel='', ax=ax)
    ax.set_xticklabels(xticks, fontsize=12)
    ax.legend(keys, loc='upper left', bbox_to_anchor=(1.0, 1.0), 
              fontsize=12, title=legend, title_fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.subplots_adjust(right=0.85)
    return ax    
xticks = [f"{v}{u}" for v, u in intervals]
keys = [a + '\n' + b for a,b in g.indices.keys()]

# Summarize unchanged midquote and last trade price, and zero-volume bins
labels = [f"tunch{v}{u}" for v, u in intervals]
result = g[labels].median()*100
ax = plot_helper(result.T, title="% Unchanged Bins (Last Trade Price)",
                 xticks=xticks, xlabel="Bin Length",
                 keys=keys, legend='Size/Exch', num=1)
plt.savefig(os.path.join(logdir, 'tunch.jpg'))

labels = [f"qunch{v}{u}" for v, u in intervals]
result = g[labels].median()*100
ax = plot_helper(result.T, title="% Unchanged Bins (Last MidQuote)",
                 xticks=xticks, xlabel="Bin Length",
                 keys=keys, legend='Size/Exch', num=2)
plt.savefig(os.path.join(logdir, 'qunch.jpg'))

labels = [f"tzero{v}{u}" for v, u in intervals]
result = g[labels].median()*100
ax = plot_helper(result.T, title="% Zero-Volume Bins",
                 xticks=xticks, xlabel="Bin Length",
                 keys=keys, legend='Size/Exch', num=3)
plt.savefig(os.path.join(logdir, 'tzero.jpg'))

labels = [f"tvar{v}{u}" for v, u in intervals]
result = g[labels].median()
result = result.div(result['tvar1s'].values, axis=0)
ax = plot_helper(result.T, title="Variance Ratio (Last Trade Price)",
                 xticks=xticks, xlabel="Bin Length",
                 keys=keys, legend='Size/Exch', num=4)
plt.savefig(os.path.join(logdir, 'tvratio.jpg'))
    
labels = [f"tvar{v}{u}" for v, u in intervals]
result = np.sqrt(g[labels].median())
ax = plot_helper(result.T, title="Daily Ret StdDev (last trade price)",
                 xticks=xticks, xlabel="Bin Length",
                 keys=keys, legend='Size/Exch', num=5)
plt.savefig(os.path.join(logdir, 'tstd.jpg'))

labels = [f"qvar{v}{u}" for v, u in intervals]
result = np.sqrt(g[labels].median())
ax = plot_helper(result.T, title="Daily Ret StdDev (midquote)",
                 xticks=xticks, xlabel="Bin Length",
                 keys=keys, legend='Size/Exch', num=6)
plt.savefig(os.path.join(logdir, 'qstd.jpg'))
plt.show()


# Intraday spreads, depths and volumes
keys = ['effective', 'realized', 'impact', 'quoted', 
        'volume', 'counts', 'offersize', 'bidsize']
for num, key in enumerate(keys):
    df = bins_df[key].drop(columns=['Round_Lot', 'Symbol'])
    df.index = list(zip(df['permno'], df['date']))
        
    # Group by market cap and exchange
    df['Size'] = pd.cut(df['decile'], [0, 3.5, 6.5, 9.5, 11],
                        labels=['large', 'medium', 'small', 'tiny'])
    df['Exchange'] = pd.cut(df['exchcd'], [0, 2.5, 4],
                            labels=['NYSE','NASDAQ'])
    df = df.drop(columns=['date', 'permno', 'decile', 'exchcd', 'siccd'])\
           .dropna().groupby(['Size', 'Exchange']).median().T
    fig, ax = plt.subplots(1, 1, num=num+1, clear=True, figsize=(8,5))
    plot_time(df.iloc[1:], title='Median '+key.capitalize(), ax=ax, 
              fontsize=14, loc='upper center', legend1=None)
    ax.legend([a + '\n' + b for a,b in df.columns], 
              loc='upper left', bbox_to_anchor=(1.0, 1.0), 
              fontsize=12)
    plt.subplots_adjust(right=0.8)
    plt.savefig(os.path.join(logdir, key + '.jpg'))
plt.show()

