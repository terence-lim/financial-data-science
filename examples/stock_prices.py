"""Stock prices, dividends, split-adjustments and identifiers

Copyright 2023, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from finds.database import SQL, Redis
from finds.busday import BusDay
from finds.structured import CRSP, Finder
from finds.recipes import fractiles
from finds.display import show, plot_date
from yahoo import get_price
from conf import credentials, VERBOSE, paths

%matplotlib qt
VERBOSE = 0      # 1
SHOW = dict(ndigits=4, latex=True)  # None

sql = SQL(**credentials['sql'], verbose=VERBOSE)
rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
imgdir = paths['images']
find = Finder(sql)    # to search identifier lookup tables
ticker = 'AAPL'
find('AAPL')

## Show dividend and split adjustments for AAPL

### Retrieve price history from yahoo finance back to 1980
yahoo_df = get_price(ticker, start_date='19800101', verbose=VERBOSE)
yahoo_df.rename_axis(ticker)

### Get price and distributions history from CRSP
found = find(ticker)      # locate names records by ticker
where = f" where permno = {found['permno'].iloc[-1]}"
dist = sql.read_dataframe(f"select * from {crsp['dist'].key} {where}")
crsp_df = sql.read_dataframe(f"select * from {crsp['daily'].key} {where}")\
             .set_index('date', inplace=False)
crsp_df

### Merge yahoo and CRSP
df = yahoo_df[['close', 'adjClose']]\
    .join(crsp_df[['prc', 'ret', 'retx', 'vol']]\
          .join(dist[dist['facpr'] != 0.0].set_index('exdt')['facpr'])\
          .join(dist[dist['divamt'] != 0.0].set_index('exdt')['divamt']),
          how='inner')\
    .drop(columns=['vol'])

### Select distribution dates where divamt and facpr are not NA
dist_dates = df.index[:1].append(df.index[~df['divamt'].isna()
                                          | ~df['facpr'].isna()])

print(f"{ticker}: {min(df.index)}-{max(df.index)}")
print('Cumulative dividend yield',
      ((1+(df['divamt']\
           .fillna(0)\
           .div(df['prc'].add(df['divamt'].fillna(0))))).prod() - 1).round(4))
print('Cumulative stock split-adjustment',
(df['facpr'].fillna(0) + 1).prod())
print('Cumulative total stock return multiple',
      ((df['ret'].fillna(0) + 1).prod()).round(1))

fig, ax = plt.subplots(num=1, clear=True, figsize=(10, 5))
plot_date(np.log10((df[['ret']].fillna(0) + 1).cumprod()
                   .join((df[['facpr']].fillna(0)+1).cumprod())),
          (1+(df['divamt'].fillna(0)\
              .div(df['prc'].abs().add(df['divamt'].fillna(0))))).cumprod()-1,
          legend1=['log10 total return', 'log10 cumulative split-adjustment'],
          legend2=['compounded dividend return'],
          loc2='lower right',
          fontsize=10,
          ax=ax,
          title=ticker + ': total returns, dividends and stock splits')
plt.tight_layout()
plt.savefig(imgdir / 'splits.jpg')


## Reconcile CRSP and Yahoo adjustments
"""
CRSP 
- prc: market closing price (negative if closing midquote)
- ret: total daily holding return
- retx: daily capital appreciation return
- facpr: factor to split-adjust price prior to ex-date
- divamt: dividend amount on ex-date

YAHOO
- close: split-adjusted price
- adjClose: split-adjust price, capital appreciation only (exclude dividend)

reconcile Yahoo and CRSP
- price: CRSP price adjusted by cumulative split factor
- adjprice: CRSP split-adjusted price excluding dividends
"""

# Cumulate factor to adjust pre-split prices prior to ex-date
facpr = (1+df['facpr'].shift(-1)).sort_index(ascending=False)\
                                 .cumprod()\
                                 .fillna(method='ffill').fillna(1)

# Split-adjust CRSP market price by cumulative factor
df['price'] = df['prc'].abs().div(facpr)
                                                          
# Reduce CRSP split-adjusted price by dividend income
adjret = (1+df['retx']).cumprod().div(((1+df['ret'])).cumprod())
df['adjprice'] = df['price'].div(adjret) * adjret.iloc[-1] 

# display first five distribution events
show(df.loc[dist_dates].head(5), caption=ticker, **SHOW)

# display 1995-2013 distribution events
show(df.loc[dist_dates[(dist_dates>19950816) & (dist_dates<20130207)]], **SHOW)

# display last five distribution events
show(df.loc[dist_dates].tail(10), **SHOW)

