"""Stock prices and adjustments

- Total stock returns: splits and dividend adjustment factors

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

import os
import glob
import time
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from finds.database import SQL, Redis
from finds.busday import BusDay
from finds.structured import CRSP, Finder
from finds.recipes import fractiles
from yahoo import get_price
from conf import credentials, VERBOSE, paths
from finds import display

VERBOSE = 1

sql = SQL(**credentials['sql'], verbose=VERBOSE)
rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
imgdir = paths['images']
find = Finder(sql)    # to search identifier lookup tables

# Show dividend and split adjustments for AAPL

# Retrieve price history from yahoo finance back to 1980
ticker = 'AAPL'
yahoo_df = get_price(ticker, start_date='19800101')

# Get price and distributions history from CRSP
found = find(ticker)      # locate names records by ticker
where = f" where permno = {found['permno'].iloc[-1]}"
dist = sql.read_dataframe(f"select * from {crsp['dist'].key} {where}")
crsp_df = sql.read_dataframe(f"select * from {crsp['daily'].key} {where}")\
             .set_index('date', inplace=False)

# Merge yahoo and CRSP
df = yahoo_df[['close', 'adjClose']]\
    .join(crsp_df[['prc', 'ret', 'retx', 'vol']]\
          .join(dist[dist['facpr'] != 0.0].set_index('exdt')['facpr'])\
          .join(dist[dist['divamt'] != 0.0].set_index('exdt')['divamt']),
    how='inner').drop(columns=['vol'])

# Select distribution dates where divamt and facpr are not NA
dist_dates = df.index[:1].append(df.index[~df['divamt'].isna()
                                          | ~df['facpr'].isna()])

# Display cumulative adjustments
print(f"{ticker}: {min(df.index)}-{max(df.index)}")
print('Cumulative dividend yield',
      ((1+(df['divamt'].fillna(0)\
                .div(df['prc'].add(df['divamt'].fillna(0))))).prod() - 1)\
      .round(4))
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
          legend2=['compounded dividend return'], loc2='lower right',
          fontsize=8, rotation=90, ax=ax,
          title=ticker + ': total returns, dividends and stock splits')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'splits' + figext))
plt.show()

# Display total adjustments

# cumulate (1+facpr) as factor to adjust past prices
# adjustment is made to pre-split price on or prior to ex-date
facpr = (1+df['facpr'].shift(-1)).sort_index(ascending=False)\
                                 .cumprod()\
                                 .fillna(method='ffill').fillna(1)

# split-adjust CRSP price by cumulative factor
df['price'] = df['prc'].abs().div(facpr)
                                                          
# adjust price for income return (= total return - capital appreciation)
adjret = (1+df['retx']).cumprod().div(((1+df['ret'])).cumprod())
df['adjprice'] = df['price'].div(adjret) * adjret.iloc[-1] 

show(df.loc[dist_dates].head(5),
     caption=f"{ticker}: Stock splits and dividends")
show(df.loc[dist_dates[(dist_dates > 19950816) & (dist_dates < 20130207)]])
show(df.loc[dist_dates].tail(10))

