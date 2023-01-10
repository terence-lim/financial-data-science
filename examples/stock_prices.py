"""Stock prices, dividends, splits and identifiers

- total returns, split-adjustments, identifier changes, share types
- SQL and Pandas

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
VERBOSE = 1      # 1
SHOW = dict(ndigits=4, latex=False) #True)  # None)

sql = SQL(**credentials['sql'], verbose=VERBOSE)
rdb = Redis(**credentials['redis'])
bd = BusDay(sql, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb=rdb, verbose=VERBOSE)
imgdir = paths['images']

## Stock dividends and splits
ticker = 'AAPL'
find = Finder(sql)    # to search identifier lookup tables
find('AAPL')

### Get price and distributions history from CRSP
found = find(ticker)      # locate names records by ticker
where = f" where permno = {found['permno'].iloc[-1]}"
dist = sql.read_dataframe(f"select * from {crsp['dist'].key} {where}")
crsp_df = sql.read_dataframe(f"select * from {crsp['daily'].key} {where}")\
             .set_index('date', inplace=False)
crsp_df

### Retrieve price history from yahoo finance back to 1980
yahoo_df = get_price(ticker, start_date='19800101', verbose=VERBOSE)
yahoo_df.rename_axis(ticker)

### Merge yahoo and CRSP by date
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
          (1+(df['divamt']\
              .fillna(0)\
              .div(df['prc']\
                   .abs()\
                   .add(df['divamt']\
                        .fillna(0))))).cumprod() - 1,
          legend1=['log10 total return', 'log10 cumulative split-adjustment'],
          legend2=['compounded dividend return'],
          loc2='lower right',
          fontsize=10,
          ax=ax,
          title=ticker + ': total returns, dividends and stock splits')
plt.tight_layout()
plt.savefig(imgdir / 'splits.jpg')


## Reconcile split-adjusted Yahoo and CRSP prices
"""
YAHOO
- close: split-adjusted price
- adjClose: split-adjust price, capital appreciation only (exclude dividend)

CRSP 
- prc: market closing price (negative if closing midquote)
- ret: total daily holding return
- retx: daily capital appreciation return
- facpr: factor to adjust price prior to ex-date
  - stocks splits and dividends: facpr = Shares(exdt) / Shares(exdt-1) - 1
  - reverse splits: -1 < facpr < 0
  - mergers and exchanges: facpr = -1, by convention
  - cash dividends and payments: facpr = 0
- divamt: dividend amount on ex-date

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

#show(df.loc[dist_dates].head(5), caption=ticker, **SHOW)
#show(df.loc[dist_dates[(dist_dates>19950816) & (dist_dates<20130207)]], **SHOW)
show(df.loc[dist_dates].tail(10), **SHOW)

## SQL: share type codes and name substring
"""
- all "INDEX" fund ETF's (share code 73)
- select from, max, min, where, like, group by, order by
"""
funds = DataFrame(**sql.run("SELECT permno, MAX(comnam) as comnam, "
                            "    MAX(ncusip) as ncusip, MAX(ticker) as ticker, "
                            "    MIN(date) as date, MAX(nameendt) as nameendt, "
                            "    MAX(permco) as permco "
                            "  FROM names "
                            "  WHERE shrcd in (73) "
                            "    AND comnam LIKE '%% INDEX %%'"
                            "  GROUP BY permno"
                            "  ORDER BY permco"))


## SQL and Pandas: find currently-active stocks with most ticker changes
counts = DataFrame(**sql.run('SELECT permno, '
                             '  MAX(nameendt) as enddate, '
                             '  COUNT(DISTINCT ticker) as counts '
                             '  FROM names '
                             '  WHERE shrcd in (10, 11) '
                             '    AND exchcd in (1, 2, 3) '
                             '  GROUP BY permno'
                             '  ORDER BY counts, enddate'))
# require to still be listed at latest date
counts = counts[counts['enddate'].eq(counts['enddate'].max())]

# select permnos with most number of ticker changes
permnos = counts[counts['counts'].eq(counts['counts'].max())]['permno'].to_list()
print(permnos)

# show all record fields for selected permnos
df = DataFrame(**sql.run(f"SELECT * from names " +
                         f"WHERE permno IN ({permnos[-1]})"))\
                         .sort_values(['permno', 'date'])
show(df, **SHOW)

## SQL and Pandas: missing and average delisting returns, by delisting code
"""
- join on, subqueries
"""
# usual filter by share (US stocks) and exchange (NYSE, Amex, Nasdaq) codes
DataFrame(**sql.run("SELECT t1.* FROM names t1 INNER JOIN "
                    "  (SELECT permno, MAX(date) as date "
                    "  FROM names GROUP BY permno) t2 "
                    "  ON t1.permno = t2.permno AND t1.date = t2.date"
                    "    WHERE shrcd in (10, 11) AND exchcd in (1, 2, 3)"))

### Last delisting after 1962, by permno
DataFrame(**sql.run("SELECT t1.* FROM delist t1 INNER JOIN "
                    "  (SELECT permno, MAX(dlstdt) as dlstdt "
                    "    FROM delist GROUP BY permno) t2 "
                    "  ON t1.permno = t2.permno AND t1.dlstdt = t2.dlstdt "
                    "    WHERE t2.dlstdt > 19621231"))

### Inner join of sub-quries
df = DataFrame(**sql.run("SELECT u1.permno, u2.dlstcd, u2.dlret FROM"
                        "  (SELECT t1.* FROM names t1 INNER JOIN "
                        "    (SELECT permno, MAX(date) as date FROM names"
                        "        GROUP BY permno) t2 "
                        "      ON t1.permno = t2.permno "
                        "        AND t1.date = t2.date "
                        "      WHERE shrcd in (10, 11) "
                        "        AND exchcd in (1, 2, 3)) u1"
                        "  INNER JOIN"
                        "  (SELECT t1.* FROM delist t1 INNER JOIN "
                        "    (SELECT permno, MAX(dlstdt) as dlstdt "
                        "      FROM delist GROUP BY permno) t2 "
                        "    ON t1.permno = t2.permno "
                        "      AND t1.dlstdt = t2.dlstdt "
                        "    WHERE t2.dlstdt > 19621231) u2"
                        "  ON u1.permno = u2.permno"))
# Pandas: summarize by delisting code
df.groupby('dlstcd').agg({'dlret': ['count', 'size', 'mean']})\
                    .assign(frac_miss=lambda x: ((x.iloc[:,1] - x.iloc[:,0])
                                                 / x.iloc[:,1]))\
                    .round(4)
