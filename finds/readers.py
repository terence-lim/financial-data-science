"""Perform web requests and retrievals

- pandas_datareader, Fama-French data library
- Loughran McDonald financial word lists, FOMC minutes, Liu and Wu yield curve

Author: Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pandas_datareader as pdr
import wget
import os
import re
import time
import requests
from pandas.api.types import is_list_like, is_datetime64_any_dtype
from pandas.api.types import is_integer_dtype, is_string_dtype, is_numeric_dtype
from pandas.api import types
from bs4 import BeautifulSoup
import config


def fetch_lm(source, label=None):
    """Retrieve and parse LoughranMcDonald dictionary and stopwords

    Parameters
    ----------
    source : str
        URL or full pathname of csv file for LoughranMcDonald_MasterDictionary
    label : str, optional.  Default is None
        if source is a stopword file, specifies associated type label

    Returns
    -------
    results : dict of {key: list of strings}
        word lists keyed by associated label, e.g. 'positive', 'generic' etc

    Notes
    -----
    https://sraf.nd.edu/textual-analysis/resources/
    """
    results = dict()
    if label is None:
        lm = pd.read_csv(source)        # main csv file
        lm['Word'] = lm['Word'].str.lower()  # all text in lower case
        lm.columns = lm.columns.str.lower()  # column names to lower
        for s in ['negative', 'positive', 'uncertainty', 'litigious', 
                  'constraining', 'superfluous', 'interesting', 'irr_verb']:
            results[s] = lm['word'][lm[s].ne(0)].tolist()
        results['strong'] = lm['word'][lm['modal'].eq(1)].tolist()
        results['moderate'] = lm['word'][lm['modal'].eq(2)].tolist()
        results['weak'] = lm['word'][lm['modal'].eq(3)].tolist()
    else:
        words = pd.read_csv(source, sep='|', encoding='latin_1')
        results[label] = words.iloc[:,0].str.lower().str.rstrip().to_list()
    return results


def fetch_fomc(url=None):
    """Retrieve FOMC minutes or catalog from Fed website

    Parameter
    ---------
    url : str, optional
        url of webpage (default is None to return dict of historical calendar)

    Returns
    -------
    result : str or dict
        text of minutes at input url, or dict of all {date: url} from Fed site
    """

    if url is not None:  # Retrieve an FOMC minutes document from input {url}
        raw = BeautifulSoup(requests.get(url).content, 'html.parser')
        minutes = "\n\n".join([p.get_text().strip() for p in raw.findAll('p')])
        return re.sub('\n+','\n', re.sub('[\r\t]',' ', minutes))

    url = 'https://www.federalreserve.gov/'  # Else retrieve catalog from site
    dateOf = lambda s: int(re.sub('\D', '', s)[-8:]) 
        
    # latest five years' minutes can be linked from a main page
    new_url = url + 'monetarypolicy/fomccalendars.htm'
    raw = BeautifulSoup(requests.get(new_url).content, 'html.parser')
    hrefs = raw.find_all('a', href=re.compile('\S+minutes\S+.htm$', re.I))
    links = [url + m.attrs['href'] for m in hrefs]

    # earlier years' minutes are linked from annual pages with this format
    old_url = url + 'monetarypolicy/fomchistorical%d.htm'
    for year in range(1993, min([dateOf(m) for m in links]) // 10000):
        raw = BeautifulSoup(requests.get(old_url % year).content, 'html.parser')
        hrefs = raw.find_all('a', href=re.compile('\S+minutes\S+.htm$', re.I))
        links += [url + m.attrs['href'].replace(url,'') for m in hrefs]
    return {dateOf(link) : link for link in links}


def fetch_FamaFrench(name=None, item=0, suffix='',
                     start=19260101, end=20271231, index_formatter=None):
    """Wrapper over pandas_datareader to retrieve FamaFrench research factors"""
    # benchmark returns from Ken French site:
    # [name, index (0 is usually value-weighted), suffix (to differentiate)]
    _sources = [
        ('F-F_Research_Data_5_Factors_2x3_daily', 0, ''),
        ('F-F_Research_Data_5_Factors_2x3', 0, '(mo)'),
        ('F-F_Research_Data_Factors_daily', 0, ''),
        ('F-F_Research_Data_Factors', 0, '(mo)'),   # "(mo)" for monthly
        ('F-F_Momentum_Factor_daily', 0, ''),
        ('F-F_Momentum_Factor', 0, '(mo)'),
        ('F-F_LT_Reversal_Factor_daily', 0, ''),
        ('F-F_LT_Reversal_Factor', 0, '(mo)'),
        ('F-F_ST_Reversal_Factor_daily', 0, ''),
        ('F-F_ST_Reversal_Factor', 0, '(mo)'),
        ('49_Industry_Portfolios_daily', 0, '49vw'), # append suffix
        ('48_Industry_Portfolios_daily', 0, '48vw'), #  to differentiate
        ('49_Industry_Portfolios_daily', 1, '49ew'), #  value-weighted vs
        ('48_Industry_Portfolios_daily', 1, '48ew')] #  equal-weighted

    if name is None:
        return _sources
    if isinstance(name, int):
        name, item, suffix = _sources[name]
    if isinstance(start, int):
        start=pd.to_datetime(start, format='%Y%m%d')
    if isinstance(end, int):
        end=pd.to_datetime(end, format='%Y%m%d')
    df = pdr.data.DataReader(name=name, data_source='famafrench',
                             start=start, end=end)[item]
    try:
        df.index = df.index.to_timestamp()   # else invalid comparison error!
    except:
        pass
    df = df[(df.index >= start) & (df.index <= end)]
    if index_formatter:
        df.index = [index_formatter(d) for d in df.index]
    df.columns = [c.rstrip() + suffix for c in df.columns]
    df.where(df > -99.99, other=np.nan, inplace=True)
    df = df / 100
    return df

from finds.busday import to_monthend
def fetch_lw(file_id='1_u9cRxmOSiwp_tFvlaORuhS-zwl935s0'):
    """Retrieve reconstructed yield curve history: Liu and Wu (2020)
    https://sites.google.com/view/jingcynthiawu/yield-data
    """
    src = ("https://drive.google.com/uc?export=download&id={}".format(file_id)
           if '.' not in file_id else file_id)
    x = pd.ExcelFile(src)
    df = x.parse()
    dates = np.where(df.iloc[:, 0].astype(str).str[0].str.isdigit())[0]
    return DataFrame(np.exp(df.iloc[dates,1:361].astype(float).values/100) - 1,
                     index=to_monthend(df.iloc[dates, 0].values),
                     columns=np.arange(1, 361))

if __name__ == "__main__":  # hack to splice in latest yahoo finance prices
    import os
    import glob
    import pandas as pd
    import numpy as np
    from pandas import DataFrame, Series, to_datetime
    import pandas_datareader as pdr
    import pandas_datareader.data as web
    from datetime import datetime
    from finds.busday import BusDay    
    from finds.database import SQL
    from finds.structured import CRSP, as_dtypes
    from yahoo import get_price
    import config

    DEBUG = True
    START_DATE = 20201201
    today = datetime.today().strftime('%Y%m%d')    
    crsp_date = config.CRSP_DATE
    downloads = config.localpath['yahoo']
    symbols = pdr.nasdaq_trader.get_nasdaq_symbols()
    print(today, crsp_date, downloads, len(symbols), DEBUG)
    
    prices = {}
    dividends = {}
    ignored = []
    for i, symbol in enumerate(symbols.index):
        try:
            assert('$' not in symbol)
            assert('^' not in symbol)
            assert(not symbols.loc[symbol, 'Test Issue'])
            ticker = symbol.replace('.','')
            df = get_price(ticker=ticker, start_date=str(START_DATE))
            assert(len(df) > 0)
        except:
            print(symbol, ' ignored')
            ignored += [symbol]
            continue

        # guess dividend dates (ascending order)
        diff = (df.close / df.adjClose).rolling(15).min()
        dates = []
        threshold = 0.02 / max(df['adjClose'])
        while True:
            idx = df.index[abs(diff - 1) >= threshold]
            if not len(idx):
                break
            idx = max(idx)
            dates += [idx]
            diff[diff.index <= idx] /= diff[idx]
        dates = sorted(dates, reverse=True)
        print(i, ticker, len(df), dates)
        
        # estimate mean dividend rates of each period
        diff = (df.close / df.adjClose) 
        div = DataFrame(data=[diff[(diff.index <= a) & (diff.index > b)].mean()
                              for a,b in zip(dates, dates[1:] + [0])],
                        index = dates,
                        columns = ['ret']).reindex(df.index).shift().dropna()
        div['ret'] /= div['ret'].shift(-1, fill_value=1)
        if np.sum(df.adjClose < 0):
            print('**** negative adjClose', ticker)
            df['ret'] = df['retx']
        else:
            df.loc[div.index, 'ret'] = df.loc[div.index, 'retx'] + div['ret']-1
            df['ret'].where(df['ret'].notnull(), df['retx'], inplace = True)
            
            # finalize (cash) dividend and price dataframes
            div = (div['ret'] - 1) * df.loc[div.index, 'close']
            div = DataFrame({'div': div, 'date': div.index,
                             'ticker':ticker}).reset_index(drop=True)
            dividends[ticker] = div
        
        df = df.reset_index().rename(columns={'index':'date'})
        df['ticker'] = ticker
        prices[ticker] = df

    outdir = os.path.join(downloads, today)
    os.makedirs(outdir, exist_ok=True)
    pd.concat(prices.values(), ignore_index=True)\
      .to_csv(os.path.join(outdir, 'prices.csv.gz'), sep='|', index=False)
    pd.concat(dividends.values(), ignore_index=True)\
      .to_csv(os.path.join(outdir, 'dividends.csv.gz'), sep='|', index=False)
    symbols.to_csv(os.path.join(outdir, 'symbols.csv.gz'), sep='|', index=False)

    paths = sorted(glob.glob(os.path.join(downloads, '2*')), reverse=True)
    prices = pd.read_csv(os.path.join(paths[0], 'prices.csv.gz'), sep='|')
    dividends = pd.read_csv(os.path.join(paths[0], 'dividends.csv.gz'), sep='|')

    for pathname in paths[1:]:
        df = pd.read_csv(os.path.join(pathname, 'prices.csv.gz'), sep='|')
        new = set(np.unique(df['ticker'])).difference(
            set(np.unique(prices['ticker'])))
        df = df[df['ticker'].isin(new)]
        prices = prices.append(df, sort=False)
        print(pathname, 'added prices', new)
        
        df = pd.read_csv(os.path.join(pathname, 'dividends.csv.gz'), sep='|')
        new = set(np.unique(df['ticker'])).difference(
            set(np.unique(dividends['ticker'])))
        df = df[df['ticker'].isin(new)]
        dividends = dividends.append(df, sort=False)
        print(pathname, 'added dividends', new)

    sql = SQL(**config.credentials['sql'], echo=config.ECHO)
    bd = BusDay(sql)
    crsp = CRSP(sql, bd, rdb=None)
    date = bd.offset(crsp_date)
        
    # get price and shrout as of last date
    price = crsp.get_section('daily', ['prc','shrout'], 'date', date, start=None)

    # get tickers to lookup permno
    tickers = crsp.get_section('names', ['tsymbol', 'date'], 'date', date,
                               start=0).reindex(price.index)
    tickers = tickers.sort_values(['tsymbol', 'date'])\
                     .drop_duplicates(keep='last')
    tickers = tickers.reset_index().set_index('tsymbol')['permno']

    # merge permnos into big prices table
    prices = prices.join(tickers, on='ticker', how='inner')

    # estimate facpr, for internal consistency, based on year-end price ratio
    facpr = prices[prices.date == date][['permno', 'close']]\
            .join(price, on='permno', how='inner')
    facpr['facpr'] = np.round((facpr['prc'].abs() / facpr['close']) - 1, 3)
    facpr = facpr[facpr['facpr'].abs().gt(0.005)
                  & (facpr['close']-facpr['prc'].abs()).abs().ge(0.009)
                  & (facpr['prc'].gt(0) |
                     (facpr['close'] - facpr['prc'].abs()).abs().gt(0.25))]

    # adjust shrout if necessary, then merge into big df of prices
    facpr['shrout'] *= (1 + facpr['facpr'])
    facpr = facpr.set_index('permno')
    price.loc[facpr.index, 'shrout'] = facpr.loc[facpr.index, 'shrout']
    prices = prices.join(price[['shrout']], on='permno', how='inner')
    
    # create monthly data df
    prices['month'] = prices['date'] // 100
    groupby = prices.groupby(['permno','month'])
    monthly = (groupby[['ret', 'retx']].prod() - 1)\
              .join(groupby['close'].last()).reset_index()
    monthly['month'] = bd.endmo(monthly['month'])
    monthly.rename(columns={'month': 'date', 'close':'prc'}, inplace=True)
    monthly = monthly[monthly['date'] > date]
    print('monthly:', len(monthly), len(np.unique(monthly['permno'])))
    
    if DEBUG:
        raise exception('debug mode: Ready to load sql?')

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
    print(prices.date.value_counts())

    # clean up facpr table and merge into dist dataframe
    dist = as_dtypes(None, {k : v.type for k,v in crsp['dist'].columns.items()})
    facpr = facpr.reset_index().drop(columns=['close','prc','shrout'])
    facpr['exdt'] = int(bd.offset(date, 1))
    facpr['distcd'] = 5523
    dist = dist.append(facpr, sort=False)
    print(np.isnan(dist).mean(axis=0))

    # clean up dividends table and merge into dist dataframe
    dividends = dividends[dividends['date'] > date]\
                .join(tickers, on='ticker', how='inner')
    dividends = dividends.rename(columns={'div':'divamt', 'date':'exdt'})\
                         .drop(columns=['ticker'])
    dividends['distcd'] = 1232
    dist = dist.append(dividends, sort=False)   # per shape $ dividends
    print(np.isnan(dist).mean(axis=0))

    # save current intermediate files
    tickers.to_csv(os.path.join(downloads, 'tickers.csv.gz'), sep='|')
    dividends.to_csv(os.path.join(downloads, 'dividends.csv.gz'), sep='|')
    facpr.to_csv(os.path.join(downloads, 'facpr.csv.gz'), sep='|')
    prices.to_csv(os.path.join(downloads, 'daily.csv.gz'), sep='|')

    # update monthly sql table
    print('monthly', sql.run('select count(*) from monthly'))
    table = crsp['monthly']
    delete = table.delete().where(table.c['date'] > date)
    sql.run(delete)
    print('monthly', sql.run('select count(*) from monthly'))
    sql.load_dataframe(table.key, monthly, index_label=None, to_sql=True,
                       if_exists='append')
    print('monthly', sql.run('select count(*) from monthly'))

    # update dist table
    print(len(dist), len(np.unique(dist.permno)), dist['exdt'].min(),
          dist['exdt'].max(), np.unique(dist.distcd))
    dist[dist.isna()] = 0
    dist = as_dtypes(dist, {k : v.type for k,v in crsp['dist'].columns.items()})
    dist.to_csv(os.path.join(downloads, 'dist.csv.gz'), sep='|', index=False)

    print('dist', sql.run('select count(*) from dist'))
    table = crsp['dist']
    delete = table.delete().where(table.c['exdt'] > date)
    sql.run(delete)
    sql.load_dataframe(table.key, dist, index_label=None, to_sql=True,
                       if_exists='append')
    print('dist', sql.run('select count(*) from dist'))

    # update daily table
    print('daily', sql.run('select count(*) from daily'))
    table = crsp['daily']
    delete = table.delete().where(table.c['date'] > date)
    sql.run(delete)
    sql.load_dataframe(table.key, prices, index_label=None, to_sql=True,
                       if_exists='append')
    print(sql.run(f"select count(*) from daily where date > {date}"))
    print(sql.run('select count(*) from daily'))

"""
./redis-cli --scan --pattern '*CRSP_2021*' | xargs ./redis-cli del
"""

