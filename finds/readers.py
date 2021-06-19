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
try:
    from settings import ECHO
except:
    ECHO = False

_h = {'User-Agent':
      'Mozilla/5.0 (X11; Linux i686; rv:64.0) Gecko/20100101 Firefox/64.0'}
_h = {"Connection": "keep-alive",
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit"
      "/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"}

def requests_get(url, params=None, retry=3, sleep=2, timeout=3, trap=False,
                 delay=0, headers=_h, echo=ECHO):
    """Wrapper over requests.get, with retry loops and delays

    Parameters
    ----------
    url : str
        URL address to request
    params : dict of {key: value} (optional), default is None
        Payload of &key=value to append to url
    headers : dict (optional)
        e.g. User-Agent, Connection and other headers parameters
    timeout : int (optional), default is 3
        Number of seconds before timing out one request try
    retry : int (optional), default is 5
        Number of times to retry request
    sleep : int (optional), default is 2
        Number of seconds to wait between retries
    trap : bool (optional), default is True
        On timed-out after retries: if True raise exception, else return False
    delay : int (optional), default is 0
        Number of seconds to initially wait
    echo : bool (optional), default is True
        whether to display verbose messages to aid debugging

    Returns
    -------
    r : requests.Response object, or None
        None if timed-out or status_code != 200
    """
    if echo:
        print(url)
    for _ in range(retry):
        try:
            r = requests.get(url, headers=headers,timeout=timeout,params=params)
            assert(r.status_code >= 200 and r.status_code <= 404)
            break
        except Exception as e:
            time.sleep(sleep)
            if echo: print(e, r.status_code)
            r = None
    if r is None:  # likely timed-out after retries:
        if trap:     # raise exception if trap, else silently return None
            raise Exception(f"requests_get: {url} {time.time()}")
        return None
    if r.status_code != 200:
        if echo: print(r.status_code, r.content)
        return None
    return r


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

if False:  # retrieve
    from settings import settings
    from finds.database import SQL
    from finds.busday import BusDay
    from finds.structured import benchmarks
    sql = SQL(**settings['sql'], echo=ECHO)
    bd = BusDay(sql)
    bench = Benchmarks(sql, bd)
    
    datasets = fetch_FamaFrench()
    print("\n".join(f"[{i}] {d}" for i, d in enumerate(datasets)))
    for name, item, suffix in datasets:
        df = fetch_FamaFrench(name=name, item=item, suffix=suffix,
                              date_formatter=bd.offset)
        for col in df.columns:
            bench.load_series(df[col], name=name, item=item)
    print(DataFrame(**sql.run('select * from ' + bench['ident'].key)))
    
