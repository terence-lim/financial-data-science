"""Convenience class and methods to access ALFRED/FRED apis and FRED-MD/FRED-QD

- FRED, ALFRED, revisions vintages
- PCA, approximate factor model, EM algorithm

Author: Terence Lim
License: MIT
"""
import os
import sys
import json
import io
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.tseries.offsets import MonthEnd, YearEnd, QuarterEnd
from datetime import datetime, date
import requests
from bs4 import BeautifulSoup
from io import StringIO
import pickle
import zipfile
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pandas.api import types
import time
from .edgar import requests_get
from .busday import str2date, to_monthend
import config

# From https://research.stlouisfed.org/econ/mccracken/fred-databases/
_fred_md_url = 'https://files.stlouisfed.org/files/htdocs/fred-md/'

def _print(*args, echo=config.ECHO, **kwargs):
    """helper to echo debugging messages"""
    if echo: print(*args, **kwargs)

def _int2date(date):
    """helper method to convert int date to FRED api string format"""
    return ([_int2date(d) for d in date] if types.is_list_like(date) 
            else "-".join(str(date)[a:b] for a, b in [[0,4], [4,6], [6,8]]))

def _date2int(date):
    """helper method to convert FRED api string format to int date"""
    return ([_date2int(d) for d in date] if types.is_list_like(date)
            else int(re.sub('\D', '', str(date)[:10])))

def multpl(page):
    """Helper method to retrieve shiller series by parsing multpl.com web page"""
    url = f"https://www.multpl.com/{page}/table/by-month"
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    tables = soup.findChildren('table')
    df = pd.read_html(tables[0].decode())[0]
    df.iloc[:,0] = str2date(df.iloc[:,0], '%b %d, %Y', '%Y%m%d')
    df['date'] = to_monthend(df.iloc[:, 0])
    df = df.sort_values('Date').groupby('date').last().iloc[:,-1]
    if not types.is_numeric_dtype(df):
        df = df.map(lambda x: re.sub('[^\d\.\-]','',x)).astype(float)
    return df


def fred_md(vintage=0, url=None, echo=config.ECHO):
    """Retrieve and parse current or vintage csv from McCracken FRED-MD site

    Parameters
    ----------
    vintage : str or int, default 0 (for current.csv)
        file name relative to base url or zipfile archive, or int date YYYYMM
    url : str, default is None
        base name of url, local file path or zipfile archive

    Returns
    -------
    df : DataFrame
        indexed by end-of-month date

    Notes
    -----
    if vintage is int: then derive vintage csv file name from input date YYYYMM
    if url is None: then derive subfolder or zip archive name, from vintage

    Examples
    --------
    md_df, mt = fredmd(csvfile='Historical FRED-MD Vintages Final/2013-12.csv',
                       url=md_url + 'Historical_FRED-MD.zip') # pre-2015
    md_df, mt = fredmd(csvfile='monthly/2015-05.csv',
                       url=md_url + 'FRED_MD.zip')            # post-2015
    """
    url_ = _fred_md_url
    if isinstance(vintage, int) and vintage:
        csvfile_ = f"{vintage // 100}-{vintage % 100:02d}.csv"
        if vintage < 201500:
            url_ = url_ + 'Historical_FRED-MD.zip'
            csvfile_ = 'Historical FRED-MD Vintages Final/' + csvfile_
        else:
            csvfile_ = 'monthly/' + csvfile_
        vintage = csvfile_
    else:
        vintage = vintage or 'monthly/current.csv'
    _print(vintage, echo=echo)
    url = url or url_
    if url.endswith('.zip'):
        if url.startswith('http'):
            url = io.BytesIO(requests.get(url).content)
        with zipfile.ZipFile(url).open(vintage) as f:
            df = pd.read_csv(f, header=0)
    else:
        df = pd.read_csv(os.path.join(url, vintage), header=0)
    df.columns = df.columns.str.rstrip('x')
    meta = dict()
    for _, row in df.iloc[:5].iterrows():
        if '/' not in row[0]:    # this row has metadata, e.g. transform codes
            label = re.sub("[^a-z]", '', row[0].lower()) # simplify label str
            meta[label] = row[1:].astype(int).to_dict()  # as dict of int codes
    df = df[df.iloc[:, 0].str.find('/') > 0]      # keep rows with valid date
    df.index = str2date(df.iloc[:, 0], '%m/%d/%Y', '%Y%m%d')
    df.index = to_monthend(df.index)
    return df.iloc[:, 1:], DataFrame(meta)

def fred_qd(vintage=0, url=None, echo=False):
    """Retrieve and parse current or vintage csv from McCracken FRED-MD site

    Parameters
    ----------
    vintage : str or int, default 0 (i.e. current.csv)
        file name relative to base url or zipfile archive, or int date YYYYMM
    url : str, default is None
        base name of url, local file path or zipfile archive

    Returns
    -------
    df : DataFrame
        indexed by end-of-month date

    Notes
    -----
    if csvfile is int: then derive vintage csv file name from input date YYYYMM
    if url is None: then derive subfolder name from vintage 
    """
    url = url or _fred_md_url
    if isinstance(vintage, int) and vintage:
        vintage = f"quarterly/{vintage // 100}-{vintage % 100:02d}.csv"
    else:
        vintage = 'quarterly/current.csv'
    _print(vintage, echo=echo)
    df = pd.read_csv(os.path.join(url, vintage), header=0)
    df.columns = df.columns.str.rstrip('x')
    meta = dict()
    for _, row in df.iloc[:5].iterrows():
        if '/' not in row[0]:    # this row has metadata, e.g. transform codes
            label = re.sub("[^a-z]", '', row[0].lower()) # simplify label str
            meta[label] = row[1:].astype(int).to_dict()  # as dict of int codes
    df = df[df.iloc[:, 0].str.find('/') > 0]      # keep rows with valid date
    df.index = str2date(df.iloc[:, 0], '%m/%d/%Y', '%Y%m%d')
    df.index = to_monthend(df.index)
    return df.iloc[:, 1:], DataFrame(meta)
        

class Alfred:
    """Base class for Alfred/Fred access, and manipulating retrieved data series

    Parameters
    ----------
    cache_ : dict
        cached series and observations
    tcode_ : dict 
        transformation codes
    Notes
    -----
    lin = Levels (No transformation) [default]
    chg = Change x(t) - x(t-1)
    ch1 = Change from Year Ago x(t) - x(t-n_obs_per_yr)
    pch = Percent Change ((x(t)/x(t-1)) - 1) * 100
    pc1 = Percent Change from Year Ago ((x(t)/x(t-n_obs_per_yr)) - 1) * 100
    pca = Compounded Annual Rate of Change (x(t)/x(t-1))**n_obs_per_yr - 1
    cch = Continuously Compounded Rate of Change (ln(x(t)) - ln(x(t-1)))
    cca = Continuously Compounded Annual Rate of Change 
            (ln(x(t)) - ln(x(t-1)))  * n_obs_per_yr
    log = Natural Log ln(x(t))
    """
    tcode_ = {1: {'diff': 0, 'log': 0},
              2: {'diff': 1, 'log': 0},
              3: {'diff': 2, 'log': 0},
              4: {'diff': 0, 'log': 1},
              5: {'diff': 1, 'log': 1},
              6: {'diff': 2, 'log': 1},
              7: {'diff': 1, 'log': 0, 'pct_change': True},
              'lin': {'diff': 0, 'log': 0},
              'chg': {'diff': 1, 'log': 0},
              'ch1': {'diff': 0, 'log': 0, 'pct_change': True, 'periods': 12},
              'pch': {'diff': 0, 'log': 0, 'pct_change': True},
              'pc1': {'diff': 0, 'log': 0, 'pct_change': True, 'periods': 12},
              'pca': {'diff': 1, 'log': 1, 'annualize': 12},
              'cch': {'diff': 1, 'log': 1},
              'cca': {'diff': 1, 'log': 1, 'annualize': 12},
              'lin': {'diff': 0, 'log': 0},
              'log': {'diff': 0, 'log': 1}}

    header_ = {
        k : {'id': k, 'title': v} for k,v in
        [['CPF3MTB3M', '3-Month Commercial Paper Minus 3-Month Treasury Bill'],
         ['CLAIMS', 'Initial Claims'],
         ['HWIURATIO', 'Ratio of Help Wanted/No. Unemployed'],
         ['HWI', 'Help Wanted Index for United States'],
         ['AMDMNO', 'New Orders for Durable Goods'],
         ['S&P 500', "S&P's Common Stock Price Index: Composite"],
         ['RETAIL', "Retail and Food Services Sales"],
         ['OILPRICE', 'Crude Oil, spliced WTI and Cushing'],
         ['COMPAPFF', "3-Month Commercial Paper Minus FEDFUNDS"],
         ['CP3M', "3-Month AA Financial Commercial Paper Rates"],
         ['CONSPI', 'Nonrevolving consumer credit to Personal Income'],
         ['S&P div yield', "S&P's Composite Common Stock: Dividend Yield"],
         ['S&P PE ratio', "S&P's Composite Common Stock: Price-Earnings Ratio"],
         ['S&P: indust', "S&P's Common Stock Price Index: Industrials"]]}

    
    @classmethod
    def transform(self, data, tcode=1, freq=None, **kwargs):
        """Classmethod to apply time series transformations

        Parameters
        ----------
        data : DataFrame
            input data
        tcode : int in {1, ..., 7}, default is 1
            transformation code 
        freq : str in {'M', 'Q', 'A'}, default is None
            set periodicity of dates
        log : int, default is 0
            number of times to take log
        diff : int, default is 0
            number of times to take difference
        pct_change : bool
            whether to apply pct_change operator
        periods : int, default is 1
            number of periods to lag for pct_change or diff operator
        annualize : int. default is 1
            annualization factor
        shift : int, default is 0
            number of rows to shift output (negative to lag)
        """
        t = {'periods':1, 'shift':0, 'pct_change':False, 'annualize':1}
        t.update(self.tcode_[tcode])
        t.update(kwargs)
        df = data.sort_index()
        if t['pct_change']:
            #df = df.pct_change(fill_method='pad')
            df = df.pct_change(fill_method=None)
            df = ((1 + df) ** t['annualize']) - 1  # by compounding
        for _ in range(t['log']):
            df = np.log(df)
        for _ in range(t['diff']):
            #df = df.fillna(method='pad').diff(periods=t['periods'])
            df = df.diff(periods=t['periods'])
            df = df * t['annualize']               # by adding
        return df.shift(t['shift'])

    alfred_api = ("https://api.stlouisfed.org/fred/{api}?series_id={series_id}"
                  "&realtime_start={start}&realtime_end={end}"
                  "&api_key={api_key}&file_type=json").format
    fred_api = ("https://api.stlouisfed.org/fred/{api}?series_id={series_id}"
                "&api_key={api_key}&file_type=json").format
    category_api = ("https://api.stlouisfed.org/fred/{api}?"
                    "category_id={category_id}&api_key={api_key}&"
                    "file_type=json{args}").format
    start = 17760704
    end = 99991231
    echo_ = config.ECHO
    api_key = None

    def header(self, series_id, column='title'):
        """Returns a column from last meta record of a series"""
        if series_id not in self.header_:
            try:
                if series_id not in self.cache_:  # load via api if not in cache
                    self.get(series_id)
                self.header_[series_id] = self[series_id]['series'].iloc[-1]
            except:
                return f"*** {series_id} ***"
        return self.header_[series_id].get(column, f"*** {series_id} ***")

    def keys(self):
        """Return id names of all loaded series data"""
        return list(self.cache_.keys())

    def values(self, columns=None):
        """Return headers (last metadata row) of all loaded series

        Parameters
        ----------
        columns: list of str, default is None
            subset of header columns to return

        Returns
        -------
        df : DataFrame
            headers of all series loaded
        """
        df = DataFrame()
        keep = ['id', 'observation_start', 'observation_end', 'frequency_short',
                'title', 'popularity', 'seasonal_adjustment_short',
                'units_short']   # default list of columns to display
        for v in self.cache_.values():
            df = df.append(v['series'].iloc[-1], ignore_index=True)
        df = df.set_index('id', drop=False)
        return df[columns or keep]

    def __init__(self, api_key, start=17760704, end=99991231, savefile=None,
                 echo=config.ECHO):
        """Create object, with api_key, for FRED access and data manipulation"""
        self.api_key = api_key
        self.start = start
        self.end = end
        self.savefile = savefile
        self.cache_ = dict()
        self.header_ = Alfred.header_.copy()
        self.echo_ = echo

    def _print(self, *args, echo=None):
        if echo or self.echo_:
            print(*args)
            
    def load(self, savefile=None):
        """Load series data to memory cache from saved file"""
        with open(savefile or self.savefile, 'rb') as f:
            self.cache_.update(**pickle.load(f))
            
        return len(self.cache_)

    def dump(self, savefile=None):
        """Save all memory-cached series data to an output file"""
        with open(savefile or self.savefile, 'wb') as f:
             pickle.dump(self.cache_, f)
        return len(self.cache_)

    def clear(self):
        self.cache_.clear()

    def pop(self, series_id):
        return self.cache_.pop(series_id, None)

    def get(self, series_id, api_key=None, start=None, end=None):
        """Retrieve metadata and full observations of a series with FRED api

        Parameters
        ----------
        series_id : str or list of str
            ids of series to retrieve

        Returns
        -------
        n : int
            length of observations dataframe
        """
        if types.is_list_like(series_id):
            return [self.get(s, start=start, end=end) for s in series_id]
        series = self.series(series_id, api_key=api_key, start=start, end=end,
                             echo=self.echo_)
        if series is None or series.empty:
            return 0
        self.cache_[series_id] = {
            'observations': self.series_observations(
                series_id, api_key=api_key, start=start, end=end,
                alfred_mode=True, echo=self.echo_),
            'series': series}
        return len(self.cache_[series_id]['observations'])

    def __call__(self, series_id, start=None, end=None, release=0,
                 vintage=99991231, label=None, realtime=False, freq=True,
                 **kwargs):
        """Select from full observations of a series and apply transforms

        Parameters
        ----------
        series_id : str or list of str
            Labels of series to retrieve
        start, end : int, default is None
            start and end period dates (inclusive) to keep
        label : str, default is None
            New label to rename returned series
        release : pd.DateOffset or int (default is 0)
            maximum release number or date offset (inclusive). If 0: latest
        vintage : int, default is None
            latest realtime_start date of observations to keep
        diff, log, pct_change : int
            number of difference, log and pct_change operations to apply
        freq : str in {'M', 'A'. 'Q', 'D', 'Y'} or bool (default is True)
            resample and replace date index with month ends at selected freqs

        Returns
        -------
        Series or DataFrame
            transformed values, name set to label if provided else series_id
        """
        if (series_id not in self.cache_ and not self.get(series_id)):
            return None
        if freq is True:
            freq = self.header(series_id, 'frequency_short')
            
        df = self.as_series(
            self[series_id]['observations'],
            release=release,
            vintage=vintage,
            start=start or self.start,
            end=end or self.end,
            freq=freq)
        if realtime:
            s = self.transform(df['value'], **kwargs).to_frame()
            s['realtime_start'] = df['realtime_start'].values
            s['realtime_end'] = df['realtime_end'].values
            return s.rename(columns={'value': label or series_id})
        return self.transform(df['value'], **kwargs).rename(label or series_id)

    def __getitem__(self, series_id):
        """Get observations and metadata for {series_id}"""
        return self.cache_.get(series_id, None)

    @classmethod
    def as_series(self, observations, release=0, vintage=99991231,
                  start=0, end=99991231, freq=None):
        """Classmethod to select a series from alfred observations set

        Parameters
        ----------
        observations: DataFrame
            from FRED 'series/observations' api call
        release : pd.DateOffset or int (default is 0)
            maximum release number or date offset (inclusive). If 0: latest
        vintage : int, default is None
            Latest realtime_start date (inclusive) allowed

        Returns
        -------
        out: Series
            value of each period date, optionally indexed by realtime_start

        Examples
        --------
        """
        df = observations.copy()
        df['value'] = pd.to_numeric(observations['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna().reset_index(drop=True)
        if freq:
            if freq.upper()[0] in ['A']:
                df['date'] += YearEnd(0)
            if freq.upper()[0] in ['S']:
                df['date'] += QuarterEnd(1)
            if freq.upper()[0] in ['Q']:
                df['date'] += QuarterEnd(0)
            if freq.upper()[0] in ['M']:
                df['date'] += MonthEnd(0)
            if freq.upper()[0] in ['B']:
                df['date'] += pd.DateOffset(days=13)
            if freq.upper()[0] in ['W']:
                df['date'] += pd.DateOffset(days=6)
        if np.any(df['realtime_start'] <= _int2date(vintage)):
            df = df[df['realtime_start'] <= _int2date(vintage)]
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.sort_values(by=['date', 'realtime_start'])

        if isinstance(release, int):  # keep latest up to max release
            df['release'] = df.groupby('date').cumcount()
            df = df[df['release'] + 1 == (release or 99999999)]\
                 .append(df.drop_duplicates('date', keep='last'))\
                 .drop_duplicates('date', keep='first')
        else:    # else latest release up through date offset
            df['release'] = (df['date'] + release).dt.strftime('%Y-%m-%d')
            df = df[df['realtime_start'] <= df['release']]\
                 .drop_duplicates('date', keep='last')
        df['date'] = df['date'].dt.strftime('%Y%m%d').astype(int)
        df['realtime_start'] = _date2int(df['realtime_start'])
        df['realtime_end'] = _date2int(df['realtime_end'])
        df = df.set_index('date').sort_index().drop(columns=['release'])
        return df[(df.index <= min(end, vintage)) & (df.index >= start)]
                 

    def series(self, series_id, api_key=None, start=None, end=None,
               echo=ECHO):
        """API wrapper to retrieve series metadata as dataframe"""
        url = self.alfred_api(api="series",
                              series_id=series_id,
                              start=_int2date(start or self.start),
                              end=_int2date(end or self.end),
                              api_key=api_key or self.api_key)
        r = requests_get(url, echo=echo)
        if r is None:
            url = self.fred_api(api="series",
                                series_id=series_id,
                                api_key=api_key or self.api_key)
            r = requests_get(url, echo=echo)
            if r is None:
                return DataFrame()
        v = json.loads(r.content)
        df = DataFrame(v['seriess'])
        df.index.name = str(datetime.now())
        return df

    def series_observations(self, series_id, api_key=None, start=None, end=None,
                            alfred_mode=False, echo=ECHO):
        """API wrapper to retrieve full observations of a series as dataframe"""
        url = self.alfred_api(api="series/observations",
                              series_id=series_id,
                              start=_int2date(start or self.start),
                              end=_int2date(end or self.end),
                              api_key=api_key or self.api_key)
        r = requests_get(url, echo=echo)
        if r is None:
            url = self.fred_api(api="series/observations",
                                series_id=series_id,
                                api_key=api_key or self.api_key)
            r = requests_get(url, echo=echo)
            if r is None:
                return DataFrame()
        contents = json.loads(r.content)
        df = DataFrame(contents['observations'])
        if alfred_mode:  # convert fred to alfred by backfilling realtime_start
            f = (df['realtime_start'].eq(contents['realtime_start']) &
                 df['realtime_end'].eq(contents['realtime_end'])).values
            df.loc[f, 'realtime_start'] = df.loc[f, 'date']
        return df

    def get_category(self, category_id, api_key=None):
        c = self.category(category_id, api="category", api_key=api_key)
        if 'categories' not in c:
            return None
        c = c['categories'][0]
        c['children'] = self.category(category_id,
                                      api="category/children",
                                      api_key=api_key).get('categories', [])
        c['series'] = []
        offset = 0
        while True:
            s = self.category(category_id,
                              api="category/series",
                              api_key=api_key,
                              offset=offset)
            if not s['seriess']:
                break
            c['series'].extend(s['seriess'])
            offset += s['limit']        
        return c
        
    def category(self, category_id, api="category", api_key=None, echo=ECHO,
                 **kwargs):
        """API wrapper to retrieve category data as dict"""
        args = "&".join([f"{k}={v}" for k,v in kwargs.items()])
        url = self.category_api(api=api,
                                category_id=category_id,
                                api_key=api_key or self.api_key,
                                args="&" + args if args else '')
        r = requests_get(url, echo=echo)
        return dict() if r is None else json.loads(r.content)


    @classmethod
    def popular(self, page=1):
        """Classmethod to web scrape popular series names, by page number"""
        assert(page > 0)
        url = f"https://fred.stlouisfed.org/tags/series?ob=pv&pageID={page}"
        data = requests.get(url).content
        soup = BeautifulSoup(data, 'lxml')
        tags = soup.findAll(name='a', attrs={'class': 'series-title'})
        details = [tag.get('href').split('/')[-1] for tag in tags]
        return details
        #tags = soup.findAll(name='input',attrs={'class':'pager-item-checkbox'})
        #details = [tag.get('value') for tag in tags]
        #return details

    fred_adjust = {'HWI': 'JTSJOL',
                   'AMDMNO': 'DGORDER',
                   'S&P 500': 'SP500',
                   'RETAIL': 'RSAFS',
                   'OILPRICE': 'MCOILWTICO',
                   'COMPAPFF': 'CPFF',
                   'CP3M': 'CPF3M',
                   'CLAIMS': 'ICNSA',  # weekly
                   'HWIURATIO': [Series.div, 'JTSJOL', 'UNEMPLOY'],
                   'CPF3MTB3M': [Series.sub, 'CPF3M', 'DTB3'],
                   'CONSPI': [Series.div, 'NONREVSL', 'PI']}

    def adjusted_series(self, series_id, start=19590101, freq='M'):
        """Retrieve a raw series to update FRED-MD dataset

        Notes
        -----
        http://www.econ.yale.edu/~shiller/data/ie_data.xls        
        """
        shiller = {'S&P div yield': 's-p-500-dividend-yield',
                   'S&P PE ratio': 'shiller-pe'}
        if series_id in ['S&P: indust']:
            s = Series()
        elif series_id in ['CLAIMS']:
            df = DataFrame(self('ICNSA'))
            df['Date'] = to_monthend(df.index)
            s = df.groupby('Date').mean().iloc[:,0]
        elif series_id in shiller.keys():
            v = shiller[series_id]
            s = multpl(v)
        elif series_id in self.fred_adjust.keys():
            v = adjust[series_id]
            s = (self(v, freq=freq) if isinstance(v, str) \
                 else v[0](self(v[1], freq=freq),
                           self(v[2], freq=freq)))
        else:
            s = self(series_id, auto_request=True, freq=freq)
        return s[s.index >= start].rename(series_id)
    
    
def pcaEM(X, kmax=None, p=2, tol=1e-12, n_iter=2000, echo=ECHO):
    """Fill in missing data with factor model and EM algorithm of
    Rubin & Thayer (1982), Stock & Watson (1998) and Bai & Ng (2002)

    Parameters
    ----------
    X : 2D array
        T observations/samples in rows, N variables/features in columns
    kmax : int, default is None
        Maximum number of factors.  If None, set to rank from SVD minus 1
    p : int in [0, 1, 2, 3], default is 2 (i.e. 'ICp2' criterion)
        If 0, number of factors is fixed as kmax.  Else picks one of three
        methods in Bai & Ng (2002) to auto-determine number in every iteration

    Returns
    -------
    x : 2D arrayint
        X with nan's replaced by PCA EM
    model : dict
        Model results 'u', 's', 'vT', 'kmax', 'converge', 'n_iter'
    """
    X = X.copy()      # passed by reference
    Y = np.isnan(X)   # identify missing entries
    assert(not np.any(np.all(Y, axis=1)))  # no row can be all missing
    assert(not np.any(np.all(Y, axis=0)))  # no column can be all missing
    for col in np.flatnonzero(np.any(Y, axis=0)): # replace with column means
        X[Y[:, col], col] = np.nanmean(X[:, col])
    M = dict()   # latest fitted model parameters
    for M['n_iter'] in range(1, n_iter + 1):
        old = X.copy()
        mean, std = X.mean(axis=0).reshape(1, -1), X.std(axis=0).reshape(1, -1)
        X = (X - mean) / std  # standardize

        # "M" step: estimate factors
        M['u'], M['s'], M['vT'] = np.linalg.svd(X)

        # auto-select number of factors if p>0 else fix number of factors
        r = BaiNg(X, p, kmax or len(M['s'])-1) if p else kmax or len(M['s'])-1

        # "E" step: update missing entries
        y = M['u'][:, :r] @ np.diag(M['s'][:r]) @ M['vT'][:r, :]  # "E" step
        X[Y] = y[Y]
        
        X = (X * std) + mean  # undo standardization
        M['kmax'] = r
        M['converge'] = np.sum((X - old)**2)/np.sum(X**2)  # diff**2/prev**2
        if echo:
            print(f"{M['n_iter']:4d} {M['converge']:8.3g} {r}")
        if M['converge'] < tol:
            break
    return X, M
    
def BaiNg(x, p=2, kmax=None, standardize=False, echo=ECHO):
    """Determine number of factors based on Bai & Ng (2002) criterion

    Parameters
    ----------
    x : 2D array
        T observations/samples in rows, N variables/features in columns
    p : int in [1, 2, 3], default is 2
        use PCp1 or PCp2 or PCp3 penalty
    kmax : int, default is None
        maximum number of factors.  If None, set to rank from SVD
    standardize : bool, default is False
        if True, then standardize data before processing (works better)

    Returns
    -------
    r : int
        best number of factors based on ICp{p} criterion, or 0 if not determined

    Notes
    -----
    See Bai and Ng (2002) and McCracken at
      https://research.stlouisfed.org/econ/mccracken/fred-databases/
    """
    if standardize:
        x = ((x-x.mean(axis=0).reshape(1,-1))/x.std(axis=0,ddof=0).reshape(1,-1))
    T, N = x.shape
    #mR2 = np.sum(marginalR2(x), axis=1)
    u, s, vT = np.linalg.svd(x, full_matrices=False)
    kmax = min(len(s), kmax or len(s))
    mR2 = [0] + list(s**2 / (N * T))   # first case is when no factors used
    var = (sum(mR2) - np.cumsum(mR2))  # variance of residuals after k components
    lnvar = np.log(np.where(var > 0, var, 1e-26))
    NT2 = (N * T)/(N + T)
    C2 = min(N, T)
    penalty = [np.log(NT2) / NT2, np.log(C2) / NT2, np.log(C2) / C2][p - 1]
    ic = (lnvar + np.arange(len(mR2))*penalty)[:(kmax + 2)]
    sign = np.sign(ic[1:] - ic[:-1])
    r = np.flatnonzero(sign>0)
    return min(r) if len(r) else 0   # first min point

def marginalR2(x, kmax=None, standardize=False):
    """Return marginal R2 of each variable from incrementally adding factors

    Parameters
    ----------
    x : 2D array
        T observations/samples in rows, N variables/features in columns
    kmax : int, default is None
        maximum number of factors.  If None, set to rank from SVD
    standardize : bool, default is False
        if True, then standardize data before processing (works better)

    Returns
    -------
    mR2 : 2D array
        each row corresponds to adding one factor component
        values are the incremental R2 for the variable in the column

    Notes
    -----
    See Bai and Ng (2002) and McCracken at
      https://research.stlouisfed.org/econ/mccracken/fred-databases/

    pca.components_[i,:] is vT[i, :]
    pca.explained_variance_ is s**2/(T-1)
    y = pca.transform(x)    # y = s * u: T x n "projection"
    beta = np.diag(pca.singular_values_) @ pca.components_  # "loadings"
    x.T @ x = beta.T @ beta is covariance matrix
    """
    if standardize:
        x = (x-x.mean(axis=0).reshape(1,-1))/x.std(axis=0,ddof=0).reshape(1, -1)
    u, s, vT = np.linalg.svd(x, full_matrices=False)
    
    # increase in R2 from adding kth (orthogonal) factor as a regressor
    mR2 = np.vstack([np.mean((u[:,k-1:k] @ u[:,k-1:k].T @ x)**2, axis=0)
                     for k in (np.arange(kmax or len(s)) + 1)])
    mR2 = mR2 / np.mean((u @ u.T @ x)**2, axis=0).reshape(1, - 1)
    return mR2

# units - stromg that indicates a data value transformation.
#   lin = Levels (No transformation) [default]
#   chg = Change x(t) - x(t-1)
#   ch1 = Change from Year Ago x(t) - x(t-n_obs_per_yr)
#   pch = Percent Change ((x(t)/x(t-1)) - 1) * 100
#   pc1 = Percent Change from Year Ago ((x(t)/x(t-n_obs_per_yr)) - 1) * 100
#   pca = Compounded Annual Rate of Change (((x(t)/x(t-1)) ** (n_obs_per_yr)) - 1) * 100
#   cch = Continuously Compounded Rate of Change (ln(x(t)) - ln(x(t-1))) * 100
#   cca = Continuously Compounded Annual Rate of Change ((ln(x(t)) - ln(x(t-1))) * 100) * n_obs_per_yr
#   log = Natural Log ln(x(t))
# Frequency
#   A = Annual
#   SA = Semiannual
#   Q = Quarterly
#   M = Monthly
#   BW = Biweekly
#   W = Weekly
#   D = Daily
# Seasonal Adjustment
#   SA = Seasonally Adjusted
#   NSA = Not Seasonally Adjusted
#   SAAR = Seasonally Adjusted Annual Rate
#   SSA = Smoothed Seasonally Adjusted
#   NA = Not Applicable
