"""Class and methods to access ALFRED/FRED apis, and FRED-MD/FRED-QD

- FRED, ALFRED: St Louis Fed api's, with revision vintages
- FRED-MD, FRED-QD: See McCracken website at St Louis Fed

  - https://research.stlouisfed.org/econ/mccracken/fred-databases/

Data are cached in memory. Vintage releases retrieved by sequence or date window.

Copyright 2022, Terence Lim

MIT License
"""
from typing import Dict, List, Tuple, Iterable
import json
import io
import re
import pickle
import zipfile
import time
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
import matplotlib.pyplot as plt
from pandas.api import types
import pandas as pd
from pandas import DataFrame, Series, Timestamp
from pandas.tseries.offsets import MonthEnd, YearEnd, QuarterEnd
from pandas.api.types import is_list_like
from .readers import requests_get
_VERBOSE = 1


#
# Helper functions: convert date formats, scrape multpl.com
#
def _int2date(date: int) -> str:
    """helper method to convert int date to FRED api string format"""
    if types.is_list_like(date):    
        return [_int2date(d) for d in date] 
    else:
        return "-".join(str(date)[a:b] for a, b in [[0,4], [4,6], [6,8]])

def _date2int(date: str) -> int:
    """helper method to convert FRED api string format to int date"""
    if types.is_list_like(date):
        return [_date2int(d) for d in date]
    else:
        return int(re.sub('\D', '', str(date)[:10]))

def _to_date(datestamps: Iterable, format: str) -> int:
    return [int(datetime.strptime(str(d), format).strftime('%Y%m%d'))
            for d in datestamps]

def _to_monthend(dates: Iterable) -> int:
    return [int((pd.to_datetime(d, format="%Y%m%d")
                 + pd.offsets.MonthEnd(0)).strftime("%Y%m%d"))
            for d in dates]

#
# Class Alfred to retrieve archival FRED series
#
class Alfred:
    """Base class for Alfred/Fred access, and manipulating retrieved data series

    Args:
      api_key: API key string registered with FRED
      start: default starting date of series
      end: default ending date of series
      savefile: name of local file to auto-save retrieved series
      verbose: whether to display messages

    Attributes:
        tcode: Reference dict of transformation codes
        _fred_url(): Formatter to construct FRED api query from key-value string
        _alfred_url(): Formatter to construct vintage FRED api query
        _category_url(): Formatter to construct FRED category api query
    """
    _header = {    # starter Dict of series descriptions
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


    def __init__(self,
                 api_key: str,
                 start: int = 17760704,
                 end: int = 99991231,
                 savefile: str = '',
                 verbose=_VERBOSE):
        """Create object, with api_key, for FRED access and data manipulation"""
        self.api_key = api_key
        self._start = start
        self._end = end
        self.savefile = str(savefile)
        self._cache = dict()
        self._header = Alfred._header.copy()
        self._verbose = verbose

    def __call__(self,
                 series_id: str,
                 start: int = 0,
                 end: int = 0,
                 release: int | pd.DateOffset = 0,
                 vintage: int = 99991231,
                 label: str = '',
                 realtime: bool = False,
                 freq: str = '',
                 **kwargs) -> Series | None:
        """Retrieve from cache, else call FRED api, and apply transforms

        Args:
            series_id: Label of series to retrieve
            start, end: Start and end period dates (inclusive) to keep
            label: New label to rename returned series
            release: release number (1 is first, 0 for latest), or 
                     latest up to maximum date offset; 
            vintage: Latest realtime_start date of observations to keep
            freq: Resample and replace date index with at periodic frequency;
                   in {'M', 'A'. 'Q', 'D', 'Y'}, else empty '' to auto select
            diff: Number of difference operations to apply
            log: Number of log operations to apply
            pct_change: Number of pct_change to apply

        Returns:
            transformed values; name is set to label if provided else series_id
        """
        assert isinstance(series_id, str)
        if (series_id not in self._cache and not self.get_series(series_id)):
            return None
        if not freq:
            freq = self.header(series_id, 'frequency_short')
            
        df = Alfred.construct_series(self[series_id]['observations'],
                                     release=release,
                                     vintage=vintage,
                                     start=start or self._start,
                                     end=end or self._end,
                                     freq=freq)
        if realtime:
            s = Alfred.transform(df['value'], **kwargs).to_frame()
            s['realtime_start'] = df['realtime_start'].values
            s['realtime_end'] = df['realtime_end'].values
            return s.rename(columns={'value': label or series_id})
        return Alfred.transform(df['value'], **kwargs)\
                     .rename(label or series_id)

    tcode = {1: {'diff': 0, 'log': 0},
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
             'log': {'diff': 0, 'log': 1}}

    # units - string that indicates a data value transformation.
    #   lin = Levels (No transformation) [default]
    #   chg = Change x(t) - x(t-1)
    #   ch1 = Change from Year Ago x(t) - x(t-n_obs_per_yr)
    #   pch = Percent Change ((x(t)/x(t-1)) - 1) * 100
    #   pc1 = Percent Change from Year Ago ((x(t)/x(t-n_obs_per_yr)) - 1) * 100
    #   pca = Compounded Annual Rate of Change (((x(t)/x(t-1))
    #                                           ** (n_obs_per_yr)) - 1) * 100
    #   cch = Cont Compounded Rate of Change (ln(x(t)) - ln(x(t-1))) * 100
    #   cca = Cont Compounded Annual Rate of Change = cch * n_obs_per_yr
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

    @staticmethod
    def transform(data: DataFrame, tcode: int | str = 1,
                  freq: str = '', **kwargs) -> DataFrame:
        """Static method to apply time series transformations

        Args:
            data: DataFrame of input data
            tcode: int transformation code in {1, ..., 7} or str. describing ho
                   to apply sequence of operators to make series stationary
            freq: str periodicity of dates in {'M', 'Q', 'A'}
            kwargs: transformation operators and number of times

        Transformation operators:

          - log (int): Number of times to take log (default 0)
          - diff (int): Number of times to take difference (default 0)
          - pct_change (bool): Whether to apply pct_change operator
          - periods (int): lags for pct_change or diff operator (default 1)
          - annualize (int): annualization multiplier (default 1)
        """

        # build up desired transformation set from input tcode and kwargs
        t = {'periods': 1, 'shift': 0, 'pct_change': False, 'annualize': 1}
        t.update(Alfred.tcode[tcode])
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

    def date_spans(self, series_id: str = 'USREC',
                   threshold: int = 0) -> List[Tuple[Timestamp, Timestamp]]:
        """Return recession span dates as tuples of Timestamp"""
        usrec = self(series_id)
        usrec.index = pd.DatetimeIndex(usrec.index.astype(str), freq='infer')
        g = (usrec > threshold) \
            | (usrec.shift(-1, fill_value=threshold) > threshold)
        g = (g != g.shift(fill_value=False)).cumsum()[g].to_frame()
        g = g.reset_index().groupby(series_id)['date'].agg(['first','last'])
        vspans = list(g.itertuples(index=False, name=None))
        return vspans

    @staticmethod
    def popular(page: int = 1):
        """Static method to web scrape popular series names, by page number"""
        assert(page > 0)
        url = f"https://fred.stlouisfed.org/tags/series?ob=pv&pageID={page}"
        data = requests_get(url).content
        soup = BeautifulSoup(data, 'lxml')
        tags = soup.findAll(name='a', attrs={'class': 'series-title'})
        details = [tag.get('href').split('/')[-1] for tag in tags]
        #tags = soup.findAll(name='input',attrs={'class':'pager-item-checkbox'})
        #details = [tag.get('value') for tag in tags]
        return details

    def header(self, series_id: str | List[str], column: str = 'title'):
        """Returns title (or other column) from last meta record of a series"""
        if is_list_like(series_id):
            return [self.header(s, column=column) for s in series_id]
        if series_id not in self._header:
            try:
                if series_id not in self._cache:  # load via api if not in cache
                    self.get_series(series_id)
                self._header[series_id] = self[series_id]['series'].iloc[-1]
            except:
                return f"*** {series_id} ***"
        return self._header[series_id].get(column, f"*** {series_id} ***")

    def observations(self, series_id: str, date: int,
                     freq: str = '') -> DataFrame:
        """Return all release of observations for a series and date

        Args:
            series_id: series name
            date: date of series
            freq: in {'M', 'A'. 'Q', 'D', 'Y', ''}
        """
        df = self._cache[series_id]['observations'].copy()
        df['date'] = pd.to_datetime(df['date'])
        date = pd.to_datetime(date, format="%Y%m%d")
        df = df.dropna().reset_index(drop=True)
        if freq:
            if freq.upper()[0] in ['A']:
                df['date'] += YearEnd(0)
                end = date + YearEnd(0)
            if freq.upper()[0] in ['S']:
                df['date'] += QuarterEnd(1)
                end = date + QuarterEnd(1)
            if freq.upper()[0] in ['Q']:
                df['date'] += QuarterEnd(0)
                end = date + QuarterEnd(0)
            if freq.upper()[0] in ['M']:
                df['date'] += MonthEnd(0)
                end = date + MonthEnd(0)
            if freq.upper()[0] in ['B']:
                df['date'] += pd.DateOffset(days=13)
                end = date + pd.DateOffset(days=13)
            if freq.upper()[0] in ['W']:
                df['date'] += pd.DateOffset(days=6)
                end = date + pd.DateOffset(days=6)
        df['date'] = df['date'].dt.strftime('%Y%m%d').astype(int)
        date = int(date.strftime('%Y%m%d'))
        end = int(end.strftime('%Y%m%d'))
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[(df['date'] >= date) & (df['date'] <= end)]\
            .sort_values('realtime_start')
        return df.set_axis(1 + np.arange(len(df))).rename_axis(index='release')

    
    #
    # Interface methods to call API wrappers
    #
    @staticmethod
    def construct_series(observations: DataFrame,
                         vintage: int = 99991231,
                         release: int | pd.DateOffset = 0,
                         start: int = 0,
                         end: int = 99991231,
                         freq: str = '') -> Series:
        """Helper to construct series from given full observations dataframe

        Args:
            observations: DataFrame from FRED 'series/observations' api call
            release: sequence num (0 for latest), or latest to max date offset
            vintage: Latest realtime_start date (inclusive) allowed
            start, end: Start and end period dates (inclusive) to keep
            freq: in {'M', 'A'. 'Q', 'D', 'Y'}, else empty '' to auto select

        Returns:
            value as of each period date, optionally indexed by realtime_start
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

        """This code is maximum release
        if isinstance(release, int):  # keep latest up to max release
            df['release'] = df.groupby('date').cumcount()
            df = pd.concat([df[df['release'] + 1 == (release or 99999999)],
                            df.drop_duplicates('date', keep='last')])\
                   .drop_duplicates('date', keep='first')
        else:    # else latest release up through date offset
            df['release'] = (df['date'] + release).dt.strftime('%Y-%m-%d')
            df = df[df['realtime_start'] <= df['release']]\
                .drop_duplicates('date', keep='last')
        """
        if not release:
            df['release'] = df.groupby('date').cumcount()
            df = df.drop_duplicates('date', keep='last')
        elif isinstance(release, int):  # keep exactly release number
            df['release'] = df.groupby('date').cumcount()
            df = df[df['release'] + 1 == release]\
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

    def get_series(self, series_id: str | List[str], api_key: str ='',
                   start: int = 0, end: int = 0) -> int:
        """Retrieve metadata and full observations of a series with FRED api

        Args:
            series_id: list of ids of series to retrieve

        Returns:
            length of observations dataframe
        """
        if types.is_list_like(series_id):
            return [self.get_series(s, start=start, end=end) for s in series_id]
        series = self.request_series(series_id,
                                     api_key=api_key,
                                     start=start,
                                     end=end,
                                     verbose=self._verbose)
        if series is None or series.empty:
            return 0
        self._cache[series_id] = {
            'observations': self.request_series_observations(series_id,
                                                             api_key=api_key,
                                                             start=start,
                                                             end=end,
                                                             alfred_mode=True,
                                                             verbose=self._verbose
            ),
            'series': series}
        return len(self._cache[series_id]['observations'])

    def get_category(self, category_id: str, api_key: str = ''):
        """Retrieve category data by calling related API
        
        Args:
            category_id: id of category to retrieve
            api_key: credentials to FRED

        Notes:

        Uses request_category method to call these FRED API on given category_id

        - 'category' API gets meta information
        - 'category/series' API gets series_ids
        - 'category/children' API gets child categories
        """
        c = self.request_category(category_id, api="category", api_key=api_key)
        if 'categories' not in c:
            return None
        c = c['categories'][0]
        c['children'] = self.request_category(category_id,
                                              api="category/children",
                                              api_key=api_key)\
                            .get('categories', [])
        c['series'] = []
        offset = 0
        while True:
            s = self.request_category(category_id,
                                      api="category/series",
                                      api_key=api_key,
                                      offset=offset)
            if not s['seriess']:
                break
            c['series'].extend(s['seriess'])
            offset += s['limit']        
        return c

    #
    # Wrappers around FRED api calls
    #

    # Format input key-values to form fred api's    
    _alfred_url = ("https://api.stlouisfed.org/fred/{api}?series_id={series_id}"
                   "&realtime_start={start}&realtime_end={end}"
                   "&api_key={api_key}&file_type=json").format
    _fred_url = ("https://api.stlouisfed.org/fred/{api}?series_id={series_id}"
                 "&api_key={api_key}&file_type=json").format
    _category_url = ("https://api.stlouisfed.org/fred/{api}?"
                    "category_id={category_id}&api_key={api_key}&"
                    "file_type=json{args}").format

    def request_series(self, series_id: str, api_key: str = '', start: int = 0,
                       end : int = 0, verbose: int = _VERBOSE) -> DataFrame:
        """Requests 'series' API for series metadata"""
        url = self._alfred_url(api="series",
                               series_id=series_id,
                               start=_int2date(start or self._start),
                               end=_int2date(end or self._end),
                               api_key=api_key or self.api_key)
        r = requests_get(url, verbose=-1)
        if r is None:
            url = self._fred_url(api="series",
                                 series_id=series_id,
                                 api_key=api_key or self.api_key)
            r = requests_get(url, verbose=verbose)
            if r is None:
                return DataFrame()
#        else:
#            self._print(url)
        v = json.loads(r.content)
        df = DataFrame(v['seriess'])
        df.index.name = str(datetime.now())
        return df

    def request_series_observations(self, series_id: str, api_key: str = '',
                                    start: int = 0, end: int = 0,
                                    alfred_mode: bool = False,
                                    verbose: int = _VERBOSE) -> DataFrame:
        """Request 'series/observations' API for full observations data"""
        url = self._alfred_url(api="series/observations",
                               series_id=series_id,
                               start=_int2date(start or self._start),
                               end=_int2date(end or self._end),
                               api_key=api_key or self.api_key)
        r = requests_get(url, verbose=-1)
        if r is None:
            url = self._fred_url(api="series/observations",
                                 series_id=series_id,
                                 api_key=api_key or self.api_key)
            r = requests_get(url, verbose=verbose)
            if r is None:
                return DataFrame()
#        else:
#            self._print(url)

        contents = json.loads(r.content)
        df = DataFrame(contents['observations'])
        if alfred_mode:  # convert fred to alfred by backfilling realtime_start
            f = (df['realtime_start'].eq(contents['realtime_start']) &
                 df['realtime_end'].eq(contents['realtime_end'])).values
            df.loc[f, 'realtime_start'] = df.loc[f, 'date']
        return df

    def request_category(self, category_id: str, api: str = "category",
                         api_key: str = '', verbose: int = _VERBOSE,
                         **kwargs) -> Dict:
        """Request 'category' and related API for category data"""
        args = "&".join([f"{k}={v}" for k,v in kwargs.items()])
        url = self._category_url(api=api,
                                 category_id=category_id,
                                 api_key=api_key or self.api_key,
                                 args="&" + args if args else '')
        r = requests_get(url, verbose=verbose)
        return dict() if r is None else json.loads(r.content)


    #
    # Internal methods to manage cache in memory
    #
    def __getitem__(self, series_id: str) -> Dict:
        """Get observations and metadata from cache for {series_id}"""
        return self._cache.get(series_id, None)

    def keys(self):
        """Return id names of all loaded series data"""
        return list(self._cache.keys())

    _columns = ['id', 'observation_start', 'observation_end',
                'frequency_short', 'title', 'popularity',
                'seasonal_adjustment_short', 'units_short']
    def values(self, columns: List[str] = _columns) -> DataFrame:
        """Return headers (last metadata row) of all loaded series

        Args:
            columns: subset of header columns to return

        Returns:
            DataFrame of latest headers of all series loaded
        """
        df = pd.concat([v['series'].iloc[[-1]] for v in self._cache.values()],
                       axis=0,
                       ignore_index=True)
        df = df.set_index('id', drop=False)
        return df[columns]
            
    def load(self, savefile: str = ''):
        """Load all series to memory from pickle file, return number loaded"""
        with open(str(savefile or self.savefile), 'rb') as f:
            self._cache.update(**pickle.load(f))
        return len(self._cache)

    def dump(self, savefile: str = '') -> int:
        """Save all memory-cached series to pickle file, return number saved"""
        with open(str(savefile or self.savefile), 'wb') as f:
             pickle.dump(self._cache, f)
        return len(self._cache)

    def clear(self):
        """Clear internal memory cache of previously loaded series"""
        self._cache.clear()

    def pop(self, series_id: str) -> Dict[str, DataFrame]:
        """Pop and return desired series, then clear from memory cache"""
        return self._cache.pop(series_id, None)


    #
    # To splice series -- see FRED-MD appendix
    #
    @staticmethod
    def multpl(page: str) -> DataFrame:
        """Helper method to retrieve shiller series by scraping multpl.com

        Args:
          page: Web page name in {'s-p-500-dividend-yield', 'shiller-pe'}

        Returns:
         Dataframe of monthly series (for updating FRED-MD)
        """
        url = f"https://www.multpl.com/{page}/table/by-month"
        soup = BeautifulSoup(requests_get(url).content, 'html.parser')
        tables = soup.findChildren('table')
        df = pd.read_html(tables[0].decode())[0]
        df.iloc[:,0] = _to_date(df.iloc[:,0], format='%b %d, %Y')
        df['date'] = _to_monthend(df.iloc[:, 0])
        df = df.sort_values('Date').groupby('date').last().iloc[:,-1]
        if not types.is_numeric_dtype(df):
            df = df.map(lambda x: re.sub('[^\d\.\-]','',x)).astype(float)
        return df

    # Reference dict for splicing/adjusting series for Fred-MD
    splice_: Dict = {'HWI': 'JTSJOL',
                     'AMDMNO': 'DGORDER',
                     'S&P 500': 'SP500',
                     'RETAIL': 'RSAFS',
                     'OILPRICE': 'MCOILWTICO',
                     'COMPAPFF': 'CPFF',
                     'CP3M': 'CPF3M',
                     'CLAIMS': 'ICNSA',  # weekly
                     'S&P div yield': 's-p-500-dividend-yield', # multpl
                     'S&P PE ratio': 'shiller-pe', # s-p-500-pe-ratio
                     'HWIURATIO': [Series.div, 'JTSJOL', 'UNEMPLOY'],
                     'CPF3MTB3M': [Series.sub, 'CPF3M', 'DTB3'],
                     'CONSPI': [Series.div, 'NONREVSL', 'PI']}

    def splice(self, series_id: str, start: int = 19590101,
               freq: str = 'M') -> Series:
        """Retrieve raw series to update a FRED-MD series
    
        e.g. Shiller series: 

        - http://www.econ.yale.edu/~shiller/data/ie_data.xls
        - multpl.com
        """
        shiller = ['S&P div yield', 'S&P PE ratio']
        if series_id in ['S&P: indust']:
            s = Series(dtype=float)
        elif series_id in ['CLAIMS']:
            df = DataFrame(self('ICNSA'))
            df['Date'] = _to_monthend(df.index)
            s = df.groupby('Date').mean().iloc[:,0]
        elif series_id in shiller:
            v = Alfred.splice_[series_id]
            s = Alfred.multpl(v)
        elif series_id in Alfred.splice_.keys():
            v = Alfred.splice_[series_id]
            if isinstance(v, str):
                s = self(v, freq=freq) 
            else:
                s = v[0](self(v[1], freq=freq), self(v[2], freq=freq))
        else:
            s = self(series_id, auto_request=True, freq=freq)
        return s[s.index >= start].rename(series_id)
    
#
# Functions to retrieve fred_md and fred_qd
#
fred_md_url = 'https://files.stlouisfed.org/files/htdocs/fred-md/'

def fred_md(vintage: int | str = 0, url: str = '',
            verbose: int = _VERBOSE)-> DataFrame:
    """Retrieve and parse current or vintage csv from McCracken FRED-MD site

    Args:
        vintage: file name relative to base url or zipfile, or int date YYYYMM
        url: base name of url, local file path or zipfile archive

    Returns:
       DataFrame indexed by end-of-month date

    Notes:

    - if vintage is int: derive vintage csv file name from input date YYYYMM
    - if url is '': derive subfolder or zip archive name, from vintage

    Examples:

    >>> md_df, mt = fred_md()  # current in monthly/current.csv
    >>> md_df, mt = fred_md('Historical FRED-MD Vintages Final/2013-12.csv',
                            url=fred_md_url+'Historical_FRED-MD.zip') # pre-2015
    >>> md_df, mt = fred_md('monthly/2015-05.csv',
                            url=fred_md_url+'FRED_MD.zip')      # post-2015
    """
    url_ = fred_md_url
    if isinstance(vintage, int) and vintage:
        csvfile = f"{vintage // 100}-{vintage % 100:02d}.csv"
        if vintage < 201500:
            url_ = url_ + 'Historical_FRED-MD.zip'
            vintage = 'Historical FRED-MD Vintages Final/' + csvfile
        else:
            vintage = 'monthly/' + csvfile
    else:
        vintage = vintage or 'monthly/current.csv'
    if _VERBOSE:
        print(vintage)
    url = url or url_
    if url.endswith('.zip'):
        if url.startswith('http'):
            url = io.BytesIO(requests_get(url).content)
        with zipfile.ZipFile(url).open(vintage) as f:
            df = pd.read_csv(f, header=0)
    else:
        df = pd.read_csv(urljoin(url, vintage), header=0)
    df.columns = df.columns.str.rstrip('x')
    meta = dict()
    for _, row in df.iloc[:5].iterrows():
        if '/' not in row[0]:    # this row has metadata, e.g. transform codes
            label = re.sub("[^a-z]", '', row[0].lower()) # simplify label str
            meta[label] = row[1:].astype(int).to_dict()  # as dict of int codes
    df = df[df.iloc[:, 0].str.find('/') > 0]      # keep rows with valid date
    df.index = _to_date(df.iloc[:, 0], format='%m/%d/%Y')
    df.index = _to_monthend(df.index)
    return df.iloc[:, 1:], DataFrame(meta)

def fred_qd(vintage: int | str = 0, url: str = '', verbose: int = 0):
    """Retrieve and parse current or vintage csv from McCracken FRED-MD site

    Args:
        vintage: file name relative to base url or zipfile, or int date YYYYMM
        url: base name of url, local file path or zipfile archive

    Returns:
       DataFrame indexed by end-of-month date

    Notes:

    - if vintage is int: derive vintage csv file name from input date YYYYMM
    - if url is '': derive subfolder or zip archive name, from vintage
    """
    url = url or fred_md_url
    if isinstance(vintage, int) and vintage:
        vintage = f"quarterly/{vintage // 100}-{vintage % 100:02d}.csv"
    else:
        vintage = 'quarterly/current.csv'
    if _VERBOSE:
        print(vintage)
    df = pd.read_csv(urljoin(url, vintage), header=0)
    df.columns = df.columns.str.rstrip('x')
    meta = dict()
    for _, row in df.iloc[:5].iterrows():
        if '/' not in row[0]:    # this row has metadata, e.g. transform codes
            label = re.sub("[^a-z]", '', row[0].lower()) # simplify label str
            meta[label] = row[1:].astype(int).to_dict()  # as dict of int codes
    df = df[df.iloc[:, 0].str.find('/') > 0]      # keep rows with valid date
    df.index = _to_date(df.iloc[:, 0], format='%m/%d/%Y')
    df.index = _to_monthend(df.index)
    return df.iloc[:, 1:], DataFrame(meta)
        

