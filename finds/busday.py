"""Implement custom daily and weekly trading day calendars and datetime methods

- pandas custom business calendar

Author: Terence Lim
License: MIT
"""
import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pandas_datareader as pdr
from pandas.tseries.holiday import USFederalHolidayCalendar
from sqlalchemy import Column, Integer
from pandas.api.types import is_list_like
from pandas.tseries.offsets import MonthEnd, YearEnd, QuarterEnd
try:
    from settings import ECHO
except:
    ECHO = False

# .to_pydatetime() - convert pandas format (Timestamp, datetime64) to datetime
# datetime.date.strftime(d, '%Y%m%d') - convert datetime to string
# np.array([self(dates)], dtype='datetime64[D]') - converts to numpy date format
# datetime.datetime(year, month, day) - returns datetime.datetime format

def to_monthend(dt):
    """Return calendar monthend date given an int date or list"""
    if is_list_like(dt):    
        return [to_monthend(d) for d in dt]
    if dt <= 9999:
        d = datetime.datetime(year=dt, month=12, day=1) + MonthEnd(0)
    elif dt <= 999999:
        d = datetime.datetime(year=dt//100, month=dt % 100, day=1) + MonthEnd(0)
    else:
        d = pd.to_datetime(str(dt)) + MonthEnd(0)
    return int(d.strftime('%Y%m%d'))

def str2date(date, informat='%Y-%m-%d', outformat='%Y%m%d'):
    """Extract int components from date strings by input and output formats

    Parameters
    ----------
    date : str or list of str
        input date strings to convert
    informat : str, default is '%F'
        date format of input string
    outformat : str or list of str, or dict of {key: str}, default is '%Y%m%d'
        date format of output.  If dict, then output is dict with same key names

    Returns
    -------
    output : int or list of int or dict of int
        int date components corresponding to outformat

    Notes
    -----
    Formats, per strptime and strftime:
    %b %B %h = input month name
    %F = %Y-%m-%d
    %T = %H:%M:%S
    %n = whitespace
    %w %W %U %V = week number
    %u = day of week (1-7)
    """
    if is_list_like(date):
        return [str2date(s, informat, outformat) for s in date]
    if isinstance(date, int):
        date = str(date)
    dt = datetime.datetime.strptime(date, informat)
    if isinstance(outformat, dict):
        return {k: int(dt.strftime(v)) for k,v in outformat.items()}
    elif is_list_like(outformat):
        return [int(dt.strftime(f)) for f in outformat]
    else:
        return int(dt.strftime(outformat))

def minibatch(x, batchsize):
    """Group the rows of x into minibatches of length batchsize"""
    return [x[i:(i + batchsize)] for i in range(0, len(x), batchsize)]
    
class BusDay:
    """Implement custom trading date calendar, based on FamaFrench and NYSE

    Parameters
    ----------
    sql : SQL instance
        SQL connection
    create : bool
        If True, recreate from FamaFrench library pdr datareader, else from SQL

    Attributes
    ----------
    busdaycalendar_ : numpy busdaycalendar object
        Customized business days, excluding trading holidays
    custom_ : pd.offsets.CDay object
        For determing offsets with custom business days
    begmo_ : pd.offsets.CBMonthBegin
        For determining custom beginning of month date
    endmo_ : pd.offsets.CBEndBegin
        For determining custom end of month date
    max_date : int
       last date of calendar
    """
    def __init__(self, sql, create=False, start=19251231, end=20401231):
        """Retrieve or create custom calendar (from F-F_Research_Data_Factors)"""
        self.sql = sql
        self.table = sql.Table('busdates',
                               Column('date', Integer, primary_key=True))
        if create:  #  reload the series using pandas reader
            f = pdr.data.DataReader(        
                name='F-F_ST_Reversal_Factor_daily', 
                data_source='famafrench',     # 'F-F_Research_Data_Factors_daily'
                start=1900, end=2050)[0].index.sort_values().unique()
            df = DataFrame(str2date(f.astype(str), '%Y-%m-%d', '%Y%m%d'),
                           columns=['date'])
            self.table.create(checkfirst=True)
            sql.load_dataframe('busdates', df)
        else:
            df = sql.read_dataframe('SELECT * FROM busdates')

        # 1. Initially, dates = actual FamaFrench busdays
        dates = pd.DatetimeIndex(sorted(list(df['date'].unique().astype(str))))
        last = pd.to_datetime(str(df.iloc[-1]['date']))

        # 2. Extend with pandas 5-day calendar from last through to 20221231
        dates = dates.append(pd.date_range(last, end=pd.to_datetime('20221231'),
                                           freq='B')[1:])

        # 3. But remove current list of anticipated NYSE holidays 
        hols = ['20210101', '20210118', '20210215', '20210402', '20210531',
                '20210705', '20210906', '20211125', '20211224', '20220117',
                '20220221', '20220415', '20220530', '20220704', '20220905',
                '20221124','20221226']
        self.max_date = max(int(max(hols)[:4])*10000+1231, max(df['date']))
        hols = pd.to_datetime(hols)
        dates = sorted(list(set(dates).difference(set(hols))))  # actual busdays

        # 4. Generate a list of all potential busdays from pandas 6-day calendar
        alldates = set(pd.date_range(dates[0], dates[-1], freq=pd.offsets.CDay(
            calendar=np.busdaycalendar('1111110'), normalize=True)))

        # 5. Finalize actual holidays: hols = all dates less actual dates
        hols = np.array(list(alldates.difference(dates)), dtype='datetime64[D]')
        hols = sorted(set(hols).union([np.datetime64('1926-01-01')]))
        
        # Custom and offset calendar objects is 6-day week less actual holidays
        self.busdaycalendar_ = np.busdaycalendar(weekmask='1111110',
                                                 holidays=hols)
        self.custom_ = pd.offsets.CDay(calendar=self.busdaycalendar_)
        self.begmo_ = pd.offsets.CBMonthBegin(calendar=self.busdaycalendar_)
        self.endmo_ = pd.offsets.CBMonthEnd(calendar=self.busdaycalendar_)

    def date_range(self, start, end, freq='daily'):
        """Return business dates at desired freq between beg and end dates

        Parameters
        ----------
        start: int
            Inclusive start of date range
        end: int
            Inclusive end of date range
        freq: str or int, optional, in {'daily', 'begmo', 'endmo', month:int}
            Default is 'daily'. If int, then returns every endmo for that month.
        """
        try:
            month = int(freq)           # annually as of calendar month end
            dates = pd.DatetimeIndex([self(self.year(d), month, 99) for d in
                                      range(self.year(start),self.year(end)+1)])
        except:
            freq = freq.lower()
            if freq.startswith("d"):    # custom business daily
                dates = pd.date_range(start=self(start), end=self(end),
                                      freq=self.custom_)
            elif freq.startswith("b"):  # custom business month begin
                dates = pd.date_range(start=self(start, day=1),
                                      end=str(self.endmo(end)),
                    freq=self.begmo_)
            elif freq.startswith("e"):  # custom business month end
                dates = pd.date_range(start=self(start, day=1),
                                      end=str(self.endmo(end)),
                                      freq=self.endmo_)
            else:                       # daily calendar
                dates = pd.DatetimeIndex([str(start), str(end)])
        return list(dates.strftime('%Y%m%d').astype(int))

    def offset(self, dates, offsets=0, end=None, roll='preceding'):
        """Return valid business date with optional offset or roll treatment

        Parameters
        ----------
        dates: int or array_like of int
            Input dates (YYYYMMDD int format)
        offsets: int, optional
            Number of business days to offset
        end: int, optional
            End of offset range (default None to only return single offset)
        roll: {'raise', 'forward', 'following', 'backward', 'preceding'}
            How to treat dates that are not a valid day (default 'preceding')
        """
        if end:  # return all dates within window [offsets, end] around {date}
            if is_list_like(dates):
                return self._map(self.offset, dates, offsets, end, roll=roll)
            return np.array([self.offset(dates, offsets=i, roll=roll)
                             for i in np.arange(offsets, end + 1)]).T
        try:
            return int(pd.to_datetime(np.busday_offset(
                dates=np.array([self(dates)], dtype='datetime64[D]'),
                offsets=offsets,
                roll=roll,
                busdaycal=self.busdaycalendar_)).strftime('%Y%m%d')[0])
        except:
            #return [self.offset(d, offsets, roll=roll) for d in dates]
            return self._map(self.offset, dates, offsets, end=end, roll=roll)

    def holding_periods(self, dates):
        """Returns beg and end dates of realized returns after each rebalance"""
        d = sorted(dates)
        return [(self.offset(b, offsets=1), self.offset(e, offsets=0))
                for b, e in zip(d[:-1], d[1:])]

    def endmo(self, date, months=0):
        """Returns business month end date/s, with optional offset in months"""
        return (self._map(self.endmo,date, months) if is_list_like(date) else
                int((self(date, day=1) + (0 * self.endmo_) +
                     (months * self.endmo_)).strftime('%Y%m%d')))

    def begmo(self, date, months=0):
        """Returns business month begin date/s, with optional offset in months"""
        return (self._map(self.begmo, date, months) if is_list_like(date) else
                int((self(date, day=1) + (0 * self.begmo_) +
                     (months * self.begmo_)).strftime('%Y%m%d')))

    def endyr(self, date, years=0):
        """Returns business year end date/s, with optional offset in years"""
        return (self._map(self.endyr, date, years) if is_list_like(date) else
                int((self(date, 12, 1) + (0 * self.endmo_) +
                     (12*years*self.endmo_)).strftime('%Y%m%d')))

    def begyr(self, date, years=0):
        """Returns business year begin date/s, with optional offset in years"""
        return (self._map(self.begyr, date, years) if is_list_like(date) else
                int((self(date, 1, 1) + (0 * self.begmo_) +
                     (12*years*self.begmo_)).strftime('%Y%m%d')))

    @classmethod
    def year(self, date):
        """Helper classmethod to extract year from input date or list"""
        if is_list_like(date):
            return self._map(self.year, date)
        date = int(date)
        while date > 9999:  # input date may exclude month and day
            date //= 100
        return date

    @classmethod
    def month(self, date):
        """Helper classmethod to extract month from input date or list"""
        if is_list_like(date):
            return self._map(self.month, date)
        date = int(date)
        while date > 999999:  # input date may exclude day
            date //= 100
        return date % 100
        
    @classmethod
    def day(self, date):
        """Helper classmethod to extract day of month from input date or list"""
        if is_list_like(date):
            return self._map(self.day, date)
        return int(date) % 100   # input date may exclude year or month

    def december_fiscal(self, date):
        """Determine lagged December fiscal year-end date, FamaFrench-style"""
        if is_list_like(date):        
            return self._map(self.dec_lag, date) 
        return self.endyr(date, -1 if self.month(date) >= 6 else -2)

    def june_universe(self, date):
        """Determine prior June universe selection date, FamaFrench-style"""
        if is_list_like(date):
            return self._map(self.jan_univ, date) 
        d = self.endmo(self.endyr(date), -6) 
        return d if date >= d else self.endmo(d, -12)

    @classmethod    
    def _map(self, func, dates, *args, **kwargs):
        """Helper method to apply a busday func to each date in list

        Examples
        --------
        busday._map(busday.offset, [20190526, 20190528], -2)
        busday._map(busday.begmo, [20190526, 20190528], -2)
        busday._map(busday.endmo, [20190526, 20190528], -2)
        """
        #return list(map(lambda d: func(int(d), int(n)), dates))
        mapped = {d: func(d, *args, **kwargs) for d in np.unique(dates)}
        return [mapped[d] for d in dates]
    
    def _datetime(self, date, month=None, day=None):
        """Wrapper for datetime method to flexibly handle long date format"""
        return datetime.datetime(self.year(date),
                                 self.month(date) if month is None else month,
                                 self.day(date) if day is None else day)
        
    def __call__(self, date, month=None, day=None):
        """Convert int date (long-form, or yr, month, day) to datetime,datetime

        Parameters
        ----------
        date : int or list-like
            Input date (YYYYMMDD long format) or year
        month : int, default is None
            If None, then month is inferred from input date
        day : int, default is None
            If None, then day is inferred from input date

        Returns
        -------
        dt: datetime object or list-like with same shape as input date
            If input DD is 0, then return custom business month begin date
            If input DD is 99, then return custom business month end date
        """
        if is_list_like(date):
            return [self(d) for d in date]
        try:
            if hasattr(date, 'to_pydatetime'):
                return date.to_pydatetime() # pandas timestamp or datetimeindex
            return self._datetime(date, month, day)
        except:  # catch if date is out of range of datetime.datetime
            try:     # if DD + 1 is <= 99, then return custom month begin date
                return (self._datetime(date//100,
                                       month=month,
                                       day=(day or self.day(date))+1)
                        + (0 * self.begmo_)).to_pydatetime()
            except:  # else must be DD == 99, so return custom month end date
                return (self._datetime(date//100, month=month, day=1) +
                        (0 * self.endmo_)).to_pydatetime()
            
def _to_values(df):
    """To return dataframe single value as scalar or multiple as numpy array"""
    return df.values if hasattr(df, 'values') else df

class Weekly(BusDay):
    """Implements custom weekly trading date calendar, ending on any day-of-week

    Parameters
    ----------
    sql : SQL instance
        SQL connection
    day : str in ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'] or int in [0,...,7]
        day to end week

    Attributes
    ----------
    weeks : DataFrame, index=weeknum
        Columns for beg, end (dates) and ismonthend
    """
    def __init__(self, sql, day, beg=19251231, end=20401231):
        """Derive weekly trading calendar, ending on specified day of week"""
        
        # call parent initializer, and retrieve daily trading dates
        super().__init__(sql, create=False)
        dates = pd.date_range(self(19251231), self(20401231), freq=self.custom_)
        
        # parse specified day-of-week, and generate weekly calender end dates
        if isinstance(day, int):
            day = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][day % 7]
        if not day.startswith('W-'):
            day = 'W-' + day.upper()
        weekly_end = pd.date_range(self(beg), self(end), freq=day)

        
        # require start(exclusve) and end(inclusive), and weekly end in dates
        dates = dates[(dates > weekly_end[0]) & (dates <= weekly_end[-1])]
        weeks = pd.Series(np.searchsorted(weekly_end, dates),
                          index=dates.astype(str).str.replace('-', ''))

        # determine beg and end business date of each week, and save
        g = weeks.index.astype(int).groupby(weeks)
        self.weeks = DataFrame.from_dict({k: {'beg': min(v), 'end': max(v)}
                                          for k,v in enumerate(g.values())},
                                         orient='index')
        m = (self.weeks['end'] // 100) % 100
        self.weeks['ismonthend'] = m != m.shift(-1) # last week fully in a month
        self.weeks.index.name='numwk'
        self.freq = day

    def date_range(self, start, end, freq='weekly'):
        """Return weekly ending trading dates within start and end range"""
        if isinstance(freq, str) and freq.lower().startswith('w'):
            return self.weeks['end'].iloc[
                self.numwk(start) : self.numwk(end) + 1].to_list()
        return super().date_range(start, end, freq=freq)

    def numwk(self, dates):
        """Return index number of weeks matching to input dates"""
        return np.searchsorted(self.weeks['end'], dates)

    def begwk(self, date, weeks=0):
        """Return beginning business week date/s"""
        return _to_values(self.weeks['beg'].iloc[self.numwk(date) + weeks])

    def endwk(self, date, weeks=0):
        """Return ending business week date/s"""
        return _to_values(self.weeks['end'].iloc[self.numwk(date) + weeks])

    def ismonthend(self, date):
        """If date/s in last complete week in any month"""
        return _to_values(self.weeks['ismonthend'].iloc[self.numwk(date)])
    
if False:  # create custom busday trading dates
    from settings import settings
    from finds.database import SQL
    from finds.busday import BusDay
    sql = SQL(**settings['sql'], echo=True)
    busday = BusDay(sql, create=True)   # set create flag as True
    
if False: # some unit tests
    from settings import settings
    from finds.database import SQL
    from finds.busday import Weekly
    sql = SQL(**settings['sql'], echo=True)
    wd = Weekly(sql, day=3, end=20201231)     # derive weekly trading calendar

    print(wd.numwk(20201230))
    print(wd.numwk(20210130))
    print(wd.numwk(20201231))
    print(wd.endwk([20201209, 20201219]))
    print(wd.endwk(20201209))
    print(wd.endmo([20201209, 20201219]))
    print(wd.endmo(20201209))
    print(wd.weeks.iloc[-1])
