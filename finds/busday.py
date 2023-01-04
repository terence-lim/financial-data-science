"""Implement custom trading-day business date calendar 

- Numpy busdaycalendar
- Pandas CustomBusinessDay and offsets
- FamaFrench daily research factors

Daily and Weekly business day classes and methods to:

- Create trading day and weekly calendars from 1925 to present.
- Construct date offsets, rebalance dates and holdings periods with 
  valid trading days at daily, weekly, monthly and annual frequencies. 
- Convert between custom busday, date strings, Pandas Timestamp, and 
  python datetime

Copyright 2022, Terence Lim

MIT License
"""
from typing import Iterable, List, Dict, Mapping, Any, Callable, Tuple
import random
import sys
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Timestamp, DatetimeIndex
import pandas_datareader as pdr
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.api.types import is_list_like
from pandas.tseries.offsets import MonthEnd
from sqlalchemy import Column, Integer
from datetime import datetime
from finds.database import SQL

_VERBOSE = 1
_MAXDATE = 20221231

# List of anticipated NYSE holidays to remove
_hols = ['20220530', '20220704', '20220905', '20221124','20221226']

def _map(func, dates, *args, **kwargs) -> List:
    """Helper classmethod to apply a func to each date in list

    Args:
        func: function to apply
        dates: int dates to apply func on
        *args: list of optional arguments 
        **kwargs: list of keyword arguments

    Notes:

        Func is applied to unique dates, then values copies in order

    Examples:

    >>> busday._map(busday.offset, [20190526, 20190528], -2)
    >>> busday._map(busday.begmo, [20190526, 20190528], -2)
    >>> busday._map(busday.endmo, [20190526, 20190528], -2)
    """
    values = {d: func(d, *args, **kwargs) for d in np.unique(dates)}
    return [values[d] for d in dates]


class BusDay:
    """Implement custom business/trading dates calendar

    Args:
        sql: SQL connection instance to store dates
        start: Start date calendar
        end: End date of calendar
        new: Recreate from Ken French library datareader, else retrieve SQL

    Notes:
        Non-trading holidays inferred from Ken French and NYSE websites
    """
    def __init__(self, 
                 sql: SQL, 
                 start: int = 19251231, 
                 end: int = _MAXDATE, 
                 new: bool = False, 
                 verbose: int = _VERBOSE):
        """Create or retrieve custom trading dates calendar"""

        self.sql = sql
        self.table = sql.Table('busdates',
                               Column('date', Integer, primary_key=True))
        if new: # reload 'F-F_Research_Data_Factors_daily' using pandas reader
            f = pdr.data.DataReader(name='F-F_ST_Reversal_Factor_daily',
                                    data_source='famafrench',
                                    start=1900, end=2050)[0]\
                                        .index.sort_values().unique()
            df = DataFrame({'date': BusDay.to_date(f.astype(str), '%Y-%m-%d')})
            self.table.create(checkfirst=True)
            sql.load_dataframe('busdates', df)
        else:
            df = sql.read_dataframe('SELECT * FROM busdates')

        # 1. Initially, actual dates = actual FamaFrench busdays
        dates = pd.DatetimeIndex(sorted(list(df['date'].unique().astype(str))))
        last = BusDay.to_datetime(df.iloc[-1]['date'])
        if verbose:
            print('Last FamaFrench Date', last)

        # 2. Extend with pandas 5-day calendar from last through to maxdate
        dates = dates.append(pd.date_range(last, 
                                           end=pd.to_datetime(str(end)),
                                           freq='B')[1:])
        # 3. But remove current list of anticipated NYSE holidays
        hols = pd.to_datetime(_hols)
        dates = sorted(set(dates).difference(set(hols)))
        
        # 4. List of all potential busdays from pandas 6-day calendar
        freq = pd.offsets.CDay(calendar=np.busdaycalendar('1111110'), 
                               normalize=True)
        alldates = set(pd.date_range(dates[0], dates[-1], freq=freq))

        # 5. Finalize actual holidays = all potential dates less actual dates
        hols = np.array(list(alldates.difference(dates)), 
                        dtype='datetime64[D]')
        hols = sorted(set(hols).union([np.datetime64('1926-01-01')]))

        # 6. Set custom cal and offsets from 6-day week less actual holidays
        self._busdaycal = np.busdaycalendar(weekmask='1111110', holidays=hols)
        self._customcal = pd.offsets.CDay(calendar=self._busdaycal)
        self._begmocal = pd.offsets.CBMonthBegin(calendar=self._busdaycal)
        self._endmocal = pd.offsets.CBMonthEnd(calendar=self._busdaycal)

    @staticmethod
    def today() -> int:
        return BusDay.to_date(datetime.now())

    @staticmethod
    def to_datetime(arg: Any, format: str = '%Y%m%d',
                    **kwargs) -> Timestamp | DatetimeIndex:
        """Wrapper over pd.to_datetime converts string to TimeStamp format"""
        return pd.to_datetime(arg, format=format, **kwargs)

    @staticmethod
    def to_date(dates: Any, format: str = '%Y-%m-%d') -> int | List[int]:
        """Construct int date from strings using input and output formats

        Args:
            date: Input date strings or Timestamp or pydatetime to convert
            format: Optional format of input date string

        Returns:
            int dates with year, month, date components according to outformat

        Formats specified as in strptime() and strftime():

        -  %b %B %h = input month name
        -  %F = %Y-%m-%d
        -  %T = %H:%M:%S
        -  %n = whitespace
        -  %w %W %U %V = week number
        -  %u = day of week (1-7)

        Examples:

        >>> to_date('12-31-1999', informat='%m-%d-%Y')
        >>> to_date(['1999-01-01', '1999-12-31'])
        >>> int(datetime.strptime(str(19991231), '%Y%m%d').strftime('%Y%m%d'))
        """
        if is_list_like(dates):
            return [BusDay.to_date(s, format) for s in dates]
        if not hasattr(dates, 'strftime'): # not already datetime or Timestamp
            dates = datetime.strptime(str(dates), format)
            return int(dates.strftime('%Y%m%d'))
        return int(dates.strftime('%Y%m%d'))

    @staticmethod
    def to_monthend(dates: int | Iterable[int]) -> int | List[int]:
        """Return calendar monthend date given an int date or list
            
        Args:
            dates: input YYYYMMDD (or first 4-, 6-digits) int date, or list

        Returns:
            Output dates converted to monthend of calendar    
        """

        if is_list_like(dates):
            return [BusDay.to_monthend(d) for d in dates]
        if dates <= 9999:
            dt = datetime(year=dates, month=12, day=1) + MonthEnd(0)
        elif dates <= 999999:
            dt = datetime(year=dates//100, month=dates%100, day=1) + MonthEnd(0)
        else:
            dt = BusDay.to_datetime(dates) + MonthEnd(0)
        return int(dt.strftime('%Y%m%d'))

    @staticmethod
    def year(date: int | Iterable[int]) -> int | List[int]:
        """Helper to extract int years from input date or list"""
        if is_list_like(date):
            return _map(BusDay.year, date)
        while date > 9999:      # input date may be missing month and day
            date //= 100
        return date

    @staticmethod
    def month(date: int | Iterable[int]) -> int | List[int]:
        """Helper to extract int months from input date or list"""
        if is_list_like(date):
            return _map(BusDay.month, date)
        while date > 999999:        # input date may be missing day
            date //= 100
        return date % 100
        
    @staticmethod
    def day(date: int | Iterable[int]) -> int | List[int]:
        """Helper to extract int day of month from input date or list"""
        if is_list_like(date):
            return _map(BusDay.day, date)
        return date % 100    # date may be missing year or month

    def __call__(self, date: Any | Iterable[Any], month: int | None = None,
                 day: int | None = None) -> datetime | List[datetime]:
        """Convert int date or pandas timestamp to pydatetime type

        Args:
            date: Input date or year or pandas Timestamp or datetime64
            month: If None, then month is inferred from input date
            day: If None, then day is inferred from input date

        Returns:
            datetime object or list-like with same shape as input date

        Notes:

        - If input DD is 00, then returns custom business month begin date
        - If input DD is 99, then returns custom business month end date
        """
        def _datetime(date, month, day):
            if hasattr(date, 'to_pydatetime'):  # is pandas timestamp
                return date.to_pydatetime()     #   or datetimeindex
            try:
                return datetime(self.year(date),
                                self.month(date) if month is None else month,
                                self.day(date) if day is None else day)

            except:  # catch if date is out of range of datetime.datetime
                try:     # if DD + 1 is <= 99, so return custom month begin
                    return (datetime(date // 100,
                                     month=month,
                                     day=(day or self.day(date))+1)
                            + (0 * self._begmocal)).to_pydatetime()
                except:  # else must be DD == 99, so return custom month end
                    return (datetime(date // 100, month=month, day=1)\
                            + (0 * self._endmo)).to_pydatetime()
        if is_list_like(date):
            return [self(d) for d in date]
        else:
            return _datetime(date, month, day) 
                       
    def offset(self, dates: int | List[int], 
                     offsets: int = 0,
                     end: int | None = None, 
                     roll: str = 'preceding') -> int | List[int]:
        """Return valid business date with optional offset or roll treatment

        Args:
            dates: Input dates in YYYYMMDD int format
            offsets: Number of business days to offset
            end: End index of offset window, None to return a single date
            roll: How to treat dates that are not a valid day, in 
                  {'raise', 'forward', 'following', 'backward', 'preceding'}
        """
        if end:  # return all dates in window [left, right] around {date}
            if is_list_like(dates):
                return _map(self.offset, dates, offsets, end, roll=roll)
            return [self.offset(dates, offsets=d, roll=roll)
                    for d in np.arange(offsets, end + 1)]
        try:
            return int(pd.to_datetime(np.busday_offset(
                dates=np.array([self(dates)], dtype='datetime64[D]'),
                offsets=offsets,
                roll=roll,
                busdaycal=self._busdaycal)).strftime('%Y%m%d')[0])
        except:
            #return [self.offset(d, offsets, roll=roll) for d in dates]
            return _map(self.offset, dates, offsets, end, roll)

    def endmo(self, date: int | List[int], months: int = 0) -> int | List[int]:
        """Return (list of) business month end date, optional months offset"""
        return (_map(self.endmo, date, months) if is_list_like(date) 
                else int((self(date, day=1) 
                         + (0 * self._endmocal) 
                         + (months * self._endmocal)).strftime('%Y%m%d')))

    def begmo(self, date: int | List[int], months: int = 0) -> int | List[int]:
        """Return (list of) business month begin date, optional months offset"""
        return (_map(self.begmo, date, months) if is_list_like(date) 
                else int((self(date, day=1)
                         + (0 * self._begmocal)
                         + (months * self._begmocal)).strftime('%Y%m%d')))

    def endyr(self, date: int | List[int], years: int = 0) -> int | List[int]:
        """Return (list of) business year end date, optional years offset"""
        return (_map(self.endyr, date, years) if is_list_like(date)
                else int((self(date, 12, 1)
                         + (0 * self._endmocal)
                         + (12 * years * self._endmocal)).strftime('%Y%m%d')))

    def begyr(self, date: int | List[int], years: int = 0) -> int | List[int]:
        """Return (list of) business year begin date, optional years offset"""
        return (_map(self.begyr, date, years) if is_list_like(date)
                else int((self(date, 1, 1)
                         + (0 * self._begmocal)
                         + (12*years*self._begmocal)).strftime('%Y%m%d')))

    def date_range(self, start: int, 
                         end: int, 
                         freq: str | int = 'daily') -> List[int]:
        """Return business dates at desired freq between beg and end dates

        Args:
            start: Inclusive start of date range
            end: Inclusive end of date range
            freq: If int, then annual ending on month-end business date,
                  else in {'daily', 'begmo', 'endmo', int month}
                  
        """
        try:
            month = int(freq)           # annually as of calendar month end
            dates = pd.DatetimeIndex(
                [self(self.year(d), month, 99) 
                 for d in range(self.year(start), self.year(end) + 1)])
        except:
            freq = freq.lower()
            if freq.startswith("d"):    # custom business daily
                dates = pd.date_range(start=self(start), 
                                      end=self(end),
                                      freq=self._customcal)
            elif freq.startswith("b"):  # custom business month begin
                dates = pd.date_range(start=self(start, day=1),
                                      end=str(self.endmo(end)),
                                      freq=self._begmocal)
            elif freq.startswith("e"):  # custom business month end
                dates = pd.date_range(start=self(start, day=1),
                                      end=str(self.endmo(end)),
                                      freq=self._endmocal)
            else:  # daily calendar
                dates = pd.DatetimeIndex([str(start), str(end)])
        return list(dates.strftime('%Y%m%d').astype(int))

    def date_tuples(self, dates: List[int]) -> List[Tuple[int, int]]:
        """Return (beg, end) holding period between rebalance dates"""
        dates = sorted(dates)
        return [(self.offset(beg, offsets=1), self.offset(end, offsets=0))
                for beg, end in zip(dates[:-1], dates[1:])]

    def december_fiscal(self, dates: int | List[int]) -> int | List[int]:
        """Return (list of) prevailing December fiscal year-end date/s"""
        if is_list_like(dates):
            return _map(self.december_fiscal, dates) 
        return self.endyr(dates, years=(self.month(dates) >= 6) - 2)

    def june_universe(self, dates: int | List[int]) -> int | List[int]:
        """Return (list of) prevailing June universe selection date/s"""
        if is_list_like(dates):
            return _map(self.june_universe, dates) 
        june = self.endmo(self.endyr(dates), months=-6) 
        return self.endmo(june, months =-12 * (dates < june))

class WeeklyDay(BusDay):
    """Generate custom weekly trading date calendar, ending on any day-of-week

    Args:
        sql: SQL connection instance
        day: ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'] or [0,...,7]
        beg: starting trading day on or after
        end: ending trading day on or before

    Attributes:
        weeks : DataFrame of weeks in rows, index is weeknum and 
                columns for beg, end dates and ismonthend indicator
    """
    def __init__(self, sql: SQL, day: str | int, beg: int = 19251231, 
                 end: int = 20401231):
        """Derive weekly trading calendar, ending on specified day of week"""
        
        # retrieve daily trading dates from parent
        super().__init__(sql, new=False)
        dates = pd.date_range(self(19251231),
                              self(_MAXDATE),
                              freq=self._customcal)
        
        # parse specified day-of-week, and generate weekly calender end dates
        if isinstance(day, int):
            day = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][day % 7]
        if not day.startswith('W-'):
            day = 'W-' + day.upper()[:3]
        weekly_end = pd.date_range(self(beg), self(end), freq=day)

        
        # require start(exclusve) and end(inclusive), and weekly end in dates
        dates = dates[(dates > weekly_end[0]) & (dates <= weekly_end[-1])]
        weeks = pd.Series(np.searchsorted(weekly_end, dates),
                          index=dates.astype(str).str.replace('-', ''))

        # determine beg and end business date of each week, and save
        g = weeks.index.astype(int).groupby(weeks)
        self.weeks = DataFrame.from_dict(
            {k: {'beg': min(v), 'end': max(v)}
             for k,v in enumerate(g.values())}, orient='index')
        m = (self.weeks['end'] // 100) % 100
        self.weeks['ismonthend'] = m != m.shift(-1)  # last week fully in month
        self.weeks.index.name = 'numwk'
        self.freq = day

    def _numwk(self, dates):
        """Return index number of weeks matching to input dates"""
        return np.searchsorted(self.weeks['end'], dates)

    def date_range(self, start: int, end: int,
                   freq: str | int = 'weekly') -> List[int]:
        """Return weekly ending trading dates within start and end range"""
        if isinstance(freq, str) and freq.lower().startswith('w'):
            return self.weeks['end']\
                .iloc[self._numwk(start) : self._numwk(end) + 1].to_list()
        return super().date_range(start, end, freq=freq)

    def begwk(self, date = int | List[int], weeks: int = 0) -> int | List[int]:
        """Return beginning business week dates, with optional offset"""
        dates = self.weeks['beg'].iloc[self._numwk(date) + weeks]
        return list(dates) if is_list_like(dates) else dates

    def endwk(self, date = int | List[int], weeks: int = 0) -> int | List[int]:
        """Return ending business week dates, with optional offset"""
        dates = self.weeks['end'].iloc[self._numwk(date) + weeks]
        return list(dates) if is_list_like(dates) else dates

    def ismonthend(self, date: int | List[int]) -> int | List[int]:
        """If dates is in last complete week in a month"""
        dates = self.weeks['ismonthend'].iloc[self._numwk(date)]
        return list(dates) if is_list_like(dates) else dates

if __name__ == "__main__":
#    from os.path import dirname, abspath
#    sys.path.insert(0, dirname(dirname(abspath(__file__))))
    from database import SQL
    from conf import credentials, VERBOSE
    VERBOSE = 1

    def update_busday():
        sql = SQL(**credentials['sql'], verbose=VERBOSE)
        bd = BusDay(sql, new=True, verbose=VERBOSE)
    
    def test_to_monthend():
        print(BusDay.to_monthend(1999))
        print(BusDay.to_monthend(199901))
        print(BusDay.to_monthend(19990101))
        print(BusDay.to_monthend([19991231, 199901]))

    def test_to_date():
        print(BusDay.to_date('12-31-1999', informat='%m-%d-%Y'))
        print(BusDay.to_date(['1999-01-01', '1999-12-31']))

    def test_daily():
        sql = SQL(**credentials['sql'], verbose=VERBOSE)
        bd = BusDay(sql)
        print(bd.offset(19990101, 0))
        print(bd.offset(19991231, -2, 3))
        print(bd.offset([19991231, 19990101, 19991231], -2, 3))
        print(BusDay.to_date(datetime(1999, 12, 31)))
        print(bd.december_fiscal(20100331))
        print(bd.december_fiscal(20100831))
        print(bd.june_universe(20100331))
        print(bd.june_universe(20100831))
        print(bd.date_tuples([20220131, 20220228, 20220331, 20220430]))

    def test_weekly():
        wd = WeeklyDay(sql, day=6)
        print(wd.weeks)
        print(wd.begwk([20220609, 20220601]))
        print(wd.endwk([20220609, 20220601]))
        print(wd.date_range(20210603, 20220610))
        print(wd.date_tuples([20220520, 20220527, 20220603, 20220610]))
        return wd
    
    """update
    update_busday()
    sql = SQL(**credentials['sql'])
    bd = BusDay(sql)
    """

    
