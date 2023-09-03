"""Implement custom trading-day and weekly business date calendars

Copyright 2022, Terence Lim

MIT License
"""
from typing import Iterable, List, Dict, Mapping, Any, Callable, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Timestamp, DatetimeIndex
import pandas_datareader as pdr
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.api.types import is_list_like
from pandas.tseries.offsets import MonthEnd
from sqlalchemy import Column, Integer
from datetime import datetime
from finds.database.sql import SQL
_VERBOSE = 1

# List of anticipated NYSE holidays
_hols = [20230102, 20230116, 20230220, 20230407, 20230529,
         20230619, 20230704, 20230904, 20231123, 20231225,
         20240101, 20240115, 20240219, 20240329, 20240527,
         20240619, 20240704, 20240902, 20241128, 20241225,
         20250101, 20250129, 20250217, 20250418, 20250526, 
         20250619, 20250704, 20250901, 20251127, 20251225] 
_MAXDATE = (max(_hols) // 10000) * 10000 + 1231

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

    >>> _map(busday.offset, [20190526, 20190528], -2)
    >>> _map(busday.begmo, [20190526, 20190528], -2)
    >>> _map(busday.endmo, [20190526, 20190528], -2)
    """
    values = {d: func(d, *args, **kwargs) for d in np.unique(dates)}
    return [values[d] for d in dates]


class BusDay:
    """Implement custom business-day and weekly dates calendar

    Args:
        sql: SQL connection instance to store dates
        start: Start date calendar
        end: End date of calendar
        hols: List of expected future holidays
        end: Ending day of week for weekly calendar: 0-6 or 'Mon'-'Sun'
        new: Recreate from French library datareader, else retrieve from SQL

    Attributes:
        weeks : DataFrame of weeks in rows, index is weeknum and 
                columns for beg, end dates and ismonthend indicator

    Notes:
        Non-trading holidays inferred from Ken French and NYSE websites
    """
    def __init__(self, 
                 sql: SQL, 
                 start: int = 19251231, 
                 end: int = _MAXDATE,
                 hols: List[int] = _hols,
                 endweek: int | str = 0, 
                 new: bool = False, 
                 verbose: int = _VERBOSE):
        """Create or retrieve custom trading dates calendar"""

        self.sql = sql
        self.table = sql.Table('busdates',
                               Column('date', Integer, primary_key=True))
        self.sql.create_all()
        if new: # reload 'F-F_Research_Data_Factors_daily' using pandas reader
            f = pdr.data.DataReader(name='F-F_ST_Reversal_Factor_daily',
                                    data_source='famafrench',
                                    start=1900, end=2050)[0]\
                        .index.sort_values().unique()
            df = DataFrame({'date': BusDay.to_date(f.astype(str), '%Y-%m-%d')})
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
        hols = pd.to_datetime(hols, format='%Y%m%d')
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


        """Derive weekly trading calendar, ending on specified day of week"""
        dates = pd.date_range(self.datetime(19251231),
                              self.datetime(_MAXDATE),
                              freq=self._customcal)
        
        # parse specified day-of-week, and generate weekly calender end dates
        if isinstance(endweek, int):
            endweek = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][endweek % 7]
        if not endweek.startswith('W-'):
            endweek = 'W-' + endweek.upper()[:3]

        weekly_end = pd.date_range(self.datetime(start),
                                   self.datetime(end),
                                   freq=endweek)
        
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
        self.freq = endweek
        
    def datetime(self,
                 date: Any | Iterable[Any],
                 month: int | None = None,
                 day: int | None = None) -> datetime | List[datetime]:
        """Convert int date or pandas timestamp to pydatetime type

        Args:
            date: Input date or year or pandas Timestamp or datetime64
            month: If None, then month is inferred from input date
            day: If None, then day is inferred from input date

        Returns:
            datetime object or list-like with same shape as input date

        Notes:

        - If input day is 00, then returns custom business month begin date
        - If input day is 99, then returns custom business month end date
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
                    return (datetime(date,   # // 100,
                                     month=month,
                                     day=(day or self.day(date))+1)
                            + (0 * self._begmocal)).to_pydatetime()
                except:  # else must be DD == 99, so return custom month end
                    return (datetime(date,   # // 100,
                                     month=month,
                                     day=1)\
                            + (0 * self._endmocal)).to_pydatetime()
        if is_list_like(date):
            return [self.datetime(d) for d in date]
        else:
            return _datetime(date, month, day) 
                       
    def offset(self,
               dates: int | List[int], 
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
                dates=np.array([self.datetime(dates)], dtype='datetime64[D]'),
                offsets=offsets,
                roll=roll,
                busdaycal=self._busdaycal)).strftime('%Y%m%d')[0])
        except:
            return _map(self.offset, dates, offsets, end, roll)

    def date_range(self, start: int, end: int, 
                   freq: str | int = 'daily') -> List[int]:
        """Return business dates at desired freq between beg and end dates

        Args:
            start: Inclusive start of date range
            end: Inclusive end of date range
            freq: If int, then annual ending on month-end business date,
                  else in {'d'-aily, 'b'-egmo, 'e'-ndmo, 'w'-eekly}
                  
        """
        try:
            month = int(freq)           # annually as of calendar month end
            dates = pd.DatetimeIndex(
                [self.datetime(self.year(d), month, 99) 
                 for d in range(self.year(start), self.year(end) + 1)])
        except:
            freq = freq.lower()
            if freq.startswith('w'):   # custom weekly 
                return self.weeks['end']\
                           .iloc[self._numwk(start) : self._numwk(end) + 1]\
                           .to_list()
            elif freq.startswith("d"):    # custom business daily
                dates = pd.date_range(start=self.datetime(start), 
                                      end=self.datetime(end),
                                      freq=self._customcal)
            elif freq.startswith("b"):  # custom business month begin
                dates = pd.date_range(start=self.datetime(start, day=1),
                                      end=str(self.endmo(end)),
                                      freq=self._begmocal)
            elif freq.startswith("e"):  # custom business month end
                dates = pd.date_range(start=self.datetime(start, day=1),
                                      end=str(self.endmo(end)),
                                      freq=self._endmocal)
            else:                       # naive daily calendar
                dates = pd.DatetimeIndex([str(start), str(end)])
        return list(dates.strftime('%Y%m%d').astype(int))

    def date_tuples(self, dates: List[int]) -> List[Tuple[int, int]]:
        """Return (beg, end) holding period between rebalance dates"""
        dates = sorted(dates)
        return [(self.offset(beg, offsets=1), self.offset(end, offsets=0))
                for beg, end in zip(dates[:-1], dates[1:])]

    @staticmethod
    def today() -> int:
        """Return today's int date"""
        return BusDay.to_date(datetime.now())

    @staticmethod
    def to_datetime(arg: Any,
                    format: str = '%Y%m%d',
                    **kwargs) -> Timestamp | DatetimeIndex:
        """Wraps pd.to_datetime to convert string to pandas TimeStamp format"""
        return pd.to_datetime(arg, format=format, **kwargs)

    @staticmethod
    def to_date(dates: Any,
                format: str = '%Y-%m-%d') -> int | List[int]:
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

        >>> to_date('12-31-1999', format='%m-%d-%Y')
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

    def endmo(self, date: int | List[int], months: int = 0) -> int | List[int]:
        """Return (list of) business month end date, optional months offset"""
        return (_map(self.endmo, date, months) if is_list_like(date) else
                int((self.datetime(date, day=1) 
                     + (0 * self._endmocal) 
                     + (months * self._endmocal)).strftime('%Y%m%d')))

    def begmo(self, date: int | List[int], months: int = 0) -> int | List[int]:
        """Return (list of) business month begin date, optional months offset"""
        return (_map(self.begmo, date, months) if is_list_like(date) else
                int((self.datetime(date, day=1)
                     + (0 * self._begmocal)
                     + (months * self._begmocal)).strftime('%Y%m%d')))

    def endyr(self, date: int | List[int], years: int = 0) -> int | List[int]:
        """Return (list of) business year end date, optional years offset"""
        return (_map(self.endyr, date, years) if is_list_like(date) else
                int((self.datetime(date, 12, 1)
                     + (0 * self._endmocal)
                     + (12 * years * self._endmocal)).strftime('%Y%m%d')))

    def begyr(self, date: int | List[int], years: int = 0) -> int | List[int]:
        """Return (list of) business year begin date, optional years offset"""
        return (_map(self.begyr, date, years) if is_list_like(date) else
                int((self.datetime(date, 1, 1)
                     + (0 * self._begmocal)
                     + (12 * years * self._begmocal)).strftime('%Y%m%d')))

    def _numwk(self, dates):
        """Return index number of weeks matching to input dates"""
        return np.searchsorted(self.weeks['end'], dates)


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

    def december_fiscal(self, dates: int | List[int]) -> int | List[int]:
        """Return (list of) Fama-French December fiscal year-end date/s"""
        if is_list_like(dates):
            return _map(self.december_fiscal, dates) 
        return self.endyr(dates, years=(self.month(dates) >= 6) - 2)

    def june_universe(self, dates: int | List[int]) -> int | List[int]:
        """Return (list of) Fama-French June universe selection date/s"""
        if is_list_like(dates):
            return _map(self.june_universe, dates) 
        june = self.endmo(self.endyr(dates), months=-6) 
        return self.endmo(june, months =-12 * (dates < june))

    

if __name__ == "__main__":
    from finds.database import SQL
    from secret import credentials
    VERBOSE = 1


    def test_to_monthend():
        print(BusDay.to_monthend(1999))
        print(BusDay.to_monthend(199901))
        print(BusDay.to_monthend(19990101))
        print(BusDay.to_monthend([19991231, 199901]))

    def test_to_date():
        print(BusDay.to_date('12-31-1999', format='%m-%d-%Y'))
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
        print(bd.weeks)
        print(bd.begwk([20220609, 20220601]))
        print(bd.endwk([20220609, 20220601]))
        print(bd.date_range(20210603, 20220610, freq='weekly'))
        print(bd.date_tuples([20220520, 20220527, 20220603, 20220610]))

    sql = SQL(**credentials['sql'])
    bd = BusDay(sql)
    test_to_date()
    test_to_monthend()
    test_daily()
    test_weekly()


# if False:   # Recreate from French library
    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    bd = BusDay(sql, new=True, verbose=VERBOSE)

    
