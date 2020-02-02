"""
the dives.structured module defines classes for manipulating structured data, stored in sql
"""
# The MIT License
#
# Copyright (c) 2020 Terence Lim (https://terence-lim.github.io/)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation he rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import dives
from dives.util import DataFrame, print_debug, fractiles
import numpy as np
import pandas as pd
import pandas_datareader

#
# BusDates is a class to support business date functions
# 
class BusDates:
    """class to manipulate business dates -- daily, monthly and weekly"""
    
    _schema = {'table' : 'busdates',             # describes schema of table stored in SQL
               'fields': [['date', 'INT(11)']],
               'primary': ['date'],
               'indexes': []}
    _source = 'F-F_Research_Data_Factors_daily'  # source of business dates from Fama-French

    def __init__(self, sql, source=None, start=19260701, end=20190630):
        """Read business dates from sql, get from Ken French website if not exists"""

        if source:  # if source specified, then reload the series using pandas reader
            f = pandas_datareader.data.DataReader(
                name = source, data_source = 'famafrench',
                start = pd.to_datetime(start, format = '%Y%m%d'),
                end = pd.to_datetime(end, format = '%Y%m%d'))[0].index.sort_values()
            df = DataFrame(data = pd.DatetimeIndex.strftime(f, '%Y%m%d').astype(int),
                           columns = ['date'])
            sql.create_table(**BusDates._schema)
            sql.load_dataframe('busdates', df)

        # internally store the business dates, and also generate monthly and weekly lookups
        self._dates = DataFrame(**sql.run('SELECT * FROM busdates')).sort_values('date')
        self._dates.index = self._dates['date']
        self._dates['mo'] = (self._dates.index // 100) % 100
        self._dates['year'] = self._dates.index // 10000
        self.beg = self._dates.index[0]
        self.end = self._dates.index[-1]

        # create monthly dataframe, as lookup
        grouped = self._dates.groupby(['year','mo'], sort=True)
        self._months = grouped.count().rename(columns = {'date' : 'count'})
        self._months['begmo'] = grouped.date.min()
        self._months['endmo'] = grouped.date.max()
        self._months['yearmo'] = self._months['endmo'] // 100
        self._months['mo'] = self._months['yearmo'] % 100
        self._months['year'] = self._months['yearmo'] // 100

        # create weekly dataframe, as lookup
        w = list(pd.date_range(start = pd.to_datetime(19000101, format = '%Y%m%d'),
                               end = pd.to_datetime(20991231, format = '%Y%m%d'), #pd.DateShift(days=6)
                               freq='W-Sat').strftime('%Y%m%d').astype(int))
        self._dates['week'] = np.searchsorted(w, self._dates.index)
        g = self._dates.index.groupby(self._dates.week)
        self._weeks = DataFrame.from_dict({k : {'beg' : min(v), 'end' : max(v)}
                                           for k,v in g.items()}, orient = 'index').sort_index()
        m = (self._weeks['end'] // 100) % 100
        self._weeks['ismonthend'] = list(m != m.shift(-1))
        
    def week_num(self, date):
        """Return index num of week containing date"""
        return self._weeks.index[
            np.searchsorted(self._weeks['end'], self.shift(date, 0))]

    def week_end(self, num):
        """Return end business date of week num"""
        return self._weeks.loc[num, 'end']
    
    def week_beg(self, num):
        """Return beg business date of week num"""
        return self._weeks.loc[num, 'beg']

    def week_range(self, beg=None, end=None):
        """Return list of week nums between beg and end dates, inclusive""" 
        if beg is None:
            beg = self._weeks.index[0]
        else:
            beg = self.week_num(beg)
        if end is None:
            end = self._weeks.index[-1]
        else:
            end = self.week_num(end)
        return list(self._weeks.index[(self._weeks.index >= beg) & (self._weeks.index <= end)])
    
    def map(self, func, dates, n=0):
        """apply {func} to each date in vector of {dates} with {n} as second argument"""
        return list(map(lambda d: func(int(d), int(n)), dates))

    def yearmo(self, date, months=0):
        """Add {months} to date, and convert date to (year, mm) tuple"""
        date = int(date)
        if date > 999999: date //= 100
        result = [(date // 100), date % 100]
        months += result[1] - 1
        return (result[0] + (months // 12), (months % 12) + 1)

    def endmo(self, date, months=0):
        """return end-of-month business date, optionally shift by number of months"""
        return self._months.loc[self.yearmo(int(date), months), 'endmo']

    def begmo(self, date, months=0):
        """return beg-of-month business date, optionally shift by number of months"""
        return self._months.loc[self.yearmo(int(date), months), 'begmo']

    def endmo_range(self, beg, end, month=None):
        """List all monthend business datess between beg/end, optionally particular month of year only"""
        if beg > 999999: beg //= 100
        if end > 999999: end //= 100
        end = min(self._months.yearmo.iloc[-1], end)  # no later than last month
        m = self._months.yearmo.ge(beg) & self._months.yearmo.le(end)
        if month is not None:
            if not isinstance(month, list): month = [month]
            m[~self._months['mo'].isin(month)] = False
        return list(self._months.endmo[m])

    def dates_range(self, beg, end):
        """list of all business dates between beg and end"""
        return list(self._dates.index[(self._dates.index >= beg) & (self._dates.index <= end)])

    def holding_periods(self, dates):
        """beg and end of range of returns holding dates corresponding to each rebalance in {dates}"""
        d = sorted(dates)
        return [(self.shift(b, shift=1), self.shift(e, shift=0)) for b,e in zip(d[:-1], d[1:])]

    def shift(self, date, shift = 0, end = None):
        """business date/s on or before {date}, with optional shift days before (<0) or after(>0)"""
        if end:  # return all dates within window [shift, end] around {date}
            return np.array([self.shift(date, i) for i in  np.arange(shift, end+1)]).T
        return self._dates.index[shift + np.searchsorted(self._dates.index, date, 'right') - 1]

#
# Base class for Structured data sets
#
class Structured(object):
    """Base class for structured data sets, underlying stored in sql.

    Parameters
    ----------
    sql : SQL instance
        connection to mysql database
    dates: BusDates instance
        class provides business date manipulation methods
    """
    _schemas = dict()       # schema for data table stored in sql, keyed by a "virtual" name

    def __init__(self, sql, dates):
        """Initializes Structured super instance"""
        self.dates = dates  # business dates
        self.sql = sql      # sql connection
        self.rdb = None     # redis connection
        self._str = "Structured"

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._str

    def schema(self, virtual=None, item=None):
        """return schema definition of a virtual table as a dict of characteristics"""
        if virtual is None:
            return list(self._schemas.keys())
        if item is None:
            return self._schemas[virtual]
        return self._schemas[virtual][item]

    def _drop_schemas(self):
        """break glass in emergency"""
        for s in self.schema():
            self.sql.drop_table(self.schema(s, 'table'))


    def _read_csv(self, virtual, csvfile, sep=',', quoting=0):
        """Read csvfile into a DataFrame, using field types in the schema 'virtual'

        Parameters
        ----------
        virtual : str
            virtual name of table
        csvfile : string
            csv file name  
        sep : string, optional
            csv file delimiter
        quoting: char, optional
            character for quoting string, 0 if not (default)

        Notes
        -----
        Blank values in boolean and int fields are set to False/0.
        Invalid/blank values in double field are coerced to NaN.
        """
        #
        # 'utf-8' codec can't decode byte 0xf6 => encoding='latin-1'
        df = pd.read_csv(csvfile, sep=sep, quoting=quoting, encoding='latin-1',
                         header=0, low_memory=False, na_filter=False)
        
        df.columns = df.columns.map(str.lower).map(str.rstrip)   # convert column names to lower case
        #df.dropna(subset = df.columns, how = 'any', inplace=True)  # don't drop na
        
        df = df[[f[0] for f in self.schema(virtual, 'fields')]]  # reorder and keep schema columns
    
        if len(self.schema(virtual, 'primary')):   # remove duplicated primary keys
            df.drop_duplicates(subset = self.schema(virtual, 'primary'), keep='first', inplace=True)
        fields = self.schema(virtual, 'fields')    # based on schema, convert dtypes
        for col in [c[0] for c in fields if 'BOOLEAN' in c[1].upper()]:
            df[col] = df[col].replace(['C', 'E', ''], False).astype(bool)
        for col in [c[0] for c in fields if 'INT' in c[1].upper()]:
            df[col] = df[col].replace(['C', 'E', ''], 0).astype(int)
        for col in [c[0] for c in fields if 'DOUBLE' in c[1].upper()]:
            df[col] = pd.to_numeric(df[col], errors = 'coerce')  # floats can be np.nan
        for col in [c[0] for c in fields if 'VARCHAR' in c[1].upper()]:
            df[col] = df[col].astype(str).str.encode('ascii','ignore').str.decode('ascii')
        print_debug('(_read_csv) %d' % (len(df)))
        return df

    def csv_to_sql(self, virtual, csvfile, sep=',', quoting=0):
        """insert ignore into a table from a csvfile to sql.

        Parameters
        ----------
        virtual : string
            virtual name of table
        csvfile : string
            csv file name
        sep : string, optional
            csv file delimiter
        quoting: char, optional
            character for quoting string, 0 if not (default)

        Notes
        -----
        Create new table, if not exists, using schema provided in the class definition
        Insert ignore, i.e. new records with duplicate key are dropped
        """
        df = self._read_csv(virtual, csvfile, sep=sep, quoting=quoting)
        print_debug("(csv_to_sql) %d" % len(df))
        if not self.sql.exists_table(self.schema(virtual, 'table')):
            self.sql.create_table(**self.schema(virtual))
        self.sql.load_dataframe(self.schema(virtual, 'table'), df, index_label=None)
        return df

    def get_series(self, permnos, field, start=19000000, end=29001231, virtual='daily'):
        """Return time series of a field for multiple permnos

        Parameters
        ----------
        permnos: list
          identifiers to filter
        field: string
          name of column to extract
        start: int, optional (default is earliest available)
          inclusive start date (YYYYMMDD)
        end: int, optional (default is latest available
          inclusive end date (YYYYMMDD)
        table: string, optional
          virtual name of table to retrieve ret (default is 'daily' table)

        Returns
        -------
        result: DataFrame (possibly empty)
          values of the desired field, indexed by date, by permnos in columns
        """
        permnos = ", ".join(["'" + str(p) + "'" for p in permnos])
        q = "SELECT date, " \
            " {permno}, " \
            " {field} " \
            " FROM {table}" \
            " WHERE date >= {start} " \
            "  AND date <= {end} " \
            "  AND {permno} IN ({permnos})" \
            "".format(permno = self._id_field,
                      field = field,
                      table = self.schema(virtual, 'table'),
                      start = int(start),
                      end = int(end),
                      permnos = permnos)
        print_debug('(get_series) ' + q)
        return DataFrame(**self.sql.run(q)).pivot(index='date',
                                                  columns=self._id_field,
                                                  values=field)


    def get_ret(self, start, end, permnos = '', virtual='daily', nocache=False):
        """Compound returns between start and end dates

        Parameters
        ----------
        start: int
          inclusive start date (YYYYMMDD)
        end: int
          inclusive end date (YYYYMMDD)
        permnos: list, optional
          list of permnos to select (default is all permnos)
        virtual: string, optional
          virtual name of table to retrieve ret (default is 'daily' table)
        nocache: bool, optional
          to suppress use of rdb cache, set to True (default is False)

        Returns
        -------
        df : DataFrame (possibly empty)
          compounded returns in column 'ret', indexed by permno

        Notes
        -----
        If start and end are first and last business dates of a month, then:
          Search range is expanded to include first and last calendar dates of respective months
          'monthly' table is retrieved (can be defined as different or same physical table in schema)
        """
        if (not nocache and self.rdb):  # If redis is opened, then check key in cache
            rkey = "_".join(["ret", self._str, str(start), str(end)])
            if self.rdb.exists(rkey):
                print_debug('(get_ret) ' + rkey)
                return self.rdb.load(rkey)

        if permnos:
            permnos = ", ".join(["'" + str(p) + "'" for p in permnos])
            permnos = " AND permno IN ({permnos})".format(permnos=permnos)

        if self.is_monthly(start, end):
            table = self.schema('monthly', 'table')
            start = (start//100)*100
            end = (end//100*100)+31
        q = "SELECT EXP(SUM(LOG(1 + ret))) - 1 AS ret, " \
            " permno " \
            " FROM {table} " \
            " WHERE ret IS NOT NULL" \
            "  AND date >= {start} " \
            "  AND date <= {end} " \
            "  {permnos} GROUP BY permno" \
            "".format(table = self.schema(virtual, 'table'),
                      start = start,
                      end = end,
                      permnos = permnos)
        print_debug('(get_ret) ' + q)
        df = DataFrame(**self.sql.run(q)).set_index('permno')   #.reindex(permnos)
        if (not nocache and self.rdb):
            self.rdb.dump(rkey, df)
        return df

    def get_compounded(self, periods, permnos, nocache=True):
        """generate series of compounded returns (in rows) for permnos (in columns)"""
        # accumulate horizontally, then finally transpose -- for monthly appraise
        result = DataFrame(index = permnos)
        for begdate, enddate in periods:
            result[enddate] = self.get_ret(begdate, enddate, nocache = nocache).reindex(permnos)
        result = result.transpose()
        return result

    def get_window(self, virtual, field, permnos, dates, beg, end):
        """Retrieve {field} for {permnos} in a window centered around {dates}

        Parameters
        ----------
        virtual : string
          virtual name of table
        field : string
          name of field to retrieve
        permnos : list
          list of identifiers to retrieve
        dates : list of int
          list of corresponding dates of center of event window 
        beg : int
          relative (inclusive) start of event window
        end : int
          relative (inclusive) end of event window

        Returns:
        --------
        df : DataFrame
          with columns named [beg...end] containing field values in event window
        """
        cols = ["day" + str(i) for i in range(1+end-beg)]
        df = DataFrame(data = [self.dates.shift(d, beg, end) for d in dates], columns=cols)
        df['date'] = list(dates)
        df['permno'] = list(permnos)
        self.sql.load_dataframe(self.sql._temp, df, if_exists = 'replace')
        df = DataFrame({'permno' : list(permnos), 'date' : list(dates)})
        for col in cols:
            q = "SELECT {temp}.permno, " \
                " {temp}.date AS date, " \
                " {field} " \
                " FROM {temp} " \
                " LEFT JOIN {table}" \
                " ON {table}.permno = {temp}.permno " \
                "  AND {table}.date = {temp}.{col}" \
                "".format(temp = self.sql._temp,
                          field = field,
                          table = self.schema(virtual, 'table'),
                          col = col)
            d = DataFrame(**self.sql.run(q))
            df[col] = list(d[field])   # was left join, so assume is in same order
        self.sql.drop_table(self.sql._temp)
        return df

    def get_many(self, virtual, permnos, dates, fields):
        """Retrieve many {fields} for lists of {permnos} and {dates}

        Parameters
        ----------
        virtual : string
          virtual name of table
        permnos : list
          list of identifiers to retrieve
        dates : list of int
          list of corresponding dates of center of event window 
        field : list string
          names of fields to retrieve

        Returns:
        --------
        df : DataFrame
          with columns named [beg...end] containing field values in event window
        """
        field = "`, `".join(list(fields))
        self.sql.load_dataframe(self.sql._temp,
                                DataFrame(data = {'permno' : list(permnos), 'date' : list(dates)}),
                                if_exists = 'replace')
        q = "SELECT {temp}.permno, " \
            " {temp}.date AS date, " \
            " `{field}` " \
            " FROM {temp} " \
            " LEFT JOIN {table}" \
            " ON {table}.permno = {temp}.permno " \
            " AND {table}.date = {temp}.date" \
            "".format(temp = self.sql._temp,
                      field = field,
                      table = self.schema(virtual, 'table'))
        df = DataFrame(**self.sql.run(q))
        self.sql.drop_table(self.sql._temp)
        return df
        
    
    def get_section(self, virtual, fields, date_field, date, start=None):
        """Return a cross-section of fields as of a single date

        Parameters
        ----------
        virtual : string
          virtual name of table to extract from
        fields: list of str 
          names of columns to return
        date_field: str 
          name of date column in the table
        date: int 
          desired date in YYYYMMDD format
        start: int (default is None), optional
          inclusive start of date range to return latest permno row. If None, then only date exactly.

        Returns
        -------
        result: DataFrame (possibly empty)
          indexed by permno (if start=None, then duplicates allowed)

        Note
        ----
        If start is not None, then the latest prevailing record for each
        between (non-inclusive) start and (inclusive) date is returned

        Examples
        --------
        t = get_section(crsp, 'shares', ['shrenddt','shrout'], 'shrsdt', pordate)
        u = get_section(crsp, 'names', ['nameendt','comnam','ticker'], 'date', pordate-10000)
        """
        assert(fields)
        key = self._id_field
        table = self.schema(virtual, 'table')
        if start is None:
            q = "SELECT {fields} " \
                " FROM {table} " \
                " WHERE {date_field} = {date}" \
                "".format(fields = ", ".join(set(fields + [key])),
                          table = table,
                          date_field = date_field,
                          date = date)
        else:
            q = "SELECT {fields} " \
                " FROM {table} JOIN" \
                "  (SELECT {key}, " \
                "    MAX({date_field}) AS {date_field} " \
                "   FROM {table} " \
                "     WHERE {date_field} <= {date} " \
                "      AND {date_field} > {start} " \
                "     GROUP BY {key}) as a" \
                "     USING ({key}, {date_field})" \
                "".format(fields =", ".join(set(fields + [key])),
                          table = table,
                          key = key,
                          date_field = date_field,
                          date = date,
                          start = start)
        print_debug('(get_section) ' + q)
        return DataFrame(**self.sql.run(q)).set_index(key)


    def is_monthly(self, beg, end):
        """Return True if beg and end business dates of any (possibly different) month"""
        if 'monthly' in self.schema():
            begyearmo = self.dates._months.loc[self.dates.yearmo(beg)]
            endyearmo = self.dates._months.loc[self.dates.yearmo(end)]
            return (begyearmo['begmo'] >= beg) and (endyearmo['endmo'] <= end)
        else:
            return False



#
# Benchmarks is subclass of Structured data for benchmark and index returns
#
class Benchmarks(Structured):
    """subclass of structured data for benchmark returns"""

    # benchmark returns from Ken French site, for each benchmark item defined by:
    #   [name, index number (0 is usually value-weighted), suffix (to differentiate if same name)]
    _sources = [('F-F_Research_Data_Factors_daily', 0 , ''),
                ('F-F_Research_Data_Factors', 0 , '(mo)'),    # append "(mo)" to monthly returns version
                ('F-F_Research_Data_5_Factors_2x3_daily', 0, ''),
                ('F-F_Research_Data_5_Factors_2x3', 0, '(mo)'),
                ('F-F_Momentum_Factor_daily', 0, ''),
                ('F-F_Momentum_Factor', 0, '(mo)'),
                ('F-F_ST_Reversal_Factor_daily', 0 , ''),
                ('F-F_ST_Reversal_Factor', 0 , '(mo)'),
                ('F-F_LT_Reversal_Factor_daily', 0 , ''),
                ('F-F_LT_Reversal_Factor', 0 , '(mo)'),
                ('49_Industry_Portfolios_daily', 0, '49vw'),  # append suffix to permno name
                ('48_Industry_Portfolios_daily', 0, '48vw'),  #  to differentiate value-weighted
                ('49_Industry_Portfolios_daily', 1, '49ew'),  #  vs equal-weighted of same benchmark
                ('48_Industry_Portfolios_daily', 1, '48ew')]

    # For benchmarks, main table of returns is referred by usual virtual name 'daily';
    # 'monthly' virtual table also points to same -- monthly returns have permnos with suffix "(mo)"
    _schemas = {'daily' : {'table': 'benchmarks',
                           'fields': [['permno', 'VARCHAR(32)'],
                                      ['date', 'INT(11)'],
                                      ['ret', 'DOUBLE']],
                           'primary': ['permno','date'],
                           'indexes': [['date','permno']]},
                'ident' : {'table' : 'benchident',
                           'fields' : [['permno','VARCHAR(32)'],
                                       ['source','VARCHAR(64)'],
                                       ['item','VARCHAR(8)']],
                           'primary': ['source', 'item', 'permno'],
                           'indexes': []},
                'monthly' : {'table': 'benchmarks'}}

    def __init__(self, sql, dates, rdb=None):
        """Return instance of a benchmarks structured data set """
        super().__init__(sql, dates)
        self._str = 'Benchmarks'
        self._id_field = 'permno'      # field name containing permanent identifier
        self.rdb = rdb
        for virtual in self.schema():  # create table if necessary
            if not self.sql.exists_table(self.schema(virtual, 'table')):
                self.sql.create_table(**self.schema(virtual))

    def append_FF(self, source, item=0, suffix = '', start=19270101, end=29001231):
        """Load FamaFrench benchmarks source from pandas_datareader to sql"""
        f = pdr.data.DataReader(source, 'famafrench',
                                start = pd.to_datetime(max(start, self.dates.beg), format='%Y%m%d'),
                                end = pd.to_datetime(min(end, self.dates.end), format='%Y%m%d'))
        f[item].index = pd.DatetimeIndex.strftime(f[item].index, '%Y%m%d').astype(int),
        ident = DataFrame(columns = ['source', 'item'])
        result = DataFrame(columns = ['source', 'numdays', 'date', 'nameendt'])
        for c in f[item].columns:
            df = f[item][[c]]
            df = df[(df[c] > -99.99) & (df.index > 0)] / 100  ### more generally to filter out bad obs!
            df.columns = ['ret']
            permno = c.rstrip() + suffix
            df['permno'] = permno
            self.sql.load_dataframe(self.schema('daily', 'table'), df, index_label='date')
            self.sql.insert(self.schema('ident', 'table'),
                            {'permno': permno, 'source': source, 'item': str(item)})
            result.loc[permno] = [source, len(df), df.index[0], df.index[-1]]
        return result

#
# PSTAT is subclass of Structured data, for Compustat Annual, Quarterly and Key Development
#
# missing: pstkl, pstkrv, txp
# criteria: f.indfmt='INDL' and f.datafmt='STD' and f.popsrc='D' and f.consol='C'
#
class PSTAT(Structured):
    """subclass for Compustat structured data: Annual, Quarterly or Key Developments
    
    Parameters
    ----------
    sql : SQL instance
        connection to SQL database
    dates: BusDates instance
        business dates class
    """
    _schemas = {'links' : {'table' : 'links',
                           'fields': [
                               ['gvkey', 'INT(11)'],    # '1000' - '319507'  (int64)
                               ['conm', 'VARCHAR(30)'],
                               ['tic', 'VARCHAR(8)'],    # 'AEF.MV.A' @ 44204 (object)
                               ['cusip', 'VARCHAR(9)'],    # '000032102' @ 0 (object)
                               ['cik', 'INT(11)'],    # '0000723576' @ 3 (object)
                               ['sic', 'SMALLINT DEFAULT 0'],    # '100' - '9998'  (int64)
                               ['naics', 'INT(11) DEFAULT 0'],    # '442110' @ 9 (object)
                               ['linkprim', 'VARCHAR(1)'],    # 'C' @ 0 (object)
                               ['liid', 'VARCHAR(3)'],    # '00X' @ 0 (object)
                               ['linktype', 'VARCHAR(2)'],    # 'NU' @ 0 (object)
                               ['lpermno', 'INT(11) DEFAULT 0'], # '25881' @ 1 (object)
                               ['lpermco', 'INT(11) DEFAULT 0'], # '23369' @ 1 (object)
                               ['linkdt', 'INT(11) DEFAULT 0'],  # '19460101' - '20171229'  (int64)
                               ['linkenddt', 'INT(11) DEFAULT 0']],    # '19700929' @ 0 (object)
                           'primary': ['gvkey','linkdt','lpermno'],
                           'indexes': [['linkdt','gvkey','lpermno'],
                                       ['cusip','linkdt','lpermno'],
                                       ['linkdt','cusip','lpermno'],
                                       ['cik','linkdt','lpermno'],
                                       ['linkdt','cik','lpermno'],
                                       ['lpermno','linkdt','cusip'],
                                       ['lpermno','linkdt','cik'],
                                       ['lpermno','linkdt','gvkey']]},
                'annual': {'table': 'annual',
                           'fields': [
                               ['gvkey', 'INT(11)'],    # '1004' - '326688'  (int64)
                               ['datadate', 'INT(11)'], # '19900630' - '20180430' (int64)
                               ['fyear', 'SMALLINT'],    # '1990' @ 0 (object)
                               ['indfmt', 'VARCHAR(4)'],    # 'INDL' @ 0 (object)
                               ['consol', 'VARCHAR(1)'],    # 'C' @ 0 (object)
                               ['popsrc', 'VARCHAR(1)'],    # 'D' @ 0 (object)
                               ['datafmt', 'VARCHAR(3)'],    # 'STD' @ 0 (object)
                               ['cusip', 'VARCHAR(9)'],    # '000361105' @ 0 (object)
                               ['fyr', 'TINYINT']] +    # '10' @ 27 (object)
                           [[s, 'DOUBLE'] for s in   ######  'pstkrv','pstkl', 'txp'
                            ['aco','act','ao','ap','at','capx','ceq','che','cogs','csho',
                             'cshrc','dcpstk','dcvt','dlc','dltt','dm','dp','drc','drlt',
                             'dv','dvt','ebit','ebitda','emp','fatb','fatl','gdwl','gwo',
                             'ib','intan','invt','lco','lct','lo','lt','ni','nopi','oancf',
                             'ob','pi','ppegt','ppent','pstk','rect','revt','sale','scstkc',
                             'spi','txditc','txfed','txfo','txt','xad','xint','xrd','xsga']]
                           + [['cik', 'BIGINT'],    # '0000001750' @ 0 (object)
                              ['prcc_f', 'DOUBLE'],    # '11' @ 0 (object)
                              ['costat', 'VARCHAR(1)'],    # 'A' @ 0 (object)
                              ['naics', 'INT(11) DEFAULT 0'],    # '423860' @ 0 (object)
                              ['sic', 'SMALLINT DEFAULT 0']],    # '100' - '9997'  (int64)
                           'primary': ['gvkey','datadate','fyear','fyr',
                                       'indfmt','consol','popsrc','datafmt'],
                           'indexes': [['datadate','gvkey','fyear','fyr',
                                        'indfmt','consol','popsrc','datafmt']]},
                'quarterly': {'table': 'quarterly',
                              'fields': [
                                ['gvkey', 'INT(11)'],    # '1003' - '326688'  (int64)
                                  ['datadate', 'INT(11)'],  # '19900131' - '20180531'
                                  ['fyearq', 'SMALLINT'],    # '1989' - '2018'  (int64)
                                  ['fqtr', 'TINYINT'],    # '1' - '4'  (int64)
                                  ['indfmt', 'VARCHAR(4)'],    # 'INDL' @ 0 (object)
                                  ['consol', 'VARCHAR(1)'],    # 'C' @ 0 (object)
                                  ['popsrc', 'VARCHAR(1)'],    # 'D' @ 0 (object)
                                  ['datafmt', 'VARCHAR(3)'],    # 'STD' @ 0 (object)
                                  ['cusip', 'VARCHAR(9)'],    # '000354100' @ 0 (object)
                                  ['datacqtr', 'VARCHAR(6)'],    # '1989Q4' @ 0 (object)
                                  ['datafqtr', 'VARCHAR(6)'],    # '1989Q4' @ 0 (object)
                                  ['rdq', 'INT(11) DEFAULT 0']] +    # '19900307' @ 4 (object)
                              [[s, 'DOUBLE'] for s in 
                               ['actq','atq','ceqq','cheq','cogsq','cshoq','dlcq','ibq','lctq',
                                'ltq','ppentq','pstkq','pstkrq','revtq','saleq','seqq','txtq','xsgaq']]
                              + [['cik', 'BIGINT'],    # '0000730052' @ 0 (object)
                                 ['costat', 'VARCHAR(1)'],    # 'I' @ 0 (object)
                                 ['prccq', 'DOUBLE'],    # '13278.0000' @ 27195 (object)
                                 ['naics', 'INT(11) DEFAULT 0'],    # '442110' @ 0 (object)
                                 ['sic', 'SMALLINT DEFAULT 0']],    # '100' - '9997'  (int64)
                              'primary': ['gvkey','datadate','fyearq','fqtr','consol'],
                              'indexes': [['datadate','gvkey','fyearq','fqtr','consol']]},
                'keydev': {'table': 'keydev',
                           'fields': [['keydevid', 'INT(11)'],
                                      ['companyid', 'INT(11)'],
                                      ['companyname', 'VARCHAR(100)'], 
                                      ['keydeveventtypeid', 'SMALLINT'],
                                      ['keydevstatusid', 'TINYINT'],
                                      ['keydevtoobjectroletypeid', 'TINYINT'],
                                      ['announcedate', 'INT(11)'],
                                      ['enterdate', 'INT(11)'],
                                      ['gvkey', 'INT(11)']],
                           'primary': ['gvkey','announcedate','keydeveventtypeid',
                                       'keydevtoobjectroletypeid','keydevid'],
                           'indexes': [['announcedate','keydeveventtypeid'],
                                       ['keydeveventtypeid','keydevtoobjectroletypeid'],
                                       ['gvkey','keydeveventtypeid']]},
                'supply': {'table': 'supply',
                           'fields': [
                               ['gvkey', 'INT(11)'],     # Supplier GVKEY
                               ['conm', 'VARCHAR(29)'],  # Supplier Name
                               ['cgvkey', 'INT(11)'],    # Customer GVKEY
                               ['cconm', 'VARCHAR(28)'], # Customer Current Name
                               ['cnms', 'VARCHAR(50)'],  # Customer Name
                               ['srcdate', 'INT(11)'],   # Source Date - Segment Customer
                               ['cid', 'SMALLINT'],      # Customer Identifier
                               ['sid', 'SMALLINT'],      # Customer Segment Identifier Link
                               ['ctype', 'VARCHAR(7)'],  # Customer Type
                               ['salecs', 'DOUBLE'],     # Customer Sales
                               ['scusip', 'VARCHAR(9)'], # Supplier CUSIP
                               ['stic', 'VARCHAR(5)'],   # Supplier Ticker Symbol
                               ['ccusip', 'VARCHAR(9)'], # Customer CUSIP
                               ['ctic', 'VARCHAR(5)']],  # Customer Ticker Symbol
                           'primary': ['gvkey','srcdate','cgvkey'],
                           'indexes': [['cgvkey','srcdate','gvkey'],
                                       ['srcdate','gvkey','cgvkey'],
                                       ['srcdate','cgvkey','gvkey']],
                           'constraints': []}}

    role = {   # Key Development role id labels
        1: 'Target',
        2: 'Advisor',
        3: 'Buyer',
        4: 'Seller',
        5: 'Transaction',
        6: 'Transaction Consideration',
        7: 'Lender',
        8: 'Participant',
        9: 'TradingItemId',
        10: 'Auditor',
        11: 'Sponsor'}
    event = {   # Key Development event id labels
        1 : 'Seeking to Sell/Divest',            # may be "not sell"
        3 : 'Seeking Acquisitions/Investments',  # 
        5 : 'Seeking Financing/Partners',        # too general, includes bank mentions
        7 : 'Bankruptcy - Other',                # good: includes contemplates and motions
        11 : 'Delayed SEC Filings',              # good
        12 : 'Delistings',                       # good, but careful of microcap
        16 : 'Executive/Board Changes - Other',
        21 : 'Discontinued Operations/Downsizings',
        22 : 'Strategic Alliances',
        23 : 'Client Announcements',
        24 : 'Regulatory Agency Inquiries',
        25 : 'Lawsuits & Legal Issues',
        26 : 'Corporate Guidance - Lowered',
        27 : 'Corporate Guidance - Raised',
        28 : 'Announcements of Earnings',
        29 : 'Corporate Guidance - New/Confirmed',
        31 : 'Business Expansions',
        32 : 'Business Reorganizations',
        36 : 'Buybacks',
        41 : 'Product-Related Announcements',
        42 : 'Debt Financing Related',
        43 : 'Restatements of Operating Results',
        44 : 'Labor-related Announcements',
        45 : 'Dividend Affirmations',
        46 : 'Dividend Increases',
        47 : 'Dividend Decreases',
        48 : 'Earnings Calls',
        49 : 'Guidance/Update Calls',
        50 : 'Shareholder/Analyst Calls',
        51 : 'Company Conference Presentations',
        52 : 'M&A Calls',
        53 : 'Stock Splits & Significant Stock Dividends',
        54 : 'Stock Dividends (<5%)',
        55 : 'Earnings Release Date',
        56 : 'Name Changes',
        57 : 'Exchange Changes',
        58 : 'Ticker Changes',
        59 : 'Auditor Going Concern Doubts',
        60 : 'Address Changes',
        61 : 'Delayed Earnings Announcements',
        62 : 'Annual General Meeting',
        63 : 'Considering Multiple Strategic Alternatives',
        64 : 'Ex-Div Date (Regular)',
        65 : 'M&A Rumors and Discussions',
        #    68 : 'Credit Rating - S&P - Upgrade',
        #    69 : 'Credit Rating - S&P - Downgrade',
        #    70 : 'Credit Rating - S&P - Not-Rated Action',
        #    71 : 'Credit Rating - S&P - New Rating',
        #    72 : 'Credit Rating - S&P - CreditWatch/Outlook Action',
        73 : 'Impairments/Write Offs',
        74 : 'Debt Defaults',
        75 : 'Index Constituent Drops',
        76 : 'Legal Structure Changes',
        77 : 'Changes in Company Bylaws/Rules',
        78 : 'Board Meeting',
        79 : 'Fiscal Year End Changes',
        80 : 'M&A Transaction Announcements',
        81 : 'M&A Transaction Closings',
        82 : 'M&A Transaction Cancellations',
        83 : 'Private Placements',
        85 : 'IPOs',
        86 : 'Follow-on Equity Offerings',
        87 : 'Fixed Income Offerings',
        88 : 'Derivative/Other Instrument Offerings',
        89 : 'Bankruptcy - Filing',
        90 : 'Bankruptcy - Conclusion',
        91 : 'Bankruptcy - Emergence/Exit',
        92 : 'End of Lock-Up Period',
        93 : 'Shelf Registration Filings',
        94 : 'Special Dividend Announced',
        95 : 'Index Constituent Adds',
        97 : 'Special/Extraordinary Shareholders Meeting',
        99 : 'Potential Privatization of Government Entities',
        100 : 'Ex-Div Date (Special)',
        101 : 'Executive Changes - CEO',
        102 : 'Executive Changes - CFO',
        #    103 : 'LCD Institutional Loan News',
        #    104 : 'LCD Trend News',
        #    105 : 'LCD Fallen Angel News',
        #    106 : 'LCD Debtor-in-possession News',
        #    107 : 'LCD Middle Market News',
        #    108 : 'LCD High-Yield Bond Story News',
        #    109 : 'LCD Leveraged Buyout News',
        #    110 : 'LCD People Story News',
        #    111 : 'LCD Sponsored Deal News',
        #    112 : 'LCD M&A News',
        #    113 : 'LCD Distressed News',
        #    114 : 'LCD Break Price News',
        #    115 : 'LCD Investment Grade Loan News',
        #    116 : 'LCD Repricing News',
        #    117 : 'LCD Dividend News',
        #    118 : 'LCD Repayment News',
        #    119 : 'LCD Mezzanine Debt News',
        #    120 : 'LCD Second-lien News',
        #    121 : 'LCD High-yield Europe News',
        #    122 : 'LCD Covenant-lite News',
        #    123 : 'LCD Cross-border Deal News',
        #    124 : 'LCD CLO News',
        #    125 : 'LCD Secondary Story News',
        #    127 : 'LCD Amendment News',
        #    128 : 'LCD Communications News',
        #    129 : 'LCD European News',
        #    130 : 'LCD Price-flex News',
        #    131 : 'LCD Global News',
        #    132 : 'LCD Ratings News',
        134 : 'Composite Units Offerings',
        135 : 'Structured Products Offerings',
        136 : 'Public Offering Lead Underwriter Change',
        137 : 'Spin-Off/Split-Off',
        138 : 'Announcements of Sales/Trading Statement',
        139 : 'Sales/Trading Statement Calls',
        140 : 'Sales/Trading Statement Release Date',
        #    141 : 'LCD Bids Wanted in Competition',
        #    142 : 'LCD Company Buys Back Outstanding Bank Debt',
        #    143 : 'LCD Debt Exchange',
        144 : 'Estimated Earnings Release Date (CIQ Derived)',
        #    145 : 'LCD Loan Credit Default Swap News',
        #    146 : 'LCD Credit Defaults Swap News',
        #    147 : 'LCD Default News',
        #    148 : 'LCD Deal Launch News',
        149 : 'Conferences',
        150 : 'Auditor Changes',
        151 : 'Buyback Update',
        152 : 'Potential Buyback',
        153 : 'Bankruptcy - Asset Sale/Liquidation',
        154 : 'Bankruptcy - Financing',
        155 : 'Bankruptcy - Reorganization',
        156 : 'Investor Activism - Proposal Related',
        157 : 'Investor Activism - Activist Communication',
        160 : 'Investor Activism - Target Communication',
        163 : 'Investor Activism - Proxy/Voting Related',
        164 : 'Investor Activism - Agreement Related',
        172 : 'Investor Activism - Nomination Related',
        177 : 'Investor Activism - Financing Option from Activist',
        187 : 'Investor Activism - Supporting Statements',
        192 : 'Analyst/Investor Day',
        194 : 'Special Calls',
        205 : 'Regulatory Authority - Regulations',
        206 : 'Regulatory Authority - Compliance',
        207 : 'Regulatory Authority - Enforcement Actions',
        #    208 : 'Macro: Releases',
        #    209 : 'Macro: General',
        #    210 : 'Macro: Auctions',
        #    211 : 'Macro: Seminars',
        #    212 : 'Macro: Holidays',
        213 : 'Dividend Cancellation',
        214 : 'Dividend Initiation',
        215 : 'Preferred Dividend',
        #    216 : 'S&P Events',
        #    217 : "Not a Keydev - Only for Timeline"
        218 : "Announcement of Interim Management Statement",
        219 : "Operating Results Release Date",
        220 : "Interim Management Statement Release Date",
        221 : "Operating Results Calls",
        222 : "Interim Management Statement Calls",
        223 : "Fixed Income Calls",
        224 : "Halt/Resume of Operations - Unusual Events",
        225 : "Corporate Guidance - Unusual Events",
        226 : "Announcement of Operating Results",
        230 : "Buyback - Change in Plan Terms",
        231 : "Buyback Tranche Update",
        232 : "Buyback Transaction Announcements",
        233 : "Buyback Transaction Cancellations",
        234 : "Buyback Transaction Closings"}

    
    def __init__(self, sql, dates):
        super().__init__(sql, dates)
        self._id_field = 'gvkey'   # field name for unique identifier
        self._str = "PSTAT"

    # get_linked to include sort_values=[], ascending=False???
    def get_linked(self, table='keydev', date_field='announcedate', 
                   fields=['companyname', 'keydeveventtypeid', 'keydevtoobjectroletypeid'],
                   where='', limit=''):
        """query pstat table, and also returned linked crsp permno

        Parameters
        ----------
        table: string
            virtual name of pstat table to query
        date_field: string
            name of date field in pstat table to query
        fields : list of string
            fields to return
        where : string
            sql where clause
        limit : int
            maximum number of records to return

        Examples
        --------
        >>> df = pstat.get_linked(table='annual', date_field='datadate',
                 fields=['ceq','pstkrv','pstkl','pstk'],
                 where='ceq > 0 and datadate>=19930104 and datadate<=20991231')

        Notes
        -----
        select keydev.companyname, keydev.keydeveventtypeid, keydev.keydevtoobjectroletypeid,  
          keydev.announcedate, keydev.gvkey, lpermno as permno
        from keydev left join links
          on keydev.gvkey = links.gvkey and links.linkdt =
            (select max(c.linkdt) as linkdt from links c
             where c.gvkey = keydev.gvkey and c.linkdt <= keydev.announcedate)
        where lpermno is not null and keydev.gvkey > 0 and links.gvkey > 0
          and announcedate >= 20180301
        limit 100;    
        """
        fields = ", ".join(["{table}.{field}".format(table=table, field=f.lower())
                            for f in list(set(fields + [self._id_field, date_field]))])
        if where:
            where = " and " + where
        if limit:
            limit = " limit " + str(limit)
        q = "SELECT {links}.{link_perm}, " \
            " {links}.{link_date}, " \
            " {fields} "\
            " FROM {table} LEFT JOIN {links} " \
            " ON {table}.{key} = {links}.{key} " \
            "  AND {links}.{link_date} = " \
            "   (SELECT MAX(c.{link_date}) AS {link_date} " \
            "    FROM {links} c" \
            "    WHERE c.{key} = {table}.{key} " \
            "     AND (c.{link_date} <= {table}.{date_field} " \
            "      OR c.{link_date} = 0)) " \
            " WHERE {link_perm} > 0 " \
            "  AND {table}.{key} > 0 {where} {limit}" \
            "".format(links = self.schema('links', 'table'),
                      link_perm = 'lpermno',
                      link_date = 'linkdt',
                      fields = fields,
                      table = table, 
                      key = self._id_field,
                      date_field = date_field,
                      where = where,
                      limit = limit)

        print_debug("(get_linked) " + q)
        return DataFrame(**self.sql.run(q)).rename(columns={'lpermno' : 'permno'})
    


#
# IBES is subclass of Structured for IBES data
#
class IBES(Structured):
    """subclass for IBES analyst estimates structured data
    
    Parameters
    ----------
    sql : SQL instance
        connection to SQL database
    dates: BusDates instance
        business dates class
    """

    _schemas = {'summary': {'table': 'summary',
                            'fields': [
                                ['ticker', 'VARCHAR(6)'],    # 'BKHT/1' @ 170725 (object)
                                ['statpers', 'INT(11)'],    # '19930114' - '20180419'  (int64)
                                ['measure', 'VARCHAR(3)'],    # 'EPS' @ 0 (object)
                                ['fpi', 'VARCHAR(1)'],    # '1' - '1'  (int64)
                                ['numest', 'TINYINT'],    # '1' - '62'  (int64)
                                ['numup', 'TINYINT'],    # '0' - '52'  (int64)
                                ['numdown', 'TINYINT'],    # '0' - '54'  (int64)
                                ['medest', 'DOUBLE'],    # '294117647.06' @ 224845 (object)
                                ['meanest', 'DOUBLE'],    # '294117647.06' @ 224845 (object)
                                ['stdev', 'DOUBLE'],    # '780324451.75' @ 262885 (object)
                                # ['highest', 'DOUBLE'],    # '294117647.06' @ 224845 (object)
                                # ['lowest', 'DOUBLE'],    # '294117647.06' @ 224845 (object)
                                ['fpedats', 'INT(11)'],    # '19851231' - '20190331'  (int64)
                                ['actual', 'DOUBLE'],
                                ['anndats_act' , 'INT(11)']],
                            'primary': ['fpi','ticker','statpers','fpedats'],
                            'indexes': [['fpi','statpers','ticker','fpedats'],
                                        ['ticker','statpers','fpi','fpedats']]},
                'ident': {'table': 'ident',
                          'fields': [
                              ['ticker', 'VARCHAR(6)'], # 'BKHT/1' @ 9706 (object)
                              ['cusip', 'VARCHAR(8)'],  # '87482X10' @ 0 (object)
                              ['oftic', 'VARCHAR(8)'],  # 'ABBV W' @ 998 (object)
                              ['cname', 'VARCHAR(32)'], # 'AMERICAN CAPITAL SENIOR FLOATING' @ 4
                              ['dilfac', 'TINYINT'],
                              ['pdi', 'VARCHAR(1)'],    # 'P' @ 0 (object)
                              ['ccopcf', 'VARCHAR(1)'],
                              ['tnthfac', 'TINYINT'],   # '0' - '1'  (int64)
                              ['instrmnt', 'VARCHAR(1)'],
                              ['exchcd', 'VARCHAR(2)'],
                              ['country', 'VARCHAR(1)'],
                              ['compflag', 'VARCHAR(1)'],
                              ['usfirm', 'TINYINT'],
                              ['sdates', 'INT(11)']],   # '19760115' - '20180419'  (int64)
                          'primary': ['ticker','sdates'],
                          'indexes': [['sdates','ticker']]},
                'adjust': {'table': 'adjust',
                           'fields': [
                               ['ticker', 'VARCHAR(6)'],    # 'BKHT/1' @ 3083 (object)
                               ['oftic', 'VARCHAR(6)'],    # 'ABBV W' @ 587 (object)
                               ['statpers', 'INT(11)'],    # '19930114' - '20180419'  (int64)
                               ['adjspf', 'DOUBLE']],    # '0' - '2400'  (float64)
                           'primary': ['ticker','statpers'],
                           'indexes': [['statpers','ticker']]},
                'surprise': {'table': 'surprise',
                             'fields': [
                                 ['ticker', 'VARCHAR(6)'],    # 'BKHT/1' @ 63625 (object)
                                 ['oftic', 'VARCHAR(6)'],    # 'ABSR.C' @ 8471 (object)
                                 ['measure', 'VARCHAR(3)'],    # 'EPS' @ 0 (object)
                                 ['fiscalp', 'VARCHAR(3)'],    # 'QTR' @ 0 (object)
                                 ['pyear', 'SMALLINT'],    # '1990' - '2018'  (int64)
                                 ['pmon', 'TINYINT'],    # '1' - '12'  (int64)
                                 ['anndats', 'INT(11)'],    # '19920421' - '20180419'  (int64)
                                 ['actual', 'DOUBLE'],    # '-54670.32967' @ 81908 (object)
                                 ['surpmean', 'DOUBLE'],    # '-23351.64835' @ 81908 (object)
                                 ['surpstdev', 'DOUBLE'],    # '169096865.73' @ 96954 (object)
                                 ['suescore', 'DOUBLE']],    # '-3475.46316' @ 2230 (object)
                             'primary': ['ticker','anndats','fiscalp','measure'],
                             'indexes': [['anndats','ticker','fiscalp','measure']]},
                'links': {'table': 'identlink',
                          'fields': [['ticker', 'VARCHAR(6)'],
                                     ['sdates', 'INT(11)'],
                                     ['permno', 'INT(11)'],
                                     ['date', 'INT(11)'],
                                     ['cname', 'VARCHAR(32)'],
                                     ['comnam', 'VARCHAR(32)'],
                                     ['cusip', 'VARCHAR(8)']],
                          'primary': ['ticker','sdates'],
                          'indexes': [['permno','sdates'], ['sdates','ticker']]}}
                
    def __init__(self, sql, dates):
        super().__init__(sql, dates)
        self._id_field ='ticker'
        self._str = "IBES"

    def write_links(self):
        """derive and save reference table by merging IBES 'ident' and CRSP 'names' on cusip-8"""
        self.sql.drop_table(self.schema('names', 'table'))
        self.sql.create_table(**self.schema('links'))
        q = "INSERT INTO {}" \
            " SELECT ident.ticker, ident.sdates, permno, date, comnam, cname, ident.cusip" \
            " FROM ident LEFT JOIN names ON ident.cusip = names.ncusip AND names.date = " \
            "  (SELECT MAX(date) FROM names c WHERE c.ncusip=ident.cusip AND c.date<=ident.sdates)" \
            "".format(self.schema('links', 'table'))
        print_debug("(write_links) " + q)
        self.sql.run(q, fetch=False)
        q = "SELECT SUM(ISNULL(permno)) AS missing, COUNT(*) AS count FROM identlink"
        return DataFrame(**self.sql.run(q, fetch=True))

    def get_linked(self, table='ident', date_field = 'statpers', fields=['cname'],
                   where='', limit=''):
        """query ibes data, and join reference table (by ibes ticker) to get CRSP permno"""
        fields = ", ".join(["{table}.{field}".format(table=table, field=f.lower())
                            for f in list(set(fields + [self._id_field, date_field]))])
        if where:
            where = " and " + where
        if limit:
            limit = " limit " + str(limit)
        q = "SELECT {links}.{link_perm}, " \
            " {links}.{link_date}, " \
            " {fields} " \
            " FROM {table} LEFT JOIN {links} " \
            "  ON {table}.{key} = {links}.{key} " \
            "   AND {links}.{link_date} = " \
            "    (SELECT MAX(c.{link_date}) AS {link_date} " \
            "     FROM {links} AS c " \
            "     WHERE c.{key} = {table}.{key} " \
            "      AND c.{link_date} <= {table}.{date_field}) " \
            " WHERE {link_perm} IS NOT NULL {where} {limit}" \
            "".format(links = self.schema('links', 'table'), 
                      link_perm = 'permno',
                      link_date = 'sdates',
                      fields = fields,
                      table = table, 
                      key = self._id_field,
                      date_field = date_field,
                      where = where,
                      limit = limit)
        print_debug("(IBES get_linked) " + q)
        return DataFrame(**self.sql.run(q, fetch=True))
    
    # where fpi='6'  /* 1 is for annual forecasts, 6 is for quarterly */
    # and statpers < ANNDATS_ACT /* only keep summarized forecasts prior to earnings annoucement
    # and measure='EPS' and not missing(medest)
    # and not missing(fpedats)  and (fpedats-statpers)>=0;
    # (fpedats-statpers)>=0;
    
#
# CRSP is sub class of Structured Data
#
class CRSP(Structured):
    """sub class for CRSP structured data

    Parameters
    ----------
    sql : SQL instance
        connection to mysql database
    dates: BusDates instance
        business dates object
    rdb : Redis instance, optional (default None)
        redis connection used by some classes methods to cache common query results
    """
    _schemas = {'daily' : {'table': 'daily',
                           'fields': [['permno','INT(11)'],
                                      ['date','INT(11)'],
                                      ['bidlo','DOUBLE'],
                                      ['askhi','DOUBLE'],
                                      ['prc','DOUBLE'],
                                      ['vol','DOUBLE'],
                                      ['ret','DOUBLE'],
                                      ['bid','DOUBLE'],
                                      ['ask','DOUBLE'],
                                      ['shrout','INT(11)'],
                                      ['openprc','DOUBLE']],
                           'primary': ['permno','date'],
                           'indexes': [['date','permno']]},
                'shares' : {'table': 'shares',
                            'fields': [['permno', 'INT(11)'],
                                       ['shrout', 'INT(11) DEFAULT 0'],
                                       ['shrsdt','INT(11)'],
                                       ['shrenddt','INT(11)']],
                            'primary': ['permno','shrsdt','shrenddt'],
                            'indexes': [['permno','shrenddt','shrsdt'],
                                        ['shrsdt','permno'],['shrenddt','permno']]},
                'delist' : {'table': 'delist',
                            'fields': [['permno', 'INT(11)'],
                                       ['dlstdt', 'INT(11)'],
                                       ['dlstcd', 'SMALLINT'],    # '100' - '591'  (int64)
                                       ['nwperm', 'INT(11)'],    # '0' - '99841'  (int64)
                                       ['nwcomp', 'INT(11)'],    # '0' - '90044'  (int64)
                                       ['nextdt', 'INT(11)'],    # 'VARCHAR(8)' '19870612' @ 0
                                       ['dlamt', 'DOUBLE'],    # '0' - '2349.5'  (float64)
                                       ['dlretx', 'DOUBLE'],    # 'DOUBLE' '-0.003648' @ 3
                                       ['dlprc', 'DOUBLE'],    # '-1315' - '2349.5'  (float64)
                                       ['dlpdt', 'INT(11)'],    # 'VARCHAR(8)' '19870612' @ 0
                                       ['dlret', 'DOUBLE']],    # 'DOUBLE' '-0.003648' @ 3
                            'primary': ['permno','dlstdt','dlstcd'],
                            'indexes': [['dlstdt','permno','dlstcd']]},
                'dist' : {'table': 'dist',
                          'fields': [['permno', 'INT(11)'],
                                     ['distcd', 'SMALLINT'],
                                     ['divamt', 'DOUBLE'],
                                     ['facpr', 'DOUBLE'],
                                     ['facshr', 'DOUBLE'],
                                     ['dclrdt', 'INT(11) DEFAULT 0'],
                                     ['exdt', 'INT(11)'],
                                     ['rcrddt', 'INT(11) DEFAULT 0'],
                                     ['paydt', 'INT(11) DEFAULT 0'],
                                     ['acperm', 'INT(11) DEFAULT 0'],
                                     ['accomp', 'INT(11) DEFAULT 0']],
                          'primary': ['permno','distcd','exdt'],
                          'indexes': [['distcd','exdt','permno'],
                                      ['permno','exdt','distcd'],
                                      ['exdt','distcd','permno']]},
                'names' : {'table': 'names',
                           'fields': [['date', 'INT(11)'],
                                      ['comnam', 'VARCHAR(32)'],
                                      ['ncusip', 'VARCHAR(8)'],
                                      ['shrcls', 'VARCHAR(1)'],
                                      ['ticker', 'VARCHAR(5)'],
                                      ['permno', 'INT(11)'],
                                      ['nameendt', 'INT(11) DEFAULT 0'],
                                      ['shrcd', 'TINYINT DEFAULT 0'],
                                      ['exchcd', 'TINYINT DEFAULT 0'],
                                      ['siccd', 'SMALLINT DEFAULT 0'],
                                      ['tsymbol', 'VARCHAR(7)'],
                                      ['naics', 'INT(11) DEFAULT 0'],
                                      ['primexch', 'VARCHAR(1)'],
                                      ['trdstat', 'VARCHAR(1)'],
                                      ['secstat', 'VARCHAR(4)'],
                                      ['permco', 'INT(11) DEFAULT 0']],
                           'primary': ['permno','date'],
                           'indexes': [['date','permno'],
                                       ['ncusip','date'],
                                       ['date','ncusip']]},
                'monthly' : {'table': 'monthly',
                             'fields': [['permno','INT(11)'],
                                        ['date','INT(11)'],
                                        ['ret','DOUBLE']],
                             'primary': ['permno','date'],
                             'indexes': [['date','permno']]}}

    def __init__(self, sql, dates, rdb=None):
        super().__init__(sql, dates)
        self._id_field ='permno'
        self._str = "CRSP"
        self.rdb = rdb

    def get_cap(self, date):
        """Compute cross-section of market capitalization as of date (YYYYMMDD)"""
        if (self.rdb):
            rkey = "_".join(["cap", self._str, str(date)])
            if self.rdb.exists(rkey):
                print_debug('(get_cap) ' + rkey)
                return self.rdb.load(rkey)
        permnos = list(self.get_section('daily', ['permno'], 'date', date).index)
        print_debug('LENGTH PERMNOS = %d' % len(permnos))
        prc = self.get_section('daily', ['permno','prc'], 'date', date).reindex(permnos)
        print_debug('NULL PRC = %d' %  sum(prc.prc.isnull()))
        shr = self.get_section('shares', ['shrout'], 'shrsdt', date, 0).reindex(permnos)
        print_debug('NULL SHR = %d' % sum(shr.shrout.isnull()))
        df = pd.Series(shr['shrout'] * prc['prc'].abs())
        print_debug('NULL CAP = %d' % sum(df.isnull()))
        df = df[df>0]
        if (self.rdb):
            self.rdb.dump(rkey, df)
        return df

    def get_universe(self, date):
        """Return valid universe of permnos as of date

        Parameters
        ----------
        date: int
          desired rebalance date (YYYYMMDD)

        Returns
        -------
        df : DataFrame
          valid universe of stocks, indexed by permno; with market cap "decile" (1..10), 
            "nyse" bool, and "siccd", "prc" and "cap" in columns

        Notes
        -----
        market cap must be available on date, with prc > 0.0
        shrcd isin [10, 11], exchcd isin [1, 2, 3]
        
        """
        if (self.rdb):
            rkey = "_".join(["universe", self._str, str(date)])
            if self.rdb.exists(rkey):
                print_debug('(get_universe rdb) ' + rkey)
                return self.rdb.load(rkey)
        df = self.get_section('daily',
                              ['permno', 'prc', 'shrout'],
                              'date',
                              date)
        df['cap'] = df['shrout'] * df['prc'].abs()
        
        df = df.join(self.get_section('names',
                                      ['shrcd', 'exchcd', 'siccd', 'naics'],
                                      'date',
                                      date,
                                      0).reindex(df.index),
                     how = 'left')
        print_debug('LENGTH PERMNOS = ' + str(len(df)))
        print_debug('PRC NULL:{} NEG:{} '.format(sum(df.prc.isnull()), sum(df.prc.le(0))))
        print_debug('SHR NULL:{} ZERO:{} '.format(sum(df.shrout.isnull()), sum(df.shrout.le(0))))
        print_debug('CAP NULL:{} ZERO:{} '.format(sum(df.cap.isnull()), sum(df.cap.le(0))))
        minprc = 0.0
        df.drop(df.index[~(df.cap.notnull() &
                           df.cap.gt(0) &
                           df.prc.abs().gt(minprc) &
                           df.shrcd.isin([10, 11]) &
                           df.exchcd.isin([1, 2, 3]))], inplace=True)
        df['nyse'] = df['exchcd'] == 1
        df['deciles'] = fractiles(df['cap'], np.arange(10,100,10), df.loc[df['nyse'], 'cap'])
        df = df[['cap','deciles','nyse','siccd', 'prc', 'naics']]   # columns to keep
        if (self.rdb):
            self.rdb.dump(rkey, df)
        return df

    def get_divamt(self, start, end):
        """Accmumulate total dividends

        Parameters
        ----------
        start: int
          inclusive start date (YYYYMMDD)
        end: int
          inclusive end date (YYYYMMDD)

        Returns
        -------
        df : DataFrame (possibly empty)
          accumulated divamt = per share divamt * shrout, indexed by permno

        """
        q = "SELECT dist.permno AS permno, " \
            " SUM(daily.shrout*dist.divamt) AS divamt " \
            "FROM dist INNER JOIN daily " \
            " ON daily.permno = dist.permno AND " \
            "  daily.date = dist.exdt " \
            " WHERE dist.divamt > 0 " \
            "  AND dist.exdt >= {start} " \
            "  AND dist.exdt <= {end} group by permno " \
            "".format(start = start, end = end)
        return DataFrame(**self.sql.run(q, fetch=True)).set_index('permno')


    def get_dlstret(self, start, end, nocache=False):
        """Compound the delisting returns between start and end dates

        Parameters
        ----------
        start: int
          inclusive start date (YYYYMMDD)
        end: int
          inclusive end date (YYYYMMDD)
        nocache: bool, optional
          to suppress use of rdb cache, set to True (default is False)

        Returns
        -------
        df : DataFrame (possibly empty)
          compounded returns in column 'ret', indexed by permno
        """
        if (not nocache and self.rdb):  # If redis is opened, then check key in cache
            rkey = "_".join(["dlst", self._str, str(start), str(end)])
            if self.rdb.exists(rkey):
                print_debug("(get_dlstret) " + rkey + " " + self._str)
                return self.rdb.load(rkey)

        q = "SELECT EXP(SUM(LOG(1 + dlret))) - 1 AS ret, " \
            " permno FROM {table} " \
            " WHERE dlret IS NOT NULL " \
            "  AND dlstdt >= {start} " \
            "  AND dlstdt <= {end} GROUP BY permno" \
            "".format(table = self.schema('delist', 'table'),
                      start = start,
                      end = end)
        print_debug('(get_dlst) ' + q)
        df = DataFrame(**self.sql.run(q, fetch=True)).set_index('permno')
        if (not nocache and self.rdb):
            self.rdb.dump(rkey, df)
        return df
    

    def portfolio_sorts(self, label, data, beg, end, window=0, month=0, minobs=50):
        """Generate monthly time series of holdings by standard sort procedure
        
        Parameters
        ----------
        label : string
            name of signal, to retrieve either from Signal sql table or {data} dataframe
        data : Signals, or DataFrame with columns ['permno', 'rebaldate', label]
            values for the signal (most recent in each window will be used if duplicated)
        beg : int
            first rebalance date (YYYYMMDD)
        end : int
            last holding date (YYYYMMDD)
        window: int, optional (default 0)
            number of months to look back for signal values. 0 (default) is exact date
        month: int, optional
            month (e.g. 6 = June) to retrieve universe and market cap. 0 (default) means every month
        minobs: int, optional
            minimum required universe size with signal values
        """
        print_debug('(portfolio_sorts) get_section {}'.format(hasattr(data, 'get_section')))
        if hasattr(data, 'get_section'):    # assumes that Signals class identified by this hasattr
            signals = data   # if query by window from sql instead, then make next line: data=None
            data = signals.load(label,
                                beg = self.dates.endmo(beg, months = -abs(window)),
                                end = end)
        pordates = self.dates.endmo_range(beg, end)    # generate monthend dates
        holdings = {pordates[-1] : DataFrame()}        # no need to generate holding on last date
        for pordate in pordates[:-1]:
            df = self.get_universe(pordate if month == 0  # whether to reset universe only on selected month
                                   else self.dates.endmo_range(pordate-10000, pordate, month)[-1])
            start = self.dates.endmo(pordate, months = -abs(window)) if window else pordate
            if data is None:   # get signal values for this window from SQL
                signal = signals.get_section(label, pordate, start=start)
            else:              # else from dataframe
                signal = data.loc[data['rebaldate'].le(pordate)                 # select rebaldates
                                  & data['rebaldate'].ge(start),                # within window of pordate
                                  ['permno','rebaldate', label]].sort_values(   # keep latest rebaldate
                                      by = ['permno', 'rebaldate']).drop_duplicates(
                                          ['permno'], keep = 'last').set_index('permno')
            if len(signal):
                df[label] = signal[label].reindex(df.index)
                df = df[df[label].notnull()]
                if len(df) >= minobs:
                    df['fractile'] = fractiles(df[label], [30,70], df[label][df.nyse])
                    permnos, weights = [],[]
                    subs = [(df.fractile == 3) & (df.deciles > 5),          # big high subportfolio
                            (df.fractile == 3) & (df.deciles <= 5),         # small high subportfolio
                            (df.fractile == 1) & (df.deciles > 5),          # big low subportfolio
                            (df.fractile == 1) & (df.deciles <= 5)]         # small low subportfolio
                    for sub, weight in zip(subs, [0.5, 0.5, -0.5, -0.5]):   # combine subportfolios
                        weights += list(weight * df.loc[sub,'cap'] / df.loc[sub,'cap'].sum())
                        permnos += list(df.index[sub])
                    holdings[pordate] = DataFrame(data=weights, index=permnos, columns=['weights'])
                    print_debug("(portfolio_sorts) %d %d" % (pordate, len(permnos)))
        return holdings


#
# subclass of Structured for derived signals data values
#
class Signals(Structured):
    """ subclass of Structured data for derived signal values

    Parameters
    ----------
    sql : SQL instance
        connection to SQL database
    dates: BusDates instance, optional (default None)
        business dates object
    """

    def __init__(self, sql, dates=None):
        super().__init__(sql, dates)
        self._id_field ='permno'
        self._str = 'Signals'

    def schema(self, signal, item=None):
        """overwrite method to return schema for the signal name"""
        _schema = {'table': '__' + signal,   # dynamically generate new table name, prefixed "__"
                   'fields': [['permno','INT(11)'],
                              ['rebaldate','INT(11)'],
                              [signal, 'DOUBLE']],
                   'primary': ['permno','rebaldate'],
                   'indexes': [['rebaldate', 'permno']]}
        if item is None:
            return _schema
        return _schema[item]
    
    def drop(self, label):
        """drop the table associated with given signal {label}"""
        self.sql.drop_table('__' + label)

    def summary(self, signal):
        """perform a proc summary by rebaldate on a signal's values"""
        return self.count_table('__' + signal, signal, key='rebaldate')

    def save(self, out, label, append=False):
        """write new sql table from dataframe of signal values
        
        Parameters
        ----------
        out : DataFrame
            signal values, with columns = ['permno','rebaldate',label]
        label : string
            name of signal. becomes name of column and of table (prefixed by '__')
        append : bool, optional (default is False)
            if True: append to table ignoring duplicate keys. False to recreate table

        Returns
        -------
        n : int
            number of rows saved

        Notes
        -----
        first removes duplicate keys, and drops null rows before saving to table
        """
        out.index.name = None # 'permno' is both index level and column label, which may be ambiguous

        # drop duplicated (permno, rebaldate)
        df = out[['permno','rebaldate',label]].sort_values(by=['permno','rebaldate',label])
        df.drop_duplicates(['permno','rebaldate'], keep='first', inplace=True) # NaNs last
        df = df.loc[np.isfinite(df[label].astype(float))]
        if not(append) or not(self.sql.exists_table(self.schema(label, 'table'))):
            self.sql.create_table(**self.schema(label))   # drop and (re)create table
        self.sql.load_dataframe(self.schema(label, 'table'), df, index_label=None)
        print_debug("(signals_write) %s %d" % (label, len(df)))
        return len(df)

    def load(self, label, beg = 19000101, end = 21001231):
        """read dataframe of signal values from sql
        
        Parameters
        ----------
        label : string
            name of signal
        beg : int, optional
            earliest inclusive date
        end : int, optional
            latest inclusive date

        Returns
        -------
        df : DataFrame
            columns = ['permno', 'rebaldate', label]
        """
        q = "SELECT permno, " \
            " rebaldate, " \
            " {label} " \
            " FROM __{label}" \
            "  WHERE rebaldate >= {beg} " \
            "   AND rebaldate <= {end}" \
            "".format(label = label,
                      beg = beg,
                      end = end)
        return DataFrame(**self.sql.run(q)).sort_values(['permno', 'rebaldate'])

    def get_section(self, label, date, start=None):
        """return cross-section of prevailing signal values available as of date

        Parameters
        ----------
        label : string
            name of signal to retrieve
        date : int
            rebalance date (YYYYMMDD)
        start : int (YYYYMMDD), optional
            non-inclusive date of start of range to search for signal value (default is exact date)
        """
        if not start:
            start = date - 1  # no (non-inclusive) start date specified, so set to prior day (hence exact)
        q = "SELECT permno, " \
            " rebaldate, "\
            " {label} "\
            " FROM __{label}" \
            "  WHERE rebaldate > {start} " \
            "   AND rebaldate <= {date}" \
            "".format(label = label,
                      start = start,
                      date = date)
        df = DataFrame(**self.sql.run(q)).sort_values(['permno', 'rebaldate'])
        return df.drop_duplicates(['permno'], keep='last').set_index('permno')

        
if __name__ == "__main__":
    import time
    
    # clean key dev:
    # original quoted label, comma-delimited file fails on pd.read_csv,
    # so use csv.reader and io.String(), and csv.writer pipe-delimited newfile
    # year = 2019
    csvfile = inpath + str(year) + '.csv.gz'
    newfile = outpath + str(year) + '.csv'

    with gzip.open(csvfile, mode = "r") as f:  # read original file as list of lines
        lines = f.readlines()  # encoding="ISO-8859-1" "latin-1"

    with open(newfile, "w") as g:
        c = csv.writer(g, delimiter="|", lineterminator="\n")  # requires file object input
        items = lines[0].decode('latin-1').split(',')
        c.writerows(list(map(str.rstrip, [items[:12]])))
        for line in lines[1:]:
            items = list(csv.reader(io.StringIO(line.decode('latin-1'), newline=None)))
            if len(items) and len(items[0]) >= 12:     # csv.reader returns list of rows
                #  print('*', items[0][0], items[0][0].isnumeric(), len(items))
                if items[0][0].isnumeric() and items[0][1].isnumeric() and \
                   items[0][3].isnumeric() and items[0][4].isnumeric() and \
                   items[0][5].isnumeric() and items[0][6].isnumeric() and \
                   items[0][8].isnumeric():
                    c.writerows([items[0][:12]])

