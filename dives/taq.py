"""
the dives.taq module defines classes and methods for manipulating TAQ tick data
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
from dives.util import DataFrame, print_debug
import numpy as np
import pandas as pd
import indexed_gzip as igzip
import gzip
import io, pickle, time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  # for date formatting in plots

try:
    import secret
    verbose = secret.value('verbose')
except:
    verbose = 0
    
def chunk_to_df(chunk, columns=None):
    """convert csv text chunk to dataframe with correct dtypes"""

    # define know dtypes for each TAQ file
    _dtypes = {'nbbo' : {np.uint64 : ['Time','Sequence_Number','Participant_Timestamp',
                                      'FINRA_ADF_Timestamp'],
                         np.float32 : ['Bid_Price','Bid_Size','Offer_Price',
                                       'Offer_Size','Best_Bid_Price','Best_Bid_Size',
                                       'Best_Offer_Price','Best_Offer_Size']},
               'trade' : {np.uint64 : ['Time','Sequence_Number','Participant_Timestamp',
                                       'Trade_Reporting_Facility_TRF_Timestamp'],
                          np.uint8 : ['Trade_Correction_Indicator',
                                      'Trade_Through_Exempt_Indicator'],
                          np.float32 : ['Trade_Volume','Trade_Price']},
               'mast' : {np.uint8 : ['TradedOnNYSEMKT','TradedOnNASDAQBX','TradedOnNSX',
                                     'TradedOnFINRA','TradedOnISE','TradedOnEdgeA',
                                     'TradedOnEdgeX','TradedOnCHX','TradedOnNYSE',
                                     'TradedOnArca','TradedOnNasdaq','TradedOnCBOE',
                                     'TradedOnPSX','TradedOnBATSY','TradedOnBATS',
                                     'TradedOnIEX'],
                         np.uint16 : ['Unit_Of_Trade','Round_Lot',
                                      'Specialist_Clearing_Number','Specialist_Post_Number'],
                         np.uint32 : ['Shares_Outstanding', 'Effective_Date'],
                         np.uint64 : ['Unit_Of_Trade','Round_Lot',
                                      'Specialist_Clearing_Number','Specialist_Post_Number',
                                      'TradedOnNYSEMKT','TradedOnNASDAQBX','TradedOnNSX',
                                      'TradedOnFINRA','TradedOnISE','TradedOnEdgeA',
                                      'TradedOnEdgeX','TradedOnCHX','TradedOnNYSE',
                                      'TradedOnArca','TradedOnNasdaq','TradedOnCBOE',
                                      'TradedOnPSX','TradedOnBATSY','TradedOnBATS',
                                      'TradedOnIEX','Effective_Date']}}

    # convert the input
    df = pd.read_csv(io.StringIO(chunk), sep='|', na_filter=False, dtype=str, 
                     header=None if columns else 0, names = columns if columns else None)

    # guess file type from column names, and use its dtypes dict
    if 'Best_Bid_Price' in df.columns:  # guess NBBO file
        dtypes = _dtypes['nbbo']
    elif 'Trade_Price' in df.columns:   # guess TRADE file
        dtypes = _dtypes['trade']
    elif 'Round_Lot' in df.columns:     # guess MASTER file
        dtypes = _dtypes['mast']
    else:
        dtypes = {}

    # coerce each column to its required dtype
    for dtype in dtypes:
        df[dtypes[dtype]] = df[dtypes[dtype]].apply(pd.to_numeric, errors='coerce')
        df[dtypes[dtype]] = df[dtypes[dtype]].fillna(0).astype(dtype)

    # for nbbo and trade: set index to be timestamp. for master: set index to be symbol
    if 'Time' in df.columns:
        df.index = list(pd.to_datetime(df.Time.astype(str), format = '%H%M%S%f'))
    elif 'Symbol' in df.columns:
        df.index = list(df.Symbol)
        
    # TAQ raw data file ended with dummy 'END' symbol        
    if 'END' in df.index:
        df.drop('END', inplace=True)
    return df

    
class TAQ(object):
    """base class to manipulate a daily TAQ .csv.gz file

    Parameters
    ----------
    filename : string
        the .csv.gz raw data file name
    index_file : string, optional (default None)
        the saved indexed-gzip index file that was previously build and exported, if any

    Notes
    -----
    NYSE historical sample files: ftp://ftp.nyxdata.com/Historical%20Data%20Samples/

    The raw daily TAQ csv.gz file can be simultaneously accessed in three different ways:

    .read() - convert and return entire file as DataFrame (should only do this for smaller Master file)

    .iter() - returns an iterator, that returns next chunk (i.e. rows with same stock symbol) as DataFrame

    .open(symbol_file) - opens raw file so a chunk (all rows for same symbol)
        can be directly accessed the .get(symbol) method.
        The symbol_file must have been pre-generated with the .create_symbols(symbol_file) method.
        The IndexedGzipFile package is used under the hood: 
           run .create_index(index_file) to pre-build index to improve access time

    Examples
    --------
    """
    def __init__(self, filename, index_file=None):
        self.filename = filename
        self.index_file = index_file   # use indexed-gzip index file, if any

    def read(self):
        """reads entire file (should only call for master file)"""
        with gzip.open(self.filename, "rt", encoding='utf-8',errors='ignore') as f:
            chunk = f.read()
        return chunk_to_df(chunk).drop_duplicates(subset=['Symbol'])

    def create_index(self, index_file):
        """generate and save an indexed-gzip index file"""
        tic = time.time()
        with igzip.IndexedGzipFile(self.filename) as f:
            f.build_full_index()
            f.export_index(index_file)    # save to file with this name {index_file}
        self.index_file = index_file
        print('Full Index created in %d secs' % (time.time() - tic))

    def create_symbols(self, symbol_file):
        """generate and save symbol lookup locations in the csv.gz file"""
        tic = time.time()        
        with igzip.IndexedGzipFile(self.filename) as f:
            f.seek(0)                                  # goto start of file
            line = f.readline().decode('latin-1')      # read first line and parse as header
            header = line.rstrip('\n').split('|')
            symbols = dict()                           # we will compute (start, size) for each symbol
            prev = None
            eof = lambda line: (not line) or line.startswith('END')   # for detecting if end-of-file
            while not eof(line):                       # iterate while not reached end-of-file
                tell = f.tell()                        # record current location
                line = f.readline().decode('latin-1')  # parse next line
                if not eof(line):
                    curr = line.rstrip('\n').split('|')[header.index('Symbol')]
                else:
                    curr = None
                if curr != prev:                       # if line has new symbol, then update location
                    if prev:
                        symbols[prev] = (symbols[prev][0], tell-symbols[prev][0])
                    if curr:
                        symbols[curr] = (tell,0)
                    prev = curr
        print('%d symbols generated: %d secs' % (len(symbols), time.time() - tic))
        with open(symbol_file, 'wb') as out:           # save dict as file named {symbol_file}
            pickle.dump(symbols, out)

    def open(self, symbol_file):
        """open indexed gzip file for direct read access, where locations are in {symbol_file}"""

        class IGZ(object):
            """Context manager procotol supporting statement: with open"""
            def __init__(self, filename, index_file, symbol_file):
                print_debug("__init__ called")
                with open(symbol_file, 'rb') as g:
                    self.symbols = pickle.load(g)
                print_debug(symbol_file + " : " + str(len(self.symbols)))
                if index_file:
                    self._f = igzip.IndexedGzipFile(filename, index_file=index_file)
                else:
                    self._f = igzip.IndexedGzipFile(filename)
                self.header = self._f.readline().decode('latin-1').replace(' ','_').rstrip('\n').split('|')
                print_debug(self.header)

            def __enter__(self):
                print_debug("__enter__ called")
                return self

            def __exit__(self, exc_type, exc_value, exc_traceback):
                print_debug("__exit__ called")
                return self.close()

            def close(self):
                self._f.close()
                return True

            def get(self, symbol):
                """jump to {symbol}'s data and direct read"""
                loc = self.symbols.get(symbol, None)
                if loc and loc[1] > 0:
                    self._f.seek(loc[0])
                    lines = self._f.read(loc[1]).decode('latin-1')
                    df = chunk_to_df(lines, columns=self.header)
                    df.index.name = symbol
                    return df
                return None
        return IGZ(self.filename, self.index_file, symbol_file)

    def iter(self):
        """return iterator for sequential access"""
        f = gzip.open(self.filename, "rt", encoding='utf-8',errors='ignore')
        line = f.readline()
        self.header = line.rstrip('\n').replace(' ','_').split('|')

        def lines_iterator(f):
            try:
                for line in f:
                    yield line
            except Exception as e:
                print("*** Got exception (lines_iterator):", e)
            yield None

        lines = lines_iterator(f)
        line = next(lines)                # read, parse and keep first line as header
        symbol = line.rstrip('\n').split('|')[self.header.index('Symbol')]
        while line and not line.startswith('END'):    # check if reached end-of-file
            self.symbol = symbol
            chunk = ["|".join(self.header) + "\n"]
            while line and symbol == self.symbol:     # read lines until new symbol
                chunk += [line]
                line = next(lines)
                if line:
                    symbol = line.rstrip('\n').split('|')[self.header.index('Symbol')]
            df = chunk_to_df("".join(chunk))          # convert this chunk to a dataframe
            df.index.name = self.symbol               # all of whose rows have same symbol
            yield df                                  # and yield to caller
        f.close()
        self.symbol = None
        yield None

_open_datetime = pd.to_datetime('1900-01-01T09:30')    # default open  9:30am
_close_datetime = pd.to_datetime('1900-01-01T16:00')   # default close 4:00pm
# np.datetime64('1900-01-01T09:30:01.000002') - np.timedelta64('30','ns')

def clean_trade(df, open_datetime = _open_datetime, close_datetime=_close_datetime):
    """remove bad trades"""
    if df is None:
        return None
    f = (df['Trade_Correction_Indicator'].eq(0) &
         df['Trade_Price'].gt(0)
         )
    g = (df.index >= open_datetime) & (df.index < close_datetime)        # within trading hours
    df = df.loc[f & g, ['Trade_Price','Trade_Volume']]
    return df

def clean_nbbo(df):
    """remove bad quotes"""
    if df is None:
        return None
    cols = ['Best_Bid_Price','Best_Bid_Size','Best_Offer_Price','Best_Offer_Size','Mid_Price']  # to keep
    df['Mid_Price'] = (df['Best_Offer_Price'] + df['Best_Bid_Price']) / 2  # compute midquote
    f = (df['Quote_Cancel_Correction'].ne('B') &
         df['Quote_Condition'].isin(['A','B','H','O','R','W']) &         
         df['Bid_Price'].gt(0) &
         df['Bid_Size'].gt(0) &
         df['Offer_Price'].gt(0) &
         df['Offer_Size'].gt(0) &
         df['Best_Bid_Price'].gt(0) &
         df['Best_Bid_Size'].gt(0) &
         df['Best_Offer_Price'].gt(0) &
         df['Best_Offer_Size'].gt(0) &
         (df['Best_Offer_Price'] > df['Best_Bid_Price']) &
         (df['Best_Offer_Price'] - df['Best_Bid_Price'] < df['Mid_Price']*0.05)
         # (df['Offer_Price'] - df['Bid_Price']).le(5)
         )
    df = df.loc[f, cols + ['Time','Sequence_Number']]      # drop bad conditions, values
    df = df.sort_values(['Time','Sequence_Number'])        # keep later of same time
    df = df.drop_duplicates(subset='Time', keep='last')[cols]
    df = df.loc[(df.shift() != df).any(axis=1)]            # keep new records only
    return df

def measure_trade(ct, cq):
    """compute sign of each trade, and associated impact metrics:

    Interleave with quote in force at -1 nanosecond. and at +5 minutes
    compute: tick test
    compute: sign test (Lee-Ready)
    compute: effective spread 2*D(P-M)/M
    compute: realized spread 2D(P-M+5)/M
    compute: price impact 2D(M+5-M)/M

    References:
    Holden and Jacobsen (2014), "Liquidity Measurement"
    """
    # align with trades with quotes prevailing at -1 nanosecond and +5 minutes
    prevailing = cq.reindex(ct.index - np.timedelta64(-1, 'ns'), method='pad')
    forward = cq.reindex(ct.index + np.timedelta64(5, 'm'), method='pad')
    aligned = {k : prevailing[k].values for k in prevailing}
    aligned['Forward'] = forward['Mid_Price'].values
        
    # tick test 
    direction = np.sign((ct['Trade_Price'] - ct['Trade_Price'].shift(+1)))
    direction[np.isnan(direction)] = 0
    
    # Lee and Ready trade-type
    leeready = np.sign(ct['Trade_Price'] - aligned['Mid_Price'])
    leeready[direction != 0] = direction[direction != 0]
    leeready[np.isnan(leeready)] = 0

    # Spreads: effective, realized, price impact
    effective = 2 * leeready*(ct['Trade_Price'] - aligned['Mid_Price']) / aligned['Mid_Price']
    realized  = 2 * leeready*(ct['Trade_Price'] - aligned['Forward']) / aligned['Mid_Price']
    impact    = 2 * leeready*(aligned['Forward'] - aligned['Mid_Price']) / aligned['Mid_Price']
        
    return DataFrame(data = dict(effective = effective,
                                 realized = realized,
                                 impact = impact,
                                 price = ct['Trade_Price'],
                                 volume = ct['Trade_Volume']),
                     index = ct.index)

def measure_liquidity(ct, cq, minutes=15, open_datetime = _open_datetime, close_datetime=_close_datetime):
    """calculate {minutes}-binned liquidity metrics"""
    rule = '{minutes}Min'.format(minutes=minutes)  # resample rule interval length
    result = DataFrame()

    # set {intervals} to be quote times, supplemented by timestamps at every timedelta as necessary
    timedelta = np.timedelta64(minutes, 'm')
    intervals = set(np.arange(open_datetime, close_datetime + timedelta, timedelta))
    intervals = pd.to_datetime(sorted(intervals.union(cq.index)))

    # compute durations of prevailing quotes, up to every {rule} interval
    quotes = cq.reindex(intervals, method='pad')
    quotes['weight'] = quotes.index.to_series().diff().astype(int).shift(-1)  # prevailing durations
    quotes = quotes[(quotes.index>=open_datetime) & (quotes.index<=close_datetime)]
    denoms = quotes['weight'].resample(rule).sum()   # all should equal length of each interval

    result['quoted']    = ((quotes['Best_Offer_Price'] - quotes['Best_Bid_Price']) *
                           quotes['weight']).resample(rule).sum() / denoms
    result['depth']     = ((quotes['Best_Offer_Size'] + quotes['Best_Bid_Size']) *
                            quotes['weight']).resample(rule).sum() / denoms
    result['offersize'] = (quotes['Best_Offer_Size'] * quotes['weight']).resample(rule).sum() / denoms
    result['bidsize']   = (quotes['Best_Bid_Size'] * quotes['weight']).resample(rule).sum() / denoms
    result['midquote']  = quotes['Mid_Price'].resample(rule, loffset=-timedelta).ffill()
    result = result.iloc[:-1]
    
    measure = measure_trade(ct, cq)
    result['volume'] = measure['volume'].resample(rule).sum()
    result['effective'] = ((measure['effective'] * measure['volume']).resample(rule).sum()
                           / result['volume'])
    result['realized']  = ((measure['realized'] * measure['volume']).resample(rule).sum()
                           / result['volume'])
    result['impact']    = ((measure['impact'] * measure['volume']).resample(rule).sum()
                           / result['volume'])
    result['vwap']      = ((measure['price'] * measure['volume']).resample(rule).sum()
                           / result['volume'])
    result.loc[result['volume']==0, ['effective', 'realized', 'impact', 'vwap']] = np.nan
    return result

def plot_taq(symbol, trade, nbbo, open_datetime = _open_datetime, close_datetime=_close_datetime):
    """plot a {symbol}'s trades and quotes"""
    t = trade.get(symbol)
    q = nbbo.get(symbol)
    ct = clean_trade(t)
    cq = clean_nbbo(q)
        
    plt.figure(figsize=(9, 6))
    plt.clf()
    ax = plt.subplot(2, 1, 1)
    f = (ct.index >= open_datetime) & (ct.index <= close_datetime)
    sns.lineplot(x=ct.index[f], y = ct.loc[f, 'Trade_Price'], ax=ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
    g = (cq.index >= open_datetime) & (cq.index <= close_datetime)
    cq.loc[g].plot(y = ['Best_Offer_Price','Best_Bid_Price'], ax=ax, style = '--', legend=False)
    ax.legend(['Trade_Price','Best_Offer_Price','Best_Bid_Price'])
    plt.xticks(rotation='horizontal')

    ax2 = plt.subplot(2, 1, 2, sharex=ax)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    sns.lineplot(x=ct.index[f], y=ct.loc[f, 'Trade_Volume'], markers=['o'], dashes=[':'], ax=ax2)
    plt.xlabel(symbol)
