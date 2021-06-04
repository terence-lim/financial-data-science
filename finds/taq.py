"""Class and methods to preprocess and analyze TAQ trade and quotes tick data

- NYSE Daily TAQ, bid-ask quotes, trade conditions, tick test

Author: Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import indexed_gzip as igzip
import gzip, io, pickle, time, re, os
import matplotlib.pyplot as plt
from .display import plot_time

def chunk_to_df(chunk, columns=None):
    """Convert a csv text chunk from TAQ to dataframe with correct dtypes"""
    _dtypes = {    # define dtypes for each TAQ file
        'nbbo': {np.uint64: ['Time','Sequence_Number','Participant_Timestamp',
                             'FINRA_ADF_Timestamp'],
                 np.float32: ['Bid_Price','Bid_Size','Offer_Price',
                              'Offer_Size','Best_Bid_Price','Best_Bid_Size',
                              'Best_Offer_Price','Best_Offer_Size']},
        'trade': {np.uint64: ['Time','Sequence_Number','Participant_Timestamp',
                              'Trade_Reporting_Facility_TRF_Timestamp'],
                  np.uint8: ['Trade_Correction_Indicator',
                             'Trade_Through_Exempt_Indicator'],
                  np.float32: ['Trade_Volume','Trade_Price']},
        'mast': {np.uint8: ['TradedOnNYSEMKT','TradedOnNASDAQBX','TradedOnNSX',
                            'TradedOnFINRA','TradedOnISE','TradedOnEdgeA',
                            'TradedOnEdgeX','TradedOnCHX','TradedOnNYSE',
                            'TradedOnArca','TradedOnNasdaq','TradedOnCBOE',
                            'TradedOnPSX','TradedOnBATSY','TradedOnBATS',
                            'TradedOnIEX'],
                 np.uint16: ['Unit_Of_Trade','Round_Lot',
                             'Specialist_Clearing_Number',
                             'Specialist_Post_Number'],
                 np.uint32: ['Shares_Outstanding', 'Effective_Date'],
                 np.uint64: ['Unit_Of_Trade','Round_Lot',
                             'Specialist_Clearing_Number',
                             'Specialist_Post_Number',
                             'TradedOnNYSEMKT','TradedOnNASDAQBX','TradedOnNSX',
                             'TradedOnFINRA','TradedOnISE','TradedOnEdgeA',
                             'TradedOnEdgeX','TradedOnCHX','TradedOnNYSE',
                             'TradedOnArca','TradedOnNasdaq','TradedOnCBOE',
                             'TradedOnPSX','TradedOnBATSY','TradedOnBATS',
                             'TradedOnIEX','Effective_Date']}}

    df = pd.read_csv(io.StringIO(chunk), sep='|', na_filter=False, dtype=str, 
                     header=None if columns else 0, names=columns or None)

    # guess file type from column names, and use its dtypes dict
    if 'Best_Bid_Price' in df.columns:  # NBBO file
        dtypes = _dtypes['nbbo']
    elif 'Trade_Price' in df.columns:   # TRADE file
        dtypes = _dtypes['trade']
    elif 'Round_Lot' in df.columns:     # MASTER file
        dtypes = _dtypes['mast']
    else:
        dtypes = {}

    for t in dtypes:  # coerce each column to its required dtype
        df[dtypes[t]] = df[dtypes[t]].apply(pd.to_numeric, errors='coerce')
        df[dtypes[t]] = df[dtypes[t]].fillna(0).astype(t)
    
    if 'Time' in df.columns:      # for nbbo and trade: set timestamp as index
        df.index = pd.to_datetime(df['Time'].astype(str), format='%H%M%S%f')
    elif 'Symbol' in df.columns:  # for master: set symbol field as index
        df.index = df['Symbol'].values
    
    if 'END' in df.index:         # terminator dummy symbol
        df.drop('END', inplace=True)
    return df

    
class TAQ(object):
    """Base class to manipulate a daily TAQ .csv.gz file

    Parameters
    ----------
    filename: str
        .csv.gz raw data file name
    index_file: str, default None
        give name of (csv.gz) index file to support indexed-gzip
    symbols_file: str, default None
        give name of (csv.gz) symbols index file to support getitem

    Notes
    -----
    NYSE historical samples: ftp://ftp.nyxdata.com/Historical%20Data%20Samples/

    Examples
    --------
    Implements three different ways to access raw daily TAQ csv.gz files, e.g.:
        trade(n) - next n lines
        iter(trade) - iterable, by chunk with same stock symbol
        trade['AAPL'] - getitem, with symbol (requires IndexedGzipFile package)
    """
    def __init__(self, taq_file, index_file=None, symbols_file=None):
        """Initalize interface to daily TAQ file"""
        self.taq_file = taq_file
        self.date = re.findall(r"[12][90]\d\d\d\d\d\d", taq_file)
        self.index_file = index_file
        self.symbols_file = symbols_file
        self.igz_file = None   # indexed gzip stream for getitem read
        
    def close(self):
        """Close getitem file handle"""
        if self.igz_file is not None:
            self.igz_file.close()
            self.igz_file = None

    def open(self, taq_file=None):
        """Open with context manager protocol for sequential line reads"""

        class File(object):
            """Context manager procotol to open for sequential read"""
            def __init__(self, filename, *args, **kwargs):
                self.gz_file = gzip.open(filename, "rt", encoding='utf-8',
                                         errors='ignore')
                self.header = self.gz_file.readline()

            def __call__(self, n):
                """Read next n lines (-1 for entire file)"""
                if (n < 0):
                    chunk = self.header + self.gz_file.read()
                    self.close()
                else:
                    chunk = [self.header]
                    for _ in range(n):
                        line = self.gz_file.readline()
                        if len(line) <= 0:
                            self.close()
                            break
                        chunk.append(line)
                    chunk = "".join(chunk)
                return chunk_to_df(chunk)
            
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, exc_traceback):
                return self.close()

            def close(self):
                return self.gz_file.close()
        return File(taq_file or self.taq_file)

    def read(self):
        """Read entire file as a DataFrame"""
        with self.open() as f:
            return f(-1)
        
    def __iter__(self):
        """Iterator to access next symbol's chunk of rows"""
        f = gzip.open(self.taq_file, "rt", encoding='utf-8', errors='ignore')
        header = f.readline().rstrip('\n').replace(' ','_').split('|')

        def line_iterator(f):
            try:
                for line in f:
                    yield line
            except Exception as e:
                print("*** Got exception (line_iterator):", e)
            yield None

        lines = line_iterator(f)
        line = next(lines)         # keep first line as header columns
        curr = line.rstrip('\n').split('|')[header.index('Symbol')]
        while line and not line.startswith('END'): # check end-of-file
            prev = curr
            chunk = []
            while line and curr == prev:  # read until new symbol
                chunk += [line]
                line = next(lines)
                if line:
                    curr = line.rstrip('\n').split('|')[header.index('Symbol')]
            df = chunk_to_df("".join(chunk), header)
            df.index.name = prev
            yield df
        f.close()
        yield None
        
    def __call__(self, symbol):
        """Return symbol location and size in daily taq gzip file"""
        if self.igz_file is None:
            self.pos = pd.read_csv(self.symbols_file, index_col=0)
            if self.index_file:
                self.igz_file = igzip.IndexedGzipFile(self.taq_file,
                                                      index_file=self.index_file)
            else:
                self.igz_file = igzip.IndexedGzipFile(self.taq_file)
            line = self.igz_file.readline().decode('latin-1')
            self.columns = line.rstrip('\n').replace(' ','_').split('|')
        return (self.pos.loc[symbol]
                if symbol in self.pos.index and self.pos.loc[symbol, 'size'] > 0
                else None)

    def __getitem__(self, symbol):
        """Get chunk of all rows for the input symbol as a data frame"""
        pos = self(symbol)
        if pos is None:
            return None
        self.igz_file.seek(pos['start'])
        lines = self.igz_file.read(pos['size']).decode('latin-1')
        df = chunk_to_df(lines, columns=self.columns)
        df.index.name = symbol
        return df

    def index_symbols(self, index_file=None, symbols_file=None):
        """Generate indexedGzip and symbols index files"""
        def _create_index(filename, index_file):
            """generate and save an indexed-gzip index file (12 secs)"""
            tic = time.time()
            try:
                with igzip.IndexedGzipFile(filename) as f:
                    f.build_full_index()
                    f.export_index(index_file)    # save to {index_file}
                print('Full Index created in %d secs' % (time.time() - tic))
            except:
                raise Exception('TAQ _create_index failed')

        def _create_symbols(filename, symbols_file):
            """generate symbol lookup locations (~100 secs)"""
            tic = time.time()
            try:
                with igzip.IndexedGzipFile(filename) as f:
                    f.seek(0)
                    line = f.readline().decode('latin-1')
                    header = line.rstrip('\n').split('|') # header in first line
                    symbols = DataFrame(columns=['start','size'],dtype=int)
                    prev = None
                    eof = lambda line: (not line) or line.startswith('END')
                    while not eof(line):     # iterate until reached end-of-file
                        tell = f.tell()                       # current location
                        line = f.readline().decode('latin-1')  # parse next line
                        if not eof(line):
                            curr = line.rstrip('\n')\
                                       .split('|')[header.index('Symbol')]
                        else:
                            curr = None
                        if curr != prev:           # new symbol: update location
                            if prev:
                                symbols.loc[prev, 'size']\
                                    = tell - symbols.loc[prev, 'start']
                            if curr:
                                symbols.loc[curr] = tell, 0
                            prev = curr
                print('%d symbols: %d secs' % (len(symbols), time.time() - tic))
                symbols.to_csv(symbols_file)
            except:
                raise Exception('TAQ _create_symbols failed')

        if index_file:
            self.index_file = index_file
        if symbols_file:
            self.symbols_file = symbols_file
        _create_index(self.taq_file, self.index_file)
        _create_symbols(self.taq_file, self.symbols_file)
        return self

#
# tick data transformation methods
#
_open_t  = pd.to_datetime('1900-01-01T09:30')    # default open  9:30am
_close_t = pd.to_datetime('1900-01-01T16:00')    # default close 4:00pm
open_t = _open_t
close_t = _close_t
# np.datetime64('1900-01-01T09:30:01.000002') - np.timedelta64('30','ns')

def clean_trade(df, open_t=_open_t, close_t=_close_t, cond="MOZBTLGWJK145789"):
    """Remove bad trades
    Notes
    -----
    Sale Conditions to exclude by default:
    M = Market Center Close Price
    O = Market Center Opening Trade
    Z = Sold (Out of Sequence)
    L = Sold Last (Late Reporting)
    B = Bunched Trade
    G = Bunched Sold Trade
    W = Average Price Trade
    4 = Derivatively Priced
    5 = Re-opening Prints
    7 = Qualified Contingent Trade
    8 = Placeholder for 611 Exempt
    9 = Corrected Consolidated Close Price per the Listing Market
    K = Rule 127 (NYSE only) or Rule 155 Trade (NYSE MKT only)
    T = Extended Hours Trade
    """
    def any_in(keys, values):
        return np.any([v in values for v in keys]) 
    
    if df is None:
        return None
    f = (df['Trade_Correction_Indicator'].eq(0) &
         df['Trade_Price'].gt(0) &
         df['Trade_Volume'].gt(0) &
         ~df['Sale_Condition'].apply(any_in, values=cond))
    if open_t:
        f &= (df.index > open_t)
    if close_t:
        f &= (df.index <= close_t)
    df = df.loc[f, ['Trade_Price','Trade_Volume', 'Sale_Condition']]
    return df

def clean_nbbo(df, keep=['Best_Bid_Price','Best_Bid_Size','Best_Offer_Price',
                         'Best_Offer_Size']):
    """remove bad quotes"""
    if df is None:
        return None    
    #df['Mid_Price'] = (df['Best_Offer_Price'] + df['Best_Bid_Price']) / 2
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
         #(df['Best_Offer_Price'] - df['Best_Bid_Price'] < df['Mid_Price']*0.05)
         (df['Offer_Price'] - df['Bid_Price']).le(5)) # spread <= $5
    df = df.loc[f, keep + ['Time','Sequence_Number']] # bad conditions, values
    df = df.sort_values(['Time','Sequence_Number'])   # keep later of same time
    df = df.drop_duplicates(subset='Time', keep='last')[keep]
    df = df.loc[(df.shift() != df).any(axis=1)]       # keep new records only
    return df


def align_trades(ct, cq, open_t=_open_t, inplace=False):
    """Align each trade with prevailing and forward quotes
    prevailing quote at -1ns, forward quote at +5m, drop quotes before open_t
    References: Holden and Jacobsen (2014), "Liquidity Measurement"
    """
    if not inplace:
        ct = ct.copy()
    f = cq.index >= (open_t or cq.index.min())
    midprice = (cq.loc[f, 'Best_Offer_Price'] + cq.loc[f, 'Best_Bid_Price']) / 2
    ct['Prevailing_Mid'] = midprice.reindex(ct.index - np.timedelta64(1, 'ns'),
                                            method='pad').values
    ct['Forward_Mid'] = midprice.reindex(ct.index + np.timedelta64(5, 'm'),
                                         method='pad').values
    ct['Tick_Test'] = np.sign((ct['Trade_Price'] -
                               ct['Trade_Price'].shift(+1))).fillna(0)
    if not inplace:
        return ct

def bin_quotes(cq, value=15, unit='m', open_t=_open_t, close_t=_close_t):
    """Resample quotes into time interval bins
    Notes
    -----
    resample(closed='left', label='right')
    agg(Series.sum, min_count=1)
    result[result.index > open_t], result[result.index <= close_t]
    """
    units = {'h':'H', 'm':'min', 's':'S', 'ms':'ms', 'us':'us', 'ns':'N'}
    if unit not in units:
        raise Exception(str(unit) + ' must be in ' + str(units.keys()))
    rule = dict(rule=str(value)+units[unit], closed='left', label='right')

    # supplement cq.index with required 'intervals' to be new quote times
    timedelta = np.timedelta64(value, unit.lower())
    intervals = np.arange(open_t, close_t + timedelta, timedelta)
    intervals = pd.to_datetime(sorted(set(intervals).union(cq.index)))

    # compute forward duration of each (supplemented) cq row
    q = cq.reindex(intervals, method='pad')  # forward fill quotes to intervals
    q['weight'] = q.index.to_series().diff().fillna(np.timedelta64(0))\
                                            .astype(int).shift(-1,fill_value=0)

    # sum up 'weight' denominator of each interval, for weighted resampling
    result = DataFrame(q['weight'].resample(**rule).sum())

    # compute weighted mean, with kludge for sum requiring at least one non-null
    weighted = lambda s: (s * q['weight'])\
               .resample(**rule)\
               .agg(Series.sum, min_count=1).div(result['weight'])
    # finally, compute weighted resampling
    result['quoted'] = weighted(q['Best_Offer_Price']-q['Best_Bid_Price'])/2
    result['depth'] = weighted(q['Best_Offer_Size']+q['Best_Bid_Size'])/2
    result['offersize'] = weighted(q['Best_Offer_Size'])
    result['bidsize'] = weighted(q['Best_Bid_Size'])
    result['mid'] = ((q['Best_Offer_Price'] + q['Best_Bid_Price']) / 2)\
                    .resample(**rule).last()
    result['retq'] = (result['mid'] / result['mid'].shift(1)) - 1
    result = result[result.index > open_t]
    result = result[result.index <= close_t]
    return result

def bin_trades(ct, value=5, unit='m', open_t=_open_t, close_t=_close_t):
    """Resample trades into time interval bins
    Notes
    -----
    resample(closed='left', label='right')
    agg(Series.sum, min_count=1)
    result[result.index > open_t], result[result.index <= close_t]
    """
    units = {'h':'H', 'm':'min', 's':'S', 'ms':'ms', 'us':'us', 'ns':'N'}
    if unit not in units:
        raise Exception(str(unit) + ' must be in ' + str(units.keys()))
    rule = dict(rule=str(value)+units[unit], closed='left', label='right')

    # extend ct.index timestamps to open_t and close_t if necessary
    timedelta = np.timedelta64(value, unit.lower())
    p = [t for t in [open_t, close_t] if t < ct.index[0] or t > ct.index[-1]]
    t = pd.concat((ct, DataFrame(index=p, columns=ct.columns))).sort_index()
    t.index.name = ct.index.name

    result = DataFrame(t['Trade_Volume']\
                       .resample(**rule)\
                       .sum())\
                       .rename(columns = {'Trade_Volume': 'volume'})
    result['counts'] = t['Trade_Price']\
                       .resample(**rule).count()
    result['last'] = t['Trade_Price']\
                     .resample(**rule)\
                     .last()\
                     .fillna(method='pad')
    result['ret'] = result['last'].div(result['last'].shift(1)) - 1 
    result['vwap'] = (t['Trade_Price'] * t['Trade_Volume'])\
                     .resample(**rule)\
                     .sum()\
                     .div(result['volume'])
                     #.agg(Series.sum, min_count=-1)\
    result['effective'] = ((t['Trade_Price'] - t['Prevailing_Mid']).abs()\
                           .mul(t['Trade_Volume'])\
                           .resample(**rule))\
                           .sum()\
                           .div(result['volume'])
                           #.agg(Series.sum, min_count=-1)\
    if 'Tick_Test' in t:
        # Lee and Ready trade sign
        leeready = np.sign(t['Trade_Price'].sub(t['Prevailing_Mid']).fillna(0))
        leeready = leeready.where(leeready.ne(0), t['Tick_Test'])
        weighted = lambda spread: (leeready * spread * t['Trade_Volume'])\
                   .resample(**rule)\
                   .sum()\
                   .div(result['volume'])
                   #.agg(Series.sum, min_count=1)\
        result['realized'] = weighted(t['Trade_Price'] - t['Forward_Mid'])
        result['impact'] = weighted(t['Forward_Mid'] - t['Prevailing_Mid'])
    return result[(result.index > open_t) & (result.index <= close_t)]

def plot_taq(left1, right1=None, left2=None, right2=None, num=None, title='',
             open_t=None, close_t=None):
    """Convenience method for 1x2 primary/secondary-y subplots of tick data"""
    if left2 is None:
        fig, ax = plt.subplots(num=num, figsize=(10, 6), clear=True)
    else:
        fig, (ax, bx) = plt.subplots(2, 1, num=num, figsize=(10,12),
                                     clear=True, sharex=True)
        plot_time(left2, right2, ax=bx, xmin=open_t, xmax=close_t)
    plot_time(left1, right1, ax=ax, title=title, xmin=open_t, xmax=close_t)
    plt.tight_layout(pad=3)

def itertaq(trades, quotes, master, open_t=_open_t, close_t=None,
            cusips=None, symbols=None, echo=True, has_shares=True):
    """Iterates over and filters daily taq trades and quotes by symbol

    Parameters
    ----------
    trades : TAQ object
        Instance of Trades input file object
    quotes : TAQ object
        Instance of NBBO input file object
    master : DataFrame
        Reference table from Master file, indexed by symbol
    cusips : list, default is None (to return every)
        List of cusips to select
    symbols : list, default is None (to return every)
        List of symbols (upper case, space separated security class) to select.
    open_t : pandas.datetime, default is 9:30am
        Earliest datetime of valid trades and quotes
    close_t : pandas.datetime, default is None
        Latest datetime to keep trades and quotes, inclusive
    echo : bool, default True
        Print echo messages for debugging
    has_shares : bool, default True
        If True, require 'Shares_Outstanding' > 0 in master table
    """
    if cusips is not None:
        cusips = list(cusips)
    if symbols is not None:
        symbols = list(symbols)
    trade = iter(trades)
    while True:
        t = next(trade)
        if t is None:
            if echo: print('trade is none')
            break
        symbol = t.index.name
        if symbol not in master.index:
            if echo: print(f"{symbol} not in master")
            continue
        name = master.loc[symbol, 'Security_Description']
        if symbols and symbol not in symbols:
            if echo: print(f"{symbol} {name} not in symbols")
            continue
        if has_shares and not master.loc[symbol, 'Shares_Outstanding'] > 0:
            if echo: print(f"{symbol} {name} has no shares")
            continue
        cusip = master.loc[symbol, 'CUSIP']
        if cusips and cusip[:8] not in cusips and cusip not in cusips:
            if echo: print(f"{symbol} {name} {cusip} not in cusips")
            continue
        if t is None or not len(t):
            if echo: print('trades is empty')            
            continue
        q = quotes[symbol]
        if q is None or not len(q):
            if echo: print('quotes is empty')            
            continue
        ct = clean_trade(t, open_t=open_t, close_t=close_t)
        cq = clean_nbbo(q)
        if not len(ct) or not len(cq):
            if echo: print('ct or cq empty')             
            continue
        if echo: print(symbol, cusip, len(t),len(q),len(ct),len(cq))
        align_trades(ct, cq, open_t=open_t, inplace=True)
        yield ct, cq, master.loc[symbol]

def opentaq(date, taqdir):
    """Helper to init and return master dataframe, trade and quote objects"""
    return (TAQ(os.path.join(taqdir,
                             f'EQY_US_ALL_REF_MASTER_{date}.gz')).read(),
            TAQ(os.path.join(taqdir, f'EQY_US_ALL_TRADE_{date}.gz'),
                os.path.join(taqdir, f'EQY_US_ALL_TRADE_{date}.gzidx'),
                os.path.join(taqdir, f'EQY_US_ALL_TRADE_{date}.csv.gz')),
            TAQ(os.path.join(taqdir, f'EQY_US_ALL_NBBO_{date}.gz'),
                os.path.join(taqdir, f'EQY_US_ALL_NBBO_{date}.gzidx'),
                os.path.join(taqdir, f'EQY_US_ALL_NBBO_{date}.csv.gz')))

if False:  # test access methods
    from finds.taq import TAQ, opentaq, clean_nbbo, clean_trade, plot_taq
    from finds.taq import open_t, close_t, bin_trades, bin_quotes, align_trades
    import os
    import numpy as np
    from settings import settings
    import matplotlib.pyplot as plt

    # daily taq files from ftp://ftp.nyxdata.com/Historical%20Data%20Samples/
    dates = [20191007, 20191008, 20180305, 20180306, 20171101]
    taqdir = os.path.join(settings['remote'], 'TAQ')

    # generate input filenames for TAQ, igzip and symbols index files
    master_file = os.path.join(taqdir, 'EQY_US_ALL_REF_MASTER_{}.gz').format
    trade_file = os.path.join(taqdir, 'EQY_US_ALL_TRADE_{}.gz').format
    nbbo_file  = os.path.join(taqdir, 'EQY_US_ALL_NBBO_{}.gz').format

    nbbo_index = os.path.join(taqdir, 'EQY_US_ALL_NBBO_{}.gzidx').format
    trade_index = os.path.join(taqdir, 'EQY_US_ALL_TRADE_{}.gzidx').format
    
    nbbo_symbols = os.path.join(taqdir, 'EQY_US_ALL_NBBO_{}.csv.gz').format
    trade_symbols = os.path.join(taqdir, 'EQY_US_ALL_TRADE_{}.csv.gz').format
    
    date = dates[0]
    master, trades, quotes = opentaq(date, taqdir)
    
    #
    # 1. Read lines
    #
    print(master)
    trade = trades.open()
    print(trade(5))
    quote = quotes.open()
    print(quote(5))

    #
    # 2. Read in sequential chunks, symbol by symbol
    #
    trade = iter(trades)
    for _ in range(5):
        df = next(trade)
        print(f'next Trade symbol {df.index.name}: {len(df)} records')
    
    quote = iter(quotes)
    for _ in range(5):
        df = next(quote)
        print(f'next Quote symbol {df.index.name}: {len(df)} records')

    #
    # 3. getitem by symbol (uses index_gzipped package)
    #
    symbol = 'VTI'
    symbol = 'AAPL' #'ZM'
    symbol = 'GS'
    t = trades[symbol]
    q = quotes[symbol]
    ct = clean_trade(t, close_t=close_t + np.timedelta64('5','m'))
    cq = clean_nbbo(q)    
    align_trades(ct, cq, inplace=True)

    plot_taq(ct[['Trade_Price', 'Prevailing_Mid']], ct['Trade_Volume'],
             (cq['Best_Offer_Price'] - cq['Best_Bid_Price'])\
             .rename('Quoted Spread'),
             ((cq['Best_Bid_Size']+cq['Best_Offer_Size'])/2).rename('Depth'),
             open_t=open_t, close_t=close_t + np.timedelta64('5','m'), num=1,
             title=f"Tick Prices, Volume, Quotes, Spreads, and Depths")
    plt.show()    

    value, unit = 5, 'm'
    timedelta = np.timedelta64(value, unit)
    bt = bin_trades(ct, value, unit, close_t=close_t + timedelta)
    bq = bin_quotes(cq, value, unit, close_t=close_t + timedelta)
    bq = bq.join(bt, how = 'left')
    plot_taq(bq[['last', 'vwap', 'mid']], bq['quoted'],
             bt['volume'], bt['counts'], num=2,
             open_t=open_t, close_t=close_t + np.timedelta64('5','m'),
             title=f"{value}{unit}-bin Prices, Quotes and Trades")
    print(f"Correlation of MidQuote and LastTrade {value}{unit}-bin returns")
    bq[['ret', 'retq']].corr()

    m1 = np.timedelta64('5','s')
    d = (t.index >= close_t) & (t.index <= close_t+m1)
    as_print(t.loc[d])
