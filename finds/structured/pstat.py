"""Compustat annual, quarterly and key developments

Copyright 2022-2024, Terence Lim

MIT License
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import time
from pandas import DataFrame, Series
from pandas.api.types import is_list_like, is_integer_dtype
from sqlalchemy import Table, Column, Index
from sqlalchemy import Integer, String, Float, SmallInteger, Boolean, BigInteger
from finds.database.sql import SQL
from finds.database.redisdb import RedisDB
from finds.structured.busday import BusDay
from finds.structured.structured import Structured
_VERBOSE = 1

class PSTAT(Structured):
    """Provide interface to Compustat structured data sets

    Args:
      sql: Connection to mysql database
      bd: Custom business days object
      name: Name of dataset is "PSTAT"
      identifier: Stocks identifier field name is "gvkey"
    Notes:

    - Screen on (INDFMT= 'INDL', DATAFMT='STD', POPSRC='D', and CONSOL='C') 
      keeps majority of records and uniquely identifies GVKEY, DATADATE.
      also include INDFMT= 'FS', CURRENCY='USD+CAD', STATUS='ACTIVE+INACTIVE') 
    """

    _role = Series({   # Key Development role id labels
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
        11: 'Sponsor',
        14: 'Host',
    }, name='role')

    _event = Series({   # Key Development event id labels
        1: 'Seeking to Sell/Divest',            # may be "not sell"
        3: 'Seeking Acquisitions/Investments',
        5: 'Seeking Financing/Partners', # too general, mentions banks
        7: 'Bankruptcy - Other',  # good: includes contemplates and motions
        11: 'Delayed SEC Filings',   # good
        12: 'Delistings',            # good, but beware of microcap
        16: 'Executive/Board Changes - Other',
        21: 'Discontinued Operations/Downsizings',
        22: 'Strategic Alliances',
        23: 'Client Announcements',
        24: 'Regulatory Agency Inquiries',
        25: 'Lawsuits & Legal Issues',
        26: 'Corporate Guidance - Lowered',
        27: 'Corporate Guidance - Raised',
        28: 'Announcements of Earnings',
        29: 'Corporate Guidance - New/Confirmed',
        31: 'Business Expansions',
        32: 'Business Reorganizations',
        36: 'Buybacks',
        41: 'Product-Related Announcements',
        42: 'Debt Financing Related',
        43: 'Restatements of Operating Results',
        44: 'Labor-related Announcements',
        45: 'Dividend Affirmations',
        46: 'Dividend Increases',
        47: 'Dividend Decreases',
        48: 'Earnings Calls',
        49: 'Guidance/Update Calls',
        50: 'Shareholder/Analyst Calls',
        51: 'Company Conference Presentations',
        52: 'M&A Calls',
        53: 'Stock Splits & Significant Stock Dividends',
        54: 'Stock Dividends (<5%)',
        55: 'Earnings Release Date',
        56: 'Name Changes',
        57: 'Exchange Changes',
        58: 'Ticker Changes',
        59: 'Auditor Going Concern Doubts',
        60: 'Address Changes',
        61: 'Delayed Earnings Announcements',
        62: 'Annual General Meeting',
        63: 'Considering Multiple Strategic Alternatives',
        64: 'Ex-Div Date (Regular)',
        65: 'M&A Rumors and Discussions',
        #    68 : 'Credit Rating - S&P - Upgrade',
        #    69 : 'Credit Rating - S&P - Downgrade',
        #    70 : 'Credit Rating - S&P - Not-Rated Action',
        #    71 : 'Credit Rating - S&P - New Rating',
        #    72 : 'Credit Rating - S&P - CreditWatch/Outlook Action',
        73: 'Impairments/Write Offs',
        74: 'Debt Defaults',
        75: 'Index Constituent Drops',
        76: 'Legal Structure Changes',
        77: 'Changes in Company Bylaws/Rules',
        78: 'Board Meeting',
        79: 'Fiscal Year End Changes',
        80: 'M&A Transaction Announcements',
        81: 'M&A Transaction Closings',
        82: 'M&A Transaction Cancellations',
        83: 'Private Placements',
        85: 'IPOs',
        86: 'Follow-on Equity Offerings',
        87: 'Fixed Income Offerings',
        88: 'Derivative/Other Instrument Offerings',
        89: 'Bankruptcy - Filing',
        90: 'Bankruptcy - Conclusion',
        91: 'Bankruptcy - Emergence/Exit',
        92: 'End of Lock-Up Period',
        93: 'Shelf Registration Filings',
        94: 'Special Dividend Announced',
        95: 'Index Constituent Adds',
        97: 'Special/Extraordinary Shareholders Meeting',
        99: 'Potential Privatization of Government Entities',
        100: 'Ex-Div Date (Special)',
        101: 'Executive Changes - CEO',
        102: 'Executive Changes - CFO',
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
        134: 'Composite Units Offerings',
        135: 'Structured Products Offerings',
        136: 'Public Offering Lead Underwriter Change',
        137: 'Spin-Off/Split-Off',
        138: 'Announcements of Sales/Trading Statement',
        139: 'Sales/Trading Statement Calls',
        140: 'Sales/Trading Statement Release Date',
        #    141 : 'LCD Bids Wanted in Competition',
        #    142 : 'LCD Company Buys Back Outstanding Bank Debt',
        #    143 : 'LCD Debt Exchange',
        144: 'Estimated Earnings Release Date (CIQ Derived)',
        #    145 : 'LCD Loan Credit Default Swap News',
        #    146 : 'LCD Credit Defaults Swap News',
        #    147 : 'LCD Default News',
        #    148 : 'LCD Deal Launch News',
        149: 'Conferences',
        150: 'Auditor Changes',
        151: 'Buyback Update',
        152: 'Potential Buyback',
        153: 'Bankruptcy - Asset Sale/Liquidation',
        154: 'Bankruptcy - Financing',
        155: 'Bankruptcy - Reorganization',
        156: 'Investor Activism - Proposal Related',
        157: 'Investor Activism - Activist Communication',
        160: 'Investor Activism - Target Communication',
        163: 'Investor Activism - Proxy/Voting Related',
        164: 'Investor Activism - Agreement Related',
        172: 'Investor Activism - Nomination Related',
        177: 'Investor Activism - Financing Option from Activist',
        187: 'Investor Activism - Supporting Statements',
        192: 'Analyst/Investor Day',
        194: 'Special Calls',
        205: 'Regulatory Authority - Regulations',
        206: 'Regulatory Authority - Compliance',
        207: 'Regulatory Authority - Enforcement Actions',
        #    208 : 'Macro: Releases',
        #    209 : 'Macro: General',
        #    210 : 'Macro: Auctions',
        #    211 : 'Macro: Seminars',
        #    212 : 'Macro: Holidays',
        213: 'Dividend Cancellation',
        214: 'Dividend Initiation',
        215: 'Preferred Dividend',
        #    216 : 'S&P Events',
        #    217 : "Not a Keydev - Only for Timeline"
        218: "Announcement of Interim Management Statement",
        219: "Operating Results Release Date",
        220: "Interim Management Statement Release Date",
        221: "Operating Results Calls",
        222: "Interim Management Statement Calls",
        223: "Fixed Income Calls",
        224: "Halt/Resume of Operations - Unusual Events",
        225: "Corporate Guidance - Unusual Events",
        226: "Announcement of Operating Results",
        230: "Buyback - Change in Plan Terms",
        231: "Buyback Tranche Update",
        232: "Buyback Transaction Announcements",
        233: "Buyback Transaction Cancellations",
        234: "Buyback Transaction Closings"
    }, name='event')
    
    @property
    def role(self):
        """Maps keydev role id to description"""
        return self._role

    @property
    def event(self):
        """Maps keydev event id to description"""
        return self._event

    
    def __init__(self,
                 sql : SQL, 
                 bd : BusDay, 
                 name : str = 'PSTAT',
                 identifier : str = 'gvkey',
                 verbose : int = _VERBOSE):
        """Initialize Compustat tables"""
        tables = {
            'links': sql.Table(
                'links',
                Column('gvkey', Integer, primary_key=True),
                Column('conm', String(30)),
                Column('tic', String(8)),
                Column('cusip', String(9)),
                Column('cik', Integer, default=0),
                Column('sic', SmallInteger, default=0),
                Column('naics', Integer, default=0),
                Column('linkprim', String(1)),
                Column('liid', String(3)),
                Column('linktype', String(2)),
                Column('lpermno', Integer, default=0),
                Column('lpermco', Integer, default=0),
                Column('linkdt', Integer, default=0, primary_key=True),
                Column('linkenddt', Integer, default=0),
                sql.Index('cusip', 'linkdt'),
                sql.Index('cik', 'linkdt'),
                sql.Index('lpermno', 'linkdt'),
            ),
            'annual': sql.Table(
                'annual',
                Column('gvkey', Integer, primary_key=True),
                Column('datadate', Integer, primary_key=True),
                Column('indfmt', String(4), primary_key=True),
                Column('consol', String(1), primary_key=True),
                Column('popsrc', String(1), primary_key=True),
                Column('datafmt', String(3), primary_key=True),
                Column('curcd', String(3), primary_key=True),
                Column('costat', String(1)),
                Column('cusip', String(9)),
                Column('addzip', String(16), default=0),  ### NEW ###
                Column('cik', BigInteger, default=0),
                Column('fyr', SmallInteger, default=0),
                Column('naics', Integer, default=0),
                Column('sic', SmallInteger, default=0),
                Column('fyear', SmallInteger, default=0),
                Column('prcc_f', Float),
                Column('sich', SmallInteger, default=0),
                *(Column(key, Float) for key in [
                    'aco', 'acox', 'act', 'ao', 'aox',
                    'ap', 'aqc', 'aqi', 'aqs', 'at',
                    'caps', 'capx', 'capxv', 'ceq', 'ceql',
                    'ceqt', 'ch', 'che', 'chech', 'cogs',
                    'cshfd', 'csho', 'cshrc', 'dc', 'dclo',
                    'dcpstk', 'dcvsr', 'dcvsub', 'dcvt', 'dd',
                    'dd1', 'dd2', 'dd3', 'dd4', 'dd5',
                    'dlc', 'dltis', 'dlto', 'dltp', 'dltt',
                    'dm', 'dn', 'do', 'dp', 'dpact',
                    'dpc', 'dpvieb', 'ds', 'dv', 'dvc',
                    'dvp', 'dvt', 'ebit', 'ebitda', 'emp',
                    'epsfx', 'epspx', 'esub', 'esubc', 'fatb',
                    'fatl', 'fca', 'fopo', 'gdwl', 'gp',
                    'gwo', 'ib', 'ibadj', 'ibc', 'ibcom',
                    'icapt', 'idit', 'intan', 'intc', 'invfg',
                    'invrm', 'invt', 'invwip', 'itcb', 'itci',
                    'ivaeq', 'ivao', 'ivch', 'ivst', 'lco',
                    'lcox', 'lct', 'lifr', 'lifrp', 'lo',
                    'lse', 'lt', 'mib', 'mibt', 'mii',
                    'mrc1', 'mrc2', 'mrc3', 'mrc4', 'mrc5',
                    'mrct', 'msa', 'ni', 'niadj', 'nopi',
                    'nopio', 'np', 'oancf', 'ob', 'oiadp',
                    'oibdp', 'pi', 'ppegt', 'ppent', 'ppeveb',
                    'prstkc', 'pstk', 'pstkc', 'pstkl', 'pstkn',
                    'pstkr', 'pstkrv', 'rea', 'reajo', 'recco',
                    'recd', 'rect', 'recta', 'rectr', 'reuna',
                    'revt', 'sale', 'scstkc', 'seq', 'spi',
                    'sppe', 'sstk', 'tlcf', 'tstk', 'tstkc',
                    'tstkn', 'txc', 'txdb', 'txdi', 'txditc',
                    'txfed', 'txfo', 'txp', 'txr', 'txs',
                    'txt', 'txw', 'wcap', 'xacc', 'xad',
                    'xido', 'xidoc', 'xint', 'xlr', 'xopr',
                    'xpp', 'xpr', 'xrd', 'xrdp', 'xrent', 'xsga'
                ]),
            ),
            'quarterly': sql.Table(
                'quarterly',
                Column('gvkey', Integer, primary_key=True),
                Column('datadate', Integer, primary_key=True),
                Column('indfmt', String(4), primary_key=True),
                Column('consol', String(1), primary_key=True),
                Column('popsrc', String(1), primary_key=True),
                Column('datafmt', String(3), primary_key=True),
                Column('costat', String(1), primary_key=True),
                Column('cusip', String(9)),
                Column('cik', BigInteger, default=0),
                Column('naics', Integer, default=0),
                Column('sic', SmallInteger, default=0),
                Column('datacqtr', String(6)),
                Column('datafqtr', String(6)),
                Column('fqtr', SmallInteger, default=0),
                Column('fyearq', SmallInteger, default=0),
                Column('rdq', Integer, default=0),
                Column('prccq', Float),
                *(Column(key, Float) for key in
                  ['actq', 'atq', 'ceqq', 'cheq', 'cogsq',
                   'cshoq', 'dlcq', 'ibq', 'lctq', 'ltq',
                   'ppentq', 'pstkq', 'pstkrq', 'revtq', 'saleq',
                   'seqq', 'txtq', 'xsgaq']),
            ),
            'keydev': sql.Table(
                'keydev',
                Column('keydevid', Integer, primary_key=True),
                Column('companyid', Integer, default=0),
                Column('companyname', String(100)),
                Column('keydeveventtypeid', SmallInteger, primary_key=True),
                Column('keydevtoobjectroletypeid', SmallInteger, primary_key=True),
                Column('announcedate', Integer, primary_key=True),
                Column('gvkey', Integer, primary_key=True),
            ),
            'customer': sql.Table(
                'customer',
                Column('gvkey', Integer, primary_key=True),  # Supplier GVKEY
                Column('conm', String(29)),                  # Supplier Name
                Column('cgvkey', Integer, primary_key=True), # Customer GVKEY
                Column('cconm', String(28)),            #Cust Current Name
                Column('cnms', String(50)),                     # Customer Name
                Column('srcdate', Integer, primary_key=True),   # Source Date
                Column('cid', SmallInteger, default=0), #Cust Identifier
                Column('sid', SmallInteger, default=0), #Cust Segment Ident Link
                Column('ctype', String(7)),  # Customer Type
                Column('salecs', Float),     # Customer Sales
                Column('scusip', String(9)), # Supplier CUSIP
                Column('stic', String(5)),   # Supplier Ticker Symbol
                Column('ccusip', String(9)), # Customer CUSIP
                Column('ctic', String(5)),   # Customer Ticker Symbol
            ),
        }
        super().__init__(sql, bd, tables, identifier=identifier, name=name,
                         verbose=verbose)

    def build_lookup(self, source: str, target: str, date_field='linkdt', 
                     dataset: str = 'links', fillna: Any = None) -> Any:
        """Build lookup function to return target identifier from source"""
        return super().build_lookup(source=source, target=target,
                                    date_field=date_field, dataset=dataset,
                                    fillna=fillna) 

    def get_permnos(self, keys: List[str], date: int, link_perm='lpermno', 
                    link_date: str ='date', permno='permno') -> DataFrame:
        """Return list of permnos mapped to gvkeys as of a date

        Args:
          keys: Input list of gvkeys to lookup
          date: Prevailing date of link        
        """

        return super().get_permnos(keys, date, link_perm='lpermno', 
                                   link_date='date', permno='permno')

    def get_linked(self, dataset: str, fields: List[str],
                   date_field: str = 'datadate', link_perm: str = 'lpermno', 
                   link_date: str = 'linkdt', where: str = '', 
                   limit: int | str | None = None) -> DataFrame:
        """Query a pstat table, and return with linked crsp permno

        Args:
          dataset: pstat dataset to query
          fields : Names of fields to return
          date_field: Name of date field in pstat table to query
          link_date: Name of link date field in 'links' table
          link_perm: Name of permno field in 'links' table
          where : Sql where clause, as sql string (optional)
          limit : Maximum number of records to return (optional)

        Returns:
          DataFrame containing result of query

        Examples:

        >>> df = pstat.get_linked(dataset='annual', date_field='datadate',
                 fields=['ceq','pstkrv','pstkl','pstk'],
                 where='ceq > 0 and datadate>=19930104 and datadate<=20991231')
        >>> df = keydev.get_linked(dataset='keydev', date_field='announcedate',
                 fields=['companyname', 'keydeveventtypeid',
                 'keydevtoobjectroletypeid'],
                 where='', limit=''):

        Notes:

        ::

            select keydev.companyname, keydev.keydeveventtypeid,
            keydev.keydevtoobjectroletypeid,
            keydev.announcedate, keydev.gvkey, lpermno as permno
            from keydev left join links
            on keydev.gvkey = links.gvkey and links.linkdt =
                (select max(c.linkdt) as linkdt from links c
                where c.gvkey = keydev.gvkey and c.linkdt <= keydev.announcedate)
            where lpermno is not null and keydev.gvkey > 0 and links.gvkey > 0
            and announcedate >= 20180301
            limit 100;
        """

        return super().get_linked(dataset=dataset, date_field=date_field,
                fields=fields, link_perm='lpermno', link_date='linkdt', 
                where=where, limit=limit)


if __name__ == "__main__":
    from secret import credentials, paths
    VERBOSE = 1

    sql = SQL(**credentials['sql'], verbose=VERBOSE)
    user = SQL(**credentials['user'], verbose=VERBOSE)
    rdb = RedisDB(**credentials['redis'])
    bd = BusDay(sql)
    pstat = PSTAT(sql, bd)
    downloads = paths['data'] / 'PSTAT'

    """
    # load links
    df = pstat.load_csv(
        'links', downloads / 'links.txt.gz', sep='\t',    
        drop={'lpermno': ['0', 0],
              'linkprim': ['N', 'J']},
        keep={'linktype': ['LC', 'LU']},  # researched and unresearched links"
        replace={'linkdt': (['C', 'E', 'B'], 0),
                 'linkenddt': (['C', 'E', 'B'], 0)})
    print(len(df), 33036)
    lag = df.shift()
    f = (lag.gvkey == df.gvkey) & (lag.lpermno != df.lpermno)
    print('permnos in links changed in ', sum(f), 'of', len(df), 1063)

    # load annual
    df = pstat.load_csv('annual', downloads / 'annual.txt.gz', sep='\t')
    print(len(df), 464753)
    print(df.isna().mean().sort_values().tail(5))
    
    # load quarterly
    df = pstat.load_csv('quarterly', downloads / 'quarterly.txt.gz', sep='\t')
    print(len(df), 1637274)
    print(df.isna().mean().sort_values().tail(5))
    
    # load keydevs
    for filename in sorted(downloads.glob('keydev*.txt.gz')):
        tic = time.time()   
        df = pstat.load_csv('keydev', downloads / filename, sep='\t',
                            drop={'gvkey': [0, '0'],
                                  'announcedate': [0, '0'],
                                  'keydevid': [0, '0']})
        print(len(df), filename, time.time() - tic)
    print(sql.run('select count(*) from keydev'), 12256909)
    """

    # load principal customers
    df = pstat.load_csv('customer', downloads / 'supplychain.csv.gz', sep=',')
    print(len(df), 107114)
        
