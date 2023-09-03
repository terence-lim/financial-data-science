"""Wrapper over BEA web api and data

- Bureau of Economic Analysis: Input-Output Use Tables

Copyright 2022, Terence Lim

MIT License
"""
from typing import Dict, List, Any
import io
import json
import re
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types
from sqlalchemy import Integer, String, Column
from pandas.api import types
from finds.database.redisdb import RedisDB
from .readers import requests_get
_VERBOSE = 1

class BEA:
    """Class and methods to retrieve BEA web api and IOUse datasets

    Args:
        rdb: Redis connection instance to cache data from BEA web site
        userid: Register with BEA to get key for their web api

    Attributes:
        bea_vintages:  Reference dict to group BEA tables by vintage year
        bea_industry: Reference Series maps BEA industry to custom descriptions
        params: Some common parameter sets for BEA web api
        
    Examples:

    >>> bea = BEA()
    >>> df = bea.get(**BEA.items['industries'])
    >>> sql.load_dataframe('gdpindustry', df, index_label=None, if_exists='replace')
    >>> df = bea.get(**BEA.items['ioUse'], year=2018)
    """

    _bea_url = 'https://apps.bea.gov/industry/xls/io-annual/'    

    bea_industry_ = Series({    # custom abbreviations of BEA industries
        '11': 'Agriculture',
        '111CA': 'Farms',
        '113FF': 'Forestry,fishing',
        '21': 'Mining',
        '211': 'Oil, gas',
        '212': 'Mining',
        '213': 'Support mining',
        '22': 'Utilities',
        '23': 'Construction',
        '31G': 'Manufacturing',
        '33DG': 'Durable goods',
        '321': 'Wood',
        '327': 'Nonmetallic',
        '331': 'Metals',
        '332': 'Fabricated metal',
        '333': 'Machinery',
        '334': 'Computer',
        '335': 'Electrical',
        '3361MV': 'Motor vehicles',
        '3364OT': 'Transport equip',
        '337': 'Furniture',
        '339': 'Manufacturing',
        '31ND': 'Nondurable goods',
        '311FT': 'Food',
        '313TT': 'Textile',
        '315AL': 'Apparel',
        '322': 'Paper',
        '323': 'Printing',
        '324': 'Petroleum, coal',
        '325': 'Chemical',
        '326': 'Plastics, rubber',
        '42': 'Wholesale',
        '44RT': 'Retail',
        '441': 'Motor dealers',
        '445': 'Food stores',
        '452': 'General stores',
        '4A0': 'Other retail',
        '48' : 'Transportation',
        '48TW': 'Transport, warehouse',
        '481': 'Air transport',
        '482': 'Rail transport',
        '483': 'Water transport',
        '484': 'Truck transport',
        '485': 'Transit ground',
        '486': 'Pipeline transport',
        '487OS': 'Other transport',
        '493': 'Warehousing, storage',
        '51': 'Information',
        '511': 'Publishing',
        '512': 'Motion picture',
        '513': 'Broadcasting',
        '514': 'Data processing',
        '52': 'Finance,insurance',
        '521CI': 'Banks',
        '523': 'Securities',
        '524': 'Insurance',
        '525': 'Funds, trusts',
        '53': 'Real estate',
        '531': 'Real estate',
        'HS': 'Housing',
        'ORE': 'Other real estate',
        '532RL': 'Rental',
        '54': 'Professional services',
        '5411': 'Legal services',
        '5415': 'Computer services',
        '5412OP': 'Misc services',
        '55': 'Management',
        '56': 'Administrative and waste management',
        '561': 'Administrative',
        '562': 'Waste management',
        '6': 'Educational services',
        '61': 'Educational',
        '62': 'Healthcare',
        '621': 'Ambulatory',
        '622HO': 'Hospitals, nursing',
        '622': 'Hospitals',
        '623': 'Nursing',
        '624': 'Social',
        '7': 'Arts, entertainment, recreation, accommodation, food svcs',
        '71': 'Arts, entertainment, recreation',
        '711AS': 'Performing arts',
        '713': 'Recreation',
        '72': 'Accommodation. food svcs',
        '721': 'Accommodation',
        '722': 'Food services',
        '81': 'Other services',
        'G': 'Government',
        'GF': 'Federal',
        'GFG': 'General government',
        'GFGD': 'Defense',
        'GFGN': 'Nondefense',
        'GFE': 'Federal enterprises',
        'GSL': 'State local',
        'GSLG': 'State local general',
        'GSLE': 'State local enterprises'})
    
    bea_vintages_ = {  # group BEA IO-Use table coarser by vintage year
        9999: {'531': ['HS','ORE']},   
        1963: {'48': ['481','482','483','484','485','486','487OS'],
               '51': ['511','512','513','514'],
               '52': ['521CI','523','524','525'],
               '54': ['5411','5415','5412OP'],
               '56': ['561','562'],
               '62': ['621','622','623','624','622HO'],
               '71': ['711AS','713']},
        1997: {'44RT': ['441', '445', '452', '4A0'],
               'GFG': ['GFGD', 'GFGN'],
               '622HO': ['622','623']}}
    
    
    params = {'gdpindustry': {'datasetname': 'GDPbyIndustry', 
                              'index': 'key',
                              'parametername': 'Industry'},
              'ioUse': {'datasetname': 'inputoutput',
                        'tableid': 259}} # commonly parameters for BEA web api

    def __init__(self, rdb: RedisDB | None, userid: str, verbose: int =_VERBOSE):
        """Open connection to BEA web api"""
        self.rdb = rdb
        self.userid = userid
        self.verbose = verbose

    @staticmethod
    def sectoring(year: int, source: str = ""):
        """Returns BEA sector definitions, based on naics, from xls on BEA website

        Args:
            year: Year of historical BEA scheme, in {1997, 1964, 1947}
            source: Source url or local file

        Notes:

        - https://www.bea.gov/industry/input-output-accounts-data
        - https://apps.bea.gov/industry/xls/io-annual/
        - Use_SUT_Framework_2007_2012_DET.xlsx 
        - Replace "HS" "ORE" with "531"
        """
        if not source:
            filename = {   # there are 3 historical schmes
                1997: 'Use_SUT_Framework_2007_2012_DET.xlsx',   
                1963: 'IoUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx',
                1947: 'IoUse_Before_Redefinitions_PRO_1947-1962_Summary.xlsx'}
            if year not in filename:
                raise Exception('bea year not in ' + str(list(filename.keys())))
            source = BEA._bea_url + filename[year]

        # get the xls and parse the NAICS Codes sheet
        x = pd.ExcelFile(source)
        df = x.parse('NAICS Codes')

        # extract main groups and labels from columns 0,1
        beg = np.where(df.iloc[:, 0].astype(str).str[0].str.isdigit())[0]
        labels = {str(df.iloc[i,0]) : df.iloc[i,1] for i in beg}

        # Kludge: some labels missing, so fill from custom abbreviations
        labels.update({k: v for k,v in BEA.bea_industry_.items()
                       if k not in labels})

        # Now extract beg and end row of groups and labels from columns 1,2
        beg, = np.where(df.iloc[:, 1].astype(str).str[0].str.isdigit()
                        | df.iloc[:, 1].astype(str).isin(["HS", "ORE"]))
        end = np.append(beg[1:], len(df.index))

        # for each group, parse naics code ranges from col 6, store in {results}
        #   e.g. '7223-4, 722514-5' --> [722300, 722514]
        # also, parse leading digits of summary name if at least len 2 (pad 0's)
        result = []
        for imin, imax in zip(beg, end):
            s = [str(c).split('-')[0]
                 for c in df.iloc[imin:imax, 6] if str(c)[0].isdigit()]
            s = [[re.sub("[^0-9]","",a).ljust(6,'0') for a in c.split(',')]
                 for c in s if c not in ['491']]   # postal industry
            new_codes = [int(c) for inner in s for c in inner]
            name = str(df.iloc[imin, 1])
            m = re.match('\d+', name)
            if m and len(m.group()) >= 2:
                new_codes.append(int(m.group().ljust(6, '0')))
            new_df = DataFrame(sorted(new_codes), columns=['code'])
            new_df['name'] = name
            new_df['description'] = df.iloc[imin, 2]
            result.append(new_df.drop_duplicates())
        result = pd.concat(result,
                           axis=0,
                           ignore_index=True)

        # replace earlier vintage years with coarser industries
        for vintage, codes in BEA.bea_vintages_.items():
            if year < vintage:
                for k,v in codes.items():
                    result.loc[result['name'].isin(v),
                               ['name','description']] = [k, labels[k]]

        # clean-up data frame and its index for return
        return result.drop_duplicates(['code']).set_index('code').sort_index()

    def get(self, datasetname: str = "", parametername: str = "",
            cache_mode: str = "rw", **kwargs) -> DataFrame:
        """Wrapper to execute common BEA web api calls

        Args:
            datasetname: Name of dataset to retrieve, e.g. 'ioUse'
            parametername: Parameter to retrieve, e.g. 'TableID'
            cache_mode: 'r' to try read from cache first, 'w' to write to cache
            kwargs: Additional parameters, such as tableid or year

        Examples:

        >>> datasetname='ioUse'
        >>> tableid=259
        >>> year = 2017
        >>> parametername = 'TableID'
        >>> bea.get()
        >>> bea.get(datasetname)
        >>> bea.get(datasetname, parametername=parametername)
        >>> bea.get('ioUse')
        >>> df = bea.get(datasetname='ioUse', tableid=259, year=2018)
        >>> df = bea.get(datasetname='GDPbyIndustry', parametername='Industry')
        """
        url = 'https://apps.bea.gov/api/data?&UserID=' + self.userid
        if not datasetname:
            url += '&method=GETDATASETLIST'
        else:
            url += '&datasetname=' + datasetname
            if parametername:
                url += '&method=GetParameterValues'
                url += '&parametername=' + parametername
            else:
                if len(kwargs) == 0:
                    url += '&method=GetParameterList'
                else:
                    url += '&method=GetData'
                    for k,v in kwargs.items():
                        if isinstance(v, list):
                            v = ",".join(v)
                        url += "&" + str(k) + "=" + str(v)
        if self.verbose: print(url, str(kwargs))

        if 'r' in cache_mode and self.rdb and self.rdb.redis.exists(url):
            if self.verbose: print('(BEA get rdb)', url)
            return self.rdb.load(url)
        response = requests_get(url)
        f = io.BytesIO(response.content)
        data = json.loads(f.read().decode('utf-8'))
        if not datasetname:
            df = DataFrame(data['BEAAPI']['Results']['Dataset'])
        elif parametername:
            df = DataFrame(data['BEAAPI']['Results']['ParamValue'])
        elif len(kwargs) == 0:
            df = DataFrame(data['BEAAPI']['Results']['Parameter'])
        else:
            df = DataFrame(data['BEAAPI']['Results'][0]['Data'])
        df.columns = df.columns.map(str.lower).map(str.rstrip)
        if 'index' in kwargs:
            df = df.set_index(kwargs['index'])
        if 'w' in cache_mode and self.rdb:
            self.rdb.dump(url, df)
        return df


    def read_ioUse_xls(self, year: int, cache_mode: str = "rw",
                       source: str | None = None) -> DataFrame:
        """Helper to load a year's ioUSE table from vintage xls on website
        
        Args:
            year: year of IoUse to fetch
            cache_mode: 'r' to try read from cache first, 'w' to write to cache
            source: url or filename to read from
        """

        filename = {  # early years' ioUse tables
            1963: 'IoUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx',
            1947: 'IoUse_Before_Redefinitions_PRO_1947-1962_Summary.xlsx'}
        url = (source or BEA._bea_url)
        url += ('' if url.endswith('/') else '/')
        url += filename[1963 if year >= 1963 else 1947] + '_' + str(year)
        if self.verbose: print(url)

        if 'r' in cache_mode and self.rdb and self.rdb.redis.exists(url):
            if self.verbose: print('(BEA get rdb)', url)
            return self.rdb.load(url)
        
        x = pd.ExcelFile(filename)   # x.sheet_names
        df = x.parse(str(year))      # parse the sheet for the desired year

        # seek cells with "Code" in top left, and startswith "T0" at corners
        top, = np.where(df.iloc[:, 0].astype(str).str.startswith('Code'))
        right, = np.where(df.iloc[top[0],:].astype(str).str.startswith('T0'))
        bottom, = np.where(df.iloc[:, 0].astype(str).str.startswith('T0'))

        # stack all data columns into {result}
        result = []
        for col in range(2, right[0]):
            out = DataFrame(data=list(df.iloc[top[0]+1:bottom[0], 0]),
                            columns=['rowcode'])
            out['datavalue'] = list(pd.to_numeric(
                df.iloc[top[0]+1:bottom[0], col], errors='coerce'))
            out['colcode'] = str(df.iloc[top[0], col])
            result.append(out[out['datavalue'].notna()])
        result = pd.concat(result,
                           axis=0,
                           ignore_index=True,
                           sort=True)
        if 'w' in cache_mode and self.rdb:
            self.rdb.dump(url, result)   # save to redis
        return result

    def read_ioUse(self, year: int, vintage: int = 0,
                   cache_mode: str = "rw") -> DataFrame:
        """Load ioUse table from BEA web api (or xls if early vintage)

        Args:
            year: Year of IO-Use to load
            vintage: Year of sectoring; allows different eras to be compared
            cache_mode: 'r' to try read from cache first, 'w' to write to cache

        Returns:
            DataFrame in stacked form, flows amounts in 'datavalue' column, and
            columns 'rowcode', 'colcode' label maker and user industry

        Notes:

        - rows, column codes that start with ('T','U','V','Other') are dropped
        - 'F': final_use, 
        - 'G': govt, 
        - 'T': total&tax, 
        - 'U': used, 
        - 'V':value_added, 
        - 'O':imports
        - generally, should drop = ('F','T','U','V','Other')

        Examples:

        >>> bea=BEA()
        >>> data = bea.read_ioUse(1996)
        >>> sql.load_dataframe('ioUse', data, index_label=None, if_exists='replace')
        >>> data = sql.select('ioUse', where={'year': 2017, 'tableid' : 259})
        >>> df = data.pivot(index='rowcode', columns='colcode', values='datavalue')
        """

        if year >= 1997: # after 1997, via web apis; before, in xls spreadsheets
            df = self.get(**self.params['ioUse'], year=year, cache_mode="rw")
            df.columns = df.columns.map(str.lower)
            df = df[['colcode','rowcode','datavalue']]
            df['datavalue'] = pd.to_numeric(df['datavalue'], errors='coerce')
            df = df[df['datavalue'] > 0]
        else:
            df = self.read_ioUse_xls(year)

        # merge industries for vintage year, using historical sectoring scheme
        if not vintage:
            vintage = year   # no vintage specified, so use same year
        for key, codes in self.bea_vintages_.items():  # step thru vintages
            if vintage < key:         # if required vintage predates scheme year
                for k,v in codes.items():
                    if self.verbose: print(k, '<--', v)
                    oldRows = df[df['rowcode'].isin(v)].drop(columns='rowcode')
                    keep = [c for c in oldRows.columns if c != 'datavalue']
                    newRows = oldRows.groupby(by=keep, as_index=False).sum()
                    newRows['rowcode'] = k
                    df = pd.concat([df, newRows],
                                   ignore_index=True,
                                   sort=True)
                    oldCols = df[df['colcode'].isin(v)].drop(columns='colcode')
                    keep = [c for c in oldCols.columns if c != 'datavalue']
                    newCols = oldCols.groupby(by=keep, as_index=False).sum()
                    newCols['colcode'] = k
                    df = pd.concat([df, newCols],
                                   ignore_index=True,
                                   axis = 0,
                                   sort=True)
                    df = df[~df['colcode'].isin(v)]
                    df = df[~df['rowcode'].isin(v)]
        keep = ('F','G')
        drop = ('T','U','V','Other')
        df = df[~df['colcode'].str.startswith(drop)
                & ~df['rowcode'].str.startswith(drop)]        
        return df

    

if __name__ == "__main__":
    from secret import credentials, paths
    from finds.database import SQL, RedisDB

    downloads = paths['scratch']
    sql = SQL(**credentials['sql'])
    rdb = None   #RedisDB(**credentials['redis'])
    bea = BEA(rdb, **credentials['bea'])

    # Get sectoring table
    df = BEA.sectoring(1997)
    
    raise Exception
    
    # Read GDPbyIndustry descriptions
    df = bea.get(datasetname='GDPbyIndustry', parametername='Industry')
    print(df)
    df = bea.get(datasetname='GDPbyIndustry', parametername='Industry',
                 index='key')
    print(df)
    

    raise Exception

    # Read ioUse table, and regroup by vintage year scheme
    years = list(range(2021, 1946, -1))
    ioUses = {}
    for vintage in [1997, 1963, 1947]:
        for year in [y for y in years if y >= vintage]:
            df = bea.read_ioUse(year, vintage=vintage)
            ioUses[(vintage, year)] = df

    """
    for scheme in [1947, 1963, 1997]:   # BEA sectoring scheme        
        # from BEA:  code -> long description
        desc = bea.get(**BEA.params['gdpindustry'])['desc']
        labels = {BEA.bea_industry_[c]:
                  (desc[c] if c in desc.index else BEA.bea_industry_[c])
                  for c in BEA.bea_industry_.index}   # code -> short label

    rdb = RedisDB(**credentials['redis'])  # None
    bea = BEA(rdb, **credentials['bea'])

    # from BEA:  code -> long description
    desc = bea.get(**BEA._params['gdpindustry'])['desc']
    labels = {BEA.bea_industry_[c]:
              (desc[c] if c in desc.index else BEA.bea_industry_[c])
              for c in BEA.bea_industry_.index}   # code -> short label

    years = list(range(1947, 2021))
    ioUses = {}
    for vintage in [1997, 1963, 1947]:
        for year in [y for y in years if y >= vintage]:
            df = bea.read_ioUse(year, vintage=vintage)
            ioUses[(vintage, year)] = df
    print(f"Sectoring vintage year {vintage}: {len(ioUses)} records")
    """
