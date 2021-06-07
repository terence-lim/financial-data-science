"""Implement industry sectoring, and convenience methods for BEA data

- pandas, requests, sqlalchemy, json
- SIC, NAICS, Fama-French industry crosswalks
- Bureau of Economic Analysis, Input-Output Use Tables

Author: Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types
import re, requests, io, zipfile, json
from sqlalchemy import Integer, String, Column
from pandas.api import types
try:
    from settings import ECHO
except:
    ECHO = False

class Sectoring:
    """Base class to implement industry sector grouping schemes

    Parameters
    ----------
    sql : SQL object
        SQL database connection
    scheme : str 
        name of sectoring scheme
        'sic1', 'sic2', 'sic3': group 4-digit sic to 1, 2, or 3-digits
        'sic': translate 6-digit naics to 4-digit sic
        'naics': translate 4-digit sic to 6-digit naics 
        'bea1997', 'bea1963', 'bea1947': group naics to historical bea schemes
        'codes48', ..., 'codes5': group 4-digit sic to FamaFrench scheme
    fillna : value, default is None
        value to return if source label is out of range
    """
    bea_industry = Series({ # custom abbreviations of BEA industries
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
    bea_vintages = {  # group BEA IO-Use table into coarser
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
    
    def __init__(self, sql, scheme, fillna=None, echo=ECHO):
        """Initiatialize a selected sectoring scheme"""
        self.scheme = scheme.lower()
        self.sql = sql
        self.fillna = fillna
        self.echo_ = ECHO
        self.table = sql.Table('sectoring',
                               Column('code', Integer, primary_key=True),
                               Column('name', String(16), primary_key=True),
                               Column('scheme', String(8), primary_key=True),
                               Column('description', String(128)))
        self.table.create(checkfirst=True)
        if self.scheme.startswith('sic') and len(self.scheme) > 3:
            digits = int(scheme[3])
            n = 4 - digits
            self.sectors = DataFrame(index=np.arange(0, 10**digits) * (10**n))
            self.sectors['name'] = np.arange(0, 10**digits)
            self.sectors['description'] = [f"{s}{'0'*n}-{s}{'9'*n}"
                                           for s in self.sectors['name']]
            self.sectors['scheme'] = f"sic{digits}"
        else:
            self.sectors = DataFrame()
            self.sectors = self.sql.read_dataframe(
                "SELECT * FROM {} WHERE scheme = '{}'".format(
                    self.table.key, self.scheme)).set_index('code')
            #self.sectors.index = self.sectors.index.astype(int)
            try:
                self.sectors['name'] = self.sectors['name'].astype(int)
            except:
                pass
            
    def _print(self, *args, echo=None):
        if echo or self.echo_:
            print(*args)
            
    def __call__(self, code):
        """Lookup sectoring group given raw input code/s"""
        if types.is_list_like(code):
            return [self(c) for c in code]
        found = np.searchsorted(self.sectors.index, code, side='right')
        return self.sectors['name'].iloc[found-1] if found > 0 else self.fillna

    def __getitem__(self, code):
        """Lookup sectoring group given raw input code/s"""
        return self(code)

    def load(self, source=None):
        """Entry method to call suitable loader for desired scheme"""
        if self.scheme == 'naics':
            return self.load_dataframe(Sectoring.get_crosswalk(
                code='SIC Code', name='NAICS Code', desc='NAICS Description',
                source=source))
        elif self.scheme == 'sic':
            return self.load_dataframe(Sectoring.get_crosswalk(
                code='NAICS Code', name='SIC Code', desc='SIC Description',
                source=source))
        elif self.scheme.startswith('codes'):
            return self.load_dataframe(Sectoring.get_famafrench(
                self.scheme, source=source))
        elif self.scheme.startswith('bea'):
            return self.load_dataframe(Sectoring.get_bea(
                int(self.scheme[3:]), source=source))
        else:
            return None

    def load_dataframe(self, df):
        """Helper to upload dataframe with code in index, name. desc to SQL"""
        if df is not None:     # columns=['name', 'description']
            self.sectors = df
            self.sectors['scheme'] = self.scheme
            delete = self.table.delete().where(
                self.table.c['scheme'] == self.scheme)
            self.sql.run(delete)
            self.sql.load_dataframe(
                table=self.table.key, df=self.sectors, index_label='code')
            return self.sectors['name'].unique()

    @staticmethod
    def get_famafrench(scheme, source=None):
        """Load FamaFrench sectoring based on sic-4, from website or zipfile

        Notes
        -----
        There are different schemes of coarseness, e.g. Siccodes5, Siccodes48
        For example, the industry definitions file for Siccodes49 looks like:
 1 Agric  Agriculture
          0100-0199 Agricultural production - crops
          0200-0299 Agricultural production - livestock
          0700-0799 Agricultural services
          0910-0919 Commercial fishing
          2048-2048 Prepared feeds for animals

 2 Food   Food Products
          2000-2009 Food and kindred products
          2010-2019 Meat products
          2020-2029 Dairy products

        Examples
        --------
        subfile = 'Siccodes5.txt'
        prefix = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        resource = _prefix + "ftp/Siccodes5.zip"
        """
        if source is None:
            prefix_ = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            source = prefix_ + f"ftp/Sic{scheme}.zip"
        if source.startswith('http'):
            response = requests.get(source)
            f = io.BytesIO(response.content)
            subfile = 'Sic' + scheme + '.txt'
            open_ = zipfile.ZipFile(f).open
        else:
            if source.endswith('.zip'):            
                open_ = zipfile.ZipFile(source).open
                subfile = 'Sic' + scheme + '.txt'
            else:
                open_ = open
                subfile = source
        labels = DataFrame(columns=['name','description','end'])
        with open_(subfile) as f:   # open text subfile from zip archive
            for line in f:
                items = line.decode('utf-8').rstrip('\n').split()
                if len(items) >= 1:
                    sic = items[0].split('-')
                    if (len(sic) == 2):      # "-" separates two sic codes
                        labels.loc[int(sic[0]),'name'] = ind   # append a row
                        labels.loc[int(sic[0]),'description'] = desc
                        labels.loc[int(sic[0]),'end'] = int(sic[1])
                        self._print(sic[0], labels.loc[int(sic[0])].values)
                    else:
                        if len(items) <= 1:
                            ind = '???'  
                        else:
                            ind = items[1]   # else is name and description
                            desc = " ".join(items[2:])
                            if ind == 'Other':
                                other = desc # "Other" often lacks description
                                
        # handle case if last sector is "Other" with no sic's:
        #   assign next sic2 not in table to be an "Other" sector
        next_sic2 = (((labels.end // 100) + 1) * 100).astype(int)
        df = DataFrame(columns = labels.columns)
        df.loc[0, ['name', 'description']] = ['Other', other]
        df.loc[max(next_sic2) ,['name', 'description']] = ['Other', other]
        if len(np.unique(labels.name)) < int(scheme[5:]):  # "Other" has no sics
            for i in range(len(labels)-1):
                if (next_sic2.iloc[i] < labels.index[i+1] and
                    next_sic2.iloc[i] not in labels.index):
                    df.loc[next_sic2.iloc[i], 'name'] = 'Other'
                    df.loc[next_sic2.iloc[i], 'description'] = other

        # clean up dataframe {sectors} and save to sql 
        sectors = labels.append(df).drop(columns=['end']).sort_index()
        return sectors
    
    @staticmethod
    def get_crosswalk(code, name, desc, source=None):
        """Load sic - naics crosswalks from naics.com"""
        if source is None:
            url = 'https://www.naics.com/'
            url = {'N' : url + 'sic-naics-crosswalk-search-results',
                   'S' : url + 'naics-to-sic-crosswalk-search-results'}
            source = requests.get(url[name[0]]).content
            
        # scrape crosswalk from naics.com website into a DataFrame
        df = pd.read_html(source, header=0)
        sectors = max(df, key=len).astype(str)
        sectors.columns = [re.sub(r'[^\x20-\x7E]',r' ', c)
                           for c in sectors.columns]

        # rename the columns, based on the direction of conversion
        sectors = sectors.rename(columns={
            code: 'code', name: 'name', desc: 'description'})

        # clean up DataFrame, set its index, and save to sql database
        keep = sectors['code'].str.isnumeric() & sectors['name'].str.isnumeric()
        sectors = sectors.loc[keep].set_index('code')
        sectors.index = sectors.index.astype(int)
        sectors['name'] = sectors['name'].astype(int)
        return sectors[['name', 'description']]
        
    @staticmethod
    def get_bea(year, source=None):
        """Load BEA definitions, based on naics codes, from xls on BEA website

        Notes
        -----
        # https://www.bea.gov/industry/input-output-accounts-data
        # https://apps.bea.gov/industry/xls/io-annual/
        #     Use_SUT_Framework_2007_2012_DET.xlsx
        # Replace "HS" "ORE" with "531"
        """
        if source is None:
            filename = {   # there are 3 historical schmes
                1997: 'Use_SUT_Framework_2007_2012_DET.xlsx',   
                1963: 'IoUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx',
                1947: 'IoUse_Before_Redefinitions_PRO_1947-1962_Summary.xlsx'}
            prefix = 'https://apps.bea.gov/industry/xls/io-annual/'  # root url
            if year not in filename:
                raise Exception('bea year not in ' + str(list(filename.keys())))
            source = prefix + filename[year]

        # get the xls and parse the NAICS Codes sheet
        x = pd.ExcelFile(source)
        self._print(x.sheet_names)
        df = x.parse('NAICS Codes')

        # extract main groups and labels from columns 0,1
        beg = np.where(df.iloc[:, 0].astype(str).str[0].str.isdigit())[0]
        labels = {str(df.iloc[i,0]) : df.iloc[i,1] for i in beg}

        # Kludge: some labels missing, so fill from custom abbreviations
        labels.update({k: v for k,v in Sectoring.bea_industry.items()
                       if k not in labels})

        # Now extract beg and end row of groups and labels from columns 1,2
        beg, = np.where(df.iloc[:, 1].astype(str).str[0].str.isdigit()
                        | df.iloc[:, 1].astype(str).isin(["HS", "ORE"]))
        end = np.append(beg[1:], len(df.index))

        # for each group, parse naics code ranges from col 6, store in {results}
        #   e.g. '7223-4, 722514-5' --> [722300, 722514]
        # also, parse leading digits of summary name if at least len 2 (pad 0's)
        result = DataFrame()
        for imin, imax in zip(beg, end):
            s = [str(c).split('-')[0]
                 for c in df.iloc[imin:imax, 6] if str(c)[0].isdigit()]
            s = [[re.sub("[^0-9]","",a).ljust(6,'0')
                  for a in c.split(',')] for c in s if c not in ['491']] # postal
            new_df = DataFrame([[int(c)] for inner in s for c in inner],
                               columns=['code'])
            name = str(df.iloc[imin, 1])
            m = re.match('\d+', name)
            if m and len(m.group()) >= 2:
                new_df = new_df.append({'code': int(m.group().ljust(6, '0'))},
                                       ignore_index=True) 
            new_df['name'] = name
            new_df['description'] = df.iloc[imin, 2]
            result = result.append(new_df.drop_duplicates(), ignore_index=True)

        # replace earlier vintage years with coarser industries
        for vintage, codes in Sectoring.bea_vintages.items():
            if year < vintage:
                for k,v in codes.items():
                    result.loc[result['name'].isin(v),
                               ['name','description']] = [k, labels[k]]

        # clean-up data frame and its index
        return result.drop_duplicates(['code']).set_index('code').sort_index()

class BEA(Sectoring):
    """Class and methods to read and manipulate BEA data

    Parameters
    ----------
    rdb : Redis connection instance, optional (default None)
        to cache data accesssed from BEA web site
    userid : string
        register for free with BEA to get key for their web apg

    Examples
    --------
    bea = BEA()
    df = bea.get(**BEA.items['industries'])
    sql.load_dataframe('gdpindustry', df, index_label=None, if_exists='replace')
    df = bea.get(**BEA.items['ioUse'], year=2018)
    """
    params_ = {    # commonly-used parameters for BEA web api
        'gdpindustry': {'datasetname': 'GDPbyIndustry', 
                        'index': 'key',
                        'parametername': 'Industry'},
        'ioUse': {'datasetname': 'inputoutput',
                  'tableid': 259}}

    def __init__(self, rdb, userid, echo=ECHO):
        """Open connection to BEA web api"""
        self.rdb = rdb
        self.userid = userid
        self.echo_ = echo

    def _print(self, *args, echo=None):
        if echo or self.echo_:
            print(*args)

    def get(self, datasetname=None, parametername=None,
            use_cache=True, **kwargs):
        """Wrapper to execute common BEA web api calls

        Examples
        --------
        datasetname='ioUse'
        tableid=259
        year = 2017
        parametername = 'TableID'
        bea.get()
        bea.get(datasetname)
        bea.get(datasetname, parametername = parametername)
        bea.get('ioUse')
        df = bea.get(datasetname='ioUse', tableid=259, year=2018)
        df = bea.get(datasetname='GDPbyIndustry', parametername='Industry')
        """
        url = 'https://apps.bea.gov/api/data?&UserID=' + self.userid
        if datasetname is None:
            url += '&method=GETDATASETLIST'
        else:
            url += '&datasetname=' + datasetname
            if parametername is not None:
                url += '&method=GetParameterValues'
                url += '&parametername=' + parametername
            else:
                if len(kwargs) == 0:
                    url += '&method=GetParameterList'
                else:
                    url += '&method=GetData'
                    for k,v in kwargs.items():
                        if isinstance(v, list): v = ",".join(v)
                        url += "&" + str(k) + "=" + str(v)
        self._print(url, str(kwargs))

        if use_cache and self.rdb and self.rdb.redis.exists(url):
            self._print('(BEA get rdb)', url)
            return self.rdb.load(url)
        response = requests.get(url)
        f = io.BytesIO(response.content)
        data = json.loads(f.read().decode('utf-8'))
        if datasetname is None:
            df = DataFrame(data['BEAAPI']['Results']['Dataset'])
        elif parametername is not None:
            df = DataFrame(data['BEAAPI']['Results']['ParamValue'])
        elif len(kwargs) == 0:
            df = DataFrame(data['BEAAPI']['Results']['Parameter'])
        else:
            df = DataFrame(data['BEAAPI']['Results']['Data'])
        df.columns = df.columns.map(str.lower).map(str.rstrip)
        if 'index' in kwargs:
            df = df.set_index(kwargs['index'])
        if use_cache is not None and self.rdb:
            self.rdb.dump(url, df)
        return df

    def read_ioUse_xls(self, year, use_cache=True,
                       prefix='https://apps.bea.gov/industry/xls/io-annual/'):
        """Helper to load a year's ioUSE table from vintage xls on website"""

        filename = {  # early year's ioUse tables
            1963: 'IoUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx',
            1947: 'IoUse_Before_Redefinitions_PRO_1947-1962_Summary.xlsx'}
        filename = prefix + filename[1963 if year >= 1963 else 1947]
        url = filename + '_' + str(year)
        self._print(url)

        if use_cache and self.rdb and self.rdb.redis.exists(url):
            self._print('(BEA get rdb)', url)
            return self.rdb.load(url)
        
        x = pd.ExcelFile(filename)   # x.sheet_names
        df = x.parse(str(year))      # parse the sheet for the desired year

        # seek cells with "Code" in top left, and startswith "T0" at corners
        top, = np.where(df.iloc[:, 0].astype(str).str.startswith('Code'))
        right, = np.where(df.iloc[top[0],:].astype(str).str.startswith('T0'))
        bottom, = np.where(df.iloc[:, 0].astype(str).str.startswith('T0'))

        # stack all data columns into {result}
        result = DataFrame(columns = ['colcode','rowcode','datavalue'])
        for col in range(2, right[0]):
            out = DataFrame(data=list(df.iloc[top[0]+1:bottom[0], 0]),
                            columns=['rowcode'])
            out['datavalue'] = list(pd.to_numeric(
                df.iloc[top[0]+1:bottom[0], col], errors='coerce'))
            out['colcode'] = str(df.iloc[top[0], col])
            result = result.append(out[out['datavalue'].notna()],
                                   ignore_index=True, sort=True)
        if use_cache is not None and self.rdb:
            self.rdb.dump(url, result)
        return result

    def read_ioUse(self, year, vintage=None, use_cache=True):
        """Load ioUse table from BEA web api (or xls if early vintage)

        Parameters
        ----------
        year : int
            year of IO-Use to load
        vintage : int, optional
            year of sectoring to apply; allows different eras to be compared
        use_cache : boolean, optional (default True)
            whether to use redis cache for data accessed from BEA api's

        Returns
        -------
        df : DataFrame
            in stacked form, flows amounts in 'datavalue' column, and
            columns 'rowcode', 'colcode' label maker and user industry

        Notes
        -----
        rows, column codes that start with ('T','U','V','Other') are dropped
        'F': final_use, 'G': govt, 'T': total&tax, 
        'U': used, 'V':value_added, 'O':imports
        generally, should drop = ('F','T','U','V','Other')

        Examples
        --------
        bea=BEA()
        data = bea.read_ioUse(1996)
        sql.load_dataframe('ioUse', data, index_label=None, if_exists='replace')
        data = sql.select('ioUse', where={'year': 2017, 'tableid' : 259})
        df = data.pivot(index='rowcode', columns='colcode', values='datavalue')
        """

        if year >= 1997: # after 1997, via web apis; before, in xls spreadsheets
            df = self.get(**self.params_['ioUse'], year=year, use_cache=True)
            df.columns = df.columns.map(str.lower)
            df = df[['colcode','rowcode','datavalue']]
            df['datavalue'] = pd.to_numeric(df['datavalue'], errors='coerce')
            df = df[df['datavalue'] > 0]
        else:
            df = self.read_ioUse_xls(year)

        # merge industries for vintage year, using historical sectoring scheme
        if vintage is None:
            vintage = year   # no vintage specified, so use same year
        for key, codes in self.bea_vintages.items():  # step thru schemes
            if vintage < key:         # if required vintage predates scheme year
                for k,v in codes.items():
                    self._print(k, '<--', v)
                    oldRows = df[df['rowcode'].isin(v)].drop(columns='rowcode')
                    keep = [c for c in oldRows.columns if c != 'datavalue']
                    newRows = oldRows.groupby(by=keep, as_index=False).sum()
                    newRows['rowcode'] = k
                    df = df.append(newRows, ignore_index=True, sort=True)
                    oldCols = df[df['colcode'].isin(v)].drop(columns='colcode')
                    keep = [c for c in oldCols.columns if c != 'datavalue']
                    newCols = oldCols.groupby(by=keep, as_index=False).sum()
                    newCols['colcode'] = k
                    df = df.append(newCols, ignore_index=True, sort=True)
                    df = df[~df['colcode'].isin(v)]
                    df = df[~df['rowcode'].isin(v)]
        keep = ('F','G')
        drop = ('T','U','V','Other')
        df = df[~df['colcode'].str.startswith(drop)
                & ~df['rowcode'].str.startswith(drop)]        
        return df

if False:
    import os
    from finds.sectors import Sectoring, BEA
    from finds.database import SQL, Redis
    from finds.structured import PSTAT, CRSP
    from finds.busday import BusDay
    from settings import settings
    
    downloads = settings['remote']
    sql = SQL(**settings['sql'])
    bd = BusDay(sql)
    rdb = Redis(**settings['redis'])  # None
    crsp = CRSP(sql, bd, rdb)
    pstat = PSTAT(sql, bd)
    bea = BEA(rdb, **settings['bea'])

    scheme = 'sic2'
    codes = Sectoring(sql, scheme)
    
    scheme = 'sic3'
    codes = Sectoring(sql, scheme)

    for scheme in [5, 10, 12, 17, 30, 38, 48, 49]:
        codes = Sectoring(sql, f"codes{scheme}")
        codes.load(source=None)

    scheme = 'SIC'  # SIC from NAICS Crosswalk
    codes = Sectoring(sql, scheme)
    codes.load(source=None)
    
    scheme = 'NAICS'  # NAICS from SIC Crosswalk
    codes = Sectoring(sql, scheme)
    codes.load(source=None)

    for scheme in [1947, 1963, 1997]:   # BEA sectoring scheme
        codes = Sectoring(sql, f"bea{scheme}")
        codes.load(source=None)

    # from BEA:  code -> long description
    desc = bea.get(**BEA.params_['gdpindustry'])['desc'] # industry descriptions

    # generate: short label -> long description
    labels = {BEA.bea_industry[code]:
              (desc[code] if code in desc.index else BEA.bea_industry[code])
              for code in BEA.bea_industry.index}  # code -> short label

    years = np.arange(1947, 2020)
    ioUses = {}
    for vintage in [1997, 1963, 1947]:
        for year in [y for y in years if y >= vintage]:
            df = bea.read_ioUse(year, vintage=vintage)
            ioUses[(vintage, year)] = df
        print(f"Sectoring vintage year {vintage}: {len(ioUses)} total records")
