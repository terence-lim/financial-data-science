"""
the dives.econs module defines classes and methods for manipulating economic data sources
"""
# The MIT License
#
# Copyright (c) 2020 Terence Lim
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
import re
import requests, io, zipfile, json

try:
    import secret
    verbose = secret.value('verbose')
    _h = secret.value('url_header')
except:
    verbose = 0
    
class Sectoring(object):
    """routines to aggregate or translate industry sectorings

    Parameters
    ----------
    method : string
        name of sectoring method, also the label stored in sql
    sql : SQL connection object
        this methods' sectoring data are stored in and retrieved from a SQL table

    Examples
    --------
    sic_from_naics = Sectoring('sic', sql)     # best guess of sic code from naic
    url = 'https://www.naics.com/naics-to-sic-crosswalk-search-results'
    sic_from_naics.load_crosswalk(url)    

    naics_from_sic = Sectoring('naics', sql)   # best guess of naics code from sic
    url = 'https://www.naics.com/sic-naics-crosswalk-search-results'
    naics_from_sic.load_crosswalk(url)

    codes=[10, 100, 1000, 10000]
    naics_from_sic.find(codes)
    sic_from_naics.find(naics_from_sic.find(codes))

    codes5_from_sic = Sectoring('codes5', sql)  # convert sic to Fama-French industry code
    _prefix = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    _resource = _prefix + "ftp/Sic" + codes5_from_sic.method + ".zip"
    _subfile = "Sic" + codes5_from_sic.method + ".txt"
    codes5_from_sic.load_famafrench(_resource, _subfile)
    
    url = 'https://www.bls.gov/ppi/ppistonagg.htm'
    html = requests.get(url,headers=_h).content
    df = pd.read_html(html)

    """
    _schema = {'table': 'sectors',               # to store translations in SQL
               'fields': [
                   ['code', 'INT(11)'],
                   ['method', 'VARCHAR(8)'],
                   ['name', 'VARCHAR(16)'],
                   ['description','VARCHAR(128)']],
               'primary': ['method','code','name'],
               'indexes': []}

    _vintages = {2200: {'531' : ['HS','ORE']},   # to group IO-Use table into coarser industries
                 1963: {'48' : ['481','482','483','484','485','486','487OS'],
                        '51' : ['511','512','513','514'],
                        '52' : ['521CI','523','524','525'],
                        '54' : ['5411','5415','5412OP'],
                        '56' : ['561','562'],
                        '62' : ['621','622','623','624','622HO'],
                        '71' : ['711AS','713']},
                 1997: {'44RT' : ['441', '445', '452', '4A0'],
                        'GFG' : ['GFGD', 'GFGN'],
                        '622HO' : ['622','623']}}
    
    _bealabels = {'11': 'Agriculture',   # my abbreviations of BEA industry names for plots
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
                  '56': 'Administrative and waste management services',
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
                  '7': 'Arts, entertainment, recreation, accommodation, and food services',
                  '71': 'Arts, entertainment, recreation',
                  '711AS': 'Performing arts',
                  '713': 'Recreation',
                  '72': 'Accommodation. food services',
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
                  'GSLE': 'State local enterprises'}

    def __init__(self, method, sql):
        self.method = method
        self.sql = sql
        self.sectors = DataFrame()
        if self.sql.exists_table(self._schema['table']):   # if exists in SQL, then load
            self.sectors = self.sql.select(
                self._schema['table'], where = {'method':method}).set_index('code')
            self.sectors.index = self.sectors.index.astype(int)
            try:
                self.sectors['name'] = self.sectors['name'].astype(int)
            except:
                pass
        else:
            self.sql.create_table(**self._schema)

    def load_crosswalk(self, url):
        """load sic to/from naics crosswalks from naics.com

        Examples
        --------
        url = 'https://www.naics.com/sic-naics-crosswalk-search-results'
        url = 'https://www.naics.com/naics-to-sic-crosswalk-search-results'
        """
        if url.find('to-sic') >= 0:      # guess direction of conversion from url string
            code_label = 'NAICS Code'
            name_label = 'SIC Code'
            desc_label = 'SIC Description'
        else:
            code_label = 'SIC Code'
            name_label = 'NAICS Code'
            desc_label = 'NAICS Description'

        # scrape crosswalk from naics.com website into a DataFrame
        df = pd.read_html(requests.get(url, headers=_h).content, header=0)
        self.sectors = max(df, key=len).astype(str)
        self.sectors.columns = [re.sub(r'[^\x20-\x7E]',r' ', c) for c in self.sectors.columns]

        # rename the columns, based on the direction of conversion
        self.sectors.rename(columns = {code_label : 'code',
                                       name_label : 'name',
                                       desc_label : 'description'},
                            inplace = True)

        # clean up DataFrame, set its index, and save to sql database
        keep = self.sectors.code.str.isnumeric() & self.sectors.name.str.isnumeric()
        self.sectors = self.sectors.loc[keep, ['code','name','description']].set_index('code')
        self.sectors.index = self.sectors.index.astype(int)
        self.sectors.name = self.sectors.name.astype(int)
        self.sectors['method'] = self.method
        self.sql.delete(self._schema['table'], where={'method' : self.method})  # flush existing table
        self.sql.load_dataframe(self._schema['table'], self.sectors, index_label='code')

    def load_bea(self, year):
        """load BEA industry sectoring definitions, from naics codes, in xls on BEA website

        Notes
        -----
        # https://www.bea.gov/industry/input-output-accounts-data
        # https://apps.bea.gov/industry/xls/io-annual/Use_SUT_Framework_2007_2012_DET.xlsx
        # Replace "HS" "ORE" with "531"

        Examples
        --------
        bea97_from_naics = Sectoring('bea1997', sql)
        bea97_from_naics.load_bea(1997)
        bea63_from_naics = Sectoring('bea1963', sql)
        bea63_from_naics.load_bea(1963)
        bea47_from_naics = Sectoring('bea1947', sql)
        bea47_from_naics.load_bea(1947)
        bea97_from_naics.find(naics_from_sic.find(codes))

        bea_from_naics = Sectoring('bea1997', sql)
        year = 1997
        """
        filename = {1997 : 'Use_SUT_Framework_2007_2012_DET.xlsx',   # there are 3 historical schmes
                    1963 : 'IoUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx',
                    1947 : 'IoUse_Before_Redefinitions_PRO_1947-1962_Summary.xlsx'}
        prefix = 'https://apps.bea.gov/industry/xls/io-annual/'      # root url

        # get the xls and parse the NAICS Codes sheet
        x = pd.ExcelFile(prefix + filename[year])
        print_debug(str(x.sheet_names))
        df = x.parse('NAICS Codes')

        # extract main groups and labels from columns 0,1. note np.where() returns a tuple, hence beg,
        beg, = np.where(df.iloc[:, 0].astype(str).str[0].str.isdigit())
        labels = {str(df.iloc[i,0]) : df.iloc[i,1] for i in beg}

        # Kludge: some labels missing, so we will in from my abbreviations
        labels = {**labels, **{k:v for k,v in self._bealabels.items() if k not in labels}}

        # Now extract beg and end row of each summary groups and labels from columns 1,2
        beg, = np.where(df.iloc[:, 1].astype(str).str[0].str.isdigit() |   # case starts with num
                        df.iloc[:, 1].astype(str).isin(["HS", "ORE"]))
        end = np.append(beg[1:], len(df.index))

        # for each summary group, parse the naics code ranges from column 6, and store in {results}
        #   e.g. '7223-4, 722514-5' --> [722300, 722514]
        result = DataFrame()
        for imin, imax in zip(beg,end):
            s = [str(c).split('-')[0] for c in df.iloc[imin:imax, 6] if str(c)[0].isdigit()]
            s = [[re.sub("[^0-9]","",a).ljust(6,'0') for a in c.split(',')] for c in s]
            new_df = DataFrame([[int(c)] for inner in s for c in inner], columns=['code'])
            new_df['name'] = str(df.iloc[imin, 1])
            new_df['description'] = df.iloc[imin, 2]
            result = result.append(new_df.drop_duplicates(), sort=False)

        # for earlier vintage years with coarser industries, replace with the coarser group
        for vintage, codes in self._vintages.items():
            if year < vintage:
                for k,v in codes.items():
                    result.loc[result['name'].isin(v), ['name','description']] = [k, labels[k]]

        # clean-up data frame and its index,  and save to sql
        self.sectors = result.drop_duplicates(['code']).set_index('code').sort_index(inplace=False)
        self.sectors['method'] = self.method
        self.sql.delete(self._schema['table'], where={'method' : self.method})  # flush existing table
        self.sql.load_dataframe(self._schema['table'], self.sectors, index_label='code')
        

    def load_famafrench(self, resource=None, subfile=None):
        """load FamaFrench industry sectoring, based on sic, from zipped txt files

        Notes
        -----
        There are multiple schemes of different coarseness, e.g. Siccodes5, Siccodes48 etc
        For example, the industry definitions file for Siccodes49 starts like this:
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
        _prefix = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        _resources = {n : {'resource' : _prefix + str(n) + '.zip',
                     'subfile' : 'Siccodes' + str(n) + '.txt'}
                      for n in [5,49]}
        resource = '/home/terence/Downloads/Siccodes5.txt'
        resource = '/home/terence/Downloads/Siccodes5.zip'
        resource = _prefix + "Data_Library/det_5_ind_port.html"
        resource = _prefix + "ftp/Siccodes5.zip"
        load_definitions(**Siccodes._resources[5])
        """
        labels = DataFrame(columns=['name','description','end'])

        response = requests.get(resource,headers=_h)  # get resource content as byte stream
        f = io.BytesIO(response.content)
        with zipfile.ZipFile(f).open(subfile) as f:   # open text subfile from zip archive
            for line in f:
                items = line.decode('utf-8').rstrip('\n').split()
                #print_debug(items, line)
                if len(items) >= 1:
                    sic = items[0].split('-')         # check if line contains "-"
                    if (len(sic) == 2):                  # "-" separates two sic codes
                        labels.loc[int(sic[0]),'name'] = ind   # append a row
                        labels.loc[int(sic[0]),'description'] = desc
                        labels.loc[int(sic[0]),'end'] = int(sic[1])
                        print_debug(sic[0] + str(labels.loc[int(sic[0])].values))
                    else:
                        if len(items) <= 1: ind = '???'  
                        else:
                            ind = items[1]               # else is industry name and description
                            desc = " ".join(items[2:])
                            if ind == 'Other':
                                other = desc             # "Other" industry often lacks description

        # handle case if last name is "Other" with no sic's
        next_row=(((labels.end//100)+1)*100).astype(int)
        new_df = DataFrame(columns = labels.columns)
        new_df.loc[0,['name','description']] = ['Other',other]
        new_df.loc[max(next_row) ,['name','description']] = ['Other',other]
        for i in range(1,len(labels)-1):
            if next_row.iloc[i] < labels.index[i+1]:
                new_df.loc[next_row.iloc[i],['name','description']] = ['Other',other]

        # clean up dataframe {sectors} and save to sql 
        self.sectors = labels.append(new_df).drop(columns=['end']).sort_index()
        self.sectors['method'] = self.method
        self.sql.delete(self._schema['table'], where={'method' : self.method})
        self.sql.load_dataframe(self._schema['table'], self.sectors, index_label='code')
    
    def find(self, codes):
        """method to lookup sectoring given a raw code"""
        return list(self.sectors['name'].iloc[np.searchsorted(self.sectors.index, codes, side='right')-1])
    
#        if isnum(codes):
#            return self.sectors['name'].iloc[argfloor(self.sectors.index, codes)]
#        return list(map(lambda x: self.sectors['name'].iloc[argfloor(self.sectors.index, x)], codes))


class BEA(Sectoring):
    """class and methods to read and manipulate BEA data

    Parameters
    ----------
    rdb : Redis connection instance, optional (default None)
        to cache data accesssed from BEA web site
    userid : string, optional (default secret.value('bea'))
        register for free with BEA to get key for their web apg

    Examples
    --------
    bea = BEA()
    df = bea.get(**BEA.items['industries'])
    sql.load_dataframe('gdpindustry', df, index_label=None, if_exists='replace')
    df = bea.get(**BEA.items['ioUse'], year=2018)
    """

    _params = {'gdpindustry' : {'datasetname' : 'GDPbyIndustry', # favorite params for BEA's web api
                                'index' : 'key',
                                'parametername' : 'Industry'},
               'ioUse' : {'datasetname' : 'inputoutput',
                          'tableid' : 259}}

    def __init__(self, userid, rdb = None):
        self.userid = userid
        self.rdb = rdb

    def get_label(self, *args):
        """lookup abbreviated label for industry {name}, more suitable for visualizing"""
        return self._bealabels[args[0]] if args else self._bealabels.keys()

    def get_params(self, *args):
        return self._params[args[0]] if args else self._params.keys()
            
    def get(self, datasetname=None, parametername=None, nocache=False, **kwargs):
        """wrapper for BEA web api

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
        print_debug(url + str(kwargs))

        if (not nocache and self.rdb and self.rdb.exists(url)):  # If redis, then check key in cache
            print_debug('(BEA get rdb) ' + url)
            return self.rdb.load(url)
        response = requests.get(url,headers=_h)      # send get request to url
        f = io.BytesIO(response.content)             # to process content as byte stream
        data = json.loads(f.read().decode('utf-8'))  # load from json string
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
        if (not nocache and self.rdb):
            self.rdb.dump(url, df)
        return df

    def load_ioUse_xls(self, year, prefix='https://apps.bea.gov/industry/xls/io-annual/'):
        """helper method to load a year's ioUSE table from old xls"""
        
        # vintage BEA site and filenames associated with early year's ioUse table
        filename = {1963 : 'IoUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx',
                    1947 : 'IoUse_Before_Redefinitions_PRO_1947-1962_Summary.xlsx'}
        filename = prefix + filename[1963 if year >= 1963 else 1947]
        print_debug(filename)

        # parse the sheet for the desired year
        x = pd.ExcelFile(filename)
        x.sheet_names
        df = x.parse(str(year))

        # seek boundary of cells with "Code" in top left, and startswith "T0" at corners
        top, = np.where(df.iloc[:, 0].astype(str).str.startswith('Code'))
        right, = np.where(df.iloc[top[0],:].astype(str).str.startswith('T0'))
        bottom, = np.where(df.iloc[:, 0].astype(str).str.startswith('T0'))

        # stack all data columns into {result}
        result = DataFrame(columns = ['colcode','rowcode','datavalue'])
        for col in range(2, right[0]):
            out = DataFrame(data=list(df.iloc[top[0]+1:bottom[0], 0]), columns=['rowcode'])
            out['datavalue'] = list(pd.to_numeric(df.iloc[top[0]+1:bottom[0], col], errors='coerce'))
            out['colcode'] = str(df.iloc[top[0], col])
            result = result.append(out[out['datavalue'].notnull()], ignore_index=True, sort=True)
        return result

    def load_ioUse(self, year, vintage=None, nocache=False):
        """load ioUse table for a year from BEA web api (or xls if early vintage)
        
        Parameters
        ----------
        year : int
            year of IO-Use to load
        vintage : int, optional
            year of sectoring to apply; allows years from different eras to be compared
        nocache : boolean, optional (default False)
            whether to use redis cache for data accessed from BEA api's

        Returns
        -------
        df : DataFrame
            in stacked form, flows amounts in 'datavalue' column, and
            columns 'rowcode', 'colcode' labeling supplier and user industry respectively

        Notes
        -----
        rows and column industry codes that start with ('T','U','V','Other') are dropped
        'F': final_use, 'G': govt, 'T': total&tax, 'U': used, 'V':value_added, 'O':imports
        generally, should drop = ('F','T','U','V','Other')

        Examples
        --------
        bea=BEA()
        data = bea.load_ioUse(1996)

        sql.load_dataframe('ioUse', data, index_label=None, if_exists='replace')
        data = sql.select('ioUse', where={'year': 2017, 'tableid' : 259})
        df = data.pivot(index='rowcode', columns='colcode', values='datavalue')
        """

        if year >= 1997:   # after 1997, available via web apis; before, in xls spreadsheets
            df = self.get(**self.get_params('ioUse'), year=year, nocache=False)
            df.columns = df.columns.map(str.lower)      # clean up column names and types
            df = df[['colcode','rowcode','datavalue']]
            df['datavalue'] = pd.to_numeric(df['datavalue'], errors='coerce')
            df = df[df['datavalue'] > 0]
        else:
            df = self.load_ioUse_xls(year)

        # merge industries depending on vintage year desired, using a historical sectoring scheme
        if vintage is None:
            vintage = year   # no vintage year specified, so use same year's definitions
        for key, codes in self._vintages.items():  # scroll through historical schemes and aggregate
            if vintage < key:                      # if required vintage year predates scheme year
                for k,v in codes.items():
                    print_debug(str(k) + ' <-- ' + str(v))
                    oldRows = df[df['rowcode'].isin(v)].drop(columns=['rowcode'])
                    keep = [c for c in oldRows.columns if c not in ['datavalue']]
                    newRows = oldRows.groupby(by=keep, as_index=False).sum()
                    newRows['rowcode'] = k
                    df = df.append(newRows, ignore_index=True, sort=True)
                    oldCols = df[df['colcode'].isin(v)].drop(columns=['colcode'])
                    keep = [c for c in oldCols.columns if c not in ['datavalue']]
                    newCols = oldCols.groupby(by=keep, as_index=False).sum()
                    newCols['colcode'] = k
                    df = df.append(newCols, ignore_index=True, sort=True)
                    df = df[~df['colcode'].isin(v)]
                    df = df[~df['rowcode'].isin(v)]
        keep = ('F','G')                # F:final_use G:govt
        drop = ('T','U','V','Other')    # T:total&tax U:used V:value_added O:imports
        df = df[~df['colcode'].str.startswith(drop) & ~df['rowcode'].str.startswith(drop)]        
        return df
        

