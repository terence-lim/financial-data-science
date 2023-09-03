"""Implement industry sectoring

- Bureau of Economic Analysis: Input-Output Use Tables
- SIC, NAICS crosswalks: https://www.naics.com/
- Fama-French industry codes: 
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"

Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types
import re
import io
import zipfile
from typing import Dict, List, Any
from sqlalchemy import Integer, String, Column
from finds.database.sql import SQL
from .readers import requests_get
from .ffreader import FFReader
from .bea import BEA

_VERBOSE = 1

class Crosswalk:

    @staticmethod
    def sectoring(code: str, name: str, desc: str,
                  source: str = "") -> DataFrame:
        """Load sic - naics crosswalks from  https://www.naics.com/

        Args:
            code: Column name for input code (i.e. map from)
            name: Column name of output code (i.e. map to)
            desc: Description of output code
            source: location to scrape (default is blank to scrape from naics.com)

        Returns:
            DataFrame indexed by input code
        """
        if not source:
            url = 'https://www.naics.com/'
            url = {'N' : url + 'sic-naics-crosswalk-search-results',
                   'S' : url + 'naics-to-sic-crosswalk-search-results'}
            source = requests_get(url[name[0]]).content
            
        # scrape crosswalk from naics.com website into a DataFrame
        df = pd.read_html(source, header=0)
        sectors = max(df, key=len).astype(str)
        sectors.columns = [re.sub(r'[^\x20-\x7E]',r' ', c)
                           for c in sectors.columns]

        # rename the columns, based on the direction of conversion
        sectors = sectors.rename(columns={code: 'code',
                                          name: 'name',
                                          desc: 'description'})

        # clean up DataFrame, set its index, and save to sql database
        keep = sectors['code'].str.isnumeric() & sectors['name'].str.isnumeric()
        sectors = sectors.loc[keep].set_index('code')
        sectors.index = sectors.index.astype(int)
        sectors['name'] = sectors['name'].astype(int)
        return sectors[['name', 'description']]
        

class Sectoring:
    """Base class to implement industry sector grouping schemes

    Args:
        sql: SQL database connection object
        scheme: name of sectoring scheme
        fillna: value to return if source label is out of range
        new: whether to recreate from source (True) or retrieve from SQL
        source: url to recreate sector scheme from

    Attributes:
        sectors: DataFrame mapping input code in index to output 'name' column

    Sectoring schemes:

    - 'sic1', 'sic2', 'sic3': group 4-digit sic to 1, 2, or 3-digits
    - 'sic': map 6-digit naics to 4-digit sic
    - 'naics': map 4-digit sic to 6-digit naics 
    - 'bea1997', 'bea1963', 'bea1947': map naics to bea schemes by vintage year
    - 'codes48', ..., 'codes5': map 4-digit sic to FamaFrench schemes
    """
    
    
    def __init__(self, sql: SQL, scheme: str, fillna: Any = None,
                 new: bool = False, source: str = ""):
        self.scheme = scheme.lower()
        self.sql = sql
        self.fillna = fillna
        self.table = sql.Table('sectoring',
                               Column('code', Integer, primary_key=True),
                               Column('name', String(16), primary_key=True),
                               Column('scheme', String(8), primary_key=True),
                               Column('description', String(128)))
        self.sql.create_all()
        if self.scheme.startswith('sic') and len(self.scheme) > 3:
            digits = int(scheme[3])
            n = 4 - digits
            self.sectors = DataFrame(index=np.arange(0, 10**digits) * (10**n))
            self.sectors['name'] = np.arange(0, 10**digits)
            self.sectors['description'] = [f"{s}{'0'*n}-{s}{'9'*n}"
                                           for s in self.sectors['name']]
            self.sectors['scheme'] = f"sic{digits}"
        elif new:
            self.load(source=source)
        else:
            q = f"SELECT * FROM {self.table.key} WHERE scheme='{self.scheme}'"
            self.sectors = self.sql.read_dataframe(q).set_index('code')
            #self.sectors.index = self.sectors.index.astype(int)
        try:
            self.sectors['name'] = self.sectors['name'].astype(int)
        except:
            pass
       
    def __getitem__(self, code: List | str | int) -> List | Any:
        """Lookup sectoring group given raw input code/s"""
        if types.is_list_like(code):
            return [self[c] for c in list(code)]
        found = np.searchsorted(self.sectors.index, code, side='right')
        return self.sectors['name'].iloc[found-1] if found > 0 else self.fillna

    def load(self, source: str = '') -> Series | None:
        """Reload sectoring map from source url or file"""

        def _load(df: DataFrame | None) -> Series | None:
            """Helper to upload dataframe with code in index, name. desc to SQL"""
            if df is not None:     # columns=['name', 'description']
                self.sectors = df
                self.sectors['scheme'] = self.scheme
                delete = self.table.delete()\
                                   .where(self.table.c['scheme'] == self.scheme)
                self.sql.run(delete)
                self.sql.load_dataframe(table=self.table.key,
                                        df=self.sectors,
                                        index_label='code')
                return self.sectors['name'].unique()
            return None
        
        if self.scheme == 'naics':
            return _load(Crosswalk.sectoring(code='SIC Code',
                                             name='NAICS Code',
                                             desc='NAICS Description',
                                             source=source))
        elif self.scheme == 'sic':
            return _load(Crosswalk.sectoring(code='NAICS Code',
                                             name='SIC Code',
                                             desc='SIC Description',
                                             source=source))
        elif self.scheme.startswith('codes'):
            return _load(FFReader.sectoring(scheme=self.scheme,
                                            source=source))
        elif self.scheme.startswith('bea'):
            return _load(BEA.sectoring(year=int(self.scheme[3:]),
                                       source=source))
        else:
            return None




if __name__ == "__main__":
    import os
    from finds.database import SQL, RedisDB
    from secret import credentials, paths

    downloads = paths['scratch']
    sql = SQL(**credentials['sql'])

    codes = {}
    scheme = 'sic2'
    codes[scheme] = Sectoring(sql, scheme=scheme)
    
    scheme = 'sic3'
    codes[scheme] = Sectoring(sql, scheme=scheme)

    new = True
    for scheme in [5, 10, 12, 17, 30, 38, 48, 49]:  # FamaFrench
        scheme = f"codes{scheme}"
        codes[scheme] = Sectoring(sql, scheme=scheme, new=new)

    scheme = 'SIC'  # SIC from NAICS Crosswalk
    codes[scheme] = Sectoring(sql, scheme=scheme, new=new)
    
    scheme = 'NAICS'  # NAICS from SIC Crosswalk
    codes[scheme] = Sectoring(sql, scheme=scheme, new=new)
        
    for scheme in [1947, 1963, 1997]:   # BEA sectoring scheme
        scheme = f"bea{scheme}"
        codes[scheme] = Sectoring(sql, scheme=scheme, new=new)

