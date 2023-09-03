"""Wraps around pandas_datareader to retrieve French data library

MIT License

Copyright 2022-2023 Terence Lim
"""
import pandas_datareader as pdr
import numpy as np
import pandas as pd
import io
import zipfile
from pandas import DataFrame, Series
from typing import Callable, List
from .readers import requests_get

class FFReader:
    """Wraps over pandas_datareader to load FamaFrench factors from website

    Attributes:
        daily, monthly: List of common FF factors/industries
    """

    daily = [
        ('F-F_Research_Data_5_Factors_2x3_daily', 0, ''),
        ('F-F_Research_Data_Factors_daily', 0, ''),
        ('F-F_Momentum_Factor_daily', 0, ''),
        ('F-F_LT_Reversal_Factor_daily', 0, ''),
        ('F-F_ST_Reversal_Factor_daily', 0, ''),
        ('49_Industry_Portfolios_daily', 0, '49vw'), # append suffix
        ('48_Industry_Portfolios_daily', 0, '48vw'), #  to differentiate
        ('49_Industry_Portfolios_daily', 1, '49ew'), #  value-weighted vs
        ('48_Industry_Portfolios_daily', 1, '48ew'), #  equal-weighted
    ]
    
    monthly = [
        ('F-F_Research_Data_5_Factors_2x3', 0, '(mo)'),
        ('F-F_Research_Data_Factors', 0, '(mo)'),   # "(mo)" for monthly
        ('F-F_Momentum_Factor', 0, '(mo)'),
        ('F-F_LT_Reversal_Factor', 0, '(mo)'),
        ('F-F_ST_Reversal_Factor', 0, '(mo)'),
    ]


    @staticmethod
    def sectoring(scheme: str, source: str = "") -> DataFrame | None:
        """Load FamaFrench sectoring based on sic-4, from website or zipfile

        Args:
          scheme: in {codes5, codes10, 12, 17, 30, 38, 48, 49}

        Notes:

        Retrieved from "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        + "ftp/Siccodes5.zip"

        For example, the industry definitions file for Siccodes49 looks like:

        ::

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
        """
        if not source:
            prefix_ = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            source = prefix_ + f"ftp/Sic{scheme}.zip"
        if source.startswith('http'):
            response = requests_get(source)
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
                        #_print(sic[0], labels.loc[int(sic[0])].values)
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
                if (next_sic2.iloc[i] < labels.index[i+1]
                    and next_sic2.iloc[i] not in labels.index):
                    df.loc[next_sic2.iloc[i], 'name'] = 'Other'
                    df.loc[next_sic2.iloc[i], 'description'] = other

        # clean up dataframe {sectors} and save to sql 
        sectors = pd.concat([labels, df], axis=0)\
                    .drop(columns=['end'])\
                    .sort_index()
        return sectors
    
    
    
    def keys() -> List[str]:
        """Return names of all available datasets"""
        return pdr.data.FamaFrenchReader(None).get_available_datasets()

    @staticmethod
    def fetch(name: str, 
              item: int = 0, 
              suffix: str = '', 
              start: int = 19260101, 
              end: int = 20271231, 
              date_formatter: Callable = lambda x: x) -> DataFrame:
        """Retrieve item and return as DataFrame

        Args:
            name: Name of research factor in Ken French website
            item: Index of item to research (e.g. 0 is usually value-weighted)
            suffix: Suffix string to append to name when stored in sql
            start: earliest date to retrieve
            end: latest date to retrieve 
            date_formatter: to reformat dates, e.g. bd.offset or bd.endmo
        """
        start = pd.to_datetime(start, format='%Y%m%d')
        end = pd.to_datetime(end, format='%Y%m%d')
        df = pdr.data.DataReader(name=name,
                                 data_source='famafrench',
                                 start=start,
                                 end=end)[item]
        try:
            df.index = df.index.to_timestamp()
        except:
            pass     # else invalid comparison error!
        df = df[(df.index >= start) & (df.index <= end)]
        df.index = [date_formatter(d) for d in df.index]
        df.columns = [c.rstrip() + suffix for c in df.columns]
        df.where(df > -99.99, other=np.nan, inplace=True)  # replace NaNs
        df = df / 100   # change percentage returns in source to decimals
        return df

if __name__ == "__main__":
    print(Series(FFReader.keys()).to_string())
    df = FFReader.sectoring('codes5')

