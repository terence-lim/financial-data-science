"""Retrieves FOMC meeting minutes

MIT License

Copyright 2022-2023 Terence Lim
"""
import requests
import re
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from bs4 import BeautifulSoup
from typing import Dict

class FOMCReader:
    """Class to retrieve FOMC minutes"""
    
    _url = 'https://www.federalreserve.gov/'  # root url

    def __init__(self, url: str = _url):
        """Initializer retrieves dates available from website

        Args:
          url: root url of Federal Reserve website
        """
        
        def dateOf(s):
            """parse date from link string"""
            return int(re.sub('\D', '', s)[-8:]) 
        
        # latest five years' minutes can be found from a main page
        new_url = url + 'monetarypolicy/fomccalendars.htm'
        raw = BeautifulSoup(markup=requests.get(new_url).content,
                            features='html.parser')
        hrefs = raw.find_all(name='a',
                             href=re.compile('\S+minutes\S+.htm$', re.I))
        links = [url + m.attrs['href'] for m in hrefs]

        # earlier years' minutes are linked from annual pages with this format
        old_url = url + 'monetarypolicy/fomchistorical%d.htm'
        for year in range(1993, min([dateOf(m) for m in links]) // 10000):
            raw = BeautifulSoup(markup=requests.get(old_url % year).content,
                                features='html.parser')
            hrefs = raw.find_all(name='a',
                                 href=re.compile('\S+minutes\S+.htm$', re.I))
            links += [url + m.attrs['href'].replace(url,'') for m in hrefs]

        self.dates = {dateOf(link) : link for link in links}

    def __len__(self):
        return len(self.dates)

    def __iter__(self):
        return iter(self.dates)
                 
    def __getitem__(self, date) -> str:
        """Retrieve FOMC minutes text from Fed website

        Args:
          date: meeting date

        Returns:
          text of minutes for meeting date
        """
        url = self.dates[date]
        raw = BeautifulSoup(markup=requests.get(url).content,
                            features='html.parser')
        minutes = "\n\n".join([p.get_text().strip()
                               for p in raw.findAll('p')])
        return re.sub('\n+','\n', re.sub('[\r\t]',' ', minutes))



