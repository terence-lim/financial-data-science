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
_VERBOSE = 1

class FOMCReader:
    """Class to retrieve FOMC minutes"""
    
    fed_url = 'https://www.federalreserve.gov/'  # Else catalog from main site
    
    @staticmethod
    def fetch(url: str = '') -> str | Dict[int, str]:
        """Retrieve FOMC minutes or catalog from Fed website

        Args:
            url: Optional webpage url to retrieve text from

        Returns:
            text of minutes, or dict of all dates and urls from Fed site
        """

        if url:                # Retrieve FOMC minutes from input url
            raw = BeautifulSoup(markup=requests.get(url).content,
                                features='html.parser')
            minutes = "\n\n".join([p.get_text().strip()
                                   for p in raw.findAll('p')])
            return re.sub('\n+','\n', re.sub('[\r\t]',' ', minutes))

        dateOf = lambda s: int(re.sub('\D', '', s)[-8:]) 
        
        # latest five years' minutes can be linked from a main page
        new_url = FOMCReader.fed_url + 'monetarypolicy/fomccalendars.htm'
        raw = BeautifulSoup(markup=requests.get(new_url).content,
                            features='html.parser')
        hrefs = raw.find_all(name='a',
                             href=re.compile('\S+minutes\S+.htm$', re.I))
        links = [FOMCReader.fed_url + m.attrs['href'] for m in hrefs]

        # earlier years' minutes are linked from annual pages with this format
        old_url = FOMCReader.fed_url + 'monetarypolicy/fomchistorical%d.htm'
        for year in range(1993, min([dateOf(m) for m in links]) // 10000):
            raw = BeautifulSoup(markup=requests.get(old_url % year).content,
                                features='html.parser')
            hrefs = raw.find_all(name='a',
                                 href=re.compile('\S+minutes\S+.htm$', re.I))
            links += [FOMCReader.fed_url
                      + m.attrs['href'].replace(FOMCReader.fed_url,'')
                      for m in hrefs]
        return {dateOf(link) : link for link in links}


