"""Retrieves FOMC meeting minutes

MIT License

Copyright 2022-2023 Terence Lim
"""
import requests
import re
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from pandas import DataFrame, Series
from bs4 import BeautifulSoup
from typing import Dict

class FOMCReader:
    """Class to retrieve FOMC minutes"""
    
    _url = 'https://www.federalreserve.gov/'  # root url

    def __init__(self, url: str = _url, delay: float = 0.1):
        """Initializer retrieves dates available from website

        Args:
          url: root url of Federal Reserve website
          delay: sleep between requests
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
        for year in tqdm(range(1993, min([dateOf(m) for m in links]) // 10000)):
            raw = BeautifulSoup(markup=requests.get(old_url % year).content,
                                features='html.parser')
            hrefs = raw.find_all(name='a',
                                 href=re.compile('\S+minutes\S+.htm$', re.I))
            links += [url + m.attrs['href'].replace(url,'') for m in hrefs]
            time.sleep(delay)

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



if __name__ == "__main__":
    from finds.database import MongoDB
    from finds.unstructured import Unstructured
    from finds.utils import Store
    from secret import credentials, paths
    VERBOSE = 1
    
    mongodb = MongoDB(**credentials['mongodb'], verbose=VERBOSE)
    print('uptime:', mongodb.client.admin.command("serverStatus")['uptime'])
    fomc = Unstructured(mongodb, 'FOMC')
    
    # retrieve keys (dates) of minutes previously retrieved and stored locally
    dates = fomc['minutes'].distinct('date')

    # fetch new minutes from FOMC site
    docs = {d: minutes[d] for d in minutes if d not in dates}
    print("New minutes:")
    pprint([f"{k}: {len(v)} chars" for k,v in docs.items()])

    def edit(text: str) -> str:
        """helper to spawn editor and write/edit/read to tempfile"""
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tmp") as f: # save temp file
            f.write(text.encode("utf-8"))
            f.flush()
            subprocess.call([os.environ.get('EDITOR','emacs'), "-nw", f.name])
            f.seek(0)
            return f.read().decode("utf-8")        # keep edited text

    if docs:
        # to edit out head and tail of each document
        results = list()   
        for date, initial_message in docs.items(): 
            edited_text = edit(initial_message)
            results.append({'date': date, 'text' : edited_text})
        results = sorted(results, key = lambda x: x['date'])   # sort by date
            
        # save edited docs
        Store(paths['scratch'] / 'fomc', ext='gz').dump(results, f"{max(docs.keys())}.json")
        for doc in results: # store docs for new dates
            fomc.insert('minutes', doc, keys=['date'])

