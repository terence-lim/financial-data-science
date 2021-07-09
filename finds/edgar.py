"""Class and methods to retrieve and analyze EDGAR text data

- SEC Edgar, 10K, 8K, MD&A, Business Descriptions
- BeautifulSoup, requests, regular expressions

Author: Terence Lim
License: MIT
"""
import lxml
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame, Series
import os, sys, time, re
import requests, zipfile, io, gzip, csv, json, unicodedata, glob
import numpy as np
import matplotlib.pyplot as plt
import config

def _print(*args, echo=config.ECHO, **kwargs):
    if echo: print(*args, **kwargs)

def requests_get(url, params=None, retry=7, sleep=2, timeout=3, delay=0.25,
                 trap=False, headers=config.headers, echo=config.ECHO):
    """Wrapper over requests.get, with retry loops and delays

    Parameters
    ----------
    url : str
        URL address to request
    params : dict of {key: value} (optional), default is None
        Payload of &key=value to append to url
    headers : dict (optional)
        e.g. User-Agent, Connection and other headers parameters
    timeout : int (optional), default is 3
        Number of seconds before timing out one request try
    retry : int (optional), default is 5
        Number of times to retry request
    sleep : int (optional), default is 2
        Number of seconds to wait between retries
    trap : bool (optional), default is True
        On timed-out after retries: if True raise exception, else return False
    delay : int (optional), default is 0
        Number of seconds to initially wait
    echo : bool (optional), default is True
        whether to display verbose messages to aid debugging

    Returns
    -------
    r : requests.Response object, or None
        None if timed-out or status_code != 200
    """
    _print(url, echo=echo)
    if delay:
        time.sleep(delay + (delay * np.random.rand()))
    for i in range(retry):
        try:
            r = requests.get(url, headers=headers,timeout=timeout,params=params)
            assert(r.status_code >= 200 and r.status_code <= 404)
            break
        except Exception as e:
            time.sleep(sleep * (2 ** i) + sleep*np.random.rand())
            _print(e, r.status_code, echo=echo)
            r = None
    if r is None:  # likely timed-out after retries:
        if trap:     # raise exception if trap, else silently return None
            raise Exception(f"requests_get: {url} {time.time()}")
        return None
    if r.status_code != 200:
        _print(r.status_code, r.content, echo=echo)
        return None
    return r

class Edgar:
    """Class to retrieve and pre-process Edgar website documents

    Attributes
    ----------
    forms_ : dict with  key in {'10-K', '10-Q', '8-K'}
      Values are list of form names str

    Notes
    -----
    https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm
    https://www.investor.gov/introduction-investing/general-resources/
      news-alerts/alerts-bulletins/investor-bulletins/how-read-8
    """
    # Create .forms_ list: EDGAR_Forms_v2.1.py from ND-SRAF / McDonald 201606
    f_10K = ['10-K', '10-K405', '10KSB', '10-KSB', '10KSB40']
    f_10KA = ['10-K/A', '10-K405/A', '10KSB/A', '10-KSB/A', '10KSB40/A']
    f_10KT = ['10-KT', '10KT405', '10-KT/A', '10KT405/A']
    f_10Q = ['10-Q', '10QSB', '10-QSB']
    f_10QA = ['10-Q/A', '10QSB/A', '10-QSB/A']
    f_10QT = ['10-QT', '10-QT/A']
    forms_ = {'10-K' : f_10K + f_10KA + f_10KT,
              '10-Q' : f_10Q + f_10QA + f_10QT,
              '8-K' : ['8-K']}
    url_prefix = 'https://www.sec.gov/Archives'

    #
    # Static methods to fetch documents from SEC Edgar website
    #    
    @staticmethod
    def basename(date, form, cik, pathname, **kwargs):
        """Construct base filename from components of the filing pathname"""
        base = os.path.split(pathname)[-1]
        return f"{date}_{form.replace('/A', '-A')}_edgar_data_{cik}_{base}"
    
    @staticmethod
    def from_path(pathname, filename=None):
        """Extract meta info from edgar pathname"""
        items = pathname.split('.')[0].split('/')
        adsh = items[-1].replace('-','')
        resource = os.path.join(*items[:-1], adsh)
        indexname = os.path.join(resource, items[-1] + '-index.html')
        return (os.path.join(resource, filename) if filename
                else {'root': Edgar.url_prefix,
                      'adsh': adsh,
                      'indexname': indexname,
                      'resource' : resource})

    @staticmethod
    def fetch_tickers(echo=config.ECHO):
        """Fetch tickers-to-cik lookup from SEC web page as a pandas Series"""
        url = 'https://www.sec.gov/include/ticker.txt'
        tickers = requests_get(url, echo=echo).text
        df = DataFrame(data = [t.split('\t') for t in tickers.split('\n')],
                       columns = ['ticker','cik'])
        return df.set_index('ticker')['cik'].astype(int)

    @staticmethod
    def fetch_index(date=None, year=None, quarter=None, echo=config.ECHO):
        """Fetch edgar daily or master index, or walk to get all daily dates"""
        # https://www.sec.gov/Archives/edgar/data/51143/0000051143-13-000007.txt
        if year and quarter:    # get full-index by year/quarter
            root = 'https://www.sec.gov/Archives/edgar/full-index/'
            url = f"{root}{year}/QTR{quarter}/master.idx"
            r = requests_get(url, echo=echo)
            if r is None:
                return None
            df = pd.read_csv(
                io.BytesIO(r.content), sep='|', quoting=3, encoding='latin-1', 
                header=None, low_memory=False, na_filter=False, #skiprows=7,
                names=['cik', 'name', 'form', 'date', 'pathname'])
            df['date'] = df['date'].str.replace('-','')
            df = df[df['date'].str.isdigit() & df['cik'].str.isdigit()]
            df = df.drop_duplicates(['pathname', 'date', 'form', 'cik'])
            df = df[df['date'].str.isdigit() & df['cik'].str.isdigit()]
            df['cik'] = df['cik'].astype(int)
            df['date'] = df['date'].astype(int)
            return df.reset_index(drop=True)
        elif date is not None:   # get daily-index
            root = 'https://www.sec.gov/Archives/edgar/daily-index/'
            q = (((date // 100) % 100) + 2) // 3
            url = f"{root}{date//10000}/QTR{q}/master.{date}.idx"
            r = requests_get(url, echo=echo)
            if r is None:
                d = ((date // 10000) % 100) + ((date % 10000) * 100)
                url = f"{root}{date//10000}/QTR{q}/master.{d:06d}.idx"
                r = requests_get(url, echo=echo)
            df = pd.read_csv(
                io.BytesIO(r.content), sep='|', quoting=3, encoding='utf-8', 
                low_memory=False, na_filter=False, header=None, #skiprows=7,
                names=['cik', 'name', 'form', 'date', 'pathname'])
            df = df[df['date'].str.isdigit() & df['cik'].str.isdigit()]
            df['cik'] = df['cik'].astype(int)
            df['date'] = df['date'].astype(int)
            return df.reset_index(drop=True)
        elif not date:
            raise Exception('Invalid arguments to fetch_index')

        # called with no arguments => fetch category tree
        leaf = {}
        queue = ['']
        while len(queue):
            sub = queue.pop()
            f = io.BytesIO(requests_get(root + sub + "index.json",
                                        echo=echo).content)
            nodes = json.loads(f.read().decode('utf-8'))['directory']['item']
            for node in nodes:
                if node['type'] == 'dir':
                    queue += [sub + node['href']]
                        #print(str(node))
                else:
                    s = node['name'].split('.')
                    if s[0] == 'master' and s[2] == 'idx':
                        d = int(s[1])
                        if d <= 129999:
                            d = (d%100)*10000 + (d//100) # 070194->940701
                        if d <= 129999:
                            d += 20000000  # 091231->20011231
                        if d <= 999999:
                            d += 19000000  # 970102->19970102
                        leaf[d] = sub + node['name']
        return Series(leaf)

    @staticmethod
    def fetch_detail(pathname, root=None, echo=config.ECHO):
        """Fetch from HTML file, containing table of hyperlinks of documents"""
        root = root or Edgar.url_prefix
        subs = Edgar.from_path(pathname)
        r = requests_get(os.path.join(root, subs['indexname']), echo=echo)
        return None if r is None else r.content

    @staticmethod
    def fetch_filing(pathname, root=None, form8k=False, echo=config.ECHO):
        """Fetch filing text from url pathname or local html file"""
        root = root or Edgar.url_prefix        
        if not isinstance(root, str):
            root = ''
        if root.startswith('http'):
            r = requests_get(os.path.join(root, pathname), echo=echo)
            if r is None:
                return None
            r = r.content
        else:
            with open(os.path.join(root, pathname), "rb") as f:
                r = f.read()
            if not r:
                return None

        soup = BeautifulSoup(r, "lxml")   #"html.parser" )
        _print('soup: %d' % len(soup.text), echo=echo)

        # remove inline xbrl's
        for x in [re.compile("ix:\S*", re.I), re.compile("xbrli:\S*", re.I)]:
            tags = soup.find_all(x)    # regex format for soup.find_all
            for tag in tags: tag.decompose()

        # remove tables, where #alphas is less than 90% of digits+alphas
        tags = soup.find_all(['table'])
        for tag in tags:
            s = tag.get_text()
            numalpha = sum(c.isalpha() for c in s)
            numdigit = sum(c.isdigit() for c in s)
            hasitem = re.search('item.[.]?[.]?7', s, re.IGNORECASE)
            if not hasitem and numalpha < 0.9*(numdigit+numalpha):
                tag.decompose()

        for tags in ['u','b','i']:
            for tag in soup.findAll(tags):
                tag.replace_with_children()

        text = soup.get_text('\n')
        _print('table: %d %d' % (len(tags), len(text)), echo=echo)

        if form8k:
            x = re.search('emerging growth company[\w\W]*? of the Exchange Act',
                          text)
            if x:
                text = text[x.end():]
            _print('form8k: %d' % len(text), echo=echo)

        # clean-up line breaks
        text = unicodedata.normalize("NFKD", text)  # Normalize
        text = '\n'.join(text.splitlines())
        text = re.sub(r'[ ]+\n', '\n', text)
        text = re.sub(r'\n[ ]+', '\n', text)
        text = re.sub(r'\n+', '\n', text)

        # Completed Stage One
        _print('normalize: %d' % len(text), echo=echo)
        return text

    @staticmethod
    def extract_filenames(detail, echo=config.ECHO):
        """Extract ordered list of .htm and .txt filenames from filing detail"""
        df_list = pd.read_html(detail)
        jdf = -1
        jrow = -1
        jcol = -1
        html_name = ''
        html_all = []
        forms = np.ravel(Edgar.forms_.values())
        for idf in range(len(df_list)):
            for irow in range(len(df_list[idf].index)):
                for icol in range(len(df_list[idf].columns)):
                    if jdf < 0 and df_list[idf].iloc[irow, icol] in forms:
                        jdf, jrow, jcol = idf, irow, icol  # likely row of form
                for icol in range(len(df_list[idf].columns)):
                    if (".htm" in str(df_list[idf].iloc[irow, icol]).lower() or
                        ".txt" in str(df_list[idf].iloc[irow, icol]).lower()):
                        for s in str(df_list[idf].iloc[irow, icol]).split():
                            if '.htm' in s.lower() or '.txt' in s.lower():
                                name = s
                        if jdf == idf and jrow == irow:
                            html_name = name
                        else:
                            html_all += [name]
        _print(f"(extract_filenames) [{jdf} {jrow} {jcol}] {html_all}",
                   echo=echo)
        return [html_name] + html_all if html_name else html_all

    
    @staticmethod
    def extract_item(text, item, echo=config.ECHO):
        """Extract text passage for {'mda10K', 'bus10K'} item from input text

        Parameters
        ----------
        text : str
            full text of filing, from which to extract passage for item
        item : str, in {'mda10K', 'bus10K', 'mda10Q'}
            choose the item to extract

        Notes
        -----
        https://www.sec.gov/fast-answers/answersreada10khtm.html
       
        Item 1 - Business
        Item 1A - Risk Factors
        Item 1B - Unresolved Staff Comments
        Item 2 - Properties
        Item 3 - Legal Proceedings
       
        PART I—FINANCIAL INFORMATION
        Item 1. Financial Statements.
        Item 2. Management’s Discussion and Analysis
        Item 3. Quantitative and Qualitative Disclosures About Market Risk.
        Item 4. Controls and Procedures.
        PART II—OTHER INFORMATION
        Item 1. Legal Proceedings.
        Item 1A. Risk Factors.
        Item 2. Unregistered Sales of Equity Securities and Use of Proceeds.
        """

        def parse_helper(text, marker, start=0, echo=config.ECHO):
            # e.g. INTC uses text titles, not  "ITEM", in their 10K:
            #  "DISCUSSION AND ANALYSIS"
            # '\nQUANTITATIVE AND QUALITATIVE DISCLOSURE']
            # Management&#146;s Discussion and Analysis

            mda = ""
            end = 0

            # Define start and end sentinels for parsing
            item_beg = marker['item_beg'].copy()
            item_end = marker['item_end'].copy()
            if start != 0:
                next_beg = marker['next_beg'].copy() # if ITEM 7A does not exist
            else:
                next_beg = []             ### Hope this helps exception ?!
            text = text[start:]

            for item7 in item_beg:   # try to find begin
                begin = item7.search(text)
                begin = begin.start() if begin else -1
                _print('item begin?', item7, begin, echo=echo)
                if begin != -1:
                    break
            if begin != -1:          # found begin
                for item7A in item_end:
                    end = item7A.search(text, pos=begin + 1)
                    end = end.start() if end else -1
                    _print('item end?', item7A, end, echo=echo)
                    if end != -1:
                        break
                if end == -1:        # ITEM 7A end does not exist
                    for item8 in next_beg:    # often get exception undefined
                        end = item8.search(text, pos=begin + 1)
                        end = end.start() if end else -1
                        _print('next begin?', item8, end, echo=echo)
                        if end != -1:
                            break
                if end > begin:       # extract this found item
                    mda = text[begin:end].strip()
                else:
                    end = 0
            _print(f"(parse_helper) {len(mda)}, {end}/{len(text)}", echo=echo)
            return mda, end

        # clean-up for item headers
        text = text.upper()
        text = text.replace('\n.\n', '.\n')
        text = text.replace('\nI\nTEM', '\nITEM')
        text = text.replace('\nITEM\n', '\nITEM ')
        text = text.replace('\nITEM  ', '\nITEM ')
        text = text.replace(':\n', '.\n')
        text = text.replace('$\n', '$')
        text = text.replace('\n%', '%')
        text = text.replace('\n', '\n\n')

        markers = {    # secret sauce: plausible regex separating the sections
            'mda10K': {
                'item_beg': [
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?7[^a-z]+', re.I),
                    re.compile('DISCUSSION AND ANALYSIS', re.I)],
                'item_end': [
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?7A', re.I),
                    re.compile('\n\s*?QUANTITATIVE AND QUALITATIVE DIS', re.I)],
                'next_beg': [re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?8', re.I)]},
            'bus10K': {
                'item_beg': [
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?1[^\w]+', re.I),
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?I[^\w]+', re.I),
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?1[^\w]+', re.I),
                    re.compile('\n\s*?BUSINESS.?\n', re.I),
                    re.compile('SUMMARY OF BUSINESS\.?\n', re.I),
                    re.compile('DESCRIPTION OF BUSINESS\.?\n', re.I),
                    re.compile('BUSINESS SUMMARY\.?\n', re.I)],
                'item_end': [
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?1A', re.I),
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?1B', re.I),
                    re.compile('UNRESOLVED STAFF COMMENTS.?\s*?\n', re.I)],
                'next_beg' : [
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?2[^0-9]+', re.I)]},
            'mda10Q': {
                'item_beg': [
                    re.compile('DISCUSSION AND ANALYSIS', re.I),
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?I[^\w]+', re.I),
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?1[^\w]+', re.I)],
                'item_end': [
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?3', re.I),
                    re.compile('CONTROLS AND PROCEDURES', re.I),
                    re.compile('\n\s*?QUANTITATIVE AND QUALITATIVE DIS', re.I)],
                'next_beg': [
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?II[^\w]+', re.I),
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?2[^\w]+', re.I)]}}
        
        start=0      # get first passage
        mda, end = parse_helper(text, markers[item], start=start, echo=echo)
        if not mda:
            start = 1
            mda, end = parse_helper(text, markers[item], start=start, echo=echo)

        best = mda   # return longest passage
        while mda and end > 0:
            start += end
            mda, end = parse_helper(text, markers[item], start=start, echo=echo)
            if mda and len(mda.encode('utf-8')) > len(best):
                best = mda
        return best


class EdgarClone(Edgar):
    """Class to clone Edgar documents locally in zipped archives

    Examples
    --------
    # zip -r 2019.zip 2019
    ed = EdgarClone(prefix=config.datapath['10X'], zipped=False)
    files = ed.open(date=2021)   # 10-K and 10-Q (in prefix)
    files = ed.open(date=2021, item='detail')
    files = ed.open(form='8-K', date=2021)
    files = ed.open(form='8-K', date=2021, item='detail')
    files = ed.open(form='10-K', item='mda10K', permno=85414) 
    files = ed.open(date=20210323)
    files = ed.open(date=20210323, item='detail')
    files = ed.open(form='8-K', date=20210323, item='detail')

    ed = EdgarClone(prefix=config.datapath['10X'], zipped=True)
    files = ed.open(date=2020)   # 10-K and 10-Q (in prefix)
    files = ed.open(date=2020, item='detail')

    files = ed.open(form='10-K', item='mda10K')   # TO ZIP
    files = ed.open(form='10-K', item='bus10K')   # TO ZIP

    Notes
    -----
    form = {'8-K', '10-K', '10-Q'}, item = {'mda10K', 'bus10K'} :

    ~/<YYYY>.zip : zipped 10-K and 10-Q docs for year YYYY
    ~/detail/<YYYY>.zip : zipped 10-K and 10-Q details for year YYYY
    ~/<form>/<item>.zip : zipped extracted items from form
    ~/<form>/<YYYY>.zip : zipped form docs for year YYYY
    ~/<form>/detail/<YYYY>.zip : zipped form details for year YYYY

    ~/detail/<YYYY>/<YYYYMMDD>/*.txt : 10-K and 10-Q details for date YYYYMMDD
    ~/<form>/<item>/<permno>/*.txt : extracted items from form for permno
    ~/<form>/<YYYY>/<YYYYMMDD>/*.txt : form docs for date YYYYMMDD
    ~/<form>/detail/<YYYY>/<YYYYMMDD>/*.txt : form details for date YYYYMMDD
    """
    def __init__(self, prefix='', zipped=True, echo=config.ECHO):
        """To access (zipped or unzipped) Edgar cloned archives"""
        self.prefix = prefix
        self.zipped = zipped
        self.echo_ = echo

    def to_path(self, basename, form, date=None, item='', permno='', cik=None):
        """Construct local full path name for cloned archive"""
        s = os.path.join(self.prefix,
                         str(form),
                         str(item), 
                         str(permno),
                         str(date // 10000) if date else '',
                         str(date) if date else '',
                         basename)
        os.makedirs(os.path.dirname(s), exist_ok=True)
        return s        

    def open(self, form='', date=None, item=None, permno=None):
        """Opens a clone archive and return list of its documents"""
        
        def from_clonepath(pathname):
            """Helper to deconstruct components of filing from clone file name"""
            items = pathname.split('_')         # components are separated by '_'
            items[0] = items[0].split('/')[-1]  # split last substring on '/'
            return {'cik': int(items[4]),
                    'form': items[1].replace('-A', '/A'), # cannot have '/'
                    'date': int(items[0]),
                    'pathname': pathname}
        self.close()

        p = os.path.join(self.prefix, form, item or '')
        if date:
            date = str(date)
            p = os.path.join(p, date[:4])
        else:
            date = ''
        r = []
        if self.zipped: # if zipped archive: then have to retrieve its namelist()
            p = p + '.zip'
            with zipfile.ZipFile(p) as clone:
                for pathname in clone.namelist():
                    if pathname.endswith('/'):
                        _print(pathname, echo=self.echo_)
                    else:
                        r.append(from_clonepath(pathname=pathname))
            df = DataFrame(r).drop_duplicates(['pathname','date','form','cik'])
            if item and form and form not in self.forms_['8-K']:
                t = DataFrame.from_dict(df['pathname'].str.split('/').to_dict(),
                                        orient='index')
                if permno:  # select by permno
                    df=df[t.iloc[:,1].astype(int)==permno].assign(permno=permno)
                else:       # select all, so explicitly assign permno col
                    df['permno'] = t.iloc[:,1].astype(int).values
            self.keys_ = df                        
            self.zipped = p
            self.archive = zipfile.ZipFile(p)
        else:
            def list_dir(*args):
                """helper to list potential document filenames in folder"""
                r = []
                for a in glob.glob(os.path.join(*args, '*')):
                    a = a.replace(self.prefix + '/', '')
                    r.append({'pathname': a} if '_' not in a else 
                             from_clonepath(pathname=a))
                return r

            if date:
                if len(date) < 5:    # 4) date is year => not leaf
                    dates = glob.glob(os.path.join(p, date + '[01]???'))
                    df = pd.concat([DataFrame(list_dir(r)) for r in dates],
                                   ignore_index=True)
                else:                # 5) date is specific 8-digit date => leaf
                    q = glob.glob(os.path.join(p, date))
                    df = pd.concat([DataFrame(list_dir(r)) for r in q],
                                   ignore_index=True)                
            else:
                if permno:   # select by permno
                    q = glob.glob(os.path.join(p, str(permno)))
                    df = pd.concat([DataFrame(list_dir(r)) for r in q],
                                   ignore_index=True)               
                else:        # select all permnos
                    q = glob.glob(os.path.join(p, '[0-9]????'))
                    df = pd.concat([DataFrame(list_dir(r))\
                                    .assign(permno=int(r.split('/')[-1]))
                                    for r in q], ignore_index=True)
            self.keys_ = df
            self.archive = p
        _print(self.zipped if self.zipped else self.archive, len(self.keys_),
               echo=self.echo_)
        return self.keys_

    def close(self):
        """Close the clone archive"""
        try:
            self.archive.close()
            self.archive = None
        except:
            pass

    def __getitem__(self, pathname):
        """Retrieves text of document file by pathname in clone archive"""
        if not isinstance(pathname, str):
            pathname = pathname['pathname']
        if self.zipped:
            with self.archive.open(pathname) as f:
                with io.TextIOWrapper(f, encoding='latin-1') as g:
                    text = g.read()
        else:
            with open(os.path.join(self.prefix, pathname)) as f:
                text = f.read()
        return text

#    
# download 10-K's, 10-Q's, and 8-Ks from Edgar
#
import time, os
import numpy as np
from pandas import DataFrame, Series
import config
from tqdm import tqdm

start_year = 2021 
start_quarter = 2 
end_year = start_year
end_quarter = start_quarter

def load_10X(start_year, start_quarter, end_year, end_quarter):  
    ed = EdgarClone(config.datapath['10X'], zipped=True)
    forms = Edgar.forms_['10-K'] + Edgar.forms_['10-Q'] + Edgar.forms_['8-K']
    logger = {}
    tic = time.time()
    quarters = list(np.arange(start_year + ((start_quarter-1)/4),
                              end_year + (end_quarter/4), 1/4))
    for year in quarters:
        y = int(year)
        files = Edgar.fetch_index(year=y, quarter=int(1 + ((year - y) * 4)))
        restart = 0 #  29486 67426 83580 111448 161102 220591

        for i, r in tqdm(files.iterrows()):
            if i >= restart and r['form'] in forms:
                form8k = '8-K' if r['form'] in Edgar.forms_['8-K'] else ''

                # To save detail page
                detail = Edgar.fetch_detail(pathname=r['pathname'])
                if detail is None:   # filing detail page missing!
                    _print("***MISSING DETAIL***", r['name'], r['pathname'])
                    continue
                s = ed.to_path(form=form8k, item='detail', cik=r['cik'],
                               date=r['date'], basename=Edgar.basename(**r))
                with open(s, 'wb') as f:
                    f.write(detail)
                _print("--- Found Detail ---", r['date'], r['form'], r['cik'],
                       f"{time.time()-tic:.0f}s *** {i}/{len(files)} ***")
                    
                # To save main filing document text
                filenames = Edgar.extract_filenames(detail)  # retrieve filings
                if form8k and ".htm" in filenames[0]:
                    lines = "\n".join(
                        [Edgar.fetch_filing(
                            Edgar.from_path(r['pathname'], f), form8k=True)
                         for f in filenames if ".htm" in f])
                else:
                    lines = Edgar.fetch_filing(
                        Edgar.from_path(r['pathname'], filenames[0]),
                        form8k=form8k)
                if not lines:
                    lines = Edgar.fetch_filing(r['pathname'], form8k=form8k)
                s = ed.to_path(date=r['date'], form=form8k, cik=r['cik'],
                               basename=Edgar.basename(**r))
                with open(s, 'wt') as f:
                    f.write(lines)
                    
                logger[r['pathname'].split('/')[-1]] = r.append(
                    Series({'len': len(lines)})).drop('pathname')
                _print("--- Saved Filing ---", *r.values, f"{i}/{len(files)}")

    logger = DataFrame.from_dict(logger, orient='index')
    return logger

#
# Extract and save MDA and BUS text
#
import time, os
import numpy as np
from pandas import DataFrame, Series
from finds.database import SQL
from finds.structured import PSTAT
from finds.busday import BusDay
import config

years = [2021]
def parse_items(years=years):
    ed = EdgarClone(config.datapath['10X'], zipped=False)
    sql = SQL(**config.credentials['sql'])
    bday = BusDay(sql)
    pstat = PSTAT(sql, bday)
    to_permno = pstat.build_lookup(target='lpermno', source='cik', fillna=0)

    items = {'10-K': ['bus10K', 'mda10K']}  # '10-Q': ['mda10Q']}
    logger = []
    for year in years:    #2019, 2021):  # Start 1998++
        rows = ed.open(date=year)
        row = rows.iloc[0]
        for i, row in rows.iterrows():
            permno = to_permno(int(row['cik']))
            if row['form'] in items and permno:
                filing = ed[row['pathname']]
                for item in items[row['form']]:
                    extract = Edgar.extract_item(filing, item)
                    s = ed.to_path(form=row['form'], permno=permno, item=item,
                                   basename=os.path.basename(row['pathname']))
                    with open(s, 'wt') as g:
                        g.write(extract)
                    r = {'year': year, 'permno': permno, 'item': item,
                         'text_c': len(filing),
                         'item_c': len(extract),
                         'text_w': len(filing.split()),
                         'item_w': len(extract.split())}
                    logger.append(r)
                    print(", ".join([f"{k}: {v}" for k,v in r.items()]))
    logger = DataFrame.from_records(logger)
