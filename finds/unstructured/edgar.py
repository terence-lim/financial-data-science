"""Class and methods to retrieve and manipulate EDGAR text data

- SEC Edgar: 10-K, 10-Q, 8-K
- MD&A and Business Descriptions items

Copyright 2022, Terence Lim

MIT License
"""
from typing import Any, Dict, List, Tuple
import lxml
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame, Series
import os
import io
import sys
import time
import zipfile
import gzip
import re
import csv
import json
import unicodedata
import requests
import glob
import numpy as np
import matplotlib.pyplot as plt
from finds.readers.readers import requests_get
_VERBOSE = 0

def _print(*args, verbose=0, **kwargs):
    if max(_VERBOSE, verbose):
        print(*args, **kwargs)

class Edgar:
    """Class to retrieve and pre-process Edgar website documents

        <localname> = YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt

        - e.g. 20211105_10-Q_edgar_data_1761312_0001558370-21-014714.txt

    10-K and 10-Q zipped archive
    - 10X/YYYY,zip

    10-K and 10-Q local file (zip -q -r 2019.zip 2019)
    - 10X/YYYY/YYYYMMDD/YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt

    10-K and 10-Q detail folder
    - 10X/detail/YYYY/YYYYMMDD/YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt

    8-K local file
    - 10X/YYYY/YYYYMMDD/YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt

    8-K detail folder
    - 10X/8-K/detail/YYYY/YYYYMMDD/

    10-K MDA local text file
    - 10X/10-K/mda10K/PERMNO/YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt

    10-K MDA zipped archive (zip -q -r mda10K.zip mda10K)
    - 10X/10-K/mda10K.zip
    """

    edgar_url = 'https://www.sec.gov/Archives/'
    ticker_url = 'https://www.sec.gov/include/ticker.txt'

    # list of forms_: EDGAR_Forms_v2.1.py from ND-SRAF / McDonald 201606
    _f10_K = ['10-K', '10-K405', '10KSB', '10-KSB', '10KSB40']
    _f10_KA = ['10-K/A', '10-K405/A', '10KSB/A', '10-KSB/A', '10KSB40/A']
    _f10_KT = ['10-KT', '10KT405', '10-KT/A', '10KT405/A']
    _f10_Q = ['10-Q', '10QSB', '10-QSB']
    _f10_QA = ['10-Q/A', '10QSB/A', '10-QSB/A']
    _f10_QT = ['10-QT', '10-QT/A']
    _forms = {'10-K' : _f10_K + _f10_KA + _f10_KT,
              '10-Q' : _f10_Q + _f10_QA + _f10_QT,
              '8-K' : ['8-K']}


    #############################################################
    #
    # Static methods to fetch documents from SEC Edgar website
    #    
    #############################################################

    @staticmethod
    def parse_pathname(pathname: str,
                       filename: str = '') -> Dict[str, str] | str:
        """Extract meta info and locations from edgar pathname

        Args:
            pathname: Main pathname from Edgar index file
            filename: Suffix to append to resource location and return

        Returns:
            Prepend resource location to filename, if desired to download;
            else dictionary of the meta and location info

        Examples:

        https://www.sec.gov/Archives/edgar/data/51143/0000051143-13-000007.txt
        """
        items = pathname.split('.')[0].split('/')
        adsh = items[-1].replace('-','')
        resource = os.path.join(*items[:-1], adsh)
        indexname = os.path.join(resource, items[-1] + '-index.html')
        if filename:   # if suffix filename, then append to resource location
            return os.path.join(resource, filename)
        return {'root': Edgar.edgar_url,
                'adsh': adsh,
                'indexname': indexname,   # filename of detail
                'resource' : resource}    # prefix for filings filenames


    @staticmethod
    def fetch_tickers(verbose: int = _VERBOSE) -> Series:
        """Fetch tickers-to-cik lookup from SEC web page as a pandas Series"""
        tickers = requests_get(Edgar.ticker_url, delay=.1, verbose=verbose).text
        df = DataFrame(data=[t.split('\t') for t in tickers.split('\n')],
                       columns=['ticker','cik'])
        return df.set_index('ticker')['cik'].astype(int)


    @staticmethod
    def fetch_index(date: int = 0, year: int = 0, quarter: int = 0,
                    verbose: int = _VERBOSE) -> Dict:
        """Fetch edgar daily index or full index, or all daily dates

        Args:
            date: Retrieve daily index for this date (unless 0)
            year, quarter: Retrieve full-index for this year/quarter (unless 0)

        Returns:
            Dict of filings meta data from daily or full index, or daily dates

        Notes:

            If no arguments, retrieve all dates by walking daily index tree
        """

        if year and quarter:    # get full-index by year/quarter
            root = 'https://www.sec.gov/Archives/edgar/full-index/'
            url = os.path.join(root,
                               str(year),
                               'QTR' + str(quarter),
                               "master.idx")
            r = requests_get(url, verbose=verbose, delay=.1)
            if r is None:
                return None
            df = pd.read_csv(io.BytesIO(r.content),
                             sep='|',
                             quoting=3,
                             encoding='latin-1', 
                             header=None,
                             low_memory=False,
                             na_filter=False, #skiprows=7,
                             dtype='str',
                             names=['cik', 'name', 'form', 'date', 'pathname'])
            df['date'] = df['date'].str.replace('-','')
            df = df[df['date'].str.isdigit() & df['cik'].str.isdigit()]
            df = df.drop_duplicates(['pathname', 'date', 'form', 'cik'])
            df = df[df['date'].str.isdigit() & df['cik'].str.isdigit()]
            df['cik'] = df['cik'].astype(int)
            df['date'] = df['date'].astype(int)
            return df.reset_index(drop=True).to_dict(orient='index')
        
        if date:   # get daily-index
            root = 'https://www.sec.gov/Archives/edgar/daily-index/'
            q = (((date // 100) % 100) + 2) // 3
            url = os.path.join(root,
                               str(date//10000),
                               'QTR' + str(q),
                               f"master.{date}.idx.gz")
            r = requests_get(url, verbose=verbose, delay=.1)
            if r is None:
                d = ((date // 10000) % 100) + ((date % 10000) * 100)
                url = os.path.join(root,
                                   str(date//10000),
                                   'QTR' + str(q),
                                   f"master.{d:06d}.idx")
                r = requests_get(url, verbose=verbose, delay=.1)
            df = pd.read_csv(io.BytesIO(r.content),
                             compression='gzip',
                             sep='|',
                             dtype='str',
                             quoting=3,
                             encoding='utf-8', 
                             low_memory=False,
                             na_filter=False,
                             header=None,
                             #skiprows=7,
                             names=['cik', 'name', 'form', 'date', 'pathname'])
            df = df[df['date'].str.isdigit() & df['cik'].str.isdigit()]
            df['cik'] = df['cik'].astype(int)
            df['date'] = df['date'].astype(int)
            return df.reset_index(drop=True).to_dict(orient='index')

        def get_nodes(url: str) -> List[Dict]:
            """helper to retrieve url directory listing"""
            f = io.BytesIO(requests_get(url,
                                        verbose=verbose,
                                        delay=.1).content)
            return json.loads(f.read().decode('utf-8'))['directory']['item']
            
        # called with no arguments => fetch category tree for daily dates
        leaf = {}
        ynodes = get_nodes(os.path.join(root, "index.json"))
        for ynode in ynodes:
            if ynode['type'] == 'dir':
                url = os.path.join(root, ynode['href'])
                qnodes = get_nodes(url + "index.json")
                for qnode in qnodes:
                    if qnode['type'] == 'dir':
                        sub = url + qnode['href']
                        nodes = get_nodes(sub + 'index.json')
                        for node in nodes:
                            if node['type'] == 'file':
                                s = node['name'].split('.')
                                if (len(s) > 2
                                    and s[0] == 'company'
                                    and s[2] == 'idx'):
                                    d = int(s[1]) 
                                    if d <= 129999:  # 070194->940701
                                        d = (d%100)*10000 + (d//100) 
                                    if d <= 129999: # 091231->20011231
                                        d += 20000000  
                                    if d <= 999999: # 970102->19970102
                                        d += 19000000  
                                    leaf[d] = sub + node['name']
        return leaf
    
    @staticmethod
    def fetch_detail(pathname: str, root: str = '',
                     verbose: int = _VERBOSE) -> bytes:
        """Fetch from HTML filename, containing table of document hyperlinks
        
        Args:
            pathname: Relative pathname to fetch
            root: Root prefix of url
        """
        url = os.path.join(root or Edgar.edgar_url,
                           Edgar.parse_pathname(pathname)['indexname'])
        r = requests_get(url, delay=.1, verbose=verbose)
        return b'' if r is None else r.content

    @staticmethod
    def fetch_filing(pathname: str, root: str = '', form: str = '',
                     features: str = 'lxml', verbose: int = _VERBOSE) -> str:
        """Fetch and parse filing text from url pathname or local html file

        Args:
            pathname: Relative pathname to fetch
            root: Root prefix of url or local directory
            features: Parser to use e.g. lxml, lxml-xml, html.parser
            form: Additional parsing to remove preamble for form='8-K'

        Returns:
            Text of body, parsed per Loughran and McDonald "Stage One"
        """

        # Retrieve html from local file or url
        root = root or Edgar.edgar_url        
        if not isinstance(root, str):
            root = ''
        if root.startswith('http'):
            r = requests_get(os.path.join(root, pathname),
                             delay=0.1,
                             verbose=verbose)
            if r is None:
                return ''
            r = r.content
        else:
            with open(os.path.join(root, pathname), "rb") as f:
                r = f.read()
            if not r:
                return ''

        soup = BeautifulSoup(r, features=features) #lxml-xml lxml html.parser
        _print('soup: %d' % len(soup.text), verbose=verbose)

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
        _print('table: %d %d' % (len(tags), len(text)), verbose=verbose)

        if form in Edgar._forms['8-K']:
            x = re.search('emerging growth company[\w\W]*? of the Exchange Act',
                          text)
            if x:
                text = text[x.end():]
            _print('form8k: %d' % len(text), verbose=verbose)

        # clean-up line breaks
        text = unicodedata.normalize("NFKD", text)  # Normalize
        text = '\n'.join(text.splitlines())
        text = re.sub(r'[ ]+\n', '\n', text)
        text = re.sub(r'\n[ ]+', '\n', text)
        text = re.sub(r'\n+', '\n', text)

        # Completed Stage One
        _print('normalize: %d' % len(text), verbose=verbose)
        return text


    @staticmethod
    def extract_filenames(detail: str, verbose: int = _VERBOSE) -> List[str]:
        """Extract ordered list of .htm and .txt filenames from filing detail

        Args:
            detail: Text of detail file

        Returns:
            List of html filenames found in the detail file
        """
        dflist = pd.read_html(detail)
        jdf = -1
        jrow = -1
        jcol = -1
        html_name = ''
        html_all = []
        forms = np.ravel(Edgar._forms.values())
        for idf in range(len(dflist)):
            for irow in range(len(dflist[idf].index)):
                for icol in range(len(dflist[idf].columns)):
                    if jdf < 0 and dflist[idf].iloc[irow, icol] in forms:
                        jdf, jrow, jcol = idf, irow, icol  # likely row of form
                for icol in range(len(dflist[idf].columns)):
                    if (".htm" in str(dflist[idf].iloc[irow, icol]).lower() 
                        or ".txt" in str(dflist[idf].iloc[irow, icol]).lower()):
                        for s in str(dflist[idf].iloc[irow, icol]).split():
                            if '.htm' in s.lower() or '.txt' in s.lower():
                                name = s
                        if jdf == idf and jrow == irow:
                            html_name = name
                        else:
                            html_all += [name]
        _print(f"(extract_filenames) [{jdf} {jrow} {jcol}] {html_all}",
                   verbose=verbose)
        return [html_name] + html_all if html_name else html_all


    @staticmethod
    def extract_item(text: str, item: str):
        """Extract mda or business description item from input text

        Args:
            text: Full text of filing, from which to extract passage for item
            item: Item to extract, in {'mda10K', 'bus10K', 'mda10Q', 'qqr10K'}

        Notes:

        https://www.sec.gov/fast-answers/answersreada10khtm.html
       
        10-Q items:

        PART I—FINANCIAL INFORMATION
        Item 1. Financial Statements.
        Item 2. Management’s Discussion and Analysis
        Item 3. Quantitative and Qualitative Disclosures About Market Risk.
        Item 4. Controls and Procedures.

        PART II—OTHER INFORMATION
        Item 1. Legal Proceedings.
        Item 1A. Risk Factors.
        Item 2. Unregistered Sales of Equity Securities and Use of Proceeds.

        10-K items:

        Part 1
        Item 1 – Business
        Item 1A – Risk Factors
        Item 1B – Unresolved Staff Comments
        Item 2 – Properties
        Item 3 – Legal Proceedings
        Item 4 – Mine Safety Disclosures
        Part 2
        Item 5 – Market
        Item 6 – Consolidated Financial Data
        Item 7 – Management's Discussion and Analysis of Financial Condition and Results of Operations
        Item 7A – Quantitative and Qualitative Disclosures about Market Risks
        Forward Looking Statements
        Item 8 – Financial Statements
        Item 9A. Controls and Procedures
        Item 9B. Other Information
        """

        def parse_helper(text, marker, start=0):
            """Helper to find all potential items"""
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
                next_beg = []             # this may helps exception
            text = text[start:]

            for item7 in item_beg:   # try to find begin
                begin = item7.search(text)
                begin = begin.start() if begin else -1
                _print('item begin?', item7, begin)
                if begin != -1:
                    break
            if begin != -1:          # found begin
                for item7A in item_end:
                    end = item7A.search(text, pos=begin + 1)
                    end = end.start() if end else -1
                    _print('item end?', item7A, end)
                    if end != -1:
                        break
                if end == -1:        # ITEM 7A end does not exist
                    for item8 in next_beg:    # often get exception undefined
                        end = item8.search(text, pos=begin + 1)
                        end = end.start() if end else -1
                        _print('next begin?', item8, end)
                        if end != -1:
                            break
                if end > begin:       # extract this found item
                    mda = text[begin:end].strip()
                else:
                    end = 0
            _print(f"(parse_helper) {len(mda)}, {end}/{len(text)}")
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
            'qqr10K': {
                'item_beg': [
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?7A', re.I),
                    re.compile('\n\s*?QUALITATIVE AND QUANTITATIVE DIS', re.I),
                    re.compile('\n\s*?QUANTITATIVE AND QUALITATIVE DIS', re.I)],
                'item_end': [
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?8', re.I),
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?9', re.I),
                    re.compile('REPORT OF INDEPENDENT', re.I),
                    re.compile('OPINION ON THE FINANCIAL', re.I),
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?III[^\w]+', re.I),
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?3[^\w]+', re.I)],
                'next_beg': [
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?8', re.I),
                    re.compile('\n\s*?I\s?T\s?E\s?M.?\s*?9', re.I),
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?III[^\w]+', re.I),
                    re.compile('\n\s*?P\s?A\s?R\s?T.?\s*?3[^\w]+', re.I)]},
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
        mda, end = parse_helper(text, markers[item], start=start)
        if not mda:
            start = 1
            mda, end = parse_helper(text, markers[item], start=start)

        best = mda   # return longest passage
        while mda and end > 0:
            start += end
            mda, end = parse_helper(text, markers[item], start=start)
            if mda and len(mda.encode('utf-8')) > len(best):
                best = mda
        return best


    #############################
    #
    # Load from SEC Edgar Website
    #
    #############################
    @staticmethod
    def get_detail_filings(pathname: str, form: str = '') -> Tuple[bytes, str]:
        """Fetch detail and concatenated filings given edgar pathname

        Args:
            pathname: Edgar pathname of filing
            form: Special parsing to exclude preamble if form in '8-K'

        Returns:
            Tuple of detail and concatenated filings text

        Notes:

        - Fetch detail text and extract filenames, with assumed primary first
        - If first filename is htm or is form8k, then fetch all concatenate
        - Else only read first (txt) file.  
        - If still fail then fetch from pathname
        """

        # Get detail page
        detail, lines = b'', ''
        detail = Edgar.fetch_detail(pathname=pathname)
        if detail:   # filing detail page missing!
            filenames = Edgar.extract_filenames(detail)  # retrieve filings
            if form in Edgar._forms['8-K'] and ".htm" in filenames[0]:
                filenames = [Edgar.parse_pathname(pathname, f)
                             for f in filenames if ".htm" in f]
                lines = "\n".join([Edgar.fetch_filing(f, form=form)
                                   for f in filenames])
            else:
                filename = Edgar.parse_pathname(pathname, filenames[0])
                lines = Edgar.fetch_filing(filename, form=form)
            if not lines:
                lines = Edgar.fetch_filing(pathname, form=form)
        else:
            pass
            _print("***MISSING DETAIL***", pathname)
        return (detail, lines)

    ###################################
    #
    # Write Locally
    #
    ###################################

    def to_localdir(self, form: str, item: str ='', date: int = 0,
                    permno: str = '') -> str:
       """Construct local dir name prefix for local archive
       
       Args:
           form: '10K' or '10Q' for items; '' for 10K/10Q filings
           item: 'detail' or 'mda10K' or 'bus10K', or 'qqr10K'
           date: Year or date; 0 for mda10K/bus10K/qqr10K
           permno: For mda10K/bus10K/qqr10K only

       Returns:
           Local folder name to store the filing or item
       """
       s = os.path.join(self.savedir,
                        form,
                        str(item), 
                        str(permno),
                        str(date // 10000) if date else '',
                        str(date) if date else '')
       os.makedirs(s, exist_ok=True)    # make directory if not exist
       return s        

    def to_localname(self, date: int, form: str, cik: str, pathname: str,
                     **kwargs) -> str:
        """Construct local filename from components and filing pathname

        Args:
            date: Filing date
            form: Type of form
            cik: Company identifier
            pathname: Edgar file pathname -- only need the last suffix, e.g.
                      edgar/data/1000045/0000950170-22-000940.txt

        Returns:
            Local filename (per Loughran-McDonald) to store associated filing.

        Examples:

        <localname> = YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt

        - e.g. 20211105_10-Q_edgar_data_1761312_0001558370-21-014714.txt
        """
        return "_".join([str(date),
                         form.replace('/A', '-A'),
                         "edgar_data",
                         str(cik),
                         os.path.split(pathname)[-1]])

    def save_detail(self, text: bytes, form: str, date: int, cik: str,
                    pathname: str, **kwargs) -> str:
        """Save text of detail file to a local filename

        Examples:

        ~10X/detail/YYYY/YYYYMMDD/<localname>

        ~10X/8-K/detail/YYYY/YYYYMMDD/<localname>

        <localname>: YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt
        """
        s = os.path.join(self.to_localdir(form='8-K' * (form in
                                                        Edgar._forms['8-K']),
                                          item='detail',
                                          date=date),
                         self.to_localname(date=date,
                                           cik=cik,
                                           form=form,
                                           pathname=pathname))
        with open(s, 'wb') as f:
            f.write(text)
        return s

    def save_item(self, text: str, form: str, item: str, permno: int,
                  pathname: str, **kwargs) -> str:
        """Save text of filing to a local filename

        Examples:

        ~10X/10-K/mda10K/PERMNO/<localname>

        ~10X/10-K/bus10K/PERMNO/<localname>

        <localname>: YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt
        """
        s = os.path.join(self.to_localdir(form=form, item=item, permno=permno),
                         pathname.split('/')[-1])
        with open(s, 'wt') as f:
            f.write(text)
        return s
            
    def save_filing(self, text: str, form: str, date: int, cik: str,
                    pathname: str, **kwargs) -> str:
        """Save text of filing to a local filename

        Examples:

        ~10X/YYYY/YYYYMMDD/<localname>

        ~10X/8-K/YYYY/YYYYMMDD/<localname>

        <localname>: YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt
        """
        s = os.path.join(self.to_localdir(form='8-K' * (form in
                                                        Edgar._forms['8-K']),
                                          date=date),
                         self.to_localname(date=date,
                                           form=form,
                                           cik=cik,
                                           pathname=pathname))
        with open(s, 'wt') as f:
            f.write(text)
        return s
            

    ###################################
    #
    # Read Locally
    #
    ###################################

    def __init__(self, savedir: str, zipped: bool = True, verbose=_VERBOSE):
        """To access (zipped or unzipped) Edgar cloned archives

        Args:
            savedir: Root folder where archives saved locally
            zipped: Whether to use zipped or unzipped version
        """
        self.savedir = str(savedir)
        self.zipped = zipped
        self.verbose = verbose

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
        
    def open(self, form: str = '', item: str = '', date: int = 0, 
             permno: int = 0) -> List:
        """Opens local (zipped or folder) archive and return list of documents
        
        Args:
            date: Year or daily date 
            item: Item in {'mda10K, 'detail', 'bus10K', 'qqr10K}
            form: Filing type in {'10-K', '10-Q', '8-K'}
            permno: Identifier of security to retrieve

        Returns:
            List filenames in selected archive

        Notes:

        - local file names are per Loughran-McDonald convention:

          <localname> = YYYYMMDD_FORM__edgar_data_CIK_ADSH.txt
             e.g. 20211105_10-Q_edgar_data_1761312_0001558370-21-014714.txt

        - filings text:

          - 10K/10Q documents are in: ~10X/YYYY/YYYYMMDD/<localname>
          - 8K documents are in: ~10X/8-K/YYYY/YYYYMMDD/<localname>

        - index details of filings:

          - 10K/10Q detail are in: ~10X/detail/YYYY/YYYYMMDD/<localname>
          - 8K detail are in: ~10X/8-K/detail/YYYY/YYYYMMDD/<localname>

        - extracted items by permno are in: ~/FORM/ITEM/PERMNO/

          - 10-K mda are in: ~10X/10-K/mda10K/PERMNO/
          - 10-K bus are in: ~10X/10-K/bus10K/PERMNO/

        - zipped archives created with: zip -q -r 2021.zip 2019

          - 2021.zip contains the year's 10-K and 10-Q filings
          - 10-K/detail/2021.zip contains the index details of those 10-X's.
          - 10-K/mda10K.zip contains extracted MD&A sections from 10-K's
          - 10-K/bus10K.zip contains extracted Business Description sections
          - 8-K/2021.zip contains the year's 8-K filings
          - 8-K/detail/2021.zip contains the index details of those 8-K's
        """
        
        def parse_pathname(pathname):
            """Helper to deconstruct components from archive file name"""
            items = pathname.split('_')         # components separated by '_'
            items[0] = items[0].split('/')[-1]  # split last substring on '/'
            return {'cik': int(items[4]),
                    'form': items[1].replace('-A', '/A'), # cannot have '/'
                    'date': int(items[0]),
                    'pathname': pathname.strip('/')}

        self.close()   # only one archive open at a time per instance

        date = str(date) if date else ''
        localpath = self.savedir
        for node in [form, item, date[:4]]:
            if node:
                localpath = os.path.join(localpath, node)

        if self.zipped:  # if zipped archive, have to retrieve its namelist()
            localpath = localpath + '.zip'
            recs = []
            with zipfile.ZipFile(localpath) as archive:
                for pathname in archive.namelist():
                    if pathname.endswith('/'):
                        self._print(pathname)
                    else:
                        recs.append(parse_pathname(pathname=pathname))
            df = DataFrame(recs).drop_duplicates(['pathname',
                                                  'date',
                                                  'form',
                                                  'cik'])
            if item and form and form not in self._forms['8-K']:
                t = DataFrame.from_dict(df['pathname'].str.split('/').to_dict(),
                                        orient='index')
                if permno:  # select by permno
                    df = df[t.iloc[:,1]==str(permno)].assign(permno=permno)
                else:       # select all, so explicitly assign permno col
                    df['permno'] = t.iloc[:,1].astype(int).values
            self.keys_ = df.to_dict('records')
            self.zipped = localpath
            self.archive = zipfile.ZipFile(localpath)
            
        else:   # else list files in folder
            def list_dir(*args):
                """helper to list potential document filenames in folder"""
                r = []
                for a in glob.glob(os.path.join(*args, '*')):
                    a = a.replace(self.savedir, '')
                    r.append({'pathname': a} if '_' not in a else 
                             parse_pathname(pathname=a))
                return r

            if date:
                if len(date) < 5:    # 4) date is year => not leaf
                    dates = glob.glob(os.path.join(localpath, date + '[01]???'))
                    df = pd.concat([DataFrame(list_dir(r)) for r in dates],
                                   ignore_index=True)
                else:                # 5) date is specific 8-digit date => leaf
                    q = glob.glob(os.path.join(localpath, date))
                    df = pd.concat([DataFrame(list_dir(r)) for r in q],
                                   ignore_index=True)                
            else:
                if permno:   # select by permno
                    q = glob.glob(os.path.join(localpath, str(permno)))
                    df = pd.concat([DataFrame(list_dir(r)) for r in q],
                                   ignore_index=True)               
                else:        # select all permnos
                    q = glob.glob(os.path.join(localpath, '[0-9]????'))
                    df = pd.concat([DataFrame(list_dir(r))\
                                    .assign(permno=int(r.split('/')[-1]))
                                    for r in q], ignore_index=True)
            self.keys_= df.to_dict('records')
            self.archive = localpath
        return self.keys_

    def close(self):
        """Close the archive"""
        try:
            self.archive.close()
            self.archive = None
        except:
            pass

    def __getitem__(self, pathname):
        """Retrieves text of document file by pathname from archive"""
        if self.zipped:
            _print(pathname)
            with self.archive.open(pathname) as stream:
                with io.TextIOWrapper(stream, encoding='latin-1') as infile:
                    text = infile.read()
        else:
            _print(self.savedir, pathname)
            with open(os.path.join(self.savedir, pathname)) as infile:
                text = infile.read()
        return text

if __name__ == "__main__":
    from finds.database import SQL
    from finds.structured import BusDay, PSTAT
    from secret import credentials, paths
    import tqdm
    import math
    
    def _test_web():
        """Access Edgar Webside

        master = Edgar.fetch_index(year=2023, quarter=1)
        - https://www.sec.gov/Archives/edgar/full-index/2023/QTR1/master.idx
        >>> {'cik': 1000045,
        >>>  'name': 'NICHOLAS FINANCIAL INC',
        >>>  'form': '10-Q',
        >>>  'date': 20230214,
        >>>  'pathname': 'edgar/data/1000045/0000950170-23-002704.txt'}

        Edgar.get_detail_filings(r['pathname'])
        - retrieve detail and filings text files
        """

        # Read index of filings in a quarter
        master = Edgar.fetch_index(year=2023, quarter=1, verbose=1)

        # retrieve its detail and actual filing text
        r = master[0]
        detail, filing = Edgar.get_detail_filings(r['pathname'])

    
    def _save_10X():
        """Sample code to load from Edgar web site and store locally

        ed.save_detail(text, form, date, cik, pathname)
        - save detail text in local file

        ed.save_filing(text, form, date, cik, pathname)
        - save filing textin local file

        """
        start_year, start_quarter = 2024, 1
        end_year, end_quarter = 2024, 1
        yq = [(math.floor(y), int((y+.25 - math.floor(y))*4))
              for y in np.arange(start_year + (start_quarter - 1) * .25,
                                 end_year + end_quarter * .25,
                                 0.25)]
        restart = {'year': 0,
                   'quarter': 0,
                   'filenum': 0}

        ed = Edgar(savedir=paths['10X'], zipped=False, verbose=1)
        forms = [f for c in ['10-K', '10-Q', '8-K'] for f in Edgar._forms[c]]
        tic = time.time()
        for year, quarter in yq:
            if year >= restart['year'] and quarter >= restart['quarter']:
                restart['quarter'] = 0
                files = Edgar.fetch_index(year=year, quarter=quarter)
                for filenum in sorted(files.keys()):
                    if filenum >= restart['filenum']:
                        r = files[filenum]
                        restart['filenum'] = 0
                        if r['form'] in forms:
                            det, fil = Edgar.get_detail_filings(r['pathname'])
                            if det:
                                ed.save_detail(text=det, **r)
                                ed.save_filing(text=fil, **r)
                                _print("--- Saved Filing ---", filenum,
                                       len(files), *r.values())

    
    def _extract_items():
        """Sample code to extract mda10K and bus10K, and store locally

        ed = Edgar(savedir, zipped=True)
        - open archive may be local files or zipped archive

        rows = ed.open(date=2022)
        - each row has keys ['cik', 'form', 'date, 'pathname']

        Edgar.extract_item(filing, item='mda10K')
        - extract item text from filing text

        ed.save_item(text, form='10-K', permno, item='mda10K', pathname)
        - save text in local file

        """ 
        ed = Edgar(savedir=paths['10X'], zipped=True, verbose=1)
        sql = SQL(**credentials['sql'])
        bday = BusDay(sql)
        pstat = PSTAT(sql, bday)
        to_permno = pstat.build_lookup(target='lpermno', source='cik')

        years = range(2023, 2022, -1)   # 1992
        items = {'10-K': ['qqr10K']}   # '10-Q': ['mda10Q']}
        #items = {'10-K': ['bus10K', 'mda10K']}
        logger = []
        for year in years: 
            rows = ed.open(date=year)
            row = rows[0]
            for i, row in enumerate(rows):
                permno = to_permno(int(row['cik']))
                if row['form'] in items and permno:
                    filing = ed[row['pathname']]
                    for item in items[row['form']]:
                        extract = Edgar.extract_item(filing, item)
                        ed.save_item(text=extract,
                                     form=row['form'],
                                     permno=permno,
                                     item=item,
                                     pathname=row['pathname'])
                        r = {'year': year,
                             'permno': permno,
                             'item': item,
                             'text_c': len(filing),
                             'item_c': len(extract),
                             'text_w': len(filing.split()),
                             'item_w': len(extract.split())}
                        logger.append(r)

