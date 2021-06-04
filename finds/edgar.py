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
from .readers import requests_get

ECHO = False

# https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm
# https://www.investor.gov/introduction-investing/general-resources/
#   news-alerts/alerts-bulletins/investor-bulletins/how-read-8

class Edgar:
    # EDGAR_Forms_v2.1.py from ND-SRAF / McDonald : 201606
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
    def __init__(self, dirname, echo=ECHO):
        self.dirname = dirname
        self.echo_ = echo

    def _print(self, *args, echo=None):
        if echo or self.echo_:
            print(*args)

    @staticmethod
    def fetch_tickers(echo=ECHO):
        """Fetch tickers-to-cik lookup from SEC web page as a pandas Series"""
        url = 'https://www.sec.gov/include/ticker.txt'
        tickers = requests_get(url, echo=echo).text
        df = DataFrame(data = [t.split('\t') for t in tickers.split('\n')],
                       columns = ['ticker','cik'])
        return df.set_index('ticker')['cik'].astype(int)

    @staticmethod
    def fetch_index(date=None, year=None, quarter=None, echo=ECHO):
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
    def fetch_detail(pathname, root=None, echo=ECHO):
        """Fetch from HTML file, containing table of hyperlinks of documents"""
        root = root or Edgar.url_prefix
        subs = Edgar.from_pathname(pathname)
        r = requests_get(os.path.join(root, subs['indexname']), echo=echo)
        return None if r is None else r.content

    @staticmethod
    def fetch_filing(pathname, root=None, form8k=False, echo=ECHO):
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
        if echo: print('soup: %d' % len(soup.text))

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
        if echo: print('table: %d %d' % (len(tags), len(text)))

        if form8k:
            x = re.search('emerging growth company[\w\W]*? of the Exchange Act',
                          text)
            if x:
                text = text[x.end():]
            if echo: print('form8k: %d' % len(text))

        # clean-up line breaks
        text = unicodedata.normalize("NFKD", text)  # Normalize
        text = '\n'.join(text.splitlines())
        text = re.sub(r'[ ]+\n', '\n', text)
        text = re.sub(r'\n[ ]+', '\n', text)
        text = re.sub(r'\n+', '\n', text)

        # Completed Stage One
        if echo: print('normalize: %d' % len(text))
        return text


    @staticmethod
    def from_pathname(pathname, filename=None):
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
    def extract_filename(detail, echo=ECHO):
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
        if echo: print(f"(extract_filename) [{jdf} {jrow} {jcol}] {html_all}")
        return [html_name] + html_all if html_name else html_all

    def archive_index(self, form='', date=None, item=None, permno=None, zip=1):
        """List index of filings in local (zipped or folders) archive
        Examples
        --------
        form, date, item, permno = '10-K', None, 'bus', None
        form, date, item, permno = '10-K', None, 'bus', 76676
        form, date, item, permno = '10-K', 2020, 'detail', None
        form, date, item, permno = '10-K', 20200427, 'detail', None
        form, date, item, permno = '8-K', 20200427, 'detail', None
        form, date, item, permno = '8-K', 20200427, None, None        
        form, date, item, permno = '10-Q', 2020, None, None        
        form, date, item, permno = '10-Q', 20200608, None, None        
        form, date, item, permno = '10-Q', 2020, 'detail', None        
        zip = False
        form, date, item, permno = '', 2020, None, None
        form, date, item, permno = '', 2019, 'detail', None
        form, date, item, permno = '10-K', None, 'mda', None
        form, date, item, permno = '8-K', 2020, None, None
        form, date, item, permno = '8-K', 2020, 'detail', None
        zip = True
        """
        def list_zip(zipname):
            """Helper to list index of zipped archive"""
            r = []
            with zipfile.ZipFile(zipname) as archive:
                for pathname in archive.namelist():
                    if pathname.endswith('/'):
                        self._print(pathname)
                    else:
                        #r.append(pathname)
                        r.append(self.from_archivepath(pathname=pathname))
            return r
        
        def list_dir(dirname, suffix='*'):
            """Helper to list files in a folder"""
            r = []
            for a in glob.glob(os.path.join(dirname, suffix)):
                a = a.replace(dirname + '/', '')
                r.append({'pathname': a} if '_' not in a else 
                         self.from_archivepath(pathname=a))
            return r

        p = os.path.join(self.dirname, form) # infer and construct full pathname
        if item is not None:    # 1) item may be mda10K, bus10K, detail
            p = os.path.join(p, item)  
        if date is not None:    # 2) date may be year
            date = str(date)
            p = os.path.join(p, date[:4] if len(date)>4 else '', str(date))
        if zip:                 # 3a) zip is True => return list_zip()
            r = list_zip(p + '.zip')
            df = DataFrame(r).drop_duplicates(['pathname','date','form','cik'])
            if date is None:
                t = DataFrame.from_dict(df['pathname'].str.split('/').to_dict(),
                                        orient='index')
                if permno is None:
                    df['permno'] = t.iloc[:,1].values
                else:
                    df=df[t.iloc[:,1].astype(int)==permno]
        else:                   # 3b) zip is False =>
            if permno is not None:   # 4) permno 
                p = os.path.join(p, str(permno))
            if date is not None and len(date) < 5: #5a) date is year => not leaf
                df = pd.concat([DataFrame(list_dir(p, os.path.basename(r)+'/*'))
                                for r in glob.glob(os.path.join(
                                        p, date+'[01]???'))], ignore_index=True)
            else: # 5b) date is None or 8-digit => is leaf, so return list_dir()
                df = DataFrame(list_dir(p))
                if date is None and permno is None:
                    df['permno'] = df['pathname'].astype(int)
        return df, p + ('.zip' if zip else '')

    @staticmethod
    def to_archivepath(date, form, cik, filename):
        """Construct file name in archive from components of the filing"""
        base = os.path.split(filename)[-1]
        return f"{date}_{form.replace('/A', '-A')}_edgar_data_{cik}_{base}"

    @staticmethod
    def from_archivepath(pathname):
        """Deconstruct components of the filing from archive file name"""
        items = pathname.split('_')        # components are separated by '_'
        items[0] = items[0].split('/')[-1] # in the last substring split on '/'
        return {'cik': int(items[4]),
                'form': items[1].replace('-A', '/A'), # basename cannot have '/'
                'date': int(items[0]),
                'pathname': pathname}

    @staticmethod
    def archive_filing(pathname, archive):
        """Read filing text from pathname in local (zip or folders) archive"""
        if archive.endswith('.zip'):
            with zipfile.ZipFile(archive).open(pathname) as f:
                with io.TextIOWrapper(f, encoding='latin-1') as g:
                    text = g.read()
                    #text = re.sub('(May|MAY)', ' ', text)  # drop 'May'
        else:
            with open(os.path.join(archive, pathname)) as f:
                text = f.read()
        return text

    @staticmethod
    def extract_item(text, item, echo=ECHO):
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

        def parse_helper(text, marker, start=0, echo=ECHO):
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
                if echo:
                    print('item begin?', item7, begin)
                if begin != -1:
                    break
            if begin != -1:          # found begin
                for item7A in item_end:
                    end = item7A.search(text, pos=begin + 1)
                    end = end.start() if end else -1
                    if echo:
                        print('item end?', item7A, end)
                    if end != -1:
                        break
                if end == -1:        # ITEM 7A end does not exist
                    for item8 in next_beg:    # often get exception undefined
                        end = item8.search(text, pos=begin + 1)
                        end = end.start() if end else -1
                        if echo:
                            print('next begin?', item8, end)
                        if end != -1:
                            break
                if end > begin:       # extract this found item
                    mda = text[begin:end].strip()
                else:
                    end = 0
            if echo: print(f"(parse_helper) {len(mda)}, {end} / {len(text)}")
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

if False: # download 10-K's, 10-Q's, and 8-Ks
    from settings import settings
    from finds.edgar import Edgar
    import time, os
    import numpy as np
    from pandas import DataFrame, Series
    ed = Edgar(settings['10X'])

    quarters = [2021] # np.arange(2012, 2019, 0.25):
    tic = time.time()
    logger = {}
    forms = ed.forms_['10-K'] + ed.forms_['10-Q'] + ed.forms_['8-K']
    for year in quarters:
        y = int(year)
        q = int(1 + ((year - y)*4))
        files = ed.fetch_index(year=y, quarter=q)
        for i, r in files.iterrows():  # 294124
            if i < 294124:
                continue
            if r['form'] in forms:
                form8k = r['form'] in ed.forms_['8-K']
                detail = ed.fetch_detail(pathname=r['pathname'])
                if detail is None:               # filing detail page missing!
                    print('****', r['name'], r['pathname'], '****')
                    continue
                s = os.path.join(ed.dirname,     # to save detail webpage
                                 '8-K' if form8k else '',
                                 'detail',
                                 str(r['date'] // 10000),
                                 str(r['date']),
                                 ed.to_archivepath(date=r['date'],
                                                   form=r['form'],
                                                   cik=r['cik'],
                                                   filename=r['pathname']))
                os.makedirs(os.path.dirname(s), exist_ok=True)
                with open(s, 'wb') as g:
                    g.write(detail)
                print(r['date'], r['form'], r['cik'], f"*{i}/{len(files)}*")

                # to save main filing document text
                f = ed.extract_filename(detail)  # retrieve main filings
                if form8k and ".htm" in f[0]:    # for 8-K's: merge all .htm*'s
                    filenames = [ed.from_pathname(r['pathname'], g)
                                 for g in f if ".htm" in g]
                    text = "\n".join([ed.fetch_filing(s, form8k=True)
                                      for s in filenames])
                else:
                    filename = ed.from_pathname(r['pathname'], f[0])
                    text = ed.fetch_filing(filename, form8k=form8k)
                if not text:
                    text = ed.fetch_filing(r['pathname'], form8k=form8k)
                s = os.path.join(ed.dirname,     # save filing document text
                                 '8-K' if form8k else '',
                                 str(r['date'] // 10000),
                                 str(r['date']),
                                 ed.to_archivepath(date=r['date'],
                                                   form=r['form'],
                                                   cik=r['cik'],
                                                   filename=r['pathname']))
                os.makedirs(os.path.dirname(s), exist_ok=True)
                with open(s, 'wt') as g:
                    g.write(text)
                logger[r['pathname'].split('/')[-1]] = r.append(
                    Series({'len': len(text)})).drop('pathname')
                print(*r.values, f"*{i}/{len(files)}*")
    logger = DataFrame.from_dict(logger, orient='index')

if False: # extract and save MDA BUS text
    from finds.database import SQL
    from finds.structured import PSTAT
    from finds.busday import BusDay
    from settings import settings
    from finds.edgar import Edgar
    ed = Edgar(settings['10X'])
    
    sql = SQL(**settings['sql'])
    bday = BusDay(sql)
    rdb = None
    pstat = PSTAT(sql, bday)
    to_permno = pstat.build_lookup(source='lpermno', target='cik', fillna=0)

    mindate = 0   # 20201001
    items = {'10-K': ['bus10K', 'mda10K'], '10-Q': ['mda10Q']}
    items = {'10-K': ['bus10K', 'mda10K']}
    logger = []
    for year in [2021]:  #2019, 2021):  # Start 1998++
        files, root = ed.archive_index(date=year, zip=False)
        for _, f in files.iterrows():
            permno = to_permno(f['cik'])
            if f['form'] in items and permno and f['date'] >= mindate:
                filing = ed.archive_filing(f['pathname'], root)
                for item in items[f['form']]:
                    extract = ed.extract_item(filing, item)
                    s = os.path.join(ed.root, item, str(permno),
                                     os.path.basename(f['pathname']))
                    os.makedirs(os.path.dirname(s), exist_ok=True)
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

# NOT USED, because from .htm rather than .txt
xdrop = [re.compile(b"<type>GRAPHIC", re.I),
         re.compile(b"<type>EXCEL", re.I),
         re.compile(b"<type>PDF", re.I),
         re.compile(b"<type>ZIP", re.I),
         re.compile(b"<type>COVER", re.I),
         re.compile(b"<type>CORRESP", re.I),
         re.compile(b"<type>JSON", re.I),
         re.compile(b"<type>XML", re.I)]
xexhibit = re.compile(b"<type>EX-", re.I)
xdocs = re.compile(b"<DOCUMENT>[\w\W]*?<", re.I)
xmda = re.compile(b"DISCUSSION AND ANALYSIS[\w\W]*?DISCUSSION AND ANALYSIS",re.I)
xtype = re.compile(b"<TYPE>[\w\W]*?<", re.I)
xbrls = [re.compile("ix:\S*", re.I), re.compile("xbrli:\S*", re.I)]

if False: # process 8K files
    """
    Save Insider Transactions
    https://www.sec.gov/cgi-bin/own-disp?action=getissuer&CIK=0000084246
    pull out <table border="0" cellspacing="0" id="transaction-report">
    except first row, drop other <tr align="left" valign="top" class="header">
    1. 8K Insider Trading Gap Event Study

       Price Change (mkt-adjusted) 3-days up to close before Report Period Date
       Price Change (mkt-adjusted) Report Period Date up to close of day after filing date
       Price Change (mkt-adjusted) 3-days from day after filing date

       With or Without Insider Activity
       By 8K Type

   2. Abnormal (absolute) Returns and Volume
      around Report Period Date and Filing Date
      by event type -- compare to average return
      especially:
    Item 4.01 – Changes in Registrant’s Certifying Accountant
    Item 3.03 – Material Modification to Rights of Security Holders
    Item 4.02 – Non-Reliance on Previously Issued
    Item 5.02 – Departure; Election of Directors; Appointment; Compensatory Arrangements
    Item 3.01 – Notice of Delisting
    Item 2.06 – Material Impairments
    Item 1.01 – Entry into a Material Definitive Agreement
    Item 1.02 – Termination of a Material Definitive Agreement
    1.03 – Bankruptcy or Receivership
    Item 2.03 – Creation of a Direct Financial Obligation
    Item 2.04 – Triggering Events
    Item 2.05 – Costs Associated with Exit

    3. Cosine, Jacard and Wordvec  Similarity of (parsed) Nouns of Bus10K in 2020, especially Zoom and Moderna
    https://arxiv.org/abs/1803.11175
    SSRN pre-trained

    import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/1?tf-hub-format=compressed"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)
similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    message_embeddings_ = session.run(similarity_message_encodings, feed_dict={similarity_input_placeholder: messages})

    corr = np.inner(message_embeddings_, message_embeddings_)
"""
