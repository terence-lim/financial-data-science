"""
the dives.unstructured module defines class and methods for unstructured datasets
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
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dives.util import print_debug
from dives.dbengines import MongoDB

try:
    import secret
    _h = secret.value('url_header')
    verbose = secret.value('verbose')
except:
    _h = None
    verbose = 1
#
# Unstructured is base class for unstructured datasets
#
class Unstructured(object):
    """Base class for manipulating an unstructured dataset

    Parameters
    ----------
    mongodb: MongoClient object
        connection to underlying mongodb where collection is stored
    name: string
        name of the collection

    Notes
    -----
    MongoDB is the nosql where documents presently stored

    """
    def __init__(self, mongodb, name):
        self.mongodb = mongodb
        self.mongo = mongodb.client[mongodb.database][name]  # make pymongo methods available
        self.name = name                                     # the name of this collection in mongodb

    def __str__(self):
        return str(self.name)

    def append_dataframe(self, df):
        """insert_many records from rows of the DataFrame {df}."""
        return len(self.mongo.insert_many(df.to_dict(orient='records')).inserted_ids)

    def insert_unique(self, keys, doc):
        """insert {doc}, after removing other instances with same values for list of {keys}"""
        result = self.mongo.delete_many({key : doc[key] for key in keys})
        self.mongo.insert_one({k:v for k,v in doc.items() if k != '_id'})  # do not re-use _id
        return result.deleted_count

    def update_key(self, key, value):
        """update_many documents that contain {key}, with new {value}; upsert if no matches"""
        return self.mongo.update_many({key : {'$exists' : True}}, {'$set' : {key : value}},
                                           upsert=True).matched_count

    def delete_key(self, key):
        """drop any documents contain {key}, regardless of value"""
        return self.mongo.delete_many({key : {'$exists' : True}}).deleted_count
    
    def find_values(self, key):
        """return list of values of {key} in all documents"""
        return [result[key] for result in self.mongo.find({key: {'$exists': True}})]

    def find_range(self, key, beg, end=None, include_id=False):
        """return dict, all documents with {key} value between {beg} and {end} inclusive"""
        if not end:
            end = beg
        if include_id:
            docs = self.mongo.find({key: {'$gte': beg, '$lte': end}})
        else:
            docs = self.mongo.find({key: {'$gte': beg, '$lte': end}}, {'_id': 0})
        return list(docs)
        #return {doc[key]: doc for doc in docs}

    def find_keys(self):
        """return the set of key names appearing in all docs"""
        return reduce(lambda all_keys, rec_keys: all_keys | set(rec_keys),
                      map(lambda d: d.keys(), self.mongo.find()), set())

    def read_fomc(self, verbose=verbose):
        """loader for FOMC minutes by scraping fed website"""

        # get_catalog() to collect list of url's, get_minutes() on each url, and insert_many() docs
        catalog = get_catalog()
        docs = {date : get_minutes(date, url) for date,url in catalog.items()}
        dates = list(docs.keys())
        print_debug("{} FOMC minutes read from {} to {}".format(len(dates), min(dates), max(dates)))
        if verbose:
            # display a random doc
            doc = docs[random.choice(docs.keys())]
            for num, line in enumerate(re.sub('\n+','\n', re.sub('[\r\t]',' ', doc['text'])).split('\n')):
                print('[%4d] %s' % (num, line[:100]))
            for date in sorted(docs.keys(), reverse=True):   # display length of text by descending date
                print(date, len(docs[date]['text']))
        self.mongo.delete()
        self.mongo.drop_indexes()
        self.mongo.insert_many(docs.values())
        self.mongo.create_index('date')
    
    def read_wordlist(self, name = 'wordlists',
                      dirname = '',
                      csvfile = 'LoughranMcDonald_MasterDictionary_2018.csv'):
        """loader for wordlists from Loughran-McDonald text files"""
        lm = pd.read_csv(dirname + csvfile)        # main csv file
        for c in ['Negative',
                  'Positive',
                  'Uncertainty',
                  'Litigious',
                  'Constraining',
                  'Superfluous',
                  'Interesting',
                  'Irr_Verb']:
            self.update_key(c.lower(), list(lm['Word'].loc[lm[c].ne(0)]))
        self.update_key('master',
                        list(lm['Word']))
        self.update_key('10X',
                        list(lm['Word'].loc[lm['Doc Count'].gt(0)]))
        self.update_key('strong',
                        list(lm['Word'].loc[lm['Modal'].eq(1)]))
        self.update_key('moderate',
                        list(lm['Word'].loc[lm['Modal'].eq(2)]))
        self.update_key('weak',
                        list(lm['Word'].loc[lm['Modal'].eq(3)]))
    
        for c in ['Generic', 'GenericLong']: # read and load the stopword txt files
            generic = pd.read_csv(dirname + 'StopWords_' + c + '.txt')
            self.update_key(c.lower(), list(generic.iloc[:,0]))
    
#
# helper methods to load FOMC minutes documents
#
def get_catalog(root_url= 'https://www.federalreserve.gov/'):
    """helper function to get dict of dates and urls of FOMC minutes from federal website"""
        
    dateOf = lambda s: int(re.sub('\D', '', s)[-8:])
        
    # latest five years' minutes can be linked from a main page
    new_url = root_url + 'monetarypolicy/fomccalendars.htm'
    raw   = BeautifulSoup(requests.get(new_url, headers=_h).content, 'html.parser')
    hrefs = raw.find_all('a', href=re.compile('\S+minutes\S+.htm$', re.IGNORECASE))
    links = [root_url + m.attrs['href'] for m in hrefs]

    # but earlier years' minutes are linked from annual pages with this url format
    old_url = root_url + 'monetarypolicy/fomchistorical%d.htm'
    for year in range(1993, min([dateOf(m) for m in links]) // 10000):
        raw = BeautifulSoup(requests.get(old_url % year, headers=_h).content, 'html.parser')
        hrefs = raw.find_all('a', href=re.compile('\S+minutes\S+.htm$', re.IGNORECASE))
        links += [root_url + m.attrs['href'].replace(root_url,'') for m in hrefs]
    return {dateOf(link) : link for link in links}

def get_minutes(date, url):
    """helper function to retrieve a minutes document for {date} from {url} at fed website"""
    raw = BeautifulSoup(requests.get(url,headers=_h).content, 'html.parser')
    minutes = "\n\n".join([p.get_text().strip() for p in raw.findAll('p')])
    return {'url' : url,
            'date' : date,
            'text' : re.sub('\n+','\n', re.sub('[\r\t]','', minutes))}


if __name__ == "__main__":
    import secret
    import time
    dirname = ''
    if False:
        mongodb = MongoDB(**secret.value('mongo'))
        #
        # kludge to parse multi-line csv, and issues with quoted labels
        #
        situations = Unstructured(mongodb, 'situations') 
        for year in range(2018, 2013, -1):
            csvfile = dirname + 'key' + str(year) + '.csv.gz'
            outfile = dirname + 'situation' + str(year) + '.csv.gz'

            with gzip.open(csvfile, mode = "rt", encoding="latin-1") as f:
                lines = f.readlines()  # "ISO-8859-1" "latin-1")
            print(year, len(lines))
            """
            l = lines.copy()
            lines = l[:400].copy()
            """
            tic = time.time()
            for i in range(len(lines)-1, 0, -1):
                lines[i] = lines[i].encode('ascii', 'ignore').decode('ascii').replace('|','.')
                if len(lines[i]) and lines[i][0] == '\n':   # delete empty lines
                    del lines[i]
                elif re.match('\d\d\d\d\d\d\d', lines[i]) is None:   # startswith number, else append
                    lines[i-1] += lines[i]
                    del lines[i]
                else:  # is valid line, so clean it: (1) replace commas in "" (2) sep='|' (3) delete \n
                    lines[i] = re.sub('"[^"]*"', lambda m: m.group(0).replace(',', ''), lines[i])
                    lines[i] = lines[i].replace(',', '|', 12)
                    lines[i] = re.sub('["\n]', ' ', re.sub('\x1a', ' ', lines[i]))
            columns = lines[0].replace('\n', '').split(',')
            items = list(csv.reader(io.StringIO("\n".join(lines[1:])), quotechar=None, delimiter='|'))
            df = DataFrame(data=items, columns=columns)
            df.columns = df.columns.map(str.lower)
            keep = ['keydevid', 'keydeveventtypeid', 'keydevtoobjectroletypeid', 'announcedate',
                    'headline', 'situation']
            gvkey = pd.to_numeric(df['gvkey'], errors='coerce')
            d = df.loc[gvkey.notnull() & gvkey.gt(0), keep]
            d[keep[:4]] = d[keep[:4]].astype(int)
            d = d.drop_duplicates(keep='first', inplace=False)
            d.to_csv(outfile, sep='|', quotechar=None, quoting=csv.QUOTE_NONE, index=False)

            d = d[d['keydeveventtypeid'].isin([101, 192, 26, 27, 65, 80, 86])]
            situations.append_dataframe(d)
            print(year, len(lines), len(d), time.time() - tic)
        situations.mongo.create_index('keydeveventtypeid')
        situations.mongo.create_index('announcedate')
        situations.mongo.create_index('keydevtoobjectroletypeid')

    
    if False: # keydev headlines
        year = 2019
        filename = dirname + 'keydev' + str(year) + '.csv.gz'
        df = pd.read_csv(filename, sep='|', low_memory=False, na_filter=False)
        df.columns = map(str.lower, df.columns)
        gvkey = pd.to_numeric(df['gvkey'], errors='coerce')
        d = df.loc[gvkey.notnull() & gvkey.gt(0), ['keydevid','keydeveventtypeid','announcedate',
                                               'keydevtoobjectroletypeid','gvkey','headline']]
        d = d.drop_duplicates(keep='first', inplace=False)
        d = d.to_dict(orient='records')
        
    if False:
        #
        # update headlines from keydev csv files
        #
        hl = Unstructured(mongo, 'headlines') 
        hl.mongo.create_index([('keydevid', pymongo.ASCENDING),
                               ('announcedate', pymongo.ASCENDING),
                               ('keydeveventtypeid',pymongo.ASCENDING),
                               ('keydevtoobjectroletypeid', pymongo.ASCENDING),
                               ('gvkey', pymongo.ASCENDING)], unique=True)

        for year in range(2019, 1988, -1):
            tic = time.time()
            filename = path + str(year) + '.csv.gz'
            df = pd.read_csv(filename, sep='|', low_memory=False, na_filter=False)
            df.columns = df.columns.map(str.lower).map(str.rstrip)
            gvkey = pd.to_numeric(df['gvkey'], errors='coerce')
            d = df[['keydevid', 'keydeveventtypeid', 'announcedate', 'keydevtoobjectroletypeid',
                    'gvkey', 'headline']][gvkey.notnull() & gvkey.gt(0)].drop_duplicates(keep='first')
            print(year, len(df), len(d), time.time() - tic)
            hl.append_dataframe(d)

    if False:
        #
        # aid removal of headers and footers of fomc minutes, by looping files through user's editor
        #
        """remove first section: personnel, authorizations, 
        annual disposition of organizational matters, procedural instructtions, directives
        keep section beginning: Review of Monetary Policy Strategy, Tools, and Communication Practices
        or Developments in Financial Markets and Open Market Operations
        or: Staff Review of the Economic Situation, 
        or: Manager of the System Open Market Account reported
        """
        fomc     = Unstructured(mongodb, 'fomc')
        minutes1 = Unstructured(mongodb, 'minutes1')

        tic = time.time()
        year = 1993                    # edit year by year
        docs    = minutes.find_range('date', (year*10000) + 101, (year*10000) + 1231)
        for doc in docs:
            date = doc['date']                 
            initial_message = doc['text'].encode("utf-8")           # original text
            with tempfile.NamedTemporaryFile(suffix=".tmp") as f:   # save in a temp file
                f.write(initial_message)
                f.flush()
                subprocess.call([os.environ.get('EDITOR','emacs'), "-nw", f.name])  # call editor
                f.seek(0)
                edited_message = f.read().decode("utf-8")           # keep edited text
                if not len(edited_message):
                    raw = input()
                doc['text'] = edited_message   
        for doc in docs:
            print(doc['date'], len(doc['text']), minutes1.insert_unique(['date'], doc)) # display check
        print(time.time() - tic, year)
        sorted(minutes1.find_values('date'), reverse=True)[-13:]
