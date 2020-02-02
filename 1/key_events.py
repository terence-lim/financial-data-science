"""
Event Studies and CAR (cumulative abnormal returns)
Text Classification

References:

"""

import matplotlib.pyplot as plt
import seaborn as sns
import time, re
import numpy as np
import pickle
from pprint import pprint
import sklearn, sklearn.feature_extraction
import dives
import dives.util
import dives.dbengines
import dives.structured
import dives.unstructured
import dives.custom

import importlib
importlib.reload(dives)
importlib.reload(dives.util)
importlib.reload(dives.structured)
importlib.reload(dives.unstructured)
importlib.reload(dives.dbengines)
importlib.reload(dives.custom)

from dives.util import DataFrame, plot_returns, plot_wordcloud
from dives.dbengines import SQL, Redis, MongoDB
from dives.structured import BusDates, Benchmarks, CRSP, PSTAT
from dives.unstructured import Unstructured
from dives.custom import CustomTokenizer, CustomClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import secret  # passwords and db connection options
sql = SQL(**secret.value('sql'))       
rdb = Redis(**secret.value('redis'))   
mongodb = MongoDB(**secret.value('mongodb'))

bd = BusDates(sql)
bench = Benchmarks(sql, bd)
crsp = CRSP(sql, bd, rdb)
keydev = PSTAT(sql, bd)
    
if False:
    beg = 19890701  #20190101
    end = 20190630
    outdir = None  #'/home/terence/Downloads/out/events/'
    tic = time.time()
    
    for eventid in [101, 192, 65, 80, 26, 27, 86]:  # keydev.event.keys():
        for roleid in [1]:   # [None] + list(keydev.role.keys()):
            role, event = "", ""
            if roleid:
                role = " and keydevtoobjectroletypeid = {} ".format(roleid)
            if eventid:
                event = " and keydeveventtypeid = {} ".format(eventid)
            df = keydev.get_linked(table = 'keydev',
                                   date_field = 'announcedate',
                                   fields=['companyname',
                                           'keydevid',
                                           'keydeveventtypeid',
                                           'keydevtoobjectroletypeid'],
                                   where = ('announcedate >= {beg} ' \
                                            ' and announcedate <= {end} {event} {role}'\
                                            ''.format(beg = beg,
                                                      end = end,
                                                      event = event,
                                                      role = role)))
    
            if len(df) > 500:  # minimum sample size

                # get stock and market returns within {left} and {right} window around announcedate
                left = -1
                right = 21
                rets = crsp.get_window('daily','ret',df['permno'],df['announcedate'],left,right)
                mkt = bench.get_window('daily','ret',['Mkt-RF']*len(df),df['announcedate'],left,right)
                rf = bench.get_window('daily', 'ret', ['RF']*len(df), df['announcedate'], left, right)
                cols = ['day' + str(d) for d in range(right-left+1)]

                # Subtract market returns (could consider beta-adjust or other suitable benchmarks)
                rets[cols] = rets[cols] - (mkt[cols] + rf[cols])
                rets = rets.loc[rets[cols].isnull().sum(axis=1).eq(0)]  # require full window (biased)
                cumrets = (1+rets[cols]).cumprod(axis=1)                # construct cumulative returns

                # Split by market cap and sample period
                cap = crsp.get_many('daily', rets['permno'], rets['date'], ['prc','shrout'])
                cumrets['cap'] = list(cap['prc'].abs() * cap['shrout'])
                cumrets['date'] = list(rets['date'])                
                cumrets = cumrets[cumrets['cap'].ge(300000)]   # drop microcaps < $300 million
                lo, mid, hi = np.percentile(cumrets['date'], [0,50,100])
                before = cumrets['date'].le(mid)
                small = cumrets['cap'].lt(2000000)             # split large/small cap at $2 billion
                cumrets.loc[~small & ~before, 'sub'] = 'Large ({:.0f}-{:.0f})'.format(mid, hi)
                cumrets.loc[small & ~before, 'sub'] = 'Small ({:.0f}-{:.0f})'.format(mid, hi)
                cumrets.loc[~small & before, 'sub'] = 'Large ({:.0f}-{:.0f}]'.format(lo, mid)
                cumrets.loc[small & before, 'sub'] = 'Small ({:.0f}-{:.0f}]'.format(lo, mid)

                plt.figure(figsize=(10,6))
                plt.clf()
                ax = plt.subplot(2,2,1)
                for iplot, sub in enumerate(np.unique(cumrets['sub'])):
                    
                    # Combine stocks with same announcement date -- lest cross-sectional errors
                    excess = cumrets[cumrets['sub'] == sub].drop(columns='cap').groupby('date').mean()

                    # Compute event-window daily average cumulative returns
                    mean = excess.mean()                          # mean cumulative return of each day
                    stderr = excess.std()/np.sqrt(len(excess))    # cross-sectional stdeerr
                    tstat = (mean[1 - left] - 1)/stderr[1 - left] # tstat of third day's cumulative return
                    
                    post = excess.iloc[:,-1] / excess.iloc[:, 1-left]         # post-announcement drift
                    post = (post.mean()-1) / (post.std()/np.sqrt(len(post)))  # t-stat of average drift
                    
                    s = " %d:%-10s" % (roleid, keydev.role[roleid]) if roleid else ''
                    print("\n**** %2d %-20s %-14s n=%5d tstat: %6.2f  post: %6.2f%s ****\n\n" %
                          (eventid, keydev.event[eventid], s, len(excess), tstat, post, sub))

                    # Plot mean cumulative returns, and 2-stderr bands
                    plt.subplot(2, 2, iplot+1, sharex=ax, sharey=ax)
                    x = np.arange(left, right+1)                  # x-axis is event day number
                    y1 = DataFrame(index = x,
                                   data = {'excess': list(mean),  # y-axis are average and 2-stderr lines
                                           '+2 stderr': list(mean+2*stderr),
                                           '-2 stderr': list(mean-2*stderr)})
                                   
                    plot_returns(x = x, y1 = y1,
                                 title = ("{event} ({id}) {role}" \
                                          "".format(event = keydev.event[eventid],
                                                    id = eventid,
                                                    role =keydev.role[roleid] if roleid else '')),
                                 label1 = "%s [n=%d]" % (sub,len(excess)),
                                 hlines = [1],
                                 vlines = [1],
                                 xskip = 1,
                                 date = None,       # whether to format x-axis 
                                 cumprod = False)   # whether to apply log(cumprod(1+r)) transformation

                    if outdir:
                        with open(outdir + 'index.html', 'at') as f:
                            f.write("<pre>\n")
                            s = " %2d:%-10s" % (roleid, keydev.role[roleid]) if roleid else ''
                            f.write("%2d:%-20s %-14s n=%5d  tstat: %6.2f post: %6.2f  %s\n" %
                                    (eventid, keydev.event[eventid], s, len(excess), tstat, post, sub))
                            f.write("</pre>\n")
                if outdir:
                    savefig = outdir + "{}_{}.jpg".format(eventid, roleid if roleid else '') 
                    with open(outdir + 'index.html', 'at') as f:                        
                        f.write('<img src="{}"><hr><p>\n'.format(savefig))
                    plt.savefig(savefig)
    print('Time elapsed : %.1f secs' % (time.time() - tic))

if False:
    situations = Unstructured(mongodb, 'situations')
    tic = time.time()
    
    # Collect all sample text into data, and labels (i.e. event id's) into y_all
    data = []
    y_all = np.empty((0,))
    for event in [101, 192, 65, 80, 26, 27, 86]:
        docs = situations.mongo.find({'$and': [{'keydeveventtypeid' : {'$eq' : event}},
                                               {'keydevtoobjectroletypeid' : { '$eq': 1}}]},
                                     {'_id' : 0})
        text = [re.sub(r'\b\w*[\d]\w*\b', ' ', d['situation'])
                for d in docs if d['situation'] and len(d['situation']) > 80]
        data = data + text
        y_all = np.append(y_all, np.repeat(event, len(text)))

    with open('/home/terence/Downloads/situations.pkl','wb') as f:
        pickle.dump([data, y_all], f)

if True:
    with open('/home/terence/Downloads/situations.pkl','rb') as f:
        data, y_all = pickle.load(f)

    # Get stopwords wordlist
    wordlist = Unstructured(mongodb, 'wordlists')
    stopwords = wordlist.find_values('genericlong')[0]   # GenericLong stop words from LoughranMcDonald
    stopwords = CustomTokenizer()(" ".join(stopwords))
        
    # Transform text data: tokenize, lower-case, stopwords, and Tfidf vectorizer
    max_df, min_df, max_features = 0.99, 20, 10000
    # set up vectorizer and models to fit
    vectorizer = TfidfVectorizer(strip_accents = 'unicode',
                                 lowercase = True,
                                 stop_words = stopwords,
                                 max_df = max_df,
                                 min_df = min_df,
                                 max_features = max_features,
                                 tokenizer = CustomTokenizer()) #, token_pattern=r'\b[^\d\W]+\b')    
    X_all = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    
    # Fit the supervised learning models
    results = []   # to store best score of each model
    for name in ['svclinearCV', 'logisticCV', 'multinomialNB', 'kerasClassifierCV']:
        clf = CustomClassifier[name]   # wrappers calculate basic autotuning and best (training) score
        clf.fit(X_all, y_all)
        train_score = clf.score(X_all, y_all)
        print(name, clf.best_score_, train_score)
        results.append([clf.best_score_, name, clf])

    # Visualize best model -- but imbalance and confusion of guidance!
    name, clf = sorted(results)[-1][1:]                         # rank and get best model
    class_names = [keydev.event[e] for e in clf.classes_]       # lookup up class names
    y_pred = clf.predict(X_all)                                 # predicted labels
    conf_mat = sklearn.metrics.confusion_matrix(y_all, y_pred)  # display confusion matrix
    print(conf_mat)
    print("\n".join([str((int(e), keydev.event[e])) for e in clf.classes_]))
    sns.heatmap(conf_mat,                # heatmap of confusion matrix
                annot = True,
                fmt = 'd',
                yticklabels = clf.classes_,
                xticklabels = clf.classes_)

    topics = wordcloud_features(clf.best_estimator_.coef_, feature_names, 10, class_names,
                                unique=False, plot=True)
    print(name)
    for label, features in zip(class_names, topics):
        print(label)
        pprint(features)
    
