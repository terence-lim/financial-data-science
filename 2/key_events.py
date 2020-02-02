"""
Key Developments

Event Studies, Cumulative Abnormal Returns (CAR)
NLP, Supervised Learning, Text Classification

References:

"""
import matplotlib.pyplot as plt
import seaborn as sns
import re
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

from dives.util import DataFrame, wordcloud_features
from dives.dbengines import SQL, Redis, MongoDB
from dives.structured import BusDates, Benchmarks, CRSP, PSTAT
from dives.unstructured import Unstructured
from dives.custom import CustomTokenizer, CustomClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import secret    # passwords etc
sql = SQL(**secret.value('sql'))       
rdb = Redis(**secret.value('redis'))   
mongodb = MongoDB(**secret.value('mongodb'))
bd = BusDates(sql)
bench = Benchmarks(sql, bd)
crsp = CRSP(sql, bd, rdb)
keydev = PSTAT(sql, bd)
outdir = secret.value('events')['dir']

#
# helper function to extract announcement window and evaluate event returns
#   Only a simple market adjustment applied --
#   may want to consider using estimated stock betas and/or other benchmarks
#
def event_study(keydev, eventid, roleid=None, beg=19890701, end=20190630, outdir='', minobs=500):
    """helper function to execute event study for list of events by id

    Parameters
    ----------
    keydev : PSTAT instance
        connection to keydev database
    eventid : int
        event id
    roleid : int, optional (default None)
        role id.  None for all roles
    beg : int, optional
        earliest announcement date
    end : int, optional
        latest announcement date
    outdir : string, optional (default '')
        directory in which to store jpg and html results.  None or blank to display only.
    minobs : int, optional (default 500)
        minimum sample size
    """

    # query keydev database, link to crsp permnos
    numobs = 0
    role = " and keydevtoobjectroletypeid = {} ".format(roleid) if roleid else ""
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
    if len(df) > minobs:  # minimum sample size
        # get stock and market returns within {left} and {right} window around announcedate
        left = -1
        right = 21
        rets = crsp.get_window('daily', 'ret', df['permno'],     # get stock returns for permnos
                               df['announcedate'], left, right)  #   around announcement date
        mkt = bench.get_window('daily', 'ret', ['Mkt-RF'] * len(df),
                               df['announcedate'], left, right)  # and market returns
        rf = bench.get_window('daily', 'ret', ['RF'] * len(df),
                              df['announcedate'], left, right)   # and risk free-returns
        cols = ['day' + str(d) for d in range(right-left+1)]

        # Subtract market returns (could consider beta-adjust or other suitable benchmarks)
        rets[cols] = rets[cols] - (mkt[cols] + rf[cols])
        rets = rets.loc[rets[cols].isnull().sum(axis=1).eq(0)]  # require full window (biased)
        cumrets = (1+rets[cols]).cumprod(axis=1)                # construct cumulative returns
        
        # Split sample by market cap (large, small) and sample period (first, second half)
        cap = crsp.get_many('daily',          # query crsp for market cap (prc*shrout) of permnos
                            rets['permno'],
                            rets['date'],
                            ['prc', 'shrout'])
        cumrets['cap'] = list(cap['prc'].abs() * cap['shrout'])
        cumrets['date'] = list(rets['date'])                
        cumrets = cumrets[cumrets['cap'].ge(300000)]   # drop microcaps < $300 million
        lo, mid, hi = np.percentile(cumrets['date'], [0,50,100])  # to halve sample period
        before = cumrets['date'].le(mid)
        small = cumrets['cap'].lt(2000000)             # split large/small cap at $2 billion
        cumrets.loc[~small & ~before, 'sub'] = 'Large ({:.0f}-{:.0f})'.format(mid, hi)
        cumrets.loc[small & ~before, 'sub'] = 'Small ({:.0f}-{:.0f})'.format(mid, hi)
        cumrets.loc[~small & before, 'sub'] = 'Large ({:.0f}-{:.0f}]'.format(lo, mid)
        cumrets.loc[small & before, 'sub'] = 'Small ({:.0f}-{:.0f}]'.format(lo, mid)
        numobs += len(cumrets)

        # average stocks' cumulative abnormal returns and compute stderrs.  display and plot
        plt.figure(figsize=(10,6))
        plt.clf()
        ax = plt.subplot(2,2,1)    # for sharing axes
        for iplot, sub in enumerate(np.unique(cumrets['sub'])):
                    
            # Combine stocks with same announcement date -- lest correlated cross-sectional errors
            excess = cumrets[cumrets['sub'] == sub].drop(columns='cap').groupby('date').mean()

            # Compute event-window daily average cumulative returns
            mean = excess.mean()                          # mean cumulative return of each day
            stderr = excess.std()/np.sqrt(len(excess))    # cross-sectional stdeerr
            tstat = (mean[1 - left] - 1)/stderr[1 - left] # tstat of third day's cumulative return
                    
            post = excess.iloc[:,-1] / excess.iloc[:, 1-left]         # post-announcement drift
            post = (post.mean()-1) / (post.std()/np.sqrt(len(post)))  # t-stat of average drift
                    
            s = " (%d) %s" % (roleid, keydev.role[roleid]) if roleid else ''
            print("[(%d) %s %s] n=%5d, tstat: %6.2f,  post: %6.2f, %s" %
                  (eventid, keydev.event[eventid], s, len(excess), tstat, post, sub))

            # Plot mean cumulative returns, and 2-stderr bands
            plt.subplot(2, 2, iplot+1, sharex=ax, sharey=ax)      # sharex and sharey
            x = np.arange(left, right+1)                          # x-axis is event day number
            plt.plot(x, mean, 'b-')
            plt.fill_between(x, mean - (2 * stderr), mean + (2 * stderr), alpha = 0.3)
            plt.legend(['excess', '+/- 2 stderr'])
            plt.axhline(1, linestyle=':', color='y')
            plt.axvline(1, linestyle=':', color='y')
            plt.title ("{event} ({id}) {role}" \
                       "".format(event = keydev.event[eventid],
                                 id = eventid,
                                 role =keydev.role[roleid] if roleid else ''),
                       fontsize = 'small')
            plt.ylabel("%s [n=%d]" % (sub, len(excess)), fontsize = 'small')
            
            if outdir:    # to save jpg and html
                with open(outdir + 'index.html', 'at') as f:
                    f.write("<pre>\n")
                    s = " (%d) %s" % (roleid, keydev.role[roleid]) if roleid else ''
                    f.write("(%d) %s %s n=%5d,  tstat: %6.2f, post: %6.2f,  %s\n" %
                            (eventid, keydev.event[eventid], s, len(excess), tstat, post, sub))
                    f.write("</pre>\n")
        if outdir:
            savefig = outdir + "{}_{}.jpg".format(eventid, roleid if roleid else '') 
            with open(outdir + 'index.html', 'at') as f:
                f.write('<img src="{}"><hr><p>\n'.format(savefig))
            plt.savefig(savefig)
        else:
            plt.show()
    return numobs
                        
if False:
    #
    # Average cumulative abnormal returns (CAR) for some events:
    #  compare announcement-window vs post-announcement-drift
    #
    events = [101, 192, 65, 80, 26, 27, 86]
    role = 1
    for event in events:
        n = event_study(keydev, event, role, outdir = None)
        print('{:4d} obs for event {:3d}. {}'.format(n, event, keydev.event[event]))
        plt.show()

if False:
    #
    # Run event study for all events and roles
    #
    events = keydev.event.keys()
    roles = [None] + list(keydev.role.keys())      # loop over all role-ids
    if outdir:
        with open(outdir + 'index.html', 'wt') as f:
            f.write('Event study CAR<br>')
            f.write(' tstat: is of 3-day average CAR around announcement date<br>')
            f.write(' post: is t-stat of average CAR from day+2 to day+21 (both days inclusive)<br>')
            f.write('<p>\n')
    for event in events:
        numobs, numroles = 0,0
        for role in roles:
            n = event_study(keydev, event, role, outdir = 'outdir')
            if n:
                numobs, numroles = numobs + n, numroles + 1
        if outdir and numobs:
            print('{:4d} obs {:1d} roles for event {:3d}. {}'.format(
                numobs, numroles, event, keydev.event[event]))
if False:
    #
    # Classify events from situation text using NLP and Supervised Learning
    #
    situations = Unstructured(mongodb, 'situations')   # unstructured text was stored in mongodb
    
    # Collect all sample text into {data}, and labels (i.e. event id's) into {y_all}
    data = []
    y_all = np.empty((0,))
    for event in [101, 192, 65, 80, 26, 27, 86]:
        docs = situations.mongo.find({'$and': [{'keydeveventtypeid' : {'$eq' : event}},
                                               {'keydevtoobjectroletypeid' : { '$eq': 1}}]},
                                     {'_id' : 0})
        text = [re.sub(r'\b\w*[\d]\w*\b', ' ', d['situation'])   # strip out numbers
                for d in docs if d['situation'] and len(d['situation']) > 80]
        data = data + text
        y_all = np.append(y_all, np.repeat(event, len(text)))

if False:
    #
    # Fit some classifiers: support vector machine, logistic regression, naive-bayes, neural nets
    #
    import warnings; warnings.filterwarnings("ignore")
    
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

    #
    # Confusion matrix of best model:
    # most confused about "Guidance lowered" vs "Guidance raised";
    # should also address imbalanced panel sizes
    #
    name, clf = sorted(results)[-1][1:]                         # rank and get best model
    class_names = [keydev.event[e] for e in clf.classes_]       # lookup up class names
    y_pred = clf.predict(X_all)                                 # predicted labels
    conf_mat = sklearn.metrics.confusion_matrix(y_all, y_pred)  # display confusion matrix
    print('Confusion matrix for best model :', name)
    print(conf_mat)
    print("\n".join([str((int(e), keydev.event[e])) for e in clf.classes_]))
    sns.heatmap(conf_mat,                # heatmap of confusion matrix
                annot = True,
                fmt = 'd',
                yticklabels = clf.classes_,
                xticklabels = clf.classes_)

    #
    # Show most important features for classifying each class
    #
    topics = wordcloud_features(clf.best_estimator_.coef_,  # matrix of coefficients is feature importances
                                10,                         # number of features
                                feature_names,
                                class_names,
                                plot = True)
    for label, features in topics.items():
        print('Top features for class =', label)
        pprint(features)
    
