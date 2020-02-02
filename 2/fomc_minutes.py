"""
Analyze FOMC minutes
  - web-scrape from federal reserve website and store as Unstructured dataset (in mongodb)
  - pre-process text: tokenization, vectorizers
  - compare topic modelling results: LDA, NMF, LSA

References:

https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

Di and Jegadeesh, Deciphering Fedspeak: The Informational Content of FOMC Meetings"
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2939937
"""
import dives
import dives.dbengines
import dives.unstructured
import dives.custom
import dives.util
from wordcloud import WordCloud, STOPWORDS
import sklearn.feature_extraction
import matplotlib.pyplot as plt
import numpy as np
import re
from pprint import pprint

import importlib
importlib.reload(dives)
importlib.reload(dives.dbengines)
importlib.reload(dives.unstructured)
importlib.reload(dives.custom)
importlib.reload(dives.util)

from dives.util import DataFrame, wordcloud_features
from dives.dbengines import MongoDB
from dives.unstructured import Unstructured
from dives.custom import CustomTokenizer

try:
    import secret
    verbose = secret.value('verbose')
except:
    raise ImportError("Create module:secret.py, with method:value(arg) to return its secret value")

mongodb = MongoDB(**secret.value('mongodb'))

if False:
    #
    # Scrape fed website.
    # Note: first section (administrative, non-economic  matters) of text were manually removed
    #

    if False:
        catalog = get_catalog()
        print("Sample of url's")
        pprint([(c,catalog[c]) for c in np.random.choice(list(catalog.keys()), 2)])
        
        docs = {date : get_minutes(date, url) for date,url in catalog.items()}
        dates = sorted(docs.keys())
        print("{} FOMC minutes read from {} to {}".format(len(dates), dates[0], dates[-1]))
        
        print("Sample of tail of latest doc")
        doc = re.sub('\n+','\n', re.sub('[\r\t]',' ', docs[dates[-1]]['text'])).split('\n')
        for num, line in enumerate(doc[-50:]):
            print('[%4d] %s' % (num+len(doc)-50, line[:70]))

    #
    # Retrieve stopwords wordlist and minutes documents
    #
    
    minutes = Unstructured(mongodb, 'minutes1')
    docs = {doc['date']: doc for doc in list(minutes.mongo.find({},{'_id':0}))}
    #for d in sorted(docs.keys()):
    #    print(d, len(docs[d]['text']))   # print doc length in chars
    #with gzip.GzipFile('/home/terence/Downloads/out/minutes.json.gz','w') as f:
    #    f.write(json.dumps(docs).encode('utf-8'))   # to save a json copy of minutes    
    dates = sorted(docs.keys())
    text = [docs[date]['text'] for date in dates]
    dates = sorted(docs.keys())
    print("{} docs read from {} to {}".format(len(dates), dates[0], dates[-1]))

    wordlist = Unstructured(mongodb, 'wordlists')    
    stopwords = wordlist.find_values('genericlong')[0]   # GenericLong stop words from LoughranMcDonald
    stopwords = ['january','february','march','april','may', 'june', 'july','august','september',
                 'october','november','december', 'first', 'second', 'third', 'fourth', 'twelve'
                 ] + CustomTokenizer()(" ".join(stopwords))
    
    #
    # Define pipelines and fit topic models:
    # tokenizers, term frequency vectorizers (tf, tfidf), models (LDA, LSA, NMF, PLSI)
    #
    max_df, min_df, max_features = 0.95, 3, 10000   # reasonable constraints for feature selection
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        strip_accents='unicode',
        lowercase=True,
        stop_words=stopwords,
        max_df=max_df, min_df=min_df, max_features=max_features,
        tokenizer=CustomTokenizer())  # token_pattern=r'\b[^\d\W]+\b'
    tf_vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        strip_accents='unicode',
        lowercase=True,
        stop_words=stopwords,
        max_df=max_df, min_df=min_df, max_features=max_features,
        tokenizer=CustomTokenizer())  # token_pattern=r'\b[^\d\W]+\b'

    n_components = 4
    algos = {'LSA' : (sklearn.decomposition.TruncatedSVD(n_components=n_components),
                      tfidf_vectorizer),
             'NMF' : (sklearn.decomposition.NMF(n_components = n_components,
                                                random_state = 42, alpha = 0.1, l1_ratio = 0.5),
                      tfidf_vectorizer),
             'LDA' : (sklearn.decomposition.LatentDirichletAllocation(n_components = n_components,
                                                                      learning_method = 'online',
                                                                      learning_offset = 50.0,
                                                                      max_iter = 5, random_state = 42),
                      tf_vectorizer),
             'PLSI' : (sklearn.decomposition.NMF(n_components = n_components,
                                                 beta_loss='kullback-leibler', solver='mu', 
                                                 alpha = 0.1, l1_ratio = 0.5,
                                                 max_iter=1000, random_state = 42),
                       tf_vectorizer),
    }

    models = dict()
    for name, (base, vectorizer) in algos.items():
        vectorized = vectorizer.fit_transform(text)
        feature_names = vectorizer.get_feature_names()
        models[name] = base.fit(vectorized)

        # transform to reduced dimension space, and plot over topic score over time
        topics = models[name].transform(vectorized)
        ax = plt.gca()
        ax.plot(np.array(dates).astype(str), topics)
        plt.xticks(np.array(dates).astype(str), rotation='vertical')
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % 12 != 0:
                label.set_visible(False)
        plt.legend(np.arange(n_components).astype(str))
        plt.title(name)
        plt.subplots_adjust(bottom=0.25)  # prevent clipping of tick labels
        plt.show()
        
    #
    # display top features [show wordcloud_features]
    #
    for name, model in models.items():
        topics = wordcloud_features(model.components_, 20, feature_names, unique=True, plot=True)
        for topic, features in topics.items():
            print('{} Topic ({}) top unique features:'.format(name, topic))
            pprint(features)
        plt.show()
