"""
Analyze FOMC minutes
  - web-scrape from federal reserve website and store as Unstructured dataset (in mongodb)
  - pre-process text with tokenization, Loughran-McDonald stopwords, vectorizers
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
    raise ImportError("Create module:secret.py, with method:value(arg) to return its secret values")

if __name__ == "__main__":
    #
    # scrape fed website for minutes
    #
    ### show and run: read_fomc(), get_catalog, get_minutes
    
    #
    # retrieve stopwords wordlist and minutes documents
    #
    mongodb = MongoDB(**secret.value('mongodb'))

    wordlist = Unstructured(mongodb, 'wordlists')    
    stopwords = wordlist.find_values('genericlong')[0]   # GenericLong stop words from LoughranMcDonald
    stopwords = ['january','february','march','april','may', 'june', 'july','august','september',
                 'october','november','december', 'first', 'second', 'third', 'fourth', 'twelve'
                 ] + CustomTokenizer()(" ".join(stopwords))

    minutes1 = Unstructured(mongodb, 'minutes1')
    docs = {doc['date']: doc for doc in list(minutes1.mongo.find({},{'_id':0}))}
    for d in sorted(docs.keys()):
        print(d, len(docs[d]['text']))   # print doc length in chars
    #with gzip.GzipFile('/home/terence/Downloads/out/minutes1.json.gz','w') as f:
    #    f.write(json.dumps(docs).encode('utf-8'))   # to save a json copy of minutes
    dates = sorted(docs.keys())
    text = [docs[date]['text'] for date in dates]

    #
    # select tokenizers, term frequency vectorizers, and models
    #
    max_df, min_df, max_features = 0.95, 3, 10000
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
    algos = {'LDA' : (sklearn.decomposition.LatentDirichletAllocation(n_components = n_components,
                                                                      learning_method = 'online',
                                                                      learning_offset = 50.0,
                                                                      max_iter = 5, random_state = 42),
                      tf_vectorizer),
             'PLSI' : (sklearn.decomposition.NMF(n_components = n_components,
                                                 beta_loss='kullback-leibler', solver='mu', 
                                                 alpha = 0.1, l1_ratio = 0.5,
                                                 max_iter=1000, random_state = 42),
                       tf_vectorizer),
             'NMF' : (sklearn.decomposition.NMF(n_components = n_components,
                                                random_state = 42, alpha = 0.1, l1_ratio = 0.5),
                      tfidf_vectorizer),
             'LSA' : (sklearn.decomposition.TruncatedSVD(n_components=n_components),
                      tfidf_vectorizer),
    }

    #
    # fit models and accumulate results
    #
    models = dict()
    for name, (base, vectorizer) in algos.items():
        vectorized = vectorizer.fit_transform(text)
        feature_names = vectorizer.get_feature_names()
        models[name] = base.fit(vectorized)

    #
    # plot topics, and display top features [show wordcloud_features]
    #
    for name, model in models.items():
        topics = wordcloud_features(model.components_, feature_names, 20, unique=True, plot=True)
        for topic, features in enumerate(topics):
            print(name + 'Topic' + topic)
            pprint(features)
