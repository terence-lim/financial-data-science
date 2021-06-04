"""Comparing supervised learning models for classification of event text

- sklearn, naivebayes, logistic, linearsvc, mlp, decisiontree, wordcloud
- text classification, S&P Key Developments, Wharton Research Data Services

Terence Lim
License: MIT
"""
import numpy as np
import os
import time
import re
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from finds.database import MongoDB
from finds.unstructured import Unstructured
from finds.structured import PSTAT
from settings import settings
mongodb = MongoDB(**settings['mongodb'])
keydev = Unstructured(mongodb, 'KeyDev')
imgdir = os.path.join(settings['images'], 'classify')
event_ = PSTAT.event_
role_ = PSTAT.role_

## Show event sample counts
counts = {ev: keydev['events'].count_documents({'keydeveventtypeid': ev})
          for ev in keydev['events'].distinct('keydeveventtypeid')}

## Retrieve headline+situation text
# - in lower case, exclude numeric
events = [28, 16, 83, 41, 81, 23, 87, 45, 80, 97,  231, 46, 31, 77, 29,
          232, 101, 42, 47, 86, 93, 3, 22, 102, 82]
corpus = {}    #[16, 83] # #[65, 80]: #[101, 192, 65, 80, 27, 86]
for event in events:
    docs = keydev['events'].find({'keydeveventtypeid':{'$eq':event}}, {'_id':0})
    corpus[event] = [re.sub(r'\b\w*[\d]\w*\b', ' ', " ".join(
        d[k] for k in ['headline', 'situation'])).lower() for d in docs]
DataFrame({e: {'event': event_[e], 'count': counts[e]} for e in corpus.keys()})

## Collect all text into data, and labels into y_all
lines = []
y_all = []
for label, event in enumerate(events):
    lines.extend(corpus[event])
    y_all.extend([event] * len(corpus[event]))
print(lines[0])

# Pre-processing and Feature Extraction
## Tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\b[^\d\W][^\d\W][^\d\W]+\b")  # word len >= 2
lines = [" ".join(tokenizer.tokenize(line)) for line in lines]
print(lines[0])

words = Counter()
for line in lines:
    words.update(line.split())
print(words.most_common(20))
DataFrame({'Unique Words': len(words)}, index=['Tokenized'])

## Zipf's Law
zipf =  Series(sorted(words.values(), reverse=True),
               index=np.arange(1,len(words)+1), name='freq')
zipf.plot(logx=True, logy=True, title='Probability Mass Function', c='C0',
          xlabel='log rank', ylabel='log freq', figsize=(7,6))
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'zipfpre.jpg'))
plt.show()

## Remove stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(s for s in stopwords.words('english') if "'" not in s)\
             .union(['yen', 'jpy', 'eur', 'dkk', 'cny', 'sfr'])
lines = [" ".join(word for word in line.split() if not word in stop_words)
         for line in lines]
print(lines[0])

words = Counter()
for line in lines:
    words.update(line.split())
print(words.most_common(10))
DataFrame({'Unique Words': len(words)}, index=['Removed Stopwords'])
        
## Lemmatize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lines = [" ".join(lemmatizer.lemmatize(word) for word in line.split())
         for line in lines]
print(lines[0])

words = Counter()
for line in lines:
    words.update(line.split())
print(words.most_common(10))
DataFrame({'Unique Words': len(words)}, index=['Lemmatized'])

## Stem
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer(language='english')
lines = [" ".join(stemmer.stem(word) for word in line.split())
         for line in lines]
print(lines[0])

words = Counter()
for line in lines:
    words.update(line.split())
print(words.most_common(10))
DataFrame({'Unique Words': len(words)}, index=['Stemmed'])
"""
[('million', 2180937), ('year', 2098973), ('ago', 1338540), ('compani', 1324416), ('announc', 1240636), ('share', 1025525), ('compar', 935672), ('earn', 765587), ('per', 729273), ('end', 654428), ('limit', 639785), ('inc', 627465), ('ltd', 622252), ('incom', 617173), ('result', 608802), ('quarter', 606674), ('oper', 565527), ('net', 560028), ('inr', 556304), ('report', 520637)]
"""
## Zip's Law
zipf =  Series(sorted(words.values(), reverse=True),
               index=np.arange(1,len(words)+1), name='freq')
zipf.plot(logx=True, logy=True, title='After Pre-Processing', c='C1',
          xlabel='log rank', ylabel='log freq', figsize=(7,6))
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'zipfpost.jpg'))
plt.show()
            
## Split Training and Test samples,
# - shuffled and stratified
#events = [28, 16, 83, 41, 81, 23, 87, 45, 80]  # Extract subset of data
#y_all, lines = Load('lines.multi')
data = [line for c, line in zip(y_all, lines) if c in events]
label = [c for c in y_all if c in events]
from sklearn.model_selection import train_test_split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=test_size, random_state=42, stratify=label)
pd.concat([Series(y_train, name='train').value_counts()/len(y_train),
           Series(y_test, name='test').value_counts()/len(y_test)], axis=1)\
           .round(3).T

## Vectorize it
# - fit on train, transform on both train and test
from sklearn.feature_extraction import text
max_df, min_df, max_features = 0.5, 200, 10000
tfidf_vectorizer = text.TfidfVectorizer(
    encoding='latin-1',
    strip_accents='unicode',
    lowercase=True,
    #stop_words=stop_words,
    max_df=max_df,
    min_df=min_df,
    max_features=max_features,
    token_pattern=r"\b[^\d\W][^\d\W][^\d\W]+\b", #r'\b[^\d\W]+\b'
)
x_train = tfidf_vectorizer.fit_transform(X_train)   # sparse array
x_test = tfidf_vectorizer.transform(X_test)
feature_names = tfidf_vectorizer.get_feature_names()
print(len(feature_names), 'train:', x_train.shape, 'test:', x_test.shape)

# Train Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
verbose=False
models={
    'naivebayes': MultinomialNB(),
    'logistic': LogisticRegression(class_weight='balanced', verbose=False,
                                   multi_class='multinomial', max_iter=500), 
    'linearsvc': LinearSVC(class_weight='balanced',
                           multi_class='ovr', verbose=verbose),
    'mlp': MLPClassifier(tol=0.001, verbose=verbose),  # No gpu in sklearn!
    'decisiontree': DecisionTreeClassifier(class_weight='balanced',
                                           random_state=0),
}
results = {}
for name, clf in models.items():
    #print('***', name, '***')
    tic = time.time()
    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train) # evaluate train set accuracy
    test_score = clf.score(x_test, y_test)    # and test set accuracy
    results.update({name: {'train_score': train_score,
                           'test_score': test_score, 
                           'elapsed': time.time() - tic}})
    #print(name, time.time() - tic, train_score, test_score)
r = DataFrame.from_dict(results, orient='index')
r.round()

## Show decision tree fitted parameters
dt = models['decisiontree']
print('Tree depth:', dt.get_depth())
print('Tree n_leaves:', dt.get_n_leaves())
x = dt.decision_path(x_test[-1])
print('Length of a test document:', x_test[-1].nnz)
print('Length of its decision path:', x.nnz)
"""
Tree depth: 348
Tree n_leaves: 52586
Length of a test document: 24
Length of its decision path: 206
"""
## Show MLP parameters
dt = models['decisiontree']
x = dt.decision_path(x_test[-1])
DataFrame({'Tree depth': dt.get_depth(), 'Tree n_leaves': dt.get_n_leaves(),
           'Length of a test document': x_test[-1].nnz, 
           'Length of its decision path': x.nnz}, index=['DecisionTree'])
"""
activation 	relu
alpha 	0.0001
beta_1 	0.9
beta_2 	0.999
hidden_layer_sizes 	100
momentum 	0.9
solver 	adam
"""
# Plot Train and Test Accuracy
from finds.display import plot_bar
y = r[['train_score', 'test_score']]
fig, ax = plt.subplots(num=1, clear=True, figsize=(10,6))
plot_bar(y, ax=ax, labels=y.round(3).astype(str).values, fontsize=12,
         ylabel='Accuracy',
         title='Classification Models Accuracy', xlabel='Model')
plt.savefig(os.path.join(imgdir, 'mse.jpg'))
plt.show()

# Plot precision, recall, f1
from sklearn import metrics
scores = {}
for ifig, (name, clf) in enumerate(models.items()):
    train = metrics.precision_recall_fscore_support(
        y_train, clf.predict(x_train), average='macro')[:3]
    test = metrics.precision_recall_fscore_support(
        y_test, clf.predict(x_test), average='macro')[:3]
    scores[name] = np.append(train, test)
y = DataFrame(scores, index=[t + '_' + s for t in ['train', 'test']
                             for s in ['precision','recall','f1']]).T
fig, ax = plt.subplots(num=1, clear=True, figsize=(10,6))
plot_bar(y, ax=ax, labels=y.round(3).astype(str).values, fontsize=12,
         ylabel='Accuracy', title='Classification Models Precision and Recall',
         xlabel='Model', rotation=45, loc='lower left')
plt.savefig(os.path.join(imgdir, 'accuracy.jpg'))
plt.show()

## Plot Feature Importances
top_n = 40
for ifig, name in enumerate(['decisiontree']):
    clf = models[name]
    imp = clf.feature_importances_.flatten()
    words = Series({feature_names[i].replace(' ','_'): imp[i]
                    for i in np.argsort(abs(imp))})
    fig, ax = plt.subplots(clear=True, num=1, figsize=(10,12))
    words.iloc[-40:].plot(kind='barh', color='C0', ax=ax)
    ax.set_xlabel('Feature Importance')
    ax.set_title(name.capitalize())
    ax.yaxis.set_tick_params(labelsize=12)
    plt.savefig(os.path.join(imgdir, f"{name}.jpg"))
plt.show()
        
    
# Plot feature coefficient values by class (linear only)
from wordcloud import WordCloud
wc = WordCloud(height=500, width=500, colormap='cool')
top_n = 20
for ifig, name in enumerate(['logistic', 'naivebayes', 'linearsvc']):
    clf = models[name]
    fig, axes = plt.subplots(5, 5, figsize=(10, 12), num=1+ifig, clear=True)
    axes = [ax for axs in axes for ax in axs]
    for topic, ax in enumerate(axes[:len(clf.classes_)]):
        print("topic %d %s:" % (topic, event_[clf.classes_[topic]]))
        importance = clf.coef_[topic, :]
        words = {feature_names[i].replace(' ','_'): importance[i]
                 for i in importance.argsort()[:-top_n - 1:-1]}
        # print(words.keys())
        ax.imshow(wc.generate_from_frequencies(words))
        ax.axis("off")
        plt.tight_layout(pad=2)
        ax.set_title(f"{event_[clf.classes_[topic]]}", fontsize='xx-small')
    plt.savefig(os.path.join(imgdir, f"{name}.jpg"))
    plt.show()
    
