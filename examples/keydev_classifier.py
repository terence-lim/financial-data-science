"""Classification of events text

- Text classification: S&P Key Developments events
- Logistic regression: Generalized Linear Models, stochastic gradient descent
- nltk: tokenizer, lemmatizer, stemmer
- sklearn metrics: accuracy, precision, recall, confusion_matrix, auc, roc_curve

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

#
# TODO: only 2018-2019 presently, include earlier situations?
# TODO: test confusion poor, predict lower only half right: no regularization
#
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from wordcloud import WordCloud
import seaborn as sns
from finds.database import MongoDB
from finds.unstructured import Unstructured
from finds.structured import PSTAT
from conf import credentials, VERBOSE, paths

VERBOSE = 0
mongodb = MongoDB(**credentials['mongodb'], verbose=VERBOSE)
keydev = Unstructured(mongodb, 'KeyDev')
imgdir = os.path.join(paths['images'], 'keydev')

events_ = PSTAT._event

# Pre-process text

## 1. convert text to lower case
events = [26, 27] #[16, 83]
corpus = {}
for event in events:
    docs = keydev['events'].find({'keydeveventtypeid': {'$eq': event}},
                                 {'_id': 0})
    corpus[event] = [doc['headline'] + ' ' + doc['situation']
                     for doc in docs]

## 2. Tokenize for BOW; lowercase, min len 16, drop numbers and punctuation
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\b[^\d\W][^\d\W][^\d\W]+\b")  # word len >= 2
for event in events:
    corpus[event] = [" ".join(tokenizer.tokenize(line)).lower()
                     for line in corpus[event] if len(line) > 16]
print(corpus[event][-1])

### 2a. Zipf's Law
word_count = Series(dtype=int, name='Unique Words')
from collections import Counter
def count_words(corpus):
    words = Counter()
    for lines in corpus.values():
        for line in lines:
            words.update(line.split())
    print(words.most_common(20))
    return words
words = count_words(corpus)
word_count['Tokenized'] = len(words)

zipf =  Series(sorted(words.values(), reverse=True),
               index=np.arange(1,len(words)+1),
               name='freq')
zipf.plot(logx=True,
          logy=True,
          title='Probability Mass Function',
          c='C0',
          xlabel='log rank',
          ylabel='log freq',
          figsize=(7, 6))
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'zipf.jpg'))
plt.show()

## 3. remove stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(s for s in stopwords.words('english') if "'" not in s)\
    .union(['co', 'ltd', 'mr', 'mrs', 'inc', 'llc', 'plc'])
for event in events:
    corpus[event] = [" ".join(word for word in line.split()
                              if not word in stop_words)
                     for line in corpus[event]]
print(corpus[event][-1])
words = count_words(corpus)
word_count['Removed Stopwords'] = len(words)

## 4.Lemmatize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
for event in events:
    corpus[event] = [" ".join(lemmatizer.lemmatize(word)
                              for word in line.split())
                     for line in corpus[event]]
print(corpus[event][-1])
words = count_words(corpus)
word_count['Lemmatized'] = len(words)


## 5. Stem
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer(language='english')
for event in events:
    corpus[event] = [" ".join(stemmer.stem(word)
                              for word in line.split())
                     for line in corpus[event]]
print(corpus[event][-1])
words = count_words(corpus)
word_count['Stemmed'] = len(words)

## 6. Split shuffled into training and test labeled data sets
X_train = []
y_train = []
X_test = []
y_test = []
split_frac = 0.9
for label, (event, lines) in enumerate(corpus.items()):
    permuted = np.random.permutation(len(lines))
    n = int(split_frac * len(lines))    # split point of train and test sets
    X_train.extend([lines[permuted[i]].split(' ') for i in range(n)])
    X_test.extend([lines[permuted[i]].split(' ') for i in range(n, len(lines))])
    y_train.extend([label] * n)
    y_test.extend([label] * (len(lines) - n))
print('train/test N/frac:', [(len(y), np.mean(y)) for y in [y_train, y_test]])


# Vocab: Construct from training set, map words to indexes, one_hot vectors

word_list = Series(words).sort_values(ascending=False)
word_list = ['_PAD_', '_UNK_'] + word_list[word_list > 2].index.to_list()
word2id = {w: i for i,w in enumerate(word_list)}
id2word = {i: w for i,w in enumerate(word_list)}
W = len(word_list)
print('vocab size and top words:', W, word_list[:20])

def one_hot(word2id, line):
    """helper to form one-hot vectors from words and indexer"""
    vector = np.zeros(len(word2id) + 1) 
    vector[[word2id[word] for word in line if word in word2id]] = 1 
    vector[-1] = 1     # and set bias term too
    return vector

#######################################################
#
# Logistic Regression by SGD, with one-hot input vectors
#
########################################################

np.random.seed(0)
out = {'nll': [],  'train_acc': [], 'test_acc': []}  # store results per epoch
T = 100                     # max epochs to iterate SGD
C = 0.1                       # regularization weight
constant = False
# whether constant or step learning rate
w = np.random.randn(W + 1)  # initialize vector for weights and bias
from tqdm import trange, tqdm
with trange(T) as epochs:
    for epoch in epochs:
        permutation = np.random.permutation(len(X_train))
        out['train_acc'].append(0)
        out['nll'].append(0)    # negative log likelihood (to minimize)
        for i in permutation:
            X = one_hot(word2id, X_train[i])    # one-hot representation
            p = 1/(1 + np.exp(-w @ X))          # predicted probability
            y = y_train[i]                      # true label
            out['train_acc'][-1] += (y == (p > 0.5)) / len(X_train)
            out['nll'][-1] -= (y * (-100 if p == 0 else np.log(p))
                               + (1-y) * (-100 if p==1 else np.log(1-p)))
            learning_rate = 1 if constant else 1 / (epoch + 1)
            w = w - (((p - y) * X) + w * C / len(permutation)) * learning_rate
        out['test_acc'].append(0)
        for i in range(len(y_test)):              # compute test accuracy
            y = y_test[i]                         # gold label
            X = one_hot(word2id, X_test[i])
            p = 1/(1 + np.exp(-w @ X))            # predicted prob
            out['test_acc'][-1] += (y == (p > 0.5)) / len(X_test)
        epochs.set_postfix(nll=round(out['nll'][-1], 0),
                           train=round(out['train_acc'][-1], 2),
                           test=round(out['test_acc'][-1], 2))
DataFrame(out).round(3)
"""
  0: 99    500.587      0.953     0.735
0.0001: 99    495.312      0.955     0.739
0.01: 99    498.714      0.957     0.728
1.0: 99    497.304      0.954     0.728
0.1: 99    500.385      0.954     0.742
"""

# Plot train and test accuracy
fig, ax = plt.subplots(num=1, clear=True, figsize=(5, 4))
ax.plot(np.arange(T).astype(str), out['nll'], color='C2')
ax.legend(['NLL'], loc='upper left')
ax.set_ylabel('Negative Log Likelihood')
bx = ax.twinx()
bx.plot(np.arange(T).astype(str), out['train_acc'], color='C0')
bx.plot(np.arange(T).astype(str), out['test_acc'], color='C1')
bx.set_xlabel('epoch')
bx.set_ylabel('accuracy')
bx.legend(['Train accuracy', 'Test accuracy'], loc='center right')
plt.xticks(np.arange(0, T, T // 10))
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'accuracy' + figext))
plt.show()

# Compute predicted probs in test set
y, yproba = [], []
for i in range(len(X_test)):
    vector = one_hot(word2id, X_test[i])
    yproba.append(1/(1 + np.exp(-w @ vector)))   # predicted proba
    y.append(y_test[i])


# Plot AUC and ROC curve in test set
import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y, yproba)
roc_auc = metrics.auc(fpr, tpr)
fig, ax = plt.subplots(num=2, clear=True, figsize=(5, 4))
ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
ax.legend(loc = 'lower right')
ax.plot([0, 1], [0, 1],'r--')
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
ax.set_title('Receiver Operating Characteristic')
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'roc' + figext))
plt.show()

# Evaluate confusion matrix: precision, recall, F1 in test set
n = len(y)
ypred = np.array(yproba) > 0.5
tn, fp, fn, tp = metrics.confusion_matrix(y, ypred).ravel()
accuracy = (tp + tn) / n     # predicted matches actual
precision = tp / (tp + fp)   # denom is number of predicted positives
recall = tp / (tp + fn)      # denom is number of positives in gold labels
f1 = 2 * precision * recall / (precision + recall)  # harmonic mean

assert(np.allclose([accuracy,
                    precision,
                    recall],   # sanity check
                   [metrics.accuracy_score(y, ypred),
                    metrics.precision_score(y, ypred, average='binary'),
                    metrics.recall_score(y, ypred, average='binary')]))

# Display accuracy in test set
output = {}
output.update({"Precision":
               {'Description': "(predicted positives correct)",
                'Calculate': f"{tp} / {tp + fp} =",
                'Value': round(precision, 4)}})
output.update({"Recall | Sensitivity | True Positive Rate":
               {'Description': "(actual positives predicted correct)",
                'Calculate': f"{tp} / {tp + fn} =",
                'Value': round(recall, 4)}})
output.update({"F1":
               {'Description': "(harmonic mean of precision and recall)",
                'Calculate': '',
                'Value': round(f1, 4)}})
output.update({"Specificity | True Negative Rate":
               {'Description': "(actual negatives predicted correct)",
                'Calculate': f"{tn} / {tn + fp} = ",
                'Value': round(tn / (tn + fp), 4)}})
output.update({"Type I error | alpha | Significance":
               {'Description': "(actual positives incorrectly predicted)",
                'Calculate': f"{fp} / {fp + tp} = ",
                'Value': round(fp / (fp + tp), 4)}})
output.update({"Type II error | beta | (1 - Power)":
               {'Description': "(actual negatives incorrectly predicted)",
                'Calculate': f"{fn} / {fn + tn} = ",
                'Value': round(fn / (fn + tn), 4)}})
print(f"=== Negative/Label=0 is {events_[events[0]]} ===")
print(f"=== Positive/Label=1 is {events_[events[1]]} ===")
pd.set_option('max_colwidth', 50)
show(DataFrame.from_dict(output, orient='index'))

# Display confusion matrix of test set
labels = [f"{i}={events_[events[i]].split()[-1]}" for i in range(len(events))]
cf = DataFrame(metrics.confusion_matrix(y, ypred),
               index=pd.MultiIndex.from_product([['Predicted'], labels]),
               columns=pd.MultiIndex.from_product([['Actual'], labels]))


# Plot confusion of test set
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(num=1, clear=True, figsize=(7, 5))
sns.heatmap(cf,
            ax=ax,
            annot=True,
            fmt='d',
            cmap='inferno',
            robust=True,
            yticklabels=events_[events],
            xticklabels=events_[events])
ax.set_title('Confusion Matrix')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.yaxis.set_tick_params(labelsize=8, rotation=0)
ax.xaxis.set_tick_params(labelsize=8, rotation=0)
plt.subplots_adjust(left=0.35, bottom=0.25)
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'confusion' + figext))
plt.show()

# Display top words in wordcloud
wc = WordCloud(width=500, height=500, stopwords=stop_words, colormap='cool')
top_k = 15
arg_sort = np.argsort(w)
fig, axes = plt.subplots(1, 2, clear=True, num=1, figsize=(7, 5))
for ax, words, e in zip(axes, [arg_sort[:top_k], arg_sort[-top_k:]], events):
    words = {id2word[word]: abs(w[word]) for word in words}
    ax.imshow(wc.generate_from_frequencies(words))
    ax.axis("off")
    plt.tight_layout(pad=3)
    ax.set_title(events_[e], fontdict=dict(fontsize=10))
plt.savefig(os.path.join(imgdir, 'words" + figext))
plt.show()
