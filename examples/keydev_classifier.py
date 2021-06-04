"""Classification of events text

- text classification, logistic regression, stochastic gradient descent
- confusion matrix, precision, recall, ROC curve
- S&P Key Developments, Wharton Research Data Services


Terence Lim
License: MIT
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from wordcloud import WordCloud, STOPWORDS
from finds.database import MongoDB
from finds.unstructured import Unstructured
from finds.structured import PSTAT
from settings import settings
mongodb = MongoDB(**settings['mongodb'])
keydev = Unstructured(mongodb, 'KeyDev')
imgdir = os.path.join(settings['images'], 'keydev')
event_ = PSTAT.event_

# Construct corpus: to lower case, strip numeric
corpus = {}
events = [26, 27] #[16, 83]
for event in events:
    docs = keydev['events'].find(
        {'keydeveventtypeid': {'$eq': event}}, {'_id': 0})
    corpus[event] = [re.sub(r'\b\w*[\d]\w*\b', ' ', " ".join(
            d[k] for k in ['headline', 'situation'])).lower() for d in docs]
DataFrame({'description': [event_[event] for event in corpus.keys()],
           'count': [len(lines) for lines in corpus.values()]},
          index=corpus.keys())

# Tokenize, and remove stopwords
stop_words = STOPWORDS.union(['co', 'ltd', 'mr', 'mrs', 'inc', 'llc'])
for event, lines in corpus.items():
    corpus[event] = [[w for w in re.findall(r"\w\w+", line)
                      if w not in stop_words] for line in lines]
    
# Split shuffled into labelled training and test sets
train_data = []
test_data = []
split_frac = 0.9
for label, (event, lines) in enumerate(corpus.items()):
    np.random.shuffle(lines)
    n = int(split_frac * len(lines))   # split point of train and test sets
    train_data.extend([(label, corpus[event][p]) for p in range(n)])
    test_data.extend([(label, corpus[event][p]) for p in range(n, len(lines))])
N = len(train_data)
print('train/test:', N, [np.mean([label for label,_ in subset])
                         for subset in [train_data, test_data]])

# Construct vocab from training set, and map words to indexes
from collections import Counter
words = Counter()
for label, line in train_data:
    words.update(set(line))
words = Series(words).sort_values(ascending=False)
words = ['_PAD_', '_UNK_'] + words[words > 2].index.to_list()
word2idx = {w: i for i,w in enumerate(words)}
idx2word = {i: w for i,w in enumerate(words)}
W = len(words)
print('vocab:', W, words[:20])

def one_hot(word2idx, words):
    """helper to form one-hot vectors from words and indexer"""
    f = np.zeros(len(word2idx) + 1)  # initialize weights vector including bias
    f[[word2idx[w] for w in words if w in word2idx]] = 1 # set each word index
    f[-1] = 1                        # and set bias term too
    return f
    
# Logistic Regression by SGD, with one-hot input vectors

# p = 1/(1 + np.exp(-w @ f)) = logistic(-w @ f) = inverse-logit
# logit = log(p / (1-p)) = logodds = "canonical link function for bernoulli"
# sigmoid = any S-curve: e.g. logistic(), tanh(), arctan(), erf()
# MLE cannot be solved analytically, so numerically with gradient SGD

np.random.seed(0)
out = {'nll': [], 'acc': [], 'test': []} # track accuracy and log likelihood 
T = 15                      # max epochs
constant = False            # whether constant or step learning rate
w = np.random.randn(W+1)    # initialize vector for weights and bias
for epoch in range(T):
    permutation = np.random.permutation(len(train_data))
    out['acc'].append(0)
    out['nll'].append(0)    # negative log likelihood (to minimize)
    for i in permutation:
        f = one_hot(word2idx, train_data[i][1])   # one-hot representation
        p = 1/(1 + np.exp(-w @ f))                # predicted probability
        y = train_data[i][0]                      # true label
        out['acc'][-1] += (y == (p > 0.5)) / len(train_data)
        out['nll'][-1] -= (y * (-100 if p==0 else np.log(p)) +
                           (1-y) * (-100 if p==1 else np.log(1-p)))
        w = w + ((y - p) * f / (1 if constant else epoch + 1))
    out['test'].append(0)
    for i in range(len(test_data)):
        y = test_data[i][0]         # gold label
        f = one_hot(word2idx, test_data[i][1])
        p = 1/(1 + np.exp(-w @ f))  # predicted prob: let label be p>0.5
        out['test'][-1] += (y == (p > 0.5)) / len(test_data) # test set accuracy
DataFrame(out).round(3)

# Plot train and test accuracy
fig, ax = plt.subplots(num=1, clear=True, figsize=(5,4))
ax.plot(np.arange(1, T+1).astype(str), out['acc'], color='C0')
ax.plot(np.arange(1, T+1).astype(str), out['test'], color='C1')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
ax.legend(['Train accuracy', 'Test accuracy'])
bx = ax.twinx()
bx.plot(np.arange(1, T+1).astype(str), out['nll'], color='C2')
bx.legend(['Negative Log Likelihood'])
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'accuracy.jpg'))
    
# Compute predicted probs in test set
y, yproba = [], []
for i in range(len(test_data)):
    f = one_hot(word2idx, test_data[i][1])
    yproba.append(1/(1 + np.exp(-w @ f)))   # predicted proba
    y.append(test_data[i][0])

# Plot AUC and ROC curve in test set
import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y, yproba)
roc_auc = metrics.auc(fpr, tpr)
fig, ax = plt.subplots(num=2, clear=True, figsize=(5,4))
ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
ax.legend(loc = 'lower right')
ax.plot([0, 1], [0, 1],'r--')
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
ax.set_title('Receiver Operating Characteristic')
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'roc.jpg'))
plt.show()

# Evaluate confusion matrix: precision, recall, F1 in test set
n = len(y)
ypred = np.array(yproba) > 0.5
tn, fp, fn, tp = metrics.confusion_matrix(y, ypred).ravel()
accuracy = (tp + tn) / n     # predicted matches actual
precision = tp / (tp + fp)   # denom is number of predicted positives
recall = tp / (tp + fn)      # denom is number of positives in gold labels
f1 = 2 * precision * recall / (precision + recall)  # harmonic mean
assert(np.allclose([accuracy, precision, recall],   # sanity check
                   [metrics.accuracy_score(y, ypred),
                    metrics.precision_score(y, ypred, average='binary'),
                    metrics.recall_score(y, ypred, average='binary')]))

# Display accuracy in test set
output = {}
output.update({"Precision":
               {'Description': "(predicted positives correct)",
                'Calculate': f"{tp} / {tp + fp} =",
                'Value': np.round(precision, 4)}})
output.update({"Recall | Sensitivity | True Positive Rate":
               {'Description': "(actual positives predicted correct)",
                'Calculate': f"{tp} / {tp + fn} =",
                'Value': np.round(recall, 4)}})
output.update({"F1":
               {'Description': "(harmonic mean of precision and recall)",
                'Calculate': '',
                'Value': np.round(f1, 4)}})
output.update({"Specificity | True Negative Rate":
               {'Description': "(actual negatives predicted correct)",
                'Calculate': f"{tn} / {tn + fp} = ",
                'Value': np.round(tn / (tn + fp), 4)}})
output.update({"Type I error | alpha | Significance":
               {'Description': "(actual positives incorrectly predicted)",
                'Calculate': f"{fp} / {fp + tp} = ",
                'Value': np.round(fp / (fp + tp), 4)}})
output.update({"Type II error | beta | (1 - Power)":
               {'Description': "(actual negatives incorrectly predicted)",
                'Calculate': f"{fn} / {fn + tn} = ",
                'Value': np.round(fn / (fn + tn), 4)}})
print(f"=== Negative/Label=0 is {event_[events[0]]} ===")
print(f"=== Positive/Label=1 is {event_[events[1]]} ===")
pd.set_option('max_colwidth', 80)
print(DataFrame.from_dict(output, orient='index').to_latex()

# Display confusion matrix of test set
labels = [f"{i}={event_[events[i]].split()[-1]}" for i in range(len(events))]
cf = DataFrame(metrics.confusion_matrix(y, ypred),
               index=pd.MultiIndex.from_product([['Predicted'], labels]),
               columns=pd.MultiIndex.from_product([['Actual'], labels]))

# Plot confusion of test set
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(num=1, clear=True, figsize=(10,6))
sns.heatmap(cf, ax=ax, annot= True, fmt='d', cmap='inferno', robust=True,
            yticklabels=event_[events], xticklabels=event_[events])
ax.set_title('Confusion Matrix')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.yaxis.set_tick_params(labelsize=8, rotation=0)
ax.xaxis.set_tick_params(labelsize=8, rotation=20)
plt.subplots_adjust(left=0.35, bottom=0.25)
plt.savefig(os.path.join(imgdir, "confusion.jpg"))
plt.show()
    
# Display top words in wordcloud
wc = WordCloud(width=500, height=500, stopwords=stop_words, colormap='cool')
top_k = 20
arg_sort = np.argsort(w)
fig, axes = plt.subplots(1, 2, clear=True, num=1, figsize=(10,6))
for ax, args, e in zip(axes, [arg_sort[:top_k], arg_sort[-top_k:]], events):
    words = {idx2word[a]: abs(w[a]) for a in args}   # top words and weights
    ax.imshow(wc.generate_from_frequencies(words))
    ax.axis("off")
    plt.tight_layout(pad=3)
    ax.set_title(event_[e])
plt.savefig(os.path.join(imgdir, "words.jpg"))
plt.show()
