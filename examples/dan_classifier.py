"""Deep Averaging Networks for text classification

- pytorch, spacy, deep averaging networks, word embeddings
- S&P Key Developments, Wharton Research Data Services

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
from finds.database import MongoDB
from finds.unstructured import Unstructured
from finds.structured import PSTAT
from settings import settings
from settings import pickle_dump, pickle_load
mongodb = MongoDB(**settings['mongodb'])
keydev = Unstructured(mongodb, 'KeyDev')
imgdir = os.path.join(settings['images'], 'classify')
event_ = PSTAT.event_
role_ = PSTAT.role_

## Retrieve headline+situation text
events = [28, 16, 83, 41, 81, 23, 87, 45, 80, 97,  231, 46, 31, 77, 29,
          232, 101, 42, 47, 86, 93, 3, 22, 102, 82]
corpus = {}    #[16, 83] # #[65, 80]: #[101, 192, 65, 80, 27, 86]

lines = []
event_all = []
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\b[^\d\W][^\d\W][^\d\W]+\b")
for event in events:
    docs = keydev['events'].find({'keydeveventtypeid':{'$eq':event}}, {'_id':0})
    doc = [tokenizer.tokenize((d['headline'] + " " + d['situation']).lower())
           for d in docs]
    lines.extend(doc)
    event_all.extend([event] * len(doc))
print(lines[1000000])
Series(event_all).value_counts().rename(index=event_).rename('count').to_frame()

## Save as gzipped csv
import csv, gzip
with gzip.open(os.path.join(imgdir, 'lines.csv.gz'), 'wt', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(lines)
with gzip.open(os.path.join(imgdir,'event_all.csv.gz'), 'wt',newline='\n') as f:
    f.write("\n".join(str(e) for e in event_all))

## Encode class labels
from sklearn.preprocessing import LabelEncoder
event_encoder = LabelEncoder().fit(event_all)    # .inverse_transform()
num_classes = len(np.unique(event_all))
y_all = event_encoder.transform(event_all)

## Split into stratified train and test indices
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(np.arange(len(y_all)), random_state=42,
                                       stratify=y_all, test_size=0.2)

## Load spacy vocab
import spacy
lang = 'en_core_web_lg'
nlp = spacy.load(lang, disable=['parser', 'tagger', 'ner', 'lemmatizer'])
for w in ['yen', 'jpy', 'eur', 'dkk', 'cny', 'sfr']:
    nlp.vocab[w].is_stop = True    # Mark customized stop words
    
n_vocab, vocab_dim = nlp.vocab.vectors.shape
print('Language:', lang, '   vocab:', n_vocab, '   dim:', vocab_dim)

## Pytorch Feed Forward Network
import torch
import torch.nn as nn
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytorch Deep Averaging Feed Forward Network
class FFNN(nn.Module):
    """Deep Averaging Network for classification"""
    def __init__(self, vocab_dim, num_classes, hidden, dropout=0.3):
        super().__init__()
        V = nn.Linear(vocab_dim, hidden[0])
        nn.init.xavier_uniform_(V.weight)
        L = [V, nn.Dropout(dropout)]
        for g, h in zip(hidden, hidden[1:] + [num_classes]):
            W = nn.Linear(g, h)
            nn.init.xavier_uniform_(W.weight)
            L.extend([nn.ReLU(), W])
        self.network = nn.Sequential(*L)
        self.classifier = nn.LogSoftmax(dim=-1)  # output is (N, C) logits

    def forward(self, x):
        """Return tensor of log probabilities"""
        return self.classifier(self.network((x)))

    def predict(self, x):
        """Return predicted int class of input tensor vector"""
        return int(torch.argmax(self(x[None, :])[0]))
            
    def save(self, filename):
        """save model state to filename"""
        return torch.save(self.state_dict(), filename)

    def load(self, filename):
        """load model name from filename"""
        self.load_state_dict(torch.load(filename, map_location='cpu'))
        return self

## Precompute word embeddings input
def form_input(line, nlp):
    """Return spacy average vector from valid words"""
    tokens = [tok.vector for tok in nlp(" ".join(line))
              if not(tok.is_stop or tok.is_punct or tok.is_oov or tok.is_space)]
    return np.array(tokens).mean(axis=0)

if False:
    X = np.zeros((len(lines), vocab_dim))
    for i, line in tqdm(enumerate(lines)):
        X[i] = form_input(line, nlp)
    pickle_dump(X, 'X.pkl', outdir=imgdir)
else:
    X = pickle_load('X.pkl', outdir=imgdir)

## Training Loops
# - Instantiate model, optimizer, scheduler, loss_function for layers
# - Loops over epochs and batches
accuracy = {}  # to store computed metrics
max_layers, hidden = 3, 300
batch_size, lr, num_lr, step_size, eval_skip = 64, 0.01, 5, 20, 1
num_epochs = step_size * num_lr
for layers in range(1, max_layers + 1):
    accuracy[layers] = {}
    model = FFNN(vocab_dim, num_classes, hidden=[hidden]*layers,
                 word_vec=word_vec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1,
                                                step_size=step_size)
    loss_function = nn.NLLLoss()
    for epoch in range(0, num_epochs):
        tic = time.time()
        idxs = [i for i in train_idx]
        random.shuffle(idxs)
        batches = [idxs[i:(i+batch_size)]
                   for i in range(0, len(idxs), batch_size)]
    
        # train by batch
        total_loss = 0.0
        model.train()
        for batch in batches:
            x = torch.FloatTensor(X[batch]).to(device)
            y = torch.tensor([y_all[idx] for idx in batch]).to(device)
            model.zero_grad()                    # reset model gradient
            log_probs = model(x)                 # run model
            loss = loss_function(log_probs, y)   # compute loss
            total_loss += float(loss)
            loss.backward()      # loss step
            optimizer.step()     # optimizer step
        scheduler.step()      # scheduler step
        model.eval()
        model.save(os.path.join(imgdir, f"dan{layers}.pt"))

        with torch.no_grad():
            print("Total loss on epoch %i: %f" % (epoch, total_loss))
            if epoch % eval_skip == 0:
                test_pred = [model.predict(torch.FloatTensor(X[i]).to(device))
                             for i in test_idx]
                test_gold = [int(y_all[i]) for i in test_idx]
                test_correct = (np.asarray(test_pred) ==
                                np.asarray(test_gold)).sum()
                train_pred = [model.predict(torch.FloatTensor(X[i]).to(device))
                              for i in train_idx]
                train_gold = [int(y_all[i]) for i in train_idx]
                train_correct = (np.asarray(train_pred) ==
                                 np.asarray(train_gold)).sum()
                accuracy[layers][epoch] {'loss': total_loss,
                                         'train': train_correct/len(train_gold),
                                         'test': test_correct/len(test_gold)}
                print(layers, epoch, int(time.time()-tic),
                      optimizer.param_groups[0]['lr'],
                      train_correct/len(train_idx), test_correct/len(test_idx))

    from sklearn import metrics
    print(model)   # show accuracy metrics for this layer
    print(pd.concat([
        Series({'Accuracy':  metrics.accuracy_score(test_gold, test_pred),
                'Precision': metrics.precision_score(test_gold, test_pred,
                                                     average='weighted'),
                'Recall': metrics.recall_score(test_gold, test_pred,
                                               average='weighted')},
               name='Test Set').to_frame().T,      
        Series({'Accuracy': metrics.accuracy_score(train_gold, train_pred),
                'Precision': metrics.precision_score(train_gold, train_pred,
                                                     average='weighted'),
                'Recall': metrics.recall_score(train_gold, train_pred,
                                               average='weighted')},
               name='Train Set').to_frame().T], axis=0).to_string())

## Confusion Matrix
from sklearn.metrics import confusion_matrix
labels = [event_[e] for e in event_encoder.classes_]
cf_train = DataFrame(confusion_matrix(train_gold, train_pred),
              index=pd.MultiIndex.from_product([['Actual'], labels]),
              columns=pd.MultiIndex.from_product([['Predicted'], labels]))
cf_test = DataFrame(confusion_matrix(test_gold, test_pred),
                    index=pd.MultiIndex.from_product([['Actual'], labels]),
                    columns=pd.MultiIndex.from_product([['Predicted'], labels]))
import seaborn as sns
for num, (title, cf) in enumerate({'Training':cf_train,'Test':cf_test}.items()):
    fig, ax = plt.subplots(num=1+num, clear=True, figsize=(10,6))
    sns.heatmap(cf, ax=ax, annot= False, fmt='d', cmap='viridis', robust=True,
                yticklabels=[f"{lab}  {e}"
                             for lab, e in zip(labels, event_encoder.classes_)],
                xticklabels=event_encoder.classes_)
    ax.set_title(f'{title} Set Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.yaxis.set_tick_params(labelsize=8, rotation=0)
    ax.xaxis.set_tick_params(labelsize=8, rotation=0)
    plt.subplots_adjust(left=0.35, bottom=0.25)
    plt.savefig(os.path.join(imgdir, f"cf_{title}.jpg"))
    plt.tight_layout()
plt.show()


## Construct vocab, and convert str to word index for GloVe DAN
from finds.learning import TextualData  # helper methods for text mining
textdata = TextData()
vocab = textdata.counter(lines)         # count words for vocab
textdata(vocab.most_common(20000))      # keep most common 20000
x_all = textdata[lines]                 # convert str to word indexes

## Save pre-processed data as gzipped csv
import csv, gzip
with gzip.open(os.path.join(imgdir, 'x_all.csv.gz'), 'wt', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(x_all)
textdata.dump('textdata.json', imgdir)  # save word index

## Get relativized glove embeddings
"""
wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip
Archive:  glove.6B.zip
  inflating: glove.6B.100d.txt       
  inflating: glove.6B.200d.txt       
  inflating: glove.6B.300d.txt       
  inflating: glove.6B.50d.txt    
"""
vocab_dim = 300
glovefile = f"/home/terence/Downloads/sshfs/glove/glove.6B.{vocab_dim}d.txt"
glove = textdata.relativize(glovefile)
pickle_dump(glove, f"glove{vocab_dim}rel.pkl", imgdir)

## train_test split stratified by y_all
textdata.form_splits(y_all, random_state=42, test_size=0.2)

## define DAN with tunable word embeddings
class DAN(nn.Module):
    """Deep Averaging Network for classification"""
    def __init__(self, vocab_dim, num_classes, hidden, embedding, dropout=0.3):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(embedding)
        self.tune(False)
        V = nn.Linear(vocab_dim, hidden[0])
        nn.init.xavier_uniform_(V.weight)
        L = [V, nn.Dropout(dropout)]
        for g, h in zip(hidden, hidden[1:] + [num_classes]):
            W = nn.Linear(g, h)
            nn.init.xavier_uniform_(W.weight)
            L.extend([nn.ReLU(), W])
        self.network = nn.Sequential(*L)
        self.classifier = nn.LogSoftmax(dim=-1)  # output is (N, C) logits

    def tune(self, requires_grad=False):
        self.embedding.weight.requires_grad = requires_grad

    def forward(self, x):
        """Return tensor of log probabilities"""
        return self.classifier(self.network(self.embedding(x)))

    def predict(self, x):
        """Return predicted int class of input tensor vector"""
        return int(torch.argmax(self(x[None, :])[0]))

layers = 2
hidden = vocab_dim   #100, 300
model = DAN(vocab_dim, num_classes, hidden=[hidden]*layers,
            embedding=torch.FloatTensor(glove)).to(device)

accuracy = dict()
for tune in [False, True]:
    accuracy[tune] = dict()
    model.tune(tune)
    batch_size, lr, num_lr, step_size, eval_skip = 64, 0.01, 5, 20, 5
    num_epochs = step_size * num_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1,
                                                step_size=step_size)
    loss_function = nn.NLLLoss()
    for epoch in range(0, num_epochs):
        tic = time.time()
        batches = textdata.form_batches(batch_size)

        # train by batch
        total_loss = 0.0
        model.train()
        for batch in tqdm(batches):
            x = textdata.form_input([x_all[idx] for idx in batch]).to(device)
            y = torch.tensor([y_all[idx] for idx in batch]).to(device)
            model.zero_grad()                    # reset model gradient
            log_probs = model(x)                 # run model
            loss = loss_function(log_probs, y)   # compute loss
            total_loss += float(loss)
            loss.backward()      # loss step
            optimizer.step()     # optimizer step
        scheduler.step()      # scheduler step
        model.eval()

        with torch.no_grad():
            print("Loss on epoch {epoch} (tune={tune}): {total_loss:.1f}")
            if epoch % eval_skip == 0:
                test_pred = [model.predict(torch.FloatTensor(
                    x_all[i]).to(device)) for i in textdata.test_idx]
                test_gold = [int(y_all[idx]) for idx in textdata.test_idx]
                test_correct = (np.asarray(test_pred) ==
                                np.asarray(test_gold)).sum()            
                train_pred = [model.predict(torch.FloatTensor(
                    x_all[i]).to(device)) for i in textdata.train_idx]
                train_gold = [int(y_all[idx]) for idx in textdata.train_idx]
                train_correct = (np.asarray(train_pred) ==
                                 np.asarray(train_gold)).sum()
                accuracy[tune][epoch] = {
                    'loss': total_loss,
                    'train': train_correct/len(train_gold),
                    'test': test_correct/len(test_gold)}
                print(tune, epoch, int(time.time()-tic),
                      optimizer.param_groups[0]['lr'],
                      train_correct/len(train_gold),
                      test_correct/len(test_gold))

from sklearn.metrics import confusion_matrix
labels = [event_[e] for e in event_encoder.classes_]
cf_train = DataFrame(confusion_matrix(train_gold, train_pred),
              index=pd.MultiIndex.from_product([['Actual'], labels]),
              columns=pd.MultiIndex.from_product([['Predicted'], labels]))
cf_test = DataFrame(confusion_matrix(test_gold, test_pred),
                    index=pd.MultiIndex.from_product([['Actual'], labels]),
                    columns=pd.MultiIndex.from_product([['Predicted'], labels]))
import seaborn as sns
for num, (title, cf) in enumerate({'Training':cf_train, 'Test':cf_test}.items()):
    fig, ax = plt.subplots(num=1+num, clear=True, figsize=(10,6))
    sns.heatmap(cf, ax=ax, annot= False, fmt='d', cmap='viridis', robust=True,
                yticklabels=[f"{label}  {e}"
                             for label,e in zip(labels, event_encoder.classes_)],
                xticklabels=event_encoder.classes_)
    ax.set_title(f'DAN GloVe {title} Set Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.yaxis.set_tick_params(labelsize=8, rotation=0)
    ax.xaxis.set_tick_params(labelsize=8, rotation=0)
    plt.subplots_adjust(left=0.35, bottom=0.25)
    plt.savefig(os.path.join(imgdir, f"glove_{title}.jpg"))
    plt.tight_layout()
plt.show()

print_skip = 2
fig, ax = plt.subplots(num=1, clear=True, figsize=(10,6))
DataFrame.from_dict({err: {(t*len(accuracy[t]) + k) * print_skip: v[err]
                           for t in [False, True]
                           for k,v in enumerate(accuracy[t].values())}
                     for err in ['train', 'test']}).plot(ax=ax)
ax.axvline((len(accuracy[False]) - 0.5) * print_skip, c='grey', alpha=0.5)
ax.set_title(f'DAN GloVe Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(['Train Set', 'Test Set','Fine-tune Weights'], loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, f"glove_accuracy.jpg"))
plt.show()


# Exploring Spacy
"""
hashkey = nlp.vocab.strings[v]: general hash table between vocab strings and ids
vec = nlp.vocab[v].vector: np array with word embedding vector from vocab string
row = nlp.vocab.vectors.key2row: dict from word's hashkey to int

emb = nn.Embedding.from_pretrained(torch.FloatTensor(nlp.vocab.vectors.data))
emb(row) == vec : equivalence of torch embedding and spacy vector

token = nlp('king man queen woman')[0]
token.lower : hashkey
token.lower_: str
token.lex_id : row of word vector
token.has_vector : has word vector representation
"""
doc = nlp('king queen man woman a23kj4j')
line = [tok.lex_id for tok in doc
        if not(tok.is_stop or tok.is_punct or tok.is_oov or tok.is_space)]

vec = (nlp.vocab['king'].vector
       - nlp.vocab['man'].vector
       + nlp.vocab['woman'].vector)
print(vec.shape)
sim = nlp.vocab.vectors.most_similar(vec[None,:], n=10)
[nlp.vocab.strings[hashkey] for hashkey in sim[0][0]]

# Load pretrained embeddings
emb = nn.Embedding.from_pretrained(torch.FloatTensor(nlp.vocab.vectors.data))

# test for Spacy.nlp and torch.embeddings
test_vocab = ['king', 'man', 'woman', 'queen', 'e9s82j']
for w in test_vocab:
    vocab_id = nlp.vocab.strings[w]
    spacy_vec = nlp.vocab[w].vector
    row = nlp.vocab.vectors.key2row.get(vocab_id, None) # dict 
    if row is None:
        print('{} is oov'.format(w))
        continue
    vocab_row = torch.tensor(row, dtype=torch.long)
    embed_vec = emb(vocab_row)
    print(np.allclose(spacy_vec, embed_vec.detach().numpy()))

for key, row in nlp.vocab.vectors.key2row.items():
    if row == 0: 
        print(nlp.vocab.strings[key])

