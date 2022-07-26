"""Deep Averaging Networks for text classification

- Feedforward Neural Networks: torch, deep averaging networks
- Word vectors: spacy, GloVe, relativize, frozen, fine-tuning

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

# jupyter-notebook --NotebookApp.iopub_data_rate_limit=1.0e12
import numpy as np
import os
import time
import re
import csv, gzip, json
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import torch
import torch.nn as nn
import random
from finds.database import MongoDB
from finds.unstructured import Unstructured
from finds.structured import PSTAT
from conf import credentials, VERBOSE, paths

mongodb = MongoDB(**credentials['mongodb'])
keydev = Unstructured(mongodb, 'KeyDev')
imgdir = os.path.join(paths['images'], 'classify')
events_ = PSTAT._event
roles_ = PSTAT._role
memdir = paths['scratch']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## Retrieve headline+situation text
events = [28, 16, 83, 41, 81, 23, 87, 45, 80, 97,  231, 46, 31, 77, 29,
          232, 101, 42, 47, 86, 93, 3, 22, 102, 82]

initial = False

if initial:
    lines = []
    event_all = []
    tokenizer = RegexpTokenizer(r"\b[^\d\W][^\d\W][^\d\W]+\b")
    for event in events:
        docs = keydev['events']\
               .find({'keydeveventtypeid': {'$eq': event}}, {'_id': 0})
        doc = [tokenizer.tokenize((d['headline'] + " " + d['situation'])\
                                  .lower())
               for d in docs]
        lines.extend(doc)
        event_all.extend([event] * len(doc))
    with gzip.open(os.path.join(memdir, 'lines.json.gz'), 'wt') as f:
        json.dump(lines, f)
    with gzip.open(os.path.join(memdir,'event_all.json.gz'), 'wt') as f:
        json.dump(event_all, f)
    print(lines[1000000])

else:
    with gzip.open(os.path.join(memdir, 'lines.json.gz'), 'rt') as f:
        lines = json.load(f)
    with gzip.open(os.path.join(memdir,'event_all.json.gz'), 'rt') as f:
        event_all = json.load(f)

## Encode class labels
from sklearn.preprocessing import LabelEncoder
events_encoder = LabelEncoder().fit(event_all)    # .inverse_transform()
num_classes = len(np.unique(event_all))
y_all = events_encoder.transform(event_all)

Series(event_all).value_counts()\
                 .rename(index=events_)\
                 .rename('count')\
                 .to_frame()

## Split into stratified train and test indices
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(np.arange(len(y_all)),
                                       random_state=42,
                                       stratify=y_all,
                                       test_size=0.2)

## Load spacy vocab
import spacy
lang = 'en_core_web_lg'
nlp = spacy.load(lang, disable=['parser', 'tagger', 'ner', 'lemmatizer'])
for w in ['inr', 'yen', 'jpy', 'eur', 'dkk', 'cny', 'sfr']:
    nlp.vocab[w].is_stop = True    # Mark customized stop words
    
n_vocab, vocab_dim = nlp.vocab.vectors.shape
print('Language:', lang, '   vocab:', n_vocab, '   dim:', vocab_dim)

## Precompute word embeddings input
def form_input(line, nlp):
    """Return spacy average vector from valid words"""
    tokens = [tok.vector for tok in nlp(" ".join(line))
              if not(tok.is_stop
                     or tok.is_punct
                     or tok.is_oov
                     or tok.is_space)]
    if len(tokens):
        return np.array(tokens).mean(axis=0)
    else:
        return np.zeros(nlp.vocab.vectors.shape[1])

args = {'dtype': 'float32'}

if initial:
    args.update({'shape': (len(lines), vocab_dim), 'mode': 'w+'})
    X = np.memmap(os.path.join(memdir,
                               "X.{}_{}".format(*args['shape'])),
                  **args)
    for i, line in tqdm(enumerate(lines)):
        X[i] = form_input(line, nlp).astype(args['dtype'])
else:
    args.update({'shape': (1224251, vocab_dim), 'mode': 'r'})
    X = np.memmap(os.path.join(memdir,
                               "X.{}_{}".format(*args['shape'])),
                  **args)


## Pytorch Feed Forward Network
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
        return self.classifier(self.network(x))

    def predict(self, x):
        """Return predicted int class of input tensor vector"""
        return torch.argmax(self(x), dim=1).int().tolist()
            
    def save(self, filename):
        """save model state to filename"""
        return torch.save(self.state_dict(), filename)

    def load(self, filename):
        """load model name from filename"""
        self.load_state_dict(torch.load(filename, map_location='cpu'))
        return self
    
## Training Loops
accuracy = {}  # to store computed metrics
max_layers, hidden = 1, 300 #3, 300
batch_sz, lr, num_lr, step_sz, eval_skip = 64, 0.01, 4, 10, 5 #3, 3, 3 #
num_epochs = step_sz * num_lr + 1
for layers in [max_layers]:

    # Instantiate model, optimizer, scheduler, loss_function
    model = FFNN(vocab_dim=vocab_dim,
                 num_classes=num_classes,
                 hidden=[hidden]*layers)\
                 .to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                gamma=0.1,
                                                step_size=step_sz)
    loss_function = nn.NLLLoss()
    accuracy[layers] = {}

    # Loop over epochs and batches
    for epoch in range(0, num_epochs):
        tic = time.time()
        idxs = [i for i in train_idx]
        random.shuffle(idxs)
        batches = [idxs[i:(i+batch_sz)] for i in range(0, len(idxs), batch_sz)]
    
        total_loss = 0.0
        model.train()
        for i, batch in enumerate(batches):    # loop over minibatches
            x = torch.FloatTensor(np.array(X[batch]))\
                     .to(device)
            y = torch.tensor(np.array([y_all[idx] for idx in batch]))\
                     .to(device)
            model.zero_grad()                    # reset model gradient
            log_probs = model(x)                 # run model
            loss = loss_function(log_probs, y)   # compute loss
            total_loss += float(loss)
            loss.backward()      # loss step
            optimizer.step()     # optimizer step
            print(i, len(batches), i/len(batches), total_loss, end='\r')
        scheduler.step()      # scheduler step
        model.eval()
        print(f"Loss on epoch {epoch}: {total_loss:.1f}")
        #model.save(os.path.join(imgdir, f"dan{layers}.pt"))

        with torch.no_grad():
            if epoch % eval_skip == 0:
                gold = np.asarray([int(y) for y in y_all])
                batches = [test_idx[i:(i+128)]
                           for i in range(0, len(test_idx), 128)]
                test_gold, test_pred = [], []
                for batch in tqdm(batches):
                    test_pred.extend(model.predict(
                        torch.FloatTensor(np.array(X[batch])).to(device)))
                    test_gold.extend(gold[batch])
                test_correct = (np.asarray(test_pred) ==
                                np.asarray(test_gold)).sum()

                batches = [train_idx[i:(i+128)]
                           for i in range(0, len(train_idx), 128)]
                train_gold, train_pred = [], []
                for batch in tqdm(batches):
                    train_pred.extend(model.predict(
                        torch.FloatTensor(np.array(X[batch])).to(device)))
                    train_gold.extend(gold[batch])
                train_correct = (np.asarray(train_pred) ==
                                 np.asarray(train_gold)).sum()
                accuracy[layers][epoch] = {
                    'loss': total_loss,
                    'train': train_correct/len(train_idx),
                    'test': test_correct/len(test_idx)}
                print(layers, epoch, int(time.time()-tic),
                      optimizer.param_groups[0]['lr'],
                      train_correct/len(train_idx), test_correct/len(test_idx))

from sklearn import metrics
print(model)   # show accuracy metrics for this layer
pd.concat([
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
           name='Train Set').to_frame().T], axis=0)
"""
           Accuracy  Precision    Recall
Test Set   0.910648   0.906198  0.910648
Train Set  0.914899   0.913187  0.914899

"""

fig, ax = plt.subplots(num=1, clear=True, figsize=(10,6))
DataFrame.from_dict({err: {k: v[err] for k,v in accuracy[max_layers].items()}
                     for err in ['train', 'test']}).plot(ax=ax)
ax.set_title(f'Accuracy of DAN with frozen embedding weights')
ax.set_xlabel('Steps')
ax.set_ylabel('Accuracy')
ax.legend(['Train Set', 'Test Set'], loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, f"frozen_accuracy" + figext))
plt.show()
    
## Confusion Matrix
from sklearn.metrics import confusion_matrix
labels = [events_[e] for e in events_encoder.classes_]
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
                yticklabels=[f"{lab}  {e}" for lab, e in
                             zip(labels, events_encoder.classes_)],
                xticklabels=events_encoder.classes_)
    ax.set_title(f'{title} Set Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.yaxis.set_tick_params(labelsize=8, rotation=0)
    ax.xaxis.set_tick_params(labelsize=8, rotation=0)
    plt.subplots_adjust(left=0.35, bottom=0.25)
    plt.savefig(os.path.join(imgdir, f"frozen_{title}" + figext))
    plt.tight_layout()
plt.show()



## DAN with GloVe embeddings and fine-tune weights
from finds.unstructured import TextualData
textdata = TextualData()  # class for text pre-processing

if initial:
    vocab = textdata.counter(lines)          # count words for vocab
    textdata(vocab.most_common(20000), 0)    # vocab is most common 20000
    textdata.dump('textdata.json', imgdir)

    x_all = textdata[lines]                  # convert str docs to word indexes
    with gzip.open(os.path.join(imgdir, 'x_all.csv.gz'), 'wt', newline="") as f:
        csv.writer(f).writerows(x_all)
else:
    x_new = []
    with gzip.open(os.path.join(imgdir, 'x_all.csv.gz'), 'rt') as f:
        for row in csv.reader(f):
            x_new.append(row)
    textdata.load('textdata.json', imgdir)
print('vocab size', textdata.n)
                
## Relativize glove embeddings
"""Load GloVe embeddings weights, and drop rows not in vocab
wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip
Archive:  glove.6B.zip
  inflating: glove.6B.100d.txt       
  inflating: glove.6B.200d.txt       
  inflating: glove.6B.300d.txt       
  inflating: glove.6B.50d.txt    
"""
vocab_dim = 300
glove_prefix = os.path.join(paths['scratch'], f"glove.6B.{vocab_dim}d")
if initial:
    glove = textdata.relativize(glove_prefix + ".txt")
    with open(glove_prefix + ".rel.pkl", "wb") as f:
        pickle.dump(glove, f)
else:
    with open(glove_prefix + ".rel.pkl", "rb") as f:
        glove = pickle.load(f)
print('glove dimensions', glove.shape)

## train_test split stratified by y_all
textdata.form_splits(y_all, random_state=42, test_size=0.2)

## define DAN with tunable word embeddings
class DAN(nn.Module):
    """Deep Averaging Network for classification"""
    def __init__(self, vocab_dim, num_classes, hidden, embedding,
                 dropout=0.3, requires_grad=False):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(embedding)
        self.embedding.weight.requires_grad = requires_grad
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
        return torch.argmax(self(x), dim=1).int().tolist()

    def save(self, filename):
        """save model state to filename"""
        return torch.save(self.state_dict(), filename)

    def load(self, filename):
        """load model name from filename"""
        self.load_state_dict(torch.load(filename, map_location='cpu'))
        return self    

layers = 2
hidden = vocab_dim   #100, 300
model = DAN(vocab_dim,
            num_classes,
            hidden=[hidden]*layers,
            embedding=torch.FloatTensor(glove)).to(device)
print(model)


## Training loop
accuracy = dict()
for tune in [False, True]:
    # define model, optimizer, scheduler, loss_function
    model.tune(tune)
    batch_sz, lr, num_lr, step_sz, eval_skip = 64, 0.01, 4, 5, 5 #3, 3, 3 #
    num_epochs = step_sz * num_lr + 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1,
                                                step_size=step_sz)
    loss_function = nn.NLLLoss()
    accuracy[tune] = dict()

    # Loop over epochs and batches
    for epoch in range(0, num_epochs):
        tic = time.time()
        batches = textdata.form_batches(batch_sz)

        total_loss = 0.0
        model.train()
        for batch in tqdm(batches):   # train by batch
            x = textdata.form_input([x_all[idx] for idx in batch]).to(device)
            y = torch.tensor([y_all[idx] for idx in batch]).to(device)
            model.zero_grad()                    # reset model gradient
            log_probs = model(x)                 # run model
            loss = loss_function(log_probs, y)   # compute loss
            total_loss += float(loss)
            loss.backward()           # loss step
            optimizer.step()          # optimizer step
        scheduler.step()          # scheduler step for learning rate
        model.eval()
        model.save(os.path.join(imgdir, f"danGloVe.pt"))
        print(f"Loss on epoch {epoch} (tune={tune}): {total_loss:.1f}")

        with torch.no_grad():
            if epoch % eval_skip == 0:
                test_pred = [model.predict(textdata.form_input(
                    [x_all[i]]).to(device))[0] for i in textdata.test_idx]
                test_gold = [int(y_all[idx]) for idx in textdata.test_idx]
                test_correct = (np.asarray(test_pred) ==
                                np.asarray(test_gold)).sum() 
                train_pred = [model.predict(textdata.form_input(
                    [x_all[i]]).to(device))[0] for i in textdata.train_idx]
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

## Confusion matrix
from sklearn.metrics import confusion_matrix
labels = [events_[e] for e in events_encoder.classes_]
cf_train = DataFrame(confusion_matrix(train_gold, train_pred),
              index=pd.MultiIndex.from_product([['Actual'], labels]),
              columns=pd.MultiIndex.from_product([['Predicted'], labels]))
cf_test = DataFrame(confusion_matrix(test_gold, test_pred),
                    index=pd.MultiIndex.from_product([['Actual'], labels]),
                    columns=pd.MultiIndex.from_product([['Predicted'], labels]))

#
# First half of sample fixed weights, second half start allow tuning
#
import seaborn as sns
for num, (title, cf) in enumerate({'Training':cf_train,'Test':cf_test}.items()):
    fig, ax = plt.subplots(num=1+num, clear=True, figsize=(10,6))
    sns.heatmap(cf, ax=ax, annot= False, fmt='d', cmap='viridis', robust=True,
                yticklabels=[f"{label}  {e}" for label, e in
                             zip(labels, events_encoder.classes_)],
                xticklabels=events_encoder.classes_)
    ax.set_title(f'DAN Tuned GloVe {title} Set Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.yaxis.set_tick_params(labelsize=8, rotation=0)
    ax.xaxis.set_tick_params(labelsize=8, rotation=0)
    plt.subplots_adjust(left=0.35, bottom=0.25)
    plt.savefig(os.path.join(imgdir, f"tuned_{title}" + figext))
    plt.tight_layout()
plt.show()

fig, ax = plt.subplots(num=1, clear=True, figsize=(10, 6))
DataFrame.from_dict({sample: {(tuning * len(accuracy[tuning]) + epoch):
                             acc[sample]
                             for tuning in [False, True]
                             for epoch, acc in enumerate(
                                     accuracy[tuning].values())}
                     for sample in ['train', 'test']}).plot(ax=ax)
ax.axvline((len(accuracy[False])), c='grey', alpha=0.5)
ax.set_title(f'Accuracy of DAN with fine-tuned GloVe weights')
ax.set_xlabel('Steps')
ax.set_ylabel('Accuracy')
ax.legend(['Train Set', 'Test Set','Start Fine-Tuning'], loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, f"tuned_accuracy" + figext))
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
if False:
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
    emb = nn.Embedding\
            .from_pretrained(torch.FloatTensor(nlp.vocab.vectors.data))

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
