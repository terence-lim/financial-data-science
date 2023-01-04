"""Link Prediction

- Random graphs
- Link prediction: resource_allocation, jaccard coefficient, 
  adamic_adar, preferential_attachment
- Accuracy: precision, recall, ROC curve, AUC, confusion matrix, 
- Imbalanced sample

Copyright 2022, Terence Lim

MIT License
"""
import zipfile
import io
import time
from itertools import chain
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import networkx.algorithms.community as nx_comm
from finds.database import SQL, requests_get, Redis
from finds.busday import BusDay
from finds.structured import PSTAT, CRSP
from finds.sectors import Sectoring
from finds.graph import graph_draw, graph_info, link_prediction
from finds.display import show
from conf import credentials, paths, VERBOSE

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None

sql = SQL(**credentials['sql'], verbose=VERBOSE)
bd = BusDay(sql, verbose=VERBOSE)
rdb = Redis(**credentials['redis'])
pstat = PSTAT(sql, bd, verbose=VERBOSE)
crsp = CRSP(sql, bd, rdb, verbose=VERBOSE)
imgdir = paths['images'] / 'tnic'

## Retrieve TNIC schemes from Hoberg and Phillips website

# https://hobergphillips.tuck.dartmouth.edu/industryclass.htm
root = 'https://hobergphillips.tuck.dartmouth.edu/idata/'
tnic_data = {}
for scheme in ['tnic2', 'tnic3']:
    source = root + scheme + '_data.zip'
    if source.startswith('http'):
        response = requests_get(source)
        source = io.BytesIO(response.content)
    with zipfile.ZipFile(source).open(scheme + "_data.txt") as f:
        tnic_data[scheme] = pd.read_csv(f, sep='\s+')
for k,v in tnic_data.items():
    print(k, v.shape)

# extract one year of both tnic schemes, merge in permno, and require in univ
year = 2019
capsize = 10  # large cap (large than NYSE median)
univ = crsp.get_universe(bd.endyr(year))
univ = univ[univ['decile'] <= capsize]   
lookup = pstat.build_lookup('gvkey', 'lpermno', fillna=0)
nodes = {}
tnic = {}
edges = {}
for scheme in ['tnic2', 'tnic3']:
    tnic[scheme] = tnic_data[scheme][tnic_data[scheme].year == year].dropna()
    gvkeys = set(tnic[scheme]['gvkey1']).union(tnic[scheme]['gvkey2'])
    df = DataFrame(index=gvkeys, data=lookup(gvkeys), columns=['permno'])
    nodes[scheme] = df[df['permno'].gt(0)
                       & df['permno'].isin(univ.index)].drop_duplicates()
nodes['tnic2'] = nodes['tnic2'][nodes['tnic2'].index.isin(nodes['tnic3'].index)]
nodes['tnic3'] = nodes['tnic3'][nodes['tnic3'].index.isin(nodes['tnic2'].index)]


# create graphs of tnic2 (full graph) and tnic3 (subgraph) schemes
for scheme in ['tnic2', 'tnic3']:
    e = tnic[scheme][tnic[scheme]['gvkey1'].isin(nodes[scheme].index) &
                     tnic[scheme]['gvkey2'].isin(nodes[scheme].index)]
    edges[scheme] = list(e[['gvkey1', 'gvkey2', 'score']]\
                         .itertuples(index=False, name=None))

results = {'info':{}}
G = {}
for (scheme, node), (_, edge) in zip(nodes.items(), edges.items()):
    print(scheme, 'nodes =', len(node), 'edges =', len(edge))

    # populate graph
    g = nx.Graph()
    g.add_nodes_from(node.index)
    g.add_weighted_edges_from(edge)

    # remove self-loops: not necessary
    g.remove_edges_from(nx.selfloop_edges(g))

    # graph info
    results['info'].update({scheme: Series(graph_info(g, fast=True))})

    # Plot degree distribution
    fig, ax = plt.subplots(clear=True, figsize=(5, 4))
    degree = nx.degree_histogram(g)
    degree = DataFrame(data={'degree': degree[1:]},   # exclude degree 0
                       index=np.arange(1, len(degree)))
    degree['bin'] = (degree.index // (2*capsize) + 1) * (2*capsize)
    degree.groupby('bin').sum().plot(kind='bar', ax=ax, fontsize=6)
    ax.set_title(f'Degree Distribution of {scheme.upper()} links {year}')
    plt.tight_layout(pad=3)
    plt.savefig(imgdir / f'degree_{scheme}_{year}.jpg')

    G[scheme] = g

show(DataFrame(results['info']),     # Display graph properties
     caption=f"Graph info of TNIC schemes {year}", **SHOW) 


## Predict links
"""
- jaccard_coefficient
- resource_allocation
- adamic_adar
- preferential_attachment
"""

links = link_prediction(G['tnic3'])

# Sanity check that tnic3 and prediction edges strictly in tnic2
def isin(e1, e2):
    """helper to count number of edges e1 are in e2"""
    num = sum([e[:2] in e2 for e in e1])
    return num, len(e1), num/len(e1)


records = [[src, tgt, *isin(G[src].edges, G[tgt].edges)]
           for src, tgt in zip(['tnic3', 'tnic2'],
                               ['tnic2', 'tnic3'])]
records.extend([[src, 'tnic2', *isin(links[src], G['tnic2'].edges)]
                for src in ['jaccard_coefficient',
                            'resource_allocation',
                            'adamic_adar',
                            'preferential_attachment']])
show(DataFrame.from_records(records,
                            columns=['source',
                                     'target',
                                     'source edges in target',
                                     'total source edges',
                                     'fraction']),
     index=False, caption="Counts of edges", **SHOW)


## Evaluate accuracy of link prediction algorithms
"""
- roc
- auc 
- confusion matrix
"""
def make_sample(prediction, edges):
    """helper to transform prediction to labels and scores for roc and auc"""
    scores = [e[-1] for e in prediction]
    label = [e[:2] for e in prediction]
    gold = [e[:2] in edges for e in prediction]
    return gold, scores, label  # actual, predicted score, predicted label


for ifig, (method, pred) in enumerate(links.items()):
    y, scores, label = make_sample(pred, G['tnic2'].edges)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(clear=True, figsize=(6.5, 5))
    plt.plot(fpr,
             tpr,
             color="darkorange",
             lw=2,
             label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver operating characteristic: {method}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(imgdir + f'{method}_auc.jpg')

    thresh = scores[sum(y)]  # set threshold to equal class size
    cm = metrics.confusion_matrix(y, [score > thresh for score in scores],
                                  normalize='pred')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {method}")
    plt.tight_layout()
    plt.savefig(imgdir / f'{method}_confusion.jpg')

