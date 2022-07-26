"""Principal Customers Network

- Graphs: ego network, induced subgraph
- Supply chain: principal customers

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

import os
import numpy as np
import pandas as pd
import networkx as nx
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from pandas.api import types
import numpy.ma as ma
from numpy.ma import masked_invalid as valid
from finds.database import SQL
from finds.graph import graph_info, graph_draw, nodes_centrality
from conf import VERBOSE, paths, credentials

VERBOSE = 0
sql = SQL(**credentials['sql'], verbose=VERBOSE)
imgdir = os.path.join(paths['images'], 'supplychain')  # None

# Retrieve principal customers info
year = 2016
cust = sql.read_dataframe(f"select gvkey, cgvkey, stic, ctic, conm, cconm "
                          f"  from customer "
                          f"    where srcdate >= {year}0101 "
                          f"      and srcdate <= {year}1231")
    
# To lookup company full name from ticker
lookup = pd.concat([Series(cust['conm'].values, cust['stic'].values),
                    Series(cust['cconm'].values, cust['ctic'].values)])\
           .drop_duplicates()

# Construct Directed Graph
vertices = set(cust['stic']).union(cust['ctic'])
edges = cust[['stic', 'ctic']].values.tolist()  # supplier --> customer

G = nx.DiGraph()
G.add_nodes_from(vertices)
G.add_edges_from(edges)

# 1. Show graph properties

graph_info(G)

# 2. Display graph

pos = graph_draw(G,
                 figsize=(12, 12),
                 savefig=os.path.join(imgdir, 'graph' + figext),
                 font_color='k',
                 node_color='y')
plt.show()

# 4. Node properties
G.remove_edges_from(nx.selfloop_edges(G))  # remove self-loops, if any

## Show top node centrality properties: nodes_centrality

centrality = DataFrame.from_dict(nodes_centrality(G))
n = 5
for c in centrality.columns:
    df = centrality[[c]].sort_values(by=c, ascending=False)[:n]
    print(pd.concat((lookup[df.index].rename('name'), df), axis=1))


# 5. Induce ego-graph of max betweenness node and neighbors
c = 'betweenness'
center = centrality.index[np.argmax(centrality[c])]
all_neighbors = list(nx.all_neighbors(G, center))  # predecessors and successors
neighbors = list(nx.neighbors(G, center))          # successors only
ego = G.subgraph([center] + all_neighbors).copy()
graph_info(ego, fast=True)

node_color = (dict.fromkeys(all_neighbors, 'b')
              | dict.fromkeys(neighbors, 'g')
              | {center: 'cyan'})

labels = ({ticker: ticker for ticker in ego.nodes}
          | {ticker: lookup[ticker] for ticker in [center] + neighbors})
graph_draw(ego,
           figsize=(10, 5),
           savefig=os.path.join(imgdir, f'{center}' + figext),
           node_size=300,
           width=1,
           node_color=node_color,
           labels=labels,
           style='-',
           title=f"Ego network for node with largest {c}: {center}") 
plt.show()

