"""Graph Centrality and BEA Input-Output Use Tables

- Centrality: eigenvector, hub, authority, pagerank,
- BEA: Input-Output Use Table, Choi and Foerster (2017)

Copyright 2022, Terence Lim

MIT License
"""
import time
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import networkx as nx
from finds.database import Redis
from finds.sectors import Sectoring, BEA
from finds.graph import graph_info, nodes_centrality, graph_draw
from finds.display import show
from conf import credentials, VERBOSE, paths

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None

LAST_YEAR = 2021
years = np.arange(1947, LAST_YEAR) 
vintages = [1997, 1963, 1947]   # when sectoring schemes were revised
rdb = Redis(**credentials['redis'])
bea = BEA(rdb, **credentials['bea'], verbose=VERBOSE)
imgdir = paths['images'] / 'bea'

# Read IOUse tables from BEA website

ioUses = dict()
for vintage in vintages:
    for year in [y for y in years if y >= vintage]:
        df = bea.read_ioUse(year, vintage=vintage)
        ioUses[(vintage, year)] = df
    print(f"{len(ioUses)} tables through sectoring vintage year {vintage}")

## Set directed edges with tail on user (table column) --> head on maker (row)
## Direction of edges point from user industry to maker, i.e. follows the money
tail = 'colcode'   # edges follow flow of payments, from column to row
head = 'rowcode'   
drop = ('F','T','U','V','Other')  # drop these codes
colors = ['lightgrey', 'darkgreen', 'lightgreen']    
yearc = {}  # collect annual table

# Populate and plot graph of first and last table years

ifig, year = 0, 1947
vintage = 1947
year0 = 1947
#vintage = 1997
#year0 = 2019
year1 = 2020
for ifig, year in enumerate([year0, year1]):
    # keep year, drop invalid rows
    ioUse = ioUses[(vintage, year)]
    data = ioUse[(~ioUse['rowcode'].str.startswith(drop) &
                  ~ioUse['colcode'].str.startswith(drop))].copy()

    # create master table of industries and measurements
    master = data[data['rowcode']==data['colcode']][['rowcode','datavalue']]\
             .set_index('rowcode')\
             .rename(columns={'datavalue': 'self'})

    # extract cross data; generate and load edges (as tuples) to graph
    data = data[(data['colcode'] != data['rowcode'])]
    data['weights'] = data['datavalue'] / data['datavalue'].sum()
    edges = data.loc[data['weights'] > 0,
                     [tail, head, 'weights']].values.tolist()
    
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges, weight='weight')
    nx_labels = BEA._bea_industry[list(G.nodes)].to_dict()

    # update master table industry flow values
    master = master.join(data.groupby(['colcode'])['datavalue'].sum(),
                         how='outer').rename(columns={'datavalue': 'user'})
    master = master.join(data.groupby(['rowcode'])['datavalue'].sum(),
                         how='outer').rename(columns={'datavalue': 'maker'})
    master = master.fillna(0).astype(int)
    # inweight~supply~authority~eigenvector~pagerank, outweight~demand~hub

    centrality = DataFrame(nodes_centrality(G))
    master = master.join(centrality, how='left')
    master['bea'] = BEA._bea_industry[master.index].to_list()
    yearc[year] = master[['pagerank', 'bea']].set_index('bea')

    # visualize graph
    score = centrality['pagerank']
    node_size = score.to_dict()
    node_color = {node: colors[0] for node in G.nodes()}
    if ifig == 0:
        center_name = score.index[score.argmax()]
    else:
        node_color.update({k: colors[2] for k in top_color})
    top_color = list(score.index[score.argsort()[-5:]])
    node_color.update(dict.fromkeys(top_color, colors[1]))
    pos = graph_draw(G,
                     num=ifig+1,
                     figsize=(10, 10),
                     center_name=center_name,
                     node_color=node_color,
                     node_size=node_size,
                     edge_color='r',
                     k=3,
                     pos=(pos if ifig else None),
                     font_size=10,
                     font_weight='semibold',
                     labels=master['bea'].to_dict(),
                     title=f"By Pagerank: {year} IO-Use ({vintage} vintage)")
    if imgdir:
        plt.savefig(imgdir / f"{year}.jpg")

## Display node centrality
c = pd.concat([yearc[year0].rank(ascending=False).astype(int),
               yearc[year1].rank(ascending=False).astype(int)],
              axis=1)
c.columns = pd.MultiIndex.from_product([[year0, year1], yearc[year0].columns])
c

## Display correlation of centrality ranks
c.corr().round(3)


# Display latest graph and node centrality
year = LAST_YEAR - 1
ioUse = ioUses[(1997, year)]
data = ioUse[(~ioUse['rowcode'].str.startswith(drop) &
              ~ioUse['colcode'].str.startswith(drop))].copy()

## extract cross data; generate and load edges (as tuples) to graph
data = data[(data['colcode'] != data['rowcode'])]
data['weights'] = data['datavalue'] / data['datavalue'].sum()
edges = data.loc[data['weights'] > 0, [tail, head, 'weights']].values.tolist()
G = nx.DiGraph()
G.add_weighted_edges_from(edges, weight='weight')

## update master table industry flow values and graph centrality measures
master = pd.concat((data[data['rowcode']
                         == data['colcode']][['rowcode','datavalue']]\
                    .set_index('rowcode')\
                    .rename(columns={'datavalue': 'self'}),
                    data.groupby(['colcode'])['datavalue'].sum()\
                    .rename('user'),
                    data.groupby(['rowcode'])['datavalue'].sum()\
                    .rename('maker')),
                   join='outer',
                   axis=1).fillna(0).astype(int)
master = master.join(DataFrame(nodes_centrality(G)), how='left')
master['bea'] = BEA._bea_industry[master.index].to_list()
show(master.drop(columns=['self', 'clustering']),
     ndigits=3,
     caption=f"Centrality of BEA Input-Output Use Table {year}")

Series(graph_info(G, fast=True))

