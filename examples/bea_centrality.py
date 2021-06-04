"""Graph Centrality and BEA Input-Output Use Tables

- igraph, network centrality, BEA Input-Output Use Table
- Choi and Foerster (2017), Bureau of Economic Analysis, and others

Terence Lim
License: MIT
"""
import os
import time
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import igraph  # pip3 install cairocffi
from igraph import Graph
from finds.database import Redis
from finds.sectors import Sectoring, BEA
from finds.graph import igraph_draw, igraph_info, igraph_centrality
from settings import settings

ECHO=False
years = np.arange(1947, 2020) 
vintages = [1997, 1963, 1947]   # when sectoring schemes were revised
rdb = Redis(**settings['redis'])
bea = BEA(rdb, **settings['bea'], echo=ECHO)
logdir = None # os.path.join(settings['images'], 'bea')

# Read IOUse tables from BEA website
ioUses = dict()
for vintage in vintages:
    for year in [y for y in years if y >= vintage]:
        df = bea.read_ioUse(year, vintage=vintage)
        ioUses[(vintage, year)] = df
    print(f"{len(ioUses)} tables through sectoring vintage year {vintage}")

# Set directed edges with tail on user (table column) and head on maker (row)
# Direction of edges point from user industry to maker, i.e. follows the money
tail = 'colcode'   # edges follow flow of payments, from column to row
head = 'rowcode'   
drop = ('F','T','U','V','Other')  # drop these codes
colors = ['lightgrey', 'darkgreen', 'lightgreen']    
yearc = {}  # collect annual table

# Populate and plot graph of first and last table years
for ifig, year in enumerate([1947, 2019]): #np.arange(1947, 2020)):
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
                     [tail, head, 'weights', 'datavalue']].values
    g = Graph.TupleList(edges, edge_attrs=['weight', 'value'], directed=True)
    g.vs['bea'] = list(BEA.bea_industry[g.vs['name']])

    Series(igraph_info(g)).to_frame().T

    # update master table industry flow values
    master = master.join(data.groupby(['colcode'])['datavalue'].sum(),
                         how='outer').rename(columns={'datavalue': 'user'})
    master = master.join(data.groupby(['rowcode'])['datavalue'].sum(),
                         how='outer').rename(columns={'datavalue': 'maker'})
    master = master.fillna(0).astype(int)
    # inweight~supply~authority~eigenvector~pagerank, outweight~demand~hub
    master = master.join(DataFrame(index=list(g.vs['name']),
                                   data=igraph_centrality(g)), how='left')
    master['bea'] = BEA.bea_industry[master.index].to_list()
    yearc[year] = master[['outweight','inweight', 'pagerank', 'hub',
                          'authority', 'bea']].set_index('bea')

    # visualize graph
    score = g.pagerank(weights='weight', damping=0.99)   
    node_size = pd.Series(data=score, index=g.vs['name']).to_dict()
    node_color = {k: colors[0] for k in g.vs["name"]}
    if ifig == 0:
        center_name = g.vs['name'][np.argmax(score)]
    else:
        node_color.update({k: colors[2] for k in top_color})
    top_color = g.vs[list(np.argsort(score)[-5:])]["name"]
    node_color.update({k: colors[1] for k in top_color})
    
    pos = igraph_draw(
        g, num=ifig+1, center_name=center_name,
        node_color=node_color, node_size=node_size, edge_color='r', k=3,
        pos=(pos if ifig else None),font_size=10, font_weight='semibold',
        labels={k:v for k,v in zip(g.vs['name'], g.vs['bea'])},
        title=f"By Pagerank: {year} IO-Use ({vintage} vintage scheme)")
    if logdir: plt.savefig(os.path.join(logdir, str(year) + '.jpg'))
    print(Series(igraph_info(g)).rename(str(year)).to_frame().T.to_string())
plt.show()

# Display node centrality
c = yearc[1947].rank(ascending=False).astype(int).join(
    yearc[2019].rank(ascending=False).astype(int), rsuffix='2019')
c.columns = pd.MultiIndex.from_product([[1947,2019], yearc[1947].columns])
c
print(c.to_latex())

# Display correlation of centrality ranks
c.corr().round(3)
print(c.corr().to_latex(float_format='%.3f'))

# Display latest graph and node centrality
vintage = year = 2019
ioUse = ioUses[(1997, year)]
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
                 [tail, head, 'weights', 'datavalue']].values
g = Graph.TupleList(edges, edge_attrs=['weight', 'value'], directed=True)
g.vs['bea'] = list(BEA.bea_industry[g.vs['name']])

# update master table industry flow values and graph centrality measures
master = master.join(data.groupby(['colcode'])['datavalue'].sum(),
                     how='outer').rename(columns={'datavalue': 'user'})
master = master.join(data.groupby(['rowcode'])['datavalue'].sum(),
                     how='outer').rename(columns={'datavalue': 'maker'})
master = master.fillna(0).astype(int)
master = master.join(DataFrame(index=list(g.vs['name']),
                               data=igraph_centrality(g)), how='left')
master['bea'] = BEA.bea_industry[master.index].to_list()

print(master.set_index('bea')\
      .drop(columns=['betweenness', 'closeness'])\
      .to_latex(float_format='%.4f'))
    
Series(igraph_info(g)).rename(str(year)).to_frame().T

