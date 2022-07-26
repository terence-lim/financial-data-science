"""Industry Sectoring and Community Detection

- Community Detection
- Text-based Network Industry Classification (Hoberg and Phillips, 2016)

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

import os
import zipfile
import io
import time
from itertools import chain
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import networkx.algorithms.community as nx_comm
from finds.database import SQL, requests_get
from finds.busday import BusDay
from finds.structured import PSTAT
from finds.sectors import Sectoring
from finds.graph import graph_draw, graph_info, nodes_centrality, \
    community_quality, community_detection, link_prediction
from conf import credentials, paths, VERBOSE

sql = SQL(**credentials['sql'])
bd = BusDay(sql)
pstat = PSTAT(sql, bd)
imgdir = os.path.join(paths['images'], 'tnic')  # None
tnic_scheme = 'tnic3'

# Retrieve TNIC scheme from Hoberg and Phillips website

## https://hobergphillips.tuck.dartmouth.edu/industryclass.htm
root = 'https://hobergphillips.tuck.dartmouth.edu/idata/'   
source = os.path.join(root, tnic_scheme + '_data.zip')
if source.startswith('http'):
    response = requests_get(source)
    source = io.BytesIO(response.content)
with zipfile.ZipFile(source).open(tnic_scheme + "_data.txt") as f:
    tnic_data = pd.read_csv(f, sep='\s+')
tnic_data

# Loop over representative years for community detection
years = [1989, 1999, 2009, 2019]  # [1999, 2019]:
collect = {'info': {}, 'modularity': {}, 'community': {}}  # to collect metrics
num = 0
for year in years:
    
    # extract one year of tnic as data frame
    tnic = tnic_data[tnic_data.year == year].dropna()
    nodes = DataFrame(index=set(tnic['gvkey1']).union(tnic['gvkey2']))

    # with gvkey, lookup permno, sic and naics codes
    for code in ['lpermno', 'sic', 'naics']:
        lookup = pstat.build_lookup('gvkey', code, fillna=0)
        nodes[code] = lookup(nodes.index)
    naics = Sectoring(sql, 'naics', fillna=0)   # supplement from crosswalk
    sic = Sectoring(sql, 'sic', fillna=0)
    nodes['naics'] = nodes['naics'].where(nodes['naics'] > 0,
                                            naics[nodes['sic']])
    nodes['sic'] = nodes['sic'].where(nodes['sic'] > 0,
                                        naics[nodes['naics']])
    Series(np.sum(nodes > 0, axis=0)).rename('Non-missing').to_frame().T

    # apply sectoring schemes, and store in nodes DataFrame
    schemes = {'sic': ([f"codes{c}" for c in [5, 10, 12, 17, 30, 38, 48, 49]]
                       + ['sic2', 'sic3']),
               'naics': ['bea1947', 'bea1963', 'bea1997']}
    codes = {}   # intermediate to combine raw sic/naics to sector scheme
    for key, sub in schemes.items():
        for scheme in sub:
            if scheme not in codes:
                fillna = 0 if scheme.startswith('sic') else ''
                codes[scheme] = Sectoring(sql, scheme, fillna=fillna)
                nodes[scheme] = codes[scheme][nodes[key]]
            nodes = nodes[nodes[scheme].ne(codes[scheme].fillna)]
    nodes

    # create edges
    edges = tnic[tnic['gvkey1'].isin(nodes.index) &
                 tnic['gvkey2'].isin(nodes.index)]
    edges = list(edges[['gvkey1', 'gvkey2', 'score']]\
                 .itertuples(index=False, name=None))

    # populate graph
    g = nx.Graph()
    g.add_weighted_edges_from(edges)

    # remove self-loops: not necessary
    g.remove_edges_from(nx.selfloop_edges(g))

    # graph info
    collect['info'][year] = Series(graph_info(g, fast=True)).rename(year)

    # Plot degree distribution
#    num = num + 1
#    fig, ax = plt.subplots(clear=True, num=num, figsize=(10,6))
#    Series(nx.degree_histogram(g)).hist(grid=False, ax=ax, bins=100)
#    ax.set_title(f'Degree Distribution of {tnic_scheme.upper()} links {year}')
#    plt.tight_layout(pad=3)
#    plt.savefig(os.path.join(imgdir, f'degree{year}' + figext))
    
    # evaluate modularity of sectoring schemes
    modularity = {}
    for scheme in sorted(chain(*schemes.values())):
        communities = nodes.loc[list(g.nodes), scheme]\
                           .reset_index()\
                           .groupby(scheme)['index']\
                           .apply(list)\
                           .to_list()    # list of list of symbols
        modularity[scheme] = community_quality(g, communities)
    df = DataFrame.from_dict(modularity, orient='index').sort_index()
    collect['modularity'][year] = df
    show(df,
         latex=False,
         caption=f"Modularity of sectoring schemes {year}")

    # detect communities and report modularity
    communities = community_detection(g)
    tic = time.time()
    quality = {}
    for key, community in communities.items():
        quality[key] = community_quality(g, community)
        print('total elapsed:', round(time.time() - tic, 0), key)
    df = DataFrame.from_dict(quality, orient='index').sort_index()
    collect['community'][year] = df
    show(df,
         latex=False,
         caption=f"Modularity community detection algorithms {year}")
    
    # Plot Fama-French codes49 industry representation as heatmap
    for ifig, detection in enumerate(['label', 'greedy', 'louvain']):
        scheme = 'codes49'
        industry = []
        for i, community in enumerate(sorted(communities[detection],
                                             key=len,
                                             reverse=True)):
            industry.append(nodes[scheme][list(community)]\
                            .value_counts()\
                            .rename(i+1))
            
        df = pd.concat(industry, axis=1)\
               .dropna(axis=0, how='all')\
               .fillna(0)\
               .astype(int)\
               .reindex(codes[scheme]\
                        .sectors['name']\
                        .drop_duplicates(keep='first'))

        num = num + 1
        fig, ax = plt.subplots(num=num, clear=True, figsize=(5, 12))
        sns.heatmap(df.iloc[:,:10],
                    square=False,
                    linewidth=.5,
                    ax=ax,
                    yticklabels=1,
                    cmap="YlGnBu",
                    robust=True)
        if scheme.startswith('bea'):
            ax.set_yticklabels(Sectoring._bea_industry[df.index], size=10)
        else:
            ax.set_yticklabels(df.index, size=10)
        ax.set_title(f'{detection.capitalize()} Community Detection {year}')
        ax.set_xlabel(f"Industry representation in communities")
        ax.set_ylabel('{scheme} industry')
        fig.subplots_adjust(left=0.4)
        plt.tight_layout(pad=3)
        plt.savefig(os.path.join(imgdir, f'{detection}_{year}' + figext))
plt.show()

# Display latest year info

show(collect['info'][2019],
     latex=False,
     caption=f"{tnic_scheme} {year} graph info:")

show(collect['community'][2019],
     latex=False,
     caption=f"{tnic_scheme} {year} community detection:")

show(collect['modularity'][2019],
     latex=False,
     caption=f"{year} modularity of industry sectoring crosswalk schemes:")
