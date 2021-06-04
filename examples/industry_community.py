"""Industry Sectoring

- igraph, community detection, modularity
- Text-based Network Industry Classification (Hoberg and Phillips, 2016)

Author: Terence Lim
License: MIT
"""
import os
import zipfile
import io
import requests
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
import igraph  # pip3 install cairocffi
from igraph import Graph
from itertools import chain
from finds.graph import igraph_info, igraph_community
from finds.database import SQL
from finds.busday import BusDay
from finds.structured import PSTAT
from finds.sectors import Sectoring
from settings import settings
    
sql = SQL(**settings['sql'])
bd = BusDay(sql)
pstat = PSTAT(sql, bd)
logdir = os.path.join(settings['images'], 'tnic')  # None
tnic_scheme = 'tnic2'

# Retrieve TNIC scheme from Hoberg and Phillips website
# https://hobergphillips.tuck.dartmouth.edu/industryclass.htm
root = 'https://hobergphillips.tuck.dartmouth.edu/idata/'   
source = os.path.join(root, tnic_scheme + '_data.zip')
if source.startswith('http'):
    response = requests.get(source)
    source = io.BytesIO(response.content)
with zipfile.ZipFile(source).open(tnic_scheme + "_data.txt") as f:
    tnic_data = pd.read_csv(f, sep='\s+')
tnic_data.head()

years = [1989, 1999, 2009, 2019]  # [1999, 2019]:
collect = {'modularity': {}, 'community': {}, 'info': {}}
for num, year in enumerate(years):
    
    # extract one year of tnic as data frame, merge in permno and sic codes
    tnic = tnic_data[tnic_data.year == year].dropna()
    vs = DataFrame(index=list(set(tnic['gvkey1']).union(set(tnic['gvkey2']))))
    for code in ['lpermno', 'sic', 'naics']:
        lookup = pstat.build_lookup('gvkey', code, fillna=0)
        vs[code] = lookup(vs.index)
    naics = Sectoring(sql, 'naics', fillna=0)   # supplement from crosswalk
    sic = Sectoring(sql, 'sic', fillna=0)
    vs['naics'] = vs['naics'].where(vs['naics'] > 0, naics[vs['sic']])
    vs['sic'] = vs['sic'].where(vs['sic'] > 0, naics[vs['naics']])
    Series(np.sum(vs > 0, axis=0)).rename('Non-missing').to_frame().T

    # apply sectoring schemes
    schemes = {'sic': ([f"codes{i}" for i in [5, 10, 12, 17, 30, 38, 48, 49]]
                       +['sic2', 'sic3']),
               'naics': ['bea1947', 'bea1963', 'bea1997']}
    codes = {}
    for key, sub in schemes.items():
        for scheme in sub:
            if scheme not in codes:
                codes[scheme] = Sectoring(sql, scheme,
                                          fillna=0 if scheme.startswith('sic')
                                          else '')
                vs[scheme] = codes[scheme][vs[key]]
            vs = vs[vs[scheme].ne(codes[scheme].fillna)]
    vs

    # create vertex (permno, naics, sic) and edge (score) attributes
    edges = tnic[tnic['gvkey1'].isin(vs.index) & tnic['gvkey2'].isin(vs.index)]
    attributes = edges['score'].values
    edges = edges[['gvkey1', 'gvkey2']].astype(str).values

    # populate igraph including attributes (note: vertex names must be str)
    g = Graph(directed=False)
    g.add_vertices(vs.index.astype(str).to_list(), vs.to_dict(orient='list'))
    g.add_edges(edges, {'score': attributes})
    degree = Series(g.vs.degree())   # to remove zero degree vertexes
    print('Deleting vertex IDs with degree 0', list(degree[degree==0].index))
    g.delete_vertices(degree[degree==0].index.to_list())
    g = g.simplify()         # remove self-loops and multi-edges
    s = Series(igraph_info(g)).rename(year)
    collect['info'][year] = s
    print(s.to_frame().T)

    # Plot degree distribution
    fig, ax = plt.subplots(clear=True, num=1, figsize=(10,6))
    Series(g.vs.degree()).hist(grid=False, ax=ax, bins=100)
    ax.set_title(f'Degree Distribution of {tnic_scheme.upper()} links {year}')
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(logdir, f'degree{year}.jpg'))
    plt.show()
    
    # evaluate modularity of sectoring schemes
    modularity = {}
    for scheme in sorted(chain(*schemes.values())):
        membership, uniques = pd.factorize(g.vs[scheme])
        modularity[scheme] = {'unique': len(uniques),
                              'modularity': g.modularity(membership)}
    f = DataFrame.from_dict(modularity, orient='index').sort_index()
    collect['modularity'][year] = f
    f

    # detect communities and report modularity
    clustering, dendogram = igraph_community(g)
    m = DataFrame.from_dict({key: {'modularity': c.modularity,
                                   'components': len(c.sizes())}
                             for key, c in clustering.items()}, orient='index')\
                 .sort_values('modularity', ascending=False)
    collect['community'][year] = m
    m
    
    # Plot industry representation at heatmap
    #.reindex(Sectoring.bea_industry.index)\
    detect = 'multilevel'
    c = clustering[detect]
    attr = 'codes49'
    indus = pd.concat([Series(c.subgraph(j).vs[attr]).value_counts().rename(i+1)
                       for i, j in enumerate(np.argsort(c.sizes())[::-1])],
                      axis=1).dropna(axis=0, how='all').fillna(0).astype(int)\
                      .reindex(codes[attr].sectors['name'].drop_duplicates(
                          keep='first'))
    fig, ax = plt.subplots(num=2+num, clear=True, figsize=(5,12))
    sns.heatmap(indus, square=False, linewidth=.5, ax=ax, yticklabels=1,
                cmap="YlGnBu", robust=True)
    if attr.startswith('bea'):
        ax.set_yticklabels(Sectoring.bea_industry[indus.index], size=10)
    else:
        ax.set_yticklabels(indus.index, size=10)
    ax.set_xlabel(f'Communities ({detect} method)')
    ax.set_title(f"Industry Representation of Communities {year}")
    fig.subplots_adjust(left=0.4)
    plt.tight_layout(pad=3)
    #plt.savefig(os.path.join(logdir, f'{attr}_{year}.jpg'))
    plt.show()

collect['modularity'][1989]

collect['community'][2019]
