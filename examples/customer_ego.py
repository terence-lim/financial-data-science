"""Principal Customers Network

- igraph, ego graph, betweenness centrality
- S&P Compustat, Wharton Research Data Services

Author: Terence Lim
License: MIT
"""
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import igraph  # pip3 install cairocffi
from igraph import Graph
from pandas.api import types
import numpy.ma as ma
from numpy.ma import masked_invalid as valid
from itertools import chain
from finds.graph import igraph_draw, igraph_info, igraph_path
from finds.graph import igraph_centrality, igraph_community
from finds.database import SQL
from settings import settings
sql = SQL(**settings['sql'])
logdir = os.path.join(settings['images'], 'supplychain')  # None

# Retrieve principal customers info
year = 2016
cust = sql.read_dataframe(
    f"select gvkey, cgvkey, stic, ctic, conm, cconm from customer"
    f" where srcdate >= {year}0101 and srcdate <= {year}1231")
    
# To lookup company full name from ticker
lookup = Series(cust['conm'].values, cust['stic'].values)\
         .append(Series(cust['cconm'].values, cust['ctic'].values))\
         .drop_duplicates()

# Construct Directed Graph
vertices = np.array(list(set(cust['stic']).union(set(cust['ctic']))))
edges = cust[['stic', 'ctic']].values   # direction is supplier to customer
g = Graph(directed=True)          # g.clear()
g.add_vertices(vertices)          # g.add_vertex()
g.add_edges(edges)                # g.add_edge()
    
# Show graph properties
dg = g.simplify()
Series(igraph_info(dg)).rename('Simple Graph').to_frame().T
print(Series(igraph_info(dg)).rename('Simple Graph')\
      .to_frame().T.to_string(float_format='{:.4f}'.format))

# Display graph
pos = igraph_draw(dg, figsize=(12,12), font_color='k', node_color='y')
plt.savefig(os.path.join(logdir, 'graph.jpg'))
plt.show()

# Maximum edge_betweeness
ebs = dg.edge_betweenness()
max_eb = max(ebs)
[(lookup[dg.es[idx].source_vertex['name']], \
  lookup[dg.es[idx].target_vertex['name']], eb)
 for idx, eb in enumerate(ebs) if eb == max_eb]
    
# Show top node centrality properties
centrality = igraph_centrality(dg)
df = DataFrame(centrality, index=dg.vs['name'])
df.index = df.index.map('{:>10s}'.format)     # set print column widths
df.columns = df.columns.map('{:16s}'.format)
n = 5
for c in df.columns:
    print()
    print(df[[c]].sort_values(by=c, ascending=False)[:n].T)

# ego-graph of max betweenness node with one-step neighbors
c = 'betweenness'
u = np.argmax(centrality[c])
v = dg.neighbors(u)
ego = dg.induced_subgraph(v + [u])
ego.summary(3)
pos=igraph_draw(
    ego, node_size=300, width=0.5, center_name=g.vs[u]['name'], figsize=(10,6),
    title=f"{dg.vs[u]['name']} has largest vertex {c}",
    node_color={ 
        **{k: 'b' for k in dg.vs[dg.neighbors(u, 'out')]['name']},
        **{k: 'g' for k in dg.vs[dg.neighbors(u, 'in')]['name']},
        **{g.vs[u]['name']: 'cyan'}},
    labels=lookup[ego.vs['name']].to_dict())
plt.savefig(os.path.join(logdir, f'{c}.jpg'))
plt.show()
    
# longest shortest path, plus neighbors that complete triangles
nodes = dg.get_diameter()    # longest shortest path
print(igraph_path(dg, nodes, dg.vs['name'], lookup))

neighbors = pd.Series(chain(*[g.neighbors(v) for v in nodes])).value_counts()
triangles = set(neighbors[neighbors > 1].index).difference(nodes)
subnodes = nodes + list(triangles) # with neighbors that complete triangles

pos = igraph_draw(
    dg.induced_subgraph(subnodes), arrowsize=20, width=1, k=4, 
    title=f"Longest shortest path has {len(nodes)} nodes",
    center_name=dg.vs[nodes[len(nodes)//2]]['name'], figsize=(10,6),
    node_color={
        **{k: 'cyan' for k in dg.vs[nodes]['name']},
        **{k: 'yellow' for k in dg.vs[triangles]['name']}},
    node_size={dg.vs[k]['name']: centrality['pagerank'][k]
               for k in subnodes},
    labels=lookup[dg.vs[subnodes]['name']].to_dict())
plt.savefig(os.path.join(logdir, 'diameter.jpg'))
plt.show()

