"""Convenience methods for igraph and networkx modules

- Network science, community detection, centrality, modularity
- igraph, networkx

Author: Terence Lim
License: MIT
"""
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import networkx as nx
import cairocffi
import igraph  # pip3 install cairocffi
from igraph import Graph
from pandas.api import types
import numpy.ma as ma

from igraph import arpack_options
arpack_options.maxiter=100000000   # increase numerical iterations

# See also https://igraph.org/news.html and
# https://igraph.org/python/doc/tutorial/index.html

def igraph_to_networkx(g, weight='weight', name='name'):
    """Convert igraph object to networkx format
    Parameters
    ----------
    g : igraph.Graph object
        may be directed or undirected, weighted or unweighted
    weight : str or None, default 'weight'
        attribute of igraph edge to use as weight
    name : str or None, default 'name'
        attribute of igraph vertex to use as label
    
    Returns
    -------
    G : networkx.Graph or Digraph object
        with vertex attributes converted as well

    Notes
    -----
    replaces igraph.Graph.to_networkx()
    igraph uses vertex and edge IDs in its core, which are integers starting 
    from zero. When deleted, IDs will be renumbered to ensure continuity: 
    1. Maintain additional list of edge or vertex attributes in the graph 
       (g.vs and g.es) or externally 
    2. g.add_vertices and g.add_edges to add name, weight and other attributes 
    """
    G = nx.DiGraph() if g.is_directed() else nx.Graph()
    names = list(g.vs[name] if name and name in g.vs.attributes()
                 else np.arange(g.vcount()))
    G.add_nodes_from(names)
    if weight and weight in g.es.attributes():
        G.add_weighted_edges_from(
            [(names[e.tuple[0]], names[e.tuple[1]], e[weight]) for e in g.es])
    else:
        G.add_edges_from([(names[e.tuple[0]], names[e.tuple[1]]) for e in g.es])
    for attrib in g.vs.attributes():
        if attrib != name:
            nx.set_node_attributes(
                G, {k: v for k,v in zip(g.vs[name], g.vs[attrib])}, name=attrib)
    return G

def igraph_draw(g, savefig=None, num=1, figsize=(10,11), pos=None, 
                arrowsize=10, arrowstyle='-|>', font_weight='bold',
                labels=None, 
                style='dotted', width=.1, font_size=8, font_family='helvetica',
                edge_color='r', k=2, node_size=20, node_scale=1, alpha=0.5,
                center_name=None, node_color='#1f78b4', title='', **kwargs):
    """Draws either igraph or networkx graph object, using networkx.draw()

    Parameters
    ----------
    DG : igraph or networkx object
        directed or undirected graph to draw
    savefig : string, optional (default None)
        jpg filename to save as
    figsize : tuple of (int, int), default is (11,11)
        plt figure size in inches
    node_scale : float, default is 1
        relative scaling of nodes
    node_size : dict (default is None)
        dictionary to look up node label for its plot size
    node_color : dict (default is None)
        dictionary to look up node label for its plot size
    labels : dict (default is None)
        dictionary to look up nodel label for its display string
    arrowsize : int, default is 10
        size of arrow head to draw edges
    arrowstyle : str, default i '-|>'
        format of arrow to draw
    width : float, default is 0.1
        width of arrow
    edge_color : str, default is 'r'
        color of edge arrow
    k : float, default is 2
        extent of separatation between nodes
    center_name : node name, default None
        label of node to place in center of plot
    title : string, default ''
        text to display in top left of plot
    pos : layout dict or callable, default is None
        layout positions to use or call. If None, use nx.spring_layout()
    style : str, 
        default 'dotted' 
    font_weight : str 
        'normal', bold' 'light'
    font_size : int
        default is 8
    **kwargs : parameters
        passed on to networkx.draw

    Returns
    -------
    pos : layout dict
        layout positions, as dict of {node name: position array}

    Notes
    -----
    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
    arrowstyle : str, optional (default='-|>')
        For directed graphs, choose the style of the arrowsheads.
    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head's length and width.    with_labels :  bool, optional (default=True)
       Set to True to draw labels on the nodes.
    nodelist : list, optional (default G.nodes())
       Draw only specified nodes
    edgelist : list, optional (default=G.edges())
       Draw only specified edges
    node_size : scalar or array, optional (default=300)
       Size of nodes.  If an array same length as nodelist.
    node_color : color or array of colors (default='#1f78b4')
       Node color. If a sequence of colors same length as nodelist
    node_shape :  string, optional (default='o')
       The shape of the node.  matplotlib.scatter marker, one of 'so^>v<dph8'.
    alpha : float, optional (default=None)
       The node and edge transparency
    cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of nodes
    vmin,vmax : float, optional (default=None)
       Minimum and maximum for node colormap scaling
    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)
    width : float, optional (default=1.0)
       Line width of edges
    edge_color : color or array of colors (default='k')
       Edge color. If a sequence of colors the same length as edgelist.    
    edge_cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of edges
    edge_vmin,edge_vmax : floats, optional (default=None)
       Minimum and maximum for edge colormap scaling
    style : string, optional (default='solid')
       Edge line style (solid|dashed|dotted,dashdot)
    labels : dictionary, optional (default=None)
      Node labels in a dictionary keyed by node of text labels
    font_size : int, optional (default=12)
       Font size for text labels
    font_color : string, optional (default='k' black)
       Font color string
    font_weight : string, optional (default='normal')
       ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
    font_family : string, optional (default='sans-serif')
       ['serif', 'monospace']
    label : string, optional
        Label for graph legend

    Examples
    -----
    https://networkx.github.io/documentation/networkx-1.11/reference/generated/
      networkx.drawing.nx_pylab.draw.html 
    help(nx.draw_networkx) or help(nx.drawing.layout)
    pos = nx.circular_layout(G)
    pos = nx.spectral_layout(G)
    pos = nx.kamada_kawai_layout(G)
    pos = nx.fruchterman_reingold_layout(G)
    pos = nx.spring_layout(G)
    node_color = '#1f78b4'
    palette = igraph.drawing.colors.ClusterColoringPalette(3)
    colors = palette.get_many(np.arange(3))
    """
    G = igraph_to_networkx(g) if isinstance(g, Graph) else g    
    plt.figure(num=num, figsize=figsize)
    plt.clf()
    if isinstance(node_size, dict):
        node_size = [node_scale * 5000 * node_size[s]/max(node_size.values())
                     for s in G.nodes()]
    if isinstance(node_color, dict):
        node_color = [node_color[s] for s in G.nodes()]
    if isinstance(labels, dict):
        labels = {k:labels[k] for k in G.nodes()}  # labels must be in nodes
    if pos is None or callable(pos):
        if center_name is None:
            fixed, p = None, None
        else:
            fixed, p = [center_name], {center_name : (0,0)}
        pos = pos(G) if callable(pos) else nx.spring_layout(
            G, fixed=fixed, pos=p, k=k/np.sqrt(nx.number_of_nodes(G)))
    nx.draw(G, pos=pos, font_size=font_size, labels=labels, 
            font_weight=font_weight, edge_color=edge_color, alpha=alpha,
            width=width, arrowsize=arrowsize, style=style,
            node_size=node_size, node_color=node_color, **kwargs)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(xmin + (xmax-xmin)/20, ymax - (ymax-ymin)/20, title)
    if savefig:
        plt.savefig(savefig)
    return pos


def dendogram_info(d, verbose=False):
    """Display steps of clustering dendogram"""
    print(d.optimal_count, d.summary())  # optimal count and merge sequence
    if verbose:
        for i in range(3):
            c = d.as_clustering(d.optimal_count + i)    # replay merges
            print(sorted(c.sizes(), reverse=True))

def cluster_info(c, attr=None):
    """Display info of subgraphs in a VertexClustering"""
    return {'modularity': c.modularity, 'n': len(c.sizes())}

def igraph_community(dg, weights=None, verbose=False):
    """Run various community detection algorithms on an undirected graph
    - fastgreedy, walktrap, eigenvector, label_propogation, multilevel, infomap
    """
    g = dg.as_undirected()
    vc, vd = dict(), dict()
    vc['components'] = g.components(mode='STRONG')           # 'WEAK'

    d = g.community_fastgreedy(weights=weights)
    vd['fastgreedy'] = d                           # VertexDendrogram class
    vc['fastgreedy'] = d.as_clustering(d.optimal_count)   # replay merges
    
    d = g.community_walktrap(weights=weights)
    vd['walktrap'] = d
    vc['walktrap'] = d.as_clustering(d.optimal_count)   # replay merges
    
    vc['eigenvector'] = g.community_leading_eigenvector(
        clusters=None, weights=weights, arpack_options=arpack_options)
    vc['label_propogation'] = g.community_label_propagation(weights=weights)
    vc['multilevel'] = g.community_multilevel(weights=weights)
    vc['infomap'] = g.community_infomap(
        edge_weights=weights, vertex_weights=None)

    if verbose:
        for name, c in vc.items():
            counts = cluster_info(c, verbose=True, title=name)
    return vc, vd

    #c=g.community_optimal_modularity()
    # GNU Linear Programming Kit to solve large integer optimization :(
    #c=g.community_edge_betweenness(directed=False)  # slow!
    #c=g.community_spinglass()  # slow!


def igraph_centrality(g, weights='weight', cost=False, damping=0.99):
    """Return dict of vertex centrality measures

    Parameters
    ----------
    g : igraph object
        may be directed or indirected, weighted or unweighted
    weights : str, default='weight'
        name of node attribute to weight edges, Set to None for unweighted
    cost : bool, default is False
        if True, then weights are costs; else weights are importances

    Returns
    -------
    result : dict of {centrality str : score float}
        if cost: 'eigenvector', 'pagerank', 'authority', 'hub' ignore weights
        if not cost: 'betweenness', 'closeness' ignore weights
    """
    if not g.is_weighted() or not weights in g.es.attribute_names():
        weights = None
    out = dict()
    #out['indegree'] = g.indegree()
    #out['outdegree'] = g.outdegree()
    out['outweight'] = g.strength(weights=weights, mode='out')
    out['inweight'] = g.strength(weights=weights, mode='in')

    # weights applicable only if importance
    if not g.is_weighted():
        out['eigenvector'] = g.eigenvector_centrality(weights=weights if not cost
                                                      else None)
    out['pagerank'] = g.pagerank(weights=weights if not cost else None,
                                 damping=damping)
    out['authority'] = g.authority_score(weights=weights if not cost else None)
    out['hub'] = g.hub_score(weights=weights if not cost else None)

    # weights applicable only if cost
    out['betweenness'] = g.betweenness(weights=weights if cost else None)
    out['closeness'] = g.closeness(weights=weights if cost else None)
    
    #out['clustering'] = g.as_undirected().transitivity_local_undirected()
    return out

def igraph_info(g, census=False):
    """Return summary dict of igraph properties"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      
        out = dict()
        out["vertices"] = g.vcount()
        out["edges"] = g.ecount()
        out["density"] = 2*g.ecount()/(g.vcount()*(g.vcount()-1))
        out["diameter"] = g.diameter()
        out["simple"] = g.is_simple()
        out["directed"] = g.is_directed()
        out["global clustering"] = g.transitivity_undirected()
        out["local clustering"] = g.transitivity_avglocal_undirected()
        components = g.components(mode=1).sizes()
        out["weak components"] = len(components)
        out["largest weak"] = max(components)
        components = g.components(mode=2).sizes()
        out["strong components"] = len(components)
        out["largest strong"] = max(components)
        if census:
            out["dyads"] = g.dyad_census() if g.is_directed() else ()
            out["triads"] = g.triad_census() if g.is_directed() else ()
    return out

def igraph_path(g, nodes, labels=None, desc=None):
    """Display edge directions in path of nodes"""
    return " ".join(
        "[{node}] {arrow}".format(
            node=u if labels is None else (labels[u] if desc is None
                                           else desc[labels[u]]),
            arrow="->" if g[u, v] else ("<-" if g[v, u] else ""))
        for u,v in zip(nodes, nodes[1:] + nodes[:1]))

if False:  # randomly explore igraph
    import igraph
    p = igraph.plot(Graph.Famous("petersen"))
    p.show()
    Graph.show(Graph.Famous("petersen"))

    _colors = ['darkred', 'darkgreen', 'darkblue', 'indigo',
               'red', 'lime', 'blue', 'blueviolet', 'salmon',
               'lawngreen', 'slateblue', 'mediumpurple', 'chocolate',
               'forestgreen', 'cyan', 'magenta', 'orange', 'teal']
    g = Graph.Barabasi(n = 20, m = 1)
    i = g.community_infomap()
    pal = igraph.drawing.colors.ClusterColoringPalette(len(i))
    g.vs['color'] = pal.get_many(i.membership)
    plot(g)
