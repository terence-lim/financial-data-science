"""Graph network convenience wrappers

- networkx (link prediction, community detection, centrality)

Copyright 2022, Terence Lim

MIT License
"""
import os
import time
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from pandas.api import types
import networkx as nx
import networkx.algorithms.community as nx_comm
from typing import Dict, Tuple, List, Any, Callable

_VERBOSE = 1
def _print(tic, *args, verbose=_VERBOSE):
    if verbose > 0:
        print(round(time.time() - tic, 0), 'secs:', *args)

def graph_info(G, fast=False):
    """Return summary of graph properties

    Args:
        fast: True to skip slow triadic census, clustering and distance measures

    Returns:
        Dict of property name and value
    """
    out = dict()

    if not fast:
        # Census
        if nx.is_directed(G):
            triads = nx.triadic_census(G)
            for k,v in triads.items():
                out['triad_' + k] = v

        # Induced subgraph of largest connected component
        if nx.is_directed(G):
            connected = nx.is_strongly_connected(G)
            #component = max(nx.strongly_connected_components(G), key=len)
        else:
            connected = nx.is_connected(G)            
            #component = max(nx.connected_components(G), key=len)
        if connected:
            out['diameter_largest_component'] = nx.diameter(G)
            out['radius_largest_component'] = nx.radius(G)
            #out['center_largest_component'] = nx.center(G)
        # Clustering
        out['transitivity'] = nx.transitivity(G)
        out['average_clustering'] = nx.average_clustering(G)

    # Components
    if nx.is_directed(G):
        out['weakly_connected'] = nx.is_weakly_connected(G)
        out['weakly_connected_components'] = \
            nx.number_weakly_connected_components(G)
        out['size_largest_weak_component'] = \
            len(max(nx.weakly_connected_components(G), key=len))
        out['strongly_connected'] = nx.is_strongly_connected(G)
        out['strongly_connected_components'] = \
            nx.number_strongly_connected_components(G)
        out['size_largest_strong_component'] = \
            len(max(nx.strongly_connected_components(G), key=len))
    else:
        out['connected'] = nx.is_connected(G)
        out['connected_components'] = nx.number_connected_components(G)
        out['size_largest_component'] = \
                    len(max(nx.connected_components(G), key=len))
    out['directed'] = nx.is_directed(G)
    out['weighted'] = nx.is_weighted(G)
    if nx.is_weighted(G):
        out['negatively_weighted'] = nx.is_negatively_weighted(G)
    out['edges'] = nx.number_of_edges(G)
    out['nodes'] = nx.number_of_nodes(G)
    out['selfloops'] = nx.number_of_selfloops(G)
    out['density'] = nx.density(G)
    return out


def graph_draw(G: nx.Graph,
               num: int = 1,
               figsize: Tuple[float, float] = (10, 10),
               savefig: str = '',
               title: str = '',
               font_weight: str = 'bold',
               font_size: int = 8,
               font_family: str = 'helvetica',
               k: float = 2.,
               pos: Callable | Dict = {},
               center_name: Any = None,
               alpha: float = 0.5,               
               arrowsize: float = 10.,
               arrowstyle: str = '-|>',
               style: str = ':',
               width: float = .5,
               edge_color: str = 'r',
               nodelist: List = [],
               node_scale: float = 1.,
               node_size: float | Dict | List = 20.,
               node_color: str | Dict | List = '#1f78b4',
               labels: List | Dict = None,
               **kwargs):
    """Convenience wrapper over nx.draw_network

    Args:
        G: Directed or undirected graph to draw
        savefig: JPG filename to save as
        figsize: Figure size in inches
        title: Text to display in top left of plot
        font_weight: {'normal', bold' 'light'}
        k: Extent of separatation between nodes
        center_name: Node to place in center of plot
        pos: Layout dict or callable, default uses nx.spring_layout()
        arrowsize: Size of arrow head to draw edges
        arrowstyle: Format of arrow to draw
        width: Width of arrow
        edge_color: Color of edge arrow
        style: Edge style in {'-', '--', '-.', ':'}
        nodelist: List of nodes to draw
        node_scale: Relative scaling of nodes
        node_size: List of node size, or a lookup dict
        node_color: List of node color, or a lookup dict
        labels: List of node labels, or a lookup dict
        **kwargs : parameters passed on to networkx.draw

    Returns:
        Dict of pos with nodes as keys and positions as values

    Other pos layouts:

    - pos = nx.circular_layout(G)
    - pos = nx.spectral_layout(G)
    - pos = nx.kamada_kawai_layout(G)
    - pos = nx.fruchterman_reingold_layout(G)
    - pos = nx.spring_layout(G)
    """
    if not nodelist:
        nodelist = list(G)
    if isinstance(node_size, dict):   # node_size must be list, or float
        maxsz = max(node_size.values())
        node_size = [node_scale * 5000 * node_size[s]/maxsz for s in nodelist]
    if isinstance(node_color, dict):  # node_color must be list, or str
        node_color = [node_color[s] for s in nodelist]
    if isinstance(labels, dict):  # all labels (dict) must be in nodes
        labels = {k: labels[k] for k in G.nodes()}  

    plt.figure(num=num, figsize=figsize)
    plt.clf()
    if not pos:
        if center_name:
            fixed, p = [center_name], {center_name : (0,0)}
        else:
            fixed, p = None, None
        if callable(pos):
            pos = pos(G)
        else:
            pos = nx.spring_layout(G,
                                   fixed=fixed,
                                   pos=p,
                                   k=k/np.sqrt(nx.number_of_nodes(G)))
        
    nx.draw(G, pos=pos, font_size=font_size, labels=labels, 
            font_weight=font_weight, edge_color=edge_color, alpha=alpha,
            width=width, arrowsize=arrowsize, style=style,
            node_size=node_size, node_color=node_color, **kwargs)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(xmin + (xmax-xmin)/20, ymax - (ymax-ymin)/20, title)
    if savefig:
        plt.savefig(savefig)
    return pos


def nodes_centrality(G, weight='weight', cost=False, alpha=0.99):
    """Return dict of vertex centrality measures

    Args:
        G: Graph may be directed or indirected, weighted or unweighted
        weight: name of edge attribute for weights, Set to None for unweighted
        cost: If True, then weights are costs; else weights are importances

    Returns:
        Dict of {label (str) : score (float)}

    Notes:

    - centrality: degree, eigenvector, closeness, betweenness
    - link_analysis: pagerank, hits

    if weight is cost: 'eigenvector', 'pagerank', 'hub', 'authority' ignore weights

    if weight is not cost: 'betweenness', 'closeness' ignore weights
    """
    out = {}
    out['clustering'] = nx.clustering(G)
    if nx.is_weighted(G, weight=weight):
        out['clustering'] = nx.clustering(G)        
    if nx.is_directed(G):
        out['in_degree'] = nx.in_degree_centrality(G)
        out['out_degree'] = nx.out_degree_centrality(G)
    else:
        out['degree'] = nx.degree_centrality(G)
        out['triangles'] = nx.triangles(G)
    if not cost and nx.is_weighted(G):
        out['eigenvector'] = \
            nx.eigenvector_centrality(G, weight=weight, max_iter=1000)
        out['pagerank'] = nx.pagerank(G, weight=weight, alpha=alpha)
    else:
        out['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
        out['pagerank'] = nx.pagerank(G, alpha=alpha)
    out['hub'], out['authority'] = nx.hits(G)
    if cost and nx.is_weighted(G):
        out['betweenness'] = nx.betweenness_centrality(G, weight=weight)
        out['closeness'] = nx.closeness_centrality(G, distance=weight)
    else:
        out['betweenness'] = nx.betweenness_centrality(G)
        out['closeness'] = nx.closeness_centrality(G)
    # if connected:
    # out['eccentricity'] = nx.eccentricity(G)  # max distance of node to other
    return out


def community_detection(G: nx.Graph, methods: List[str] = [],
                        weight: str | None = None, resolution: float = 1.,
                        verbose: int = _VERBOSE) -> Dict[str, List[List]]:
    """Run built-in community detection algorithms on an undirected graph

    Args:
        G: Undirected networkx Graph
        methods: Community detection algorithms to run, 
                 in {'label', 'louvain', 'greedy'}
        weight: Name of edge attribute for weights
        resolution: Stopping rule

    Returns:
        Dictionary, keyed by algorithm, of communities lists
    """
    tic = time.time()
    results = {}
    if not methods or 'label' in methods:
        results['label'] = nx_comm.label_propagation_communities(G)
        _print(tic, 'label_propogation')

    if not methods or 'louvain' in methods:
        results['louvain'] = nx_comm.louvain_communities(
            G, weight=weight, resolution=resolution)
        _print(tic, 'louvain')

    if not methods or 'greedy' in methods:
        results['greedy'] = nx_comm.greedy_modularity_communities(
            G, weight=weight, resolution=resolution)
        _print(tic, 'greedy')

    return results

def community_quality(G: nx.Graph, communities: List[List],
                      methods: List[str] = []) -> Dict[str, float | int]:
    """Run built-in community performance metrics

    Args:
        G: Undirected networkx Graph
        communties: Communities list of lists
        method: List of metrics in {'modularity', 'quality'}

    Returns:
        Dictionary, keyed by metric label, of metric values
    """
    results = {'communities': len(communities)}
    if not methods or 'modularity' in methods:
        results['modularity'] = nx_comm.modularity(G, communities)
    if not methods or 'quality' in methods:
        results['coverage'], results['performance'] = \
            nx_comm.partition_quality(G, communities)
    return results

def link_prediction(G, verbose=_VERBOSE):
    """Run built-in link prediction algorihms

    Returns:
        Dictionary, keyed by algorithm label, of List of edge-score 3-tuples
    """
    def links(links):
        return sorted(links, key=lambda x: x[2], reverse=True)
    #[link for link in links if link[2] > 0],

    tic = time.time()
    resource = links(nx.resource_allocation_index(G))
    _print(tic, 'resource_allocations')
    
    jaccard = links(nx.jaccard_coefficient(G))
    _print(tic, 'jaccard_coefficient')
                     
    adamic = links(nx.adamic_adar_index(G))
    _print(tic, 'adamic_adar')
                    
    preferential = links(nx.preferential_attachment(G))
    _print(tic, 'preferential_attachment')
    
    return {'resource_allocation': resource,
            'jaccard_coefficient': jaccard,
            'adamic_adar': adamic,
            'preferential_attachment': preferential}

"""
- iterate over Graph edges and attributes:
for edge_tuple, attributes_dict in G.edges.items()

- iterate over Graph nodes and attributes:
for node, attributes_dict in G.nodes.items()

- neighbors of a Node:
nx.all_neighgors
nx.neighbors
list(g[1004])
G.degree[1004]

G = nx.Graph(social_pairs)
G = nx.DiGraph(social_pairs)
G.add_nodes_from(vertices)
G.add_edges_from(edges)
"""
