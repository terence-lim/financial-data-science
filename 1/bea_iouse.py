"""
BEA Input-Output Use Tables
Network Science
Lead-lag cross-momentum

References:

https://www.sci.unich.it/~francesc/teaching/network/

Choi and Foerster, 2017, "The Changing Input-Output Network Structure of the U.S. Economy", Federal Reserve Bank of Kansas City

Menzly, Lior, and Oguzhan Ozbas, 2010, Market segmentation and cross-predictability of returns, Journal of Finance 65, 1555–1580.
"""
import dives
import dives.dbengines
import dives.econs
import dives.util
import dives.structured
import dives.evaluate
import networkx as nx
import networkx.algorithms # import approximation, community
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
from pprint import pprint

import importlib
importlib.reload(dives)
importlib.reload(dives.dbengines)
importlib.reload(dives.econs)
importlib.reload(dives.util)
importlib.reload(dives.structured)
importlib.reload(dives.evaluate)

from dives.util import DataFrame, NamedDict, sort_values
from dives.econs import Sectoring, BEA
from dives.dbengines import SQL, Redis
from dives.structured import BusDates, Benchmarks, CRSP, Signals
from dives.evaluate import BackTest, run_backtest

import secret

sql = SQL(**secret.value('sql'))
rdb = Redis(**secret.value('redis'))
bea = BEA(**secret.value('bea'), rdb=rdb)

bd = BusDates(sql)
signals = Signals(sql)
bench = Benchmarks(sql, bd)
crsp = CRSP(sql, bd, rdb)
backtest = BackTest(sql, bench, 'RF')

if False:
    """
    1. Show rankings for all measures, and explain, for 1947, 1963, 1997, 2018, and biggest change in rank
    2. bar charts -- show code
    2. network graph them
    3. plot authorities score for big change industries over time

    Two reasons for directed from user to supplier: (1) more similar to web networks of information:
    an industry cannot really force other industries buy its product (2) follows the money
    """
    
    # Use table: user industries (colcodes), supplier commodities (rowcodes)

    # Read industry names and definitions
    gdpindustry = bea.get(**BEA.items['gdpindustry'])        # get industry code and description from BEA
    desc = gdpindustry.to_dict()['desc']                     # put into a dict()
    descr = {v : desc[k] for k,v in BEA._bealabels.items()}  # assign description to alternative code labels

    # in dives.econs
    def ioUse_sum(ioUse, drop = ('F','T','U','V','Other'), by_user=False, exclude_self=True,
                  labels = BEA._bealabels):
        """return total flows by industry

        Parameters
        ----------
        ioUse : DataFrame
            with stacked datavalues, returned by load_ioUse()
        drop : list of str, optional (default is ('F','T','U','V','Other'))
            do not sum over these industries
        by_user : bool, optional (default False)
            if True: sum by user industry. else sum by supplier industry
        """
        sum_by, sum_over = ('colcode', 'rowcode') if by_user else ('rowcode', 'colcode')
        data = ioUse[(~ioUse[sum_over].str.startswith(drop) &
                      ~ioUse[sum_by].str.startswith(drop))].copy()    # industries to drop
        label = lambda x : x if labels is None else labels[x]         # use original or provided labels
        s = DataFrame(data.groupby([sum_by])['datavalue'].sum())      # total sum by industry
        x = s - data.loc[data[sum_by] == data[sum_over],              # excluding own industry flow
                         [sum_by,'datavalue']].set_index(sum_by).reindex(s.index, fill_value=0)
        return ({label(k) : x.loc[k, 'datavalue'] for k in x.index} if exclude_self else
                {label(k) : s.loc[k, 'datavalue'] for k in s.index})


    # can be a dives.econs.BEA
    def ioUse_graph(ioUse, drop = ('F','T','U','V','Other'), labels=BEA._bealabels,
                    exclude_self = True, normalize_user = None, from_user = True):
        """
        Notes
        -----
        'colcode' is user industry, 'rowcode' is supplier industry
        """
        head, tail = ('rowcode','colcode') if from_user else ('colcode','rowcode')  # set edge direction
        data = ioUse[(~ioUse[tail].str.startswith(drop))].copy()                    # drop these industries
        if exclude_self:                                                            # exclude self loops
            data = data[(data[tail] != data[head])]
        if normalize_user:                 # normalize edge weights by user industry total
            group = data.groupby(['colcode'])
            data['weights'] = data['datavalue'] / group['datavalue'].transform('sum')
        elif normalize_user is not None:   # normalize edge weights by supplier industry total
            group = data.groupby(['rowcode'])
            data['weights'] = data['datavalue'] / group['datavalue'].transform('sum')
        else:                              # normalize by grand total
            data['weights'] = data['datavalue'] / np.sum(data['datavalue'])
        label = lambda x : x if labels is None else labels[x]   # if node labels provided
        edges = [(label(c), label(r), w) for c,r,w in data[[tail, head, 'weights']].values]
        DG = nx.DiGraph()
        DG.add_weighted_edges_from(edges)
        if exclude_self:    # redundant given earlier condition code checked already
            DG.remove_edges_from(list(nx.selfloop_edges(DG)))  # to remove self loops
        return DG

    # can be a dives.util
    def graph_draw(DG, out=None, figsize=(11,11), nodesize=None, labels=None,
                   center=None, nodelist = [], title = '', **kwargs):
        """
        # help(nx.draw_networkx) help(nx.drawing.layout)
        # circular_layout spring_layout kamada_kawai_layout spectral_layout
        """
        if figsize:
            plt.figure(figsize=(11, 11))
        if nodesize is None:
            (hubs, authorities) = nx.hits(DG)
            nodesize = authorities
        if center:
            fixed = [center]
            pos = {center : (0,0)}
            node_color = ['red' if n in nodelist else '#1f78b4' for n in DG.nodes()]
        else:
            node_color = '#1f78b4'
            pos = None
            fixed = None
        ns = [10000*nodesize[n]/max(nodesize.values()) for n in DG.nodes()]
        nx.draw(DG, with_labels=True, font_size='8', font_weight='bold', 
                edge_color='b', width=0.1, style='dotted', arrowsize=10, alpha=0.5, 
                node_size=ns, labels=labels, **kwargs, node_color=node_color,
                pos=nx.spring_layout(DG, fixed=fixed, pos=pos,
                                     k=2/np.sqrt(nx.number_of_nodes(DG))))
        xmin, xmax, ymin, ymax = plt.axis()
        plt.text(xmin + (xmax-xmin)/20, ymax - (ymax-ymin)/20, title)  # kludge to add title with plt.text
        if out:
            plt.savefig(out)

    # can be in dives.util
    def graph_calc(dg):
        """Calculate static graph statistics: centrality, density, links analysis"""
        (hubs, authorities) = nx.hits(dg, max_iter=1000)
        return {'hubs' : hubs,
                'authorities' : authorities,
                'indegree' : nx.in_degree_centrality(dg),
                'pagerank' : nx.pagerank(dg, max_iter=1000),
                'inweight' : {k:v for k,v in dg.in_degree(weight='weight')},
                }
#                'density': nx.density(dg)}


    ioUses = NamedDict(['vintage','year'])
    years = np.arange(1947, 2019) #[1947, 1963, 1997, 2018]
    vintages = [1947, 1963, 1997]
    for year in years:
        for vintage in [v for v in vintages if v <= year]:
            df = ioUse=bea.load_ioUse(year, vintage=vintage)
            ioUses.replace(vintage=vintage, year=year, ioUse=df)

    """
    show 2018 iouse: value, inweight,... value
    cat bar graph of 2018 authorities and inweight
    plot (1947 vs 2018: 1947) - top changes
    plot authorities from 1947 to 2018: 1947 of top and bottom five rank changes
    """
    # Summarize flows and centralities for 2018, using latest sectoring scheme
    df = ioUses.match(vintage = 1997, year = 2018)['ioUse']
    flows = DataFrame()
    flows['use'] = pd.Series(ioUse_sum(df, by_user=True, exclude_self=True))
    flows['make'] = pd.Series(ioUse_sum(df, by_user=False, exclude_self=True))
    flows['self'] = pd.Series(ioUse_sum(df, by_user=True, exclude_self=False)) - flows['use']
    dg = ioUse_graph(df)      # make directed network graph from ioUse stacked dataframe
    calc = graph_calc(dg)     # calculate node centrality statistics
    for metric in calc.keys():
        flows[metric] = pd.Series(calc[metric])
    flows[np.isnan(flows)] = 0

    # Print summary of 2018
    np.round(flows.corr(),2)  # although positive, centrality measures not perfectly correlated
    print(flows.to_string())

    # Plot centralities for 2018
    bars = DataFrame()
    for metric in ['authorities','hubs', 'inweight']:
        bar = flows[metric].reset_index().rename(columns = {metric : 'centrality'})
        bar['metric'] = metric
        bars = bars.append(bar, ignore_index=True)
    sns.catplot(x='index', y='centrality', hue='metric', data=bars, kind='bar', 
                height=6, aspect=2, legend_out=False,     # size : height, aspect, legend_out
                order=flows[metric].sort_values().index)  # order industries by inweight
    plt.xticks(rotation=90, fontsize=6)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    # Print top 5 authorities of 1947, and display in 1947 and 2018 graphs, using 1947 scheme
    vintage = 1947          # fix sectoring scheme at year 1947
    metric = 'authorities'  # rank by this metric
    n = 3
    ranked = DataFrame()
    for year in np.arange(1947, 2019):
        df = ioUses.match(vintage=vintage, year=year)['ioUse']  # apply 1947 sectoring to each year's flow
        dg = ioUse_graph(df)                                    # make directed network graph
        calc = graph_calc(dg)                                   # calculate centrality statistics
        ranked[year] =  pd.Series(calc[metric])                 # rank nodes by metric
        ranked['rank' + str(year)] = ranked[year].rank(ascending=False)
        if year == 1947:
            top_nodes = sort_values(calc[metric], reverse=True, items=False)[:n]  # top n nodes
            print('Top', metric, 'in', year, ':')
            print(pd.Series(calc[metric])[top_nodes])
        if year in [1947, 2018]:
            graph_draw(dg,
                       center = top_nodes[0],     # center graph at the top node from 1947
                       nodelist = top_nodes,      # color top n nodes from 1947
                       nodesize = calc[metric],   # node size is current year's metric value
                       title = "{} Flows (sectoring:{}) nodesize='{}'".format(year, vintage, metric))


    # Print 5 biggest changes ranks 1947-2018. Farms thru late 70's, admin & RE from mid 80's
    df = -(ranked['rank2018'] - ranked['rank1947']).sort_values()
    nodes = list(df.iloc[:n].index) + list(df.iloc[-n:].index)
    print(np.round(ranked.loc[nodes, [1947, 2018, 'rank1947','rank2018']], 4).to_string())
    df = ranked.loc[nodes, np.arange(1947, 2019)].T    # ['rank' + str(y) for y in np.arange(1947, 2019)]
    np.sqrt(df).plot(kind='line', title='sqrt of ' + metric + ' score')
    plt.show()


    # stash in util.py
    def node_distances(dg1, dg2, distance=sp.spatial.distance.euclidean):
        """return dict of each node's adjacency dissimiliarity in two graph instances"""
        nodes = list(dg1.nodes())
        a1 = nx.linalg.graphmatrix.adjacency_matrix(dg1, nodelist=nodes)
        a2 = nx.linalg.graphmatrix.adjacency_matrix(dg2, nodelist=nodes)
        return {nodes[i]: distance(a1[:,i].todense(), a2[:, i].todense())
                for i in range(len(nodes))}

    # stash in util.py
    def edge_subgraph(dg, head=None, tail=None):
        g = nx.DiGraph()
        g.add_edges_from([(e[0],e[1]) for e in dg.edges()
                          if ((tail is None) or e[1] in tail) and ((head is None) or e[0] in head)])
        return g
    
            
    """
        edges = [tuple(row) for row in data[['colcode','rowcode','datavalue']].values]
        DG = nx.DiGraph()
        DG.add_weighted_edges_from(edges)
        cent = nx.degree_centrality(DG)
        eig = nx.eigenvector_centrality(DG)
        node_size = [cent[n]*300/max(cent.values()) for n in DG.nodes()]
        nx.draw(DG, with_labels=True, style='dotted', font_size='8', font_weight='bold',
                edge_color='b', alpha=0.5, node_size=node_size,
                pos=nx.kamada_kawai_layout(DG))
        plt.show()
        
        # centrality
        indegree = nx.in_degree_centrality(DG)
        sorted([(v, k, descr[k]) for k,v in indegree.items()], reverse=False)
        eig = nx.eigenvector_centrality(DG)
        sorted([(v, k, descr[k]) for k,v in eig.items()], reverse=False)
        #katz = nx.katz_centrality(DG)
        #sorted([(v, k, descr[k]) for k,v in katz.items()], reverse=False)

        # link analysis
        (hubs, authorities) = nx.hits(DG)
        sorted([(v, k, descr[k]) for k,v in hubs.items()], reverse=False)
        sorted([(v, k, descr[k]) for k,v in authorities.items()], reverse=False)
        pagerank = nx.pagerank(DG)
        sorted([(v, k, descr[k]) for k,v in pagerank.items()], reverse=False)

        # cores
        cores = nx.algorithms.core.core_number(DG)
    
        # connectivity
        nx.is_semiconnected(DG)
        nx.is_strongly_connected(DG)
        nx.is_weakly_connected(DG)
        nx.number_attracting_components(DG)
        nx.number_strongly_connected_components(DG)
        g = sorted(nx.strongly_connected_components(DG), key=len, reverse=True)
        dg = nx.subgraph(DG, g[0])   # keep strongly connected component

        # summary statistic
        #nx.reciprocity(DG)
        #nx.average_clustering(DG)
        #nx.average_degree_connectivity(DG)
        density = nx.density(DG)

        # distances
        nx.diameter(dg)          # infinite path length when digraph is not strongly connected
        centers = nx.center(dg) #ditto
        [descr[c] for c in centers] 
        barycenter = nx.barycenter(dg)
        [descr[c] for c in barycenter]
        periphery = nx.periphery(dg)
        [descr[c] for c in periphery]

        # communities: multiple levels, but simply carve out one least connected
        girvan = list(community.girvan_newman(dg))
        """

if True: # BEA cross-momentum
    rebalbeg, rebalend = 19490101, 20190630
    naics_from_sic = Sectoring('naics', sql)   # maps from sic to naics code
    
    out = DataFrame()
    prev = 0            # previous beayear loaded
    for pordate in bd.endmo_range(rebalbeg, rebalend):
        beg = bd.endmo(pordate, -6)
        start = bd.shift(beg, 1)

        # use flows from two years prior, and prevailing sectoring scheme (e.g. bea1947, bea1963 etc)
        beayear = (pordate//10000) - 2
        if beayear != prev:
            prev = beayear
            ioUse = bea.load_ioUse(beayear)
            if beayear > 1997:
                beayear = 1997
            elif beayear > 1963:
                beayear = 1963
            else:
                beayear = 1947
            bea_from_naics = Sectoring('bea{}'.format(beayear), sql)

        # get crsp universe stocks' past return, and naics code (for bea sectoring)
        df = crsp.get_universe(pordate)
        df['weight'] = crsp.get_cap(beg).reindex(df.index)
        df['ret'] = crsp.get_ret(start, pordate).reindex(df.index)
        f = ~df['naics'].gt(0)                             # if naics missing, guess from sic code
        df.loc[f, 'naics'] = naics_from_sic.find(df.loc[f, 'siccd'])
        df['bea'] = bea_from_naics.find(df['naics'])       # assign bea code from naics
        df = df[df['weight'].gt(0) & df['ret'].notnull() & df['bea'].notnull()]  # drop missing rows

        # compute bea industry return as 'beamom'
        group = df.groupby(df['bea'])
#        df['ret'] *= df['weight']              # 'bearet' is stocks' weighted industry return
#        rets = DataFrame(group['ret'].sum() / group['weight'].sum(), columns=['bearet'])
        rets = DataFrame(group['ret'].mean()).rename(columns={'ret':'beamom'})  # eql-wtd industry return

        # merge into ioUse table: user (colcode) and maker (rowcode) industry returns
        result = ioUse.join(rets.rename(columns = {'beamom':'maker'}),
                            on='rowcode',
                            how='left')
        result = result.join(rets.rename(columns={'beamom':'user'}),
                             on='colcode',
                             how='left')
        result = result[result['user'].notnull() &
                        result['maker'].notnull() &
                        result['datavalue'].ge(0)]

        # compute flow-weighted user (sum user, by rowcode) and maker (sum maker, by colcode) momentum
        result['user'] = result['user'] * result['datavalue']
        group = result.groupby(result['rowcode'])   # groupby rowcode (maker), so colcodes are its users
        rets['user'] = group['user'].sum() / group['datavalue'].sum()
        
        result['maker'] = result['maker'] * result['datavalue']
        group = result.groupby(result['colcode'])   # groupby colcode (user), so rowcodes are its makers
        rets['maker'] = group['maker'].sum() / group['datavalue'].sum()

        # merge maker, user and own beamom industry returns into stocks df by 'bea' code
        df = df.join(rets, on='bea', how='left')

        # append DataFrame['index','date','bearet','customer','supplier']
        df['rebaldate'] = pordate
        out = out.append(df[['rebaldate','beamom','user','maker']].reset_index(), ignore_index=True)
    out = out.rename(columns={'index':'permno'})

    outdir = ''
    html = ''
    saved = dict()
    benchnames = ['Mom(mo)','Mkt-RF(mo)']
    rebalbeg, rebalend = 19490101, 20190630
    for signal in ['maker','user','beamom']:
        saved[signal] = run_backtest(backtest, crsp, signal, 0, benchnames, rebalbeg, rebalend,
                                     data=out, outdir=outdir, html=html)

