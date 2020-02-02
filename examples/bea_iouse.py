"""
BEA Input-Output Use Tables
Network Science
Lead-lag cross-momentum

References:

https://www.sci.unich.it/~francesc/teaching/network/

Choi and Foerster, 2017, "The Changing Input-Output Network Structure of the U.S. Economy", Federal Reserve Bank of Kansas City

Menzly, Lior, and Oguzhan Ozbas, 2010, Market segmentation and cross-predictability of returns, Journal of Finance 65, 1555–1580.
"""
# The MIT License
#
# Copyright (c) 2020 Terence Lim (https://terence-lim.github.io/)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation he rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

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
import pandas as pd
from pprint import pprint

import importlib
importlib.reload(dives)
importlib.reload(dives.dbengines)
importlib.reload(dives.econs)
importlib.reload(dives.util)
importlib.reload(dives.structured)
importlib.reload(dives.evaluate)

from dives.util import DataFrame, NamedDict, sort_values, graph_calc, graph_draw
from dives.econs import Sectoring, BEA
from dives.dbengines import SQL, Redis
from dives.structured import BusDates, Benchmarks, CRSP, Signals
from dives.evaluate import BackTest, run_backtest

import secret   # passwords etc
sql = SQL(**secret.value('sql'))
rdb = Redis(**secret.value('redis'))
bea = BEA(**secret.value('bea'), rdb=rdb)
bd = BusDates(sql)
signals = Signals(sql)
bench = Benchmarks(sql, bd)
crsp = CRSP(sql, bd, rdb)
backtest = BackTest(sql, bench, 'RF')

if False:
    ### Read industry names and definitions
    gdpindustry = bea.get(**bea.get_params('gdpindustry'))    # get industry description from BEA
    desc = gdpindustry.to_dict()['desc']                      # put into a dict()
    descr = {bea.get_label(k) : (desc[k] if k in desc else bea.get_label(k))
             for k in bea.get_label()}                        # assign description to alternative labels

    ### grab all years ioUse table from BEA website, and transform by vintage sectoring schemes
    ioUses = NamedDict(['vintage','year'])
    years = np.arange(1947, 2019) 
    vintages = [1997, 1963, 1947]   # when sectoring scheme revised
    for vintage in vintages:
        for year in [y for y in years if y >= vintage]:
            df = ioUse=bea.load_ioUse(year, vintage=vintage)
            ioUses.replace(vintage=vintage, year=year, ioUse=df)
        print('For {} sectoring: {} years were read'.format(
            vintage, len(ioUses.search(vintage=vintage))))

    ### Helper function to sum ioUse flows by industry
    def ioUse_sum(ioUse, drop = ('F','T','U','V','Other'), by='rowcode',
                  exclude_self=True, get_label=bea.get_label):
        """return networkx DiGraph instance with directed edges from table

        Parameters
        ----------
        ioUse : DataFrame
            stacked flows in column ['datavalue'], maker industry in ['rowcode'], user in ['colcode']
        drop : list of str, optional (default is ('F','T','U','V','Other'))
            do not sum over these industries
        head : string, optional (default is 'rowcode')
            directed edge head ends at this column: 'rowcode' (default) for maker
        tail : string, optional (default is 'colcode')
            directed edge tail starts from this column: 'colcode' (default) for user
        normalize : string, optional (default is None):
            column name to sum by as denominator to normalize flows: 
        exclude_self : bool, optional (default True)
            whether to exclude self-flows in sum
        get_label : function, optional (default is bea.get_label)
            function that returns node label name, set to None to use original codes
        """
        data = ioUse[(~ioUse['colcode'].str.startswith(drop) &
                      ~ioUse['rowcode'].str.startswith(drop))]       # codes to drop
        if exclude_self:                                             # exclude self loops
            data = data[(data['rowcode'] != data['colcode'])]
        df = pd.Series(data.groupby([by])['datavalue'].sum())      # total sum by industry
        if get_label:
            df.index = [get_label(x) for x in df.index]
        return df

    ### Helper function convert ioUse flows to directed networkx graph
    def ioUse_graph(ioUse, drop = ('F','T','U','V','Other'), head = 'rowcode', tail = 'colcode', 
                    get_label = bea.get_label, exclude_self = True, normalize  = None):
        """total flows by user- or maker-industry

        Parameters
        ----------
        ioUse : DataFrame
            stacked flows in column ['datavalue'], maker industry in ['rowcode'], user in ['colcode']
        drop : list of str, optional (default is ('F','T','U','V','Other'))
            do not sum over industry codes starting with these strings
        by : string, optional (default 'rowcode')
            'rowcode' (default) to sum by maker industry. 'colcode' ro sum by user industry
        exclude_self : bool, optional (default True)
            whether to exclude self-flows in sum
        get_label : function, optional (default is bea.get_label)
            function that returns node label name, set to None to use original codes
        """
        data = ioUse[(~ioUse[tail].str.startswith(drop) &
                      ~ioUse[head].str.startswith(drop))].copy()  # drop these industries
        if exclude_self:                                          # exclude self loops
            data = data[(data[tail] != data[head])]
        if normalize:
            group = data.groupby([normalize])
            data['weights'] = data['datavalue'] / group['datavalue'].transform('sum')
        else:
            data['weights'] = data['datavalue'] / data['datavalue'].sum()
        if get_label is None:
            get_label = lambda x : x
        edges = [(get_label(c), get_label(r), w) for c,r,w in data[[tail, head, 'weights']].values]
        DG = nx.DiGraph()
        DG.add_weighted_edges_from(edges)
        if exclude_self:        # redundant given earlier condition code checked already
            DG.remove_edges_from(list(nx.selfloop_edges(DG)))      # to remove self loops
        return DG

    ### Summarize flows and centralities for 2018, using latest sectoring scheme
    df = ioUses.match(year = 2018, vintage = 1997)['ioUse']
    dg = ioUse_graph(df, head='rowcode', tail='colcode', normalize='')    # make directed network graph
    calc = graph_calc(dg)     # calculate node centrality statistics
    flows = DataFrame()
    flows['user']  = ioUse_sum(df, by='colcode', exclude_self=True)
    flows['maker'] = ioUse_sum(df, by='rowcode', exclude_self=True)
    flows['self']  = ioUse_sum(df, by='rowcode', exclude_self=False) - flows['maker']
    for metric in calc.keys():
        flows[metric] = pd.Series(calc[metric])
    flows[np.isnan(flows)] = 0
    print(flows.to_string())

    ### Although positive, centrality measures will not be perfectly correlated
    print(np.round(flows.corr(),2))  
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

    ### Print top 3 authorities of 1947, and display in 1947 and 2018 graphs, using 1947 scheme
    vintage = 1947          # sectoring scheme fixed at year 1947
    metric = 'authorities'     #'authorities'  # rank by this metric
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


    ### Print 3 biggest change in relative ranks 1947-2018.
    """Peak is 2006, then several years of variability through 2011"""
    df = -(ranked['rank2018'] - ranked['rank1947']).sort_values()
    nodes = list(df.iloc[:n].index) #+ list(df.iloc[-n:].index)
    print(np.round(ranked.loc[nodes, [1947, 2018, 'rank1947','rank2018']], 4).to_string())
    df = ranked.loc[nodes, np.arange(1947, 2019)].T    # ['rank' + str(y) for y in np.arange(1947, 2019)]
    df.plot(kind='line', title=metric + ' score', grid=True)
    plt.show()

    
            
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

if False: # BEA cross-momentum
    ### Load stock returns, and compute BEA industry own momentum and cross-momentum
    """cross-momentum is flow-weighted average of user (or maker) industry returns last month"""
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
        print('{} rows for {}'.format(len(df), pordate))
    out = out.rename(columns={'index':'permno'})
    print('{} unique dates, {} records'.format(len(np.unique(out['rebaldate'])), len(out)))

    ### Run backtests with the three forms of industry momentum
    outdir = ''
    html = ''
    saved = dict()
    benchnames = ['Mom(mo)','Mkt-RF(mo)']
    rebalbeg, rebalend = 19490101, 20190630
    for signal in ['maker','user','beamom']:
        saved[signal] = run_backtest(backtest, crsp, signal, 0, benchnames, rebalbeg, rebalend,
                                     data=out, outdir=outdir, html=html)
