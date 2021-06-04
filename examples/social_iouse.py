"""Social Network Analysis of BEA Industries

- BEA IOUse Accounts Tables
- Social Relations Regression Model

Terence Lim
License: MIT
"""
import os
import time
import numpy as np
import numpy as np    
import numpy.ma as ma
from numpy.ma import masked_invalid as valid
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import igraph  # pip3 install cairocffi
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from finds.pyR import PyR    
from finds.busday import BusDay
from finds.database import SQL, Redis
from finds.structured import CRSP, PSTAT
from finds.sectors import Sectoring, BEA
from finds.graph import igraph_draw
from settings import settings
ECHO = True
sql = SQL(**settings['sql'])
bd = BusDay(sql)
rdb = Redis(**settings['redis'])
crsp = CRSP(sql, bd, rdb)
pstat = PSTAT(sql, bd)
bea = BEA(rdb, **settings['bea'], echo=ECHO)
logdir = None # os.path.join(settings['images'], 'bea')
years = np.arange(1947, 2020) 
vintages = [1997, 1963, 1947]   # when sectoring schemes were revised

# Read IOUse tables from BEA website
ioUses = dict()
for vintage in vintages:
    for year in [y for y in years if y >= vintage]:
        df = bea.read_ioUse(year, vintage=vintage)
        ioUses[(vintage, year)] = df
    print(f"{len(ioUses)} tables through sectoring vintage year {vintage}")
    
# Smooth average flows over several years, for social relations regression
tgt = 'colcode'   # head of edge is user industry (table column)
src = 'rowcode'   # tail of edge is maker industry (table row)
drop = ('F','T','U','V','Other')  # drop these codes
vintage = 1997    # latest sectoring scheme vintage year

# Total up production of all src->tgt; normalize; as edges in DiGraph
total = DataFrame()
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
for year in years:
    # keep year, drop invalid rows
    ioUse = ioUses[(vintage, year)]
    data = ioUse[(~ioUse['rowcode'].str.startswith(drop) &
                  ~ioUse['colcode'].str.startswith(drop))].copy()
    data = data[(data['colcode'] != data['rowcode'])]
    data['year'] = year
    total = total.append(data[['year', 'rowcode', 'colcode', 'datavalue']])
d = total.groupby(['rowcode', 'colcode'])['datavalue'].sum().reset_index()
d['weights'] = d['datavalue'] / d['datavalue'].sum()
edges = d.loc[d['weights'] > 0, [src, tgt, 'weights', 'datavalue']].values
g = Graph.TupleList(edges, edge_attrs=['weight','value'], directed=True)
g.vs['bea'] = list(BEA.bea_industry[g.vs['name']])

# visualize graph
score = g.strength(weights='weight', mode='in')
node_size = pd.Series(data=score, index=g.vs['name']).to_dict()
node_color = {k: 'green' for k in g.vs["name"]}
center_name = g.vs['name'][np.argmax(score)]
top_color = g.vs[list(np.argsort(score)[-5:])]["name"]
node_color.update({k: 'cyan' for k in top_color})
pos = igraph_draw(
    g, num=1, center_name=center_name,
    node_color=node_color, node_size=node_size, edge_color='r', k=2,
    pos=None, font_size=10, figsize=(11,12),
    labels={k:v for k,v in zip(g.vs['name'], g.vs['bea'])},
    title=f"Production Flows {list(total['year'].unique())}")
plt.show()
   
# Construct monthly BEA industry returns for the same period of years
codes = Sectoring(sql, f"bea{vintage}", fillna='')
naics = pstat.build_lookup('lpermno', 'naics', fillna=0)
caps, counts, rets = [], [], []
for year in years:
    date = bd.endyr(year - 1)
    univ = crsp.get_universe(date)
    univ['bea'] = codes[naics(univ.index, date)]
    univ = univ[univ['bea'].ne('')]
    grouped = univ.groupby('bea')
    caps.append(grouped['cap'].sum().rename(year))
    counts.append(grouped['cap'].count().rename(year))
        
    months = bd.date_range(date, bd.endyr(year), 'endmo')
    for rebaldate, end in zip(months[:-1], months[1:]):
        r = pd.concat([crsp.get_ret(bd.begmo(end), end),
                       crsp.get_cap(rebaldate, use_permco=False),
                       univ['bea']], axis=1, join='inner').dropna()
        grp = r.groupby('bea')   # industry ret is sum of weighted rets
        r['wtdret'] = r['ret'].mul(r['cap'].div(grp['cap'].transform('sum')))
        rets.append(grp['wtdret'].sum(min_count=1).rename(end))
        print(end, len(r), r['wtdret'].sum() / len(grp))

# collect and average market caps, counts and returns
caps = pd.concat(caps, axis=1).mean(axis=1)     # average cap over years
counts = pd.concat(counts, axis=1).mean(axis=1) # average count
rets = pd.concat(rets, axis=1)

# create node variables: count and cap (will take logs of)
nodevars = pd.concat([caps.rename('cap'), counts.rename('count')], axis=1)
rets = rets.T[nodevars.index]    # ensure same order of industries
n = len(nodevars.index)
nodevars.index = BEA.bea_industry[nodevars.index]

# create dyadic variables: correlations of monthly, lead and lagged returns
corrs = rets.join(rets.shift(), rsuffix='r').corr().to_numpy()
corr = corrs[:n, :n]   # corr is top-left block
lead = corrs[:n, n:]   # lead is top-right block
lag = corrs[n:, :n]    # lag is bottom-left block
np.fill_diagonal(corr, np.nan)   # drop own-correlations
np.fill_diagonal(lead, np.nan)
np.fill_diagonal(lag, np.nan)
dyadvars = np.stack((corr, lead, lag), axis=-1)


# Social Network Analysis using R and `Amen' package
# - see Hoff (2018)

# By convention: R objects to have '_ro' suffix
stats_ro    = importr('stats')
base_ro     = importr('base')
matrix_ro   = ro.r['matrix']
t_ro        = ro.r['t']
anova_ro    = ro.r['anova']
lm_ro       = ro.r['lm']
summary_ro  = ro.r['summary']
plot_ro     = ro.r['plot']
    
amen_ro     = importr('amen')
ame_ro      = ro.r['ame']  # default nscan=10000, odens=25 => 400 samples
circplot_ro = ro.r['circplot']

# Assemble flow edges
Y = PyR(g.get_adjacency('value').data, names=[list(g.vs['bea'])]*2)
Y = Y[nodevars.index, nodevars.index]
Y.assign(np.log(Y.values + 1))    # take logs of flow edges
Y[:5,:5]  # looks reasonable?

# Run ANOVA
rowsector_ro = matrix_ro(Y.rownames, Y.nrow, Y.ncol)
colsector_ro = t_ro(rowsector_ro)
formula_ro = ro.Formula("c(Y) ~ c(Rowsector) + c(Colsector)")
formula_ro.environment['Rowsector'] = rowsector_ro
formula_ro.environment['Colsector'] = colsector_ro
formula_ro.environment['Y'] = Y.ro
results_ro = anova_ro(lm_ro(formula_ro))
print(results_ro)
    
# display user effects
muhat = np.nanmean(Y.ro)
Ahat = PyR(Y.frame.mean(axis=1) - muhat, names='Ahat')
print(Ahat.frame['Ahat'].sort_values(ascending=False)[:6])

# display make effects
Bhat = PyR(Y.frame.mean(axis=0) - muhat, names='Bhat')
print(Bhat.frame['Bhat'].sort_values(ascending=False)[:6]) 

# But ignores corr of random effects, fundamental characteristic of dyads
#print(np.cov(Ahat.values, Bhat.values))
print(np.corrcoef(Ahat.values, Bhat.values)[0,1])
outer = Y.values - (muhat + np.add.outer(Ahat.values, Bhat.values))
#print(ma.cov(valid(outer.ravel()), valid(outer.T.ravel())).data)
print(ma.corrcoef(valid(outer.ravel()), valid(outer.T.ravel()))[0,1])

# Run Social Relations Model
fit_SRM_ro = ame_ro(Y.ro, plot=False, print=False)
Fit_SRM = PyR(fit_SRM_ro)
summary_ro(fit_SRM_ro)
plot_ro(fit_SRM_ro)
if logdir: PyR.savefig(os.path.join(logdir, 'srm.png'))
    
# Compare empirical and model estimates
print(muhat, np.nanmean(Fit_SRM['BETA'].values))  # overall mean
print(np.cov(Ahat.values, Bhat.values))           # mean covariances
vcmean = Fit_SRM['VC'][:, :4].frame.mean()        # posterior variance parms
print(vcmean[:3])

# Residual Dyadic Correlation
print(vcmean['cab'] / (np.sqrt(vcmean['va']) * np.sqrt(vcmean['vb'])))
print(ma.corrcoef(valid(outer.ravel()), valid(outer.T.ravel()))[0,1])    
print(np.mean(Fit_SRM['VC'][:, 3].values))

# Run Social Relations Regression Model (SSRM)
Xn = PyR(np.log(nodevars))  # nodevars are logs of caps and counts
Xd = PyR(dyadvars, names=[nodevars.index]*2 + [['corr', 'lead', 'lag']])
fit_srrm_ro = ame_ro(Y.ro, Xd=Xd.ro, Xr=Xn.ro, Xc=Xn.ro,
                     plot=False, print=False)
Fit_srrm = PyR(fit_srrm_ro)
summary_ro(fit_srrm_ro)
plot_ro(fit_srrm_ro)
if logdir: PyR.savefig(os.path.join(logdir, 'ssrm.png'))

gof = Fit_srrm['GOF'].frame.iloc[:1,:]  # actual in first row of gof
gof.loc['mean', :] = np.nanmean(Fit_srrm['GOF'].values[1:, :], axis=0)
gof.loc['std', :] = np.nanstd(Fit_srrm['GOF'].values[1:, :], axis=0)
gof

# Run OLS Model
fit_rm_ro = ame_ro(Y.ro, Xd=Xd.ro, Xr=Xn.ro, Xc=Xn.ro, print=False,
                   plot=False, rvar=False, cvar=False, dcor=False)
summary_ro(fit_rm_ro)
plot_ro(fit_rm_ro)
if logdir: PyR.savefig(os.path.join(logdir, 'ols.png'))
    
# Run SRRM with latent multiplicative effects
fit_ame2_ro = ame_ro(Y.ro, Xd=Xd.ro, Xr=Xn.ro, Xc=Xn.ro, R=2,
                     plot=False, print=False)
Fit_ame2 = PyR(fit_ame2_ro)
summary_ro(fit_ame2_ro)
plot_ro(fit_ame2_ro)
if logdir: PyR.savefig(os.path.join(logdir, 'multiplicative.png'))

# Circular plots of multiplicative factor SRRM
circplot_ro(Y.ro, U=fit_ame2_ro.rx2['U'], V=fit_ame2_ro.rx2['V'],
            row_names=Y.rownames, col_names=Y.colnames,
            plotnames=True, pscale=0.75)
if logdir: PyR.savefig(os.path.join(logdir, 'circ.png'))
