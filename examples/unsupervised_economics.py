"""Unsupervised learning models for clustering economic series

- KMeans, agglomerative, spectral clustering, nearest neighbors, PCA
- sklearn, FRED-MD

Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import re
import time
from datetime import datetime
from finds.alfred import fred_md, Alfred
from settings import settings
imgdir = os.path.join(settings['images'], 'ts')

# Load and pre-process time series from FRED
alf = Alfred(api_key=settings['fred']['api_key'])
usrec = alf('USREC', freq='m')   # to indicate recession periods in the plots
usrec.index = pd.DatetimeIndex(usrec.index.astype(str), freq='infer')
g = (usrec != usrec.shift()).cumsum()[usrec.gt(0)].to_frame()
g = g.reset_index().groupby('USREC')['date'].agg(['first','last'])
vspans = [(v[0], v[1]) for k, v in g.iterrows()]

# Retrieve FRED-MD series
mdf, mt = fred_md(202104)        # from vintage April 2020
beg = 19600301
end = 20191231 # 20191231

# Apply tcode transformations, DatetimeIndex, and sample beg:end
df = mdf
t = mt['transform']
transformed = []
for col in df.columns:
    transformed.append(alf.transform(df[col], tcode=t[col], freq='m'))
df = pd.concat(transformed, axis=1).iloc[2:]
data = df[(df.index >= beg) & (df.index <= end)].dropna(axis=1)
data.index = pd.DatetimeIndex(data.index.astype(str), freq='infer')
cols = list(data.columns)
data

## helpers to estimate explained variance
from collections import namedtuple
def lm(x, y, add_constant=True, flatten=True):
    """Calculate linear multiple regression model results as namedtuple"""
    def f(a):
        """helper to optionally flatten 1D array"""
        if not flatten or not isinstance(a, np.ndarray):
            return a
        if len(a.shape) == 1 or a.shape[1] == 1:
            return float(a) if a.shape[0] == 1 else a.flatten()
        return a.flatten() if a.shape[0] == 1 else a
    X = np.array(x)
    Y = np.array(y)
    if len(X.shape) == 1 or X.shape[0]==1:
        X = X.reshape((-1,1))
    if len(Y.shape) == 1 or Y.shape[0]==1:
        Y = Y.reshape((-1,1))
    if add_constant:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    b = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    out = {'coefficients': f(b)}
    out['fitted'] = f(X @ b)
    out['residuals'] = f(Y) - out['fitted']
    out['rsq'] = f(np.var(out['fitted'], axis=0)) / f(np.var(Y, axis=0))
    out['rvalue'] = f(np.sqrt(out['rsq']))
    out['stderr'] = f(np.std(out['residuals'], axis=0))
    return namedtuple('LinearModel', out.keys())(**out)

def cluster_rsq(y, centers, labels=None):
    """Calculate within-cluster average rsquared of y to its labelled center"""
    if labels is None:
        return np.mean(lm(y=y.T, x=centers.T).rsq)
    labels = labels.flatten()
    return np.sum([np.sum(lm(y=y.T[:, labels == p], x=centers.T[:,p]).rsq)
                   for p in np.unique(labels)]) / len(labels)

from sklearn.neighbors import NearestNeighbors
def print_nearest(centers, neighbors, labels, n_neighbors=3):
    """Display each center's nearest labeled neighbors, and variance explained"""
    
    c = centers / centers.std(ddof=0, axis=1)[:, None]   # standardize centers
    l = labels.flatten()
    neigh = NearestNeighbors(n_neighbors, radius=None, algorithm='brute')
    for k in range(c.shape[0]):
        # find nearest labelled neighbors of center[k]
        sz = np.sum(l == k)
        r2 = cluster_rsq(neighbors[l==k], c[[k]])
        print(f"Cluster:{k:2d}  size:{sz}  R2:{r2:.3f}")
        if n_neighbors:
            neigh.fit(neighbors[l==k])
            dist, nearest = neigh.kneighbors(c[[k]], min(sz, n_neighbors),
                                             return_distance=True)
            for arg in nearest[0]:
                col = np.array(cols)[l==k][arg]
                print(f"{col:16s} {alf.header(col)}")
            print()
    print()


# Clustering economic variables
"""
- input y shall have shape (n_samples=variables, n_features=monthly values)
- StandardScaler standardizes by column: note we want to normalize each variable
  hence first apply StandardScaler to (monthly values, variables), then tranpose
"""
from sklearn.preprocessing import StandardScaler
y = StandardScaler().fit_transform(np.asarray(data)).T
within_rsq = dict()    # collect average within-cluster rsquareds
total_rsq = dict()     # collect total rsquareds
max_clusters = 8

#
# KMeans clustering with MiniBatch
#
from sklearn.cluster import MiniBatchKMeans, KMeans

modelname = 'KMeans'
rsq, ttl = dict(), dict()
for n_clusters in range(2, max_clusters+1):
#    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=6, max_iter=100)
    km = KMeans(n_clusters=n_clusters, max_iter=100)    
    best = 0.0
    for trial in range(10):  # kmeans may get stuck in local optimums
        km.fit(y)
        r = cluster_rsq(y, km.cluster_centers_, km.labels_)
        if r > best:
            best = r
            kmeans = deepcopy(km)
            rsq[n_clusters] = best
            ttl[n_clusters] = lm(y=y.T, x=km.cluster_centers_.T).rsq.mean()
within_rsq[modelname] = rsq
total_rsq[modelname] = ttl

# show clusters
print(f"Model [{modelname}] Average Within-clusters Rsquare:",
      f"{cluster_rsq(y, kmeans.cluster_centers_, kmeans.labels_):.3f}")
print(f"Total Rsquare: {lm(y=y.T, x=kmeans.cluster_centers_.T).rsq.mean():.3f}")
print_nearest(kmeans.cluster_centers_, neighbors=y, labels=kmeans.labels_)

#
# PCA components as cluster centers
#
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

modelname = 'PCA'
rsq, ttl = dict(), dict()
for n_clusters in range(2, max_clusters+1):
    pca = PCA(n_components=n_clusters).fit(y)
    
    ## assign label as nearest center
    centers = pca.components_ / pca.components_.std(ddof=0, axis=1)[:, None]
    neigh = NearestNeighbors(1, algorithm='brute', radius=None).fit(centers)
    dists, labels = neigh.kneighbors(y, return_distance=True)
    
    rsq[n_clusters] = cluster_rsq(y, centers, labels)
    ttl[n_clusters] = lm(y=y.T, x=centers.T).rsq.mean()
within_rsq[modelname] = rsq
total_rsq[modelname] = ttl

# show clusters
print(f"Model [{modelname}] Average Within-cluster Rsquare:",
      f"{cluster_rsq(y, centers, labels):.03f}")
print(f"Total Rsquare: {lm(y=y.T, x=centers.T).rsq.mean():.3f}")
print_nearest(centers, neighbors=y, labels=labels)

#
# Spectral clustering
#
from sklearn.cluster import SpectralClustering

modelname = 'Spectral'
rsq, ttl = dict(), dict()
for n_clusters in range(2, max_clusters+1):
    spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
                                  affinity="nearest_neighbors").fit(y)
    
    # assign centers as medians of each feature (monthly values)
    centers = np.hstack([StandardScaler()\
                         .fit_transform(np.median(y[spectral.labels_ == c],
                                                  axis=0).reshape(-1,1))
                         for c in range(max(spectral.labels_+1))]).T
    
    rsq[n_clusters] = cluster_rsq(y, centers, spectral.labels_)
    ttl[n_clusters] = lm(y=y.T, x=centers.T).rsq.mean()
within_rsq[modelname] = rsq
total_rsq[modelname] = ttl

# show clusters
print(f"Model [{modelname}] Average Within-cluster Rsquare:",
      f"{cluster_rsq(y, centers, spectral.labels_):.03f}")
print(f"Total Rsquare: {lm(y=y.T, x=centers.T).rsq.mean():.3f}")
print_nearest(centers, neighbors=y, labels=spectral.labels_)

#
# Ward Hierarchical Clustering
#
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
connectivity = kneighbors_graph(y, n_neighbors=len(y)//2, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)

from sklearn.cluster import SpectralClustering

modelname = 'Ward'
rsq, ttl = dict(), dict()
for n_clusters in range(2, max_clusters+1):
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                                   connectivity=connectivity).fit(y)

    # assign centers as means of each feature (monthly values)
    centers = np.hstack([StandardScaler()\
                         .fit_transform(np.mean(y[ward.labels_ == c],
                                                axis=0).reshape(-1,1))
                     for c in range(max(ward.labels_+1))]).T

    rsq[n_clusters] = cluster_rsq(y, centers, ward.labels_)
    ttl[n_clusters] = lm(y=y.T, x=centers.T).rsq.mean()
within_rsq[modelname] = rsq
total_rsq[modelname] = ttl

# show clusters
print(f"Model [{modelname}] Average Within-cluster Rsquare:",
      f"{cluster_rsq(y, centers, ward.labels_):.03f}")
print(f"Total Rsquare: {lm(y=y.T, x=centers.T).rsq.mean():.3f}")
print_nearest(centers, neighbors=y, labels=ward.labels_)

#
# AverageLinkage Hierarchical Clustering
# with cityblock distance
#
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
connectivity = kneighbors_graph(y, n_neighbors=len(y)//2, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)

modelname = 'AvgLink'
rsq, ttl = dict(), dict()
for n_clusters in range(2, max_clusters+1):
    avglink = AgglomerativeClustering(linkage="average", affinity="cityblock",
                                      n_clusters=n_clusters,
                                      connectivity=connectivity).fit(y)

    # assign centers as means of each feature (monthly values)
    centers = np.hstack([StandardScaler()\
                         .fit_transform(np.mean(y[avglink.labels_ == c],
                                                axis=0).reshape(-1,1))
                     for c in range(max(avglink.labels_+1))]).T

    rsq[n_clusters] = cluster_rsq(y, centers, avglink.labels_)
    ttl[n_clusters] = lm(y=y.T, x=centers.T).rsq.mean()
within_rsq[modelname] = rsq
total_rsq[modelname] = ttl

# show clusters
# PCA seeks to maximize total rsquare
# KMeans (with Euclidean distance) seeks to maximize within-cluster rsquare
# Note for the Average Linking method, we used Manhattan rather than Euclidean distances, 
# hence appears poorer through ``variance explained'' (or Euclidean) metrics
print(f"Model [{modelname}] Average Within-cluster Rsquare:",
      f"{cluster_rsq(y, centers, avglink.labels_):.03f}")
print(f"Total Rsquare: {lm(y=y.T, x=centers.T).rsq.mean():.3f}")
print_nearest(centers, neighbors=y, labels=avglink.labels_)

## Plot within-cluster and total variance explained
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,5), num=1, clear=True)
for v, title, ax in zip([DataFrame(within_rsq), DataFrame(total_rsq)],
                        ['Average Within-Cluster Variance Explained',
                         'Total Variance Explained'], axes.ravel()):
    v.plot(ax=ax, style='-', rot=0)
    ax.set_title(title, {'fontsize':9})
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'clusters.jpg'))
plt.show()
