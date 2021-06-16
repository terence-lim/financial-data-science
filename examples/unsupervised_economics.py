"""Unsupervised learning: clustering and outlier detection for economic series

- KMeans, agglomerative, spectral clustering, nearest neighbors, PCA
- isolated forest, minimum covariance determinant, local outlier factor
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
from finds.solve import lm, is_inlier
from settings import settings
imgdir = os.path.join(settings['images'], 'ts')

# Load and pre-process time series from FRED
alf = Alfred(api_key=settings['fred']['api_key'])
usrec = alf('USREC', freq='m')   # to indicate recession periods in the plots
usrec.index = pd.DatetimeIndex(usrec.index.astype(str), freq='infer')

g = usrec.astype(bool) | usrec.shift(-1, fill_value=0).astype(bool)
g = (g != g.shift(fill_value=0)).cumsum()[g].to_frame()
g = g.reset_index().groupby('USREC')['date'].agg(['first','last'])
vspans = [(v[0], v[1]) for k, v in g.iterrows()]

# Retrieve FRED-MD series
mdf, mt = fred_md(202104)        # from vintage April 2020
beg = 19600301
end = 20201231 # 20191231

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

# Outlier Detection: single time series
"""
- 10iq (interquartile range), or other multiple, around the median
- Tukey proposed 1.5iq, and 3iq for farout, beyond 1Q and 3Q values
"""
outliers = DataFrame({method: (~is_inlier(data, method=method)).mean(axis=1)
                      for method in ['farout', 'iq10']}, index=data.index)

print("Months with most data points identified as time-series outliers")
for label in outliers.columns:
    print(outliers[[label]].sort_values(by=label, ascending=False)[:5].T) 
    print()


# Outlier Detection -- multidimensional features
""" 
Isolation Forest
- Isolates observations by randomly selecting a feature and then
  randomly selecting a split value: normality is measured by the
  number of splittings required to isolate a sample -- single feature

Minimum Covariance Determinant (MCD)
- Assume that the regular data come from a known distribution, such as
  Gaussian, define outlying observations which stand far enough from
  the fit shape - global multidimensional

Local Outlier Factor (LOF) 
- Computes a score reflecting the local density deviation of a given
  data point with respect to its neighbors, as the measure of
  abnormality -- local multidimensional
"""
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detects(label=None, frac=None):
    """list of outlier detection algorithms to select from"""
    algo = {'isoForest': IsolationForest(contamination=frac, random_state=42),
            'MCD': EllipticEnvelope(contamination=frac),
            'LOF': LocalOutlierFactor(n_neighbors=20, contamination=frac)}
    return list(algo.keys()) if label is None else algo[label]

## Compute and store detections
spans = dict() 
for label in detects():
    spans[label] = dict()
    tic = time.time()
    for frac in [0.005, 0.02]:  # specify contamination fractions
        y = detects(label, frac).fit_predict(data)
        spans[label][frac] = [(d - pd.DateOffset(months=1), d)
                              for d in data.index[y<0]]
    print(f"{label}: {time.time()-tic:.0f} secs elapsed")

## Display outlier regions
fig, axes = plt.subplots(nrows=3, ncols=1, num=1, clear=True, figsize=(9,10))
for (label, span), ax in zip(spans.items(), axes):
    outliers.plot(ax=ax)
    for i, frac in enumerate(sorted(span.keys(), reverse=True)):
        for a,b in span[frac]:
            ax.axvspan(a, b, alpha=0.2 + 0.6*i, color='m')
    ax.legend(['% ' + c for c in outliers.columns] + [label],
              loc='upper left')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"Outlier Detection of Monthly Observations")
plt.savefig(os.path.join(imgdir, 'anomaly.jpg'))
plt.show()


# Clustering economic series
"""
- input y should have shape (n_samples=variables, n_features=monthly values)
- StandardScaler standardizes by column: but we want to normalize by variable
  hence first apply StandardScaler to (monthly values, variables), then tranpose
"""
from sklearn.preprocessing import StandardScaler
y = StandardScaler().fit_transform(np.asarray(data)).T
within_rsq = dict()    # collect average within-cluster rsquareds
total_rsq = dict()     # collect total rsquareds
max_clusters = 8

# helpers to show explained variance
def cluster_rsq(y, centers, labels=None):
    """Calculate within-cluster average rsquared of y to its labelled center"""
    if labels is None:
        return np.mean(lm(y=y.T, x=centers.T).rsq)
    labels = labels.flatten()
    return np.sum([np.sum(lm(y=y.T[:, labels == p], x=centers.T[:,p]).rsq)
                   for p in np.unique(labels)]) / len(labels)

from sklearn.neighbors import NearestNeighbors
def print_nearest(centers, neighbors, labels, n_neighbors=3):
    """Display each center's labeled neighbors, and variance explained"""
    
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


## KMeans clustering - default initialization is k-means++

from sklearn.cluster import KMeans

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


## PCA components as cluster centers

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


## Spectral clustering

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


## Ward Hierarchical Clustering

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


## AverageLinkage Hierarchical Clustering

# with cityblock distance
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
print(f"Model [{modelname}] Average Within-cluster Rsquare:",
      f"{cluster_rsq(y, centers, avglink.labels_):.03f}")
print(f"Total Rsquare: {lm(y=y.T, x=centers.T).rsq.mean():.3f}")
print_nearest(centers, neighbors=y, labels=avglink.labels_)


## Plot within-cluster and total variance explained
"""
- PCA seeks to maximize total rsquare

- KMeans (with Euclidean distance) seeks to maximize within-cluster rsquare

- Average Linking method used Manhattan rather than Euclidean distances, hence 
  appears poorer through ``variance explained'' (or Euclidean) metrics
"""
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,5), num=1, clear=True)
for v, title, ax in zip([DataFrame(within_rsq), DataFrame(total_rsq)],
                        ['Average Within-Cluster Variance Explained',
                         'Total Variance Explained'], axes.ravel()):
    v.plot(ax=ax, style='-', rot=0)
    ax.set_xlabel('number of clusters')
    ax.set_xlabel('in-sample variance explained')
    ax.set_title(title, {'fontsize':9})
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'clusters.jpg'))
plt.show()
