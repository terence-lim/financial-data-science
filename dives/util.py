"""
the dives.util module is a small collection of useful functions and classes
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

import pandas as pd
import numpy as np
import scipy as sp
from wordcloud import WordCloud, STOPWORDS
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  # for date formatting in plots

try:
    import secret
    verbose = secret.value('verbose')   # pre-set value for verbose to print_debug
except:
    verbose = 0
    
def print_debug(q, verbose=verbose):
    """Print if verbose"""
    if (verbose):
        print('debug:', q)
    
def as_print(c):
    """Convert all not printable characters in string to space"""
    return re.sub(r'[^\x20-\x7E]',r' ', c)

def sort_values(x, reverse=False, items=True):
    """sort dictionary by its values

    Parameters
    ----------
    x : dict
        dictionary to sort by value
    reverse : boolean, optional (default False)
        sort order.  True to reverse sort in descending order
    items : boolean, optional (default True)
        set to False to return a list of key values only

    Return
    ------
    tuples : list of tuples
        sorted list of (key, value) tuples
    """
    return [(k,v) if items else k
            for k, v in sorted(x.items(), key = lambda item: item[1], reverse=reverse)]


def graph_draw(DG, out=None, figsize=(11,11), nodesize=None, labels=None,
               center=None, nodelist = [], title = '', **kwargs):
    """wrapper for networkx.draw

    Parameters
    ----------
    DG : DiGraph instance
        directed graph to draw
    out : string, optional (default None)
        jpg filename to save as
    figsize : tuple of (int, int), default is (11,11)
        plt figure size in inches
    nodesize : dict (default is None)
        dictionary to look up node label for its plot size
    labels : dict (default is None)
        dictionary to look up nodel label for its display string
    center : node
        label of node to place in center of plot
    nodelist : list of nodes
        list of nodes to color red
    title : string
        text to display in top left of plot
    **kwargs : parameters
        passed on to networkx.draw

    Notes
    -----
    spring_layout: see help(nx.draw_networkx) or help(nx.drawing.layout) for 
      circular_layout kamada_kawai_layout spectral_layout
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


def graph_calc(dg):
    """Calculate static graph statistics: centrality, links analysis"""
    (hubs, authorities) = nx.hits(dg, max_iter=1000)
    return {'hubs' : hubs,
            'authorities' : authorities,
            'indegree' : nx.in_degree_centrality(dg),
            'pagerank' : nx.pagerank(dg, max_iter=1000),
            'inweight' : {k:v for k,v in dg.in_degree(weight='weight')},
    }
#            'density': nx.density(dg)}

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
    """retrieves subgraph of edges with {head} and {tail} in respective argument lists"""
    g = nx.DiGraph()
    g.add_edges_from([(e[0],e[1]) for e in dg.edges()
                      if ((tail is None) or e[1] in tail) and ((head is None) or e[0] in head)])
    return g

def wordcloud_features(coefs, n, feature_names, class_names=None, unique=False,
                       plot=True, verbose=False):
    """extract top features for each class from fitted components, plot wordcloud

    Parameters
    ----------
    coefs : iterable
        feature scores for each class of fitted model
    n : int
        max number of top features
    feature_names : list of strings
        names of each feature
    class_names : list of strings, optional (default = None)
        printable names of the classes
    unique : boolean, optional (default = False)
        whether to keep only unique features not in other class's top features
    plot : boolean, optional (default = True)
        whether to display word clouds of top features
    verbose : boolean, optional (default = False)
        whether to print top features names and scores

    Returns
    -------
    out : dict() of (topic, features) tuple
        top features, as sorted tuples of (feature_name, coefficient), for each topic
    """
    # generate list of topic top features: coef (i.e. modelcompoents_) is iterable over topics
    features = [{feature_names[c] : coef[c] for c in np.argsort(coef)[-n:]}  # 
                for coef in coefs]

    # display features for each topic
    out = dict()
    for topic_idx, topic_features in enumerate(features):

        # select features for this topic that are not in other topics
        other_idx = set(np.arange(len(features))) - set([topic_idx])  # indexes of the other topics
        other_features = [y for x in [features[i] for i in other_idx] for y in x]
        words = {feature : score
                 for feature, score in topic_features.items()
                 if feature not in other_features}
        topic = class_names[topic_idx] if class_names else str(topic_idx)
        
        # display as word cloud and sorted list (sort_values)
        plt.figure()
        plt.imshow(WordCloud(stopwords = STOPWORDS).generate_from_frequencies(words))
        plt.axis("off")
        plt.title('Topic: {}'.format(topic))
        out[topic] = sort_values(words, reverse=True)
        if verbose:
            print('Topic ({}) : top {}features:'.format(topic, 'unique ' if unique else ''))
            pprint(out[topic])
    return out

def fractiles(values, pctiles, keys = None):
    """Return the fractile indices that values assigned into

    Parameters
    ----------
    values : array_like
      Input array
    pctiles : array_like
      Percentile values to determine breakpoint values
    keys : array_like, optional
      Optional alternate values to determine percentile breakpoints from

    Returns
    -------
    fractiles : list of ints
      list of fractile assignments {1,.., len(pctiles)} s.t. value <= lowest pctile

    Examples:
    ---------
    """
    if keys is None:
        keys = values
    keys = np.array(keys)[~np.isnan(keys)]
    bp = list(np.percentile(keys, sorted(pctiles))) + [np.inf]
    return 1 + np.searchsorted(bp, values, side='right')


def isnum(d):
    """return boolean whether is int or not"""
    try:
        int(d)
        return True
    except:
        return False

def plot_wordcloud(coef, feature_names, title = None, top_features=10):
    """plot wordcloud of top features"""
    top_coefficients = np.argsort(-coef)[:top_features]
    words = {feature_names[c] : coef[c] for c in top_coefficients}
    wordcloud = WordCloud(stopwords = STOPWORDS).generate_from_frequencies(words)
    #    wordcloud = WordCloud(stopwords = STOPWORDS).generate(text)
    plt.clf()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(title)
    return words


class NamedDict(dict):
    """Subclass extends dict, to allow for named, multiple keys

    Parameters:
    ----------
    keys : list of string
        names to identify the multiple fields that form a dict key tuple
    values: list of dict (optional, default is [])
        initial members of the NamedDict. 
        Each item is a dict: items that are in key fields will be used to construct the key tuple,
          items that are not will be subset as its value in the NamedDict.

    Notes:
    -----
    Useful, say, for slotting in and accumulating labeled results, keyed by experiment meta detail

    Examples:
    --------
    box = NamedDict(keys=['permno','date'],
                    values=[{'permno':1, 'date':2, 'a':3},
                            {'permno':4, 'date':'d', 'a':'b','c':5}])
    box.match(permno=1, date=2)             # exact match returns item
    box.match(permno=4, date='d')           # exact match returns item
    box.match(permno=4, date='e')           # exact match fails to return any item
    box.replace(permno=4, date='d', a='e')  # update the item
    box.search(permno=1)                    # list of items where required fields match a key
    box.search()                            # list all items (no fields required means always True)
    """
    def __init__(self, keys, values=[]):
        self._keys = keys
        for value in values:
            self[self._getkey(**value)] = self._getvalue(**value)
        super().__init__()
        
    def _getkey(self, **keys):
        """helper method to construct key tuple from its arguments"""
        return tuple(keys[k] if k in keys else None for k in self._keys)
    
    def _getvalue(self, **values):
        """helper method to subset the value item, as a dict, from its arguments"""
        return {k: values[k] for k in values if k not in self._keys}

    def _matchkey(self, key, **keys):
        """helper method to check if any fields in {keys} do not match those in {key} tuple"""
        for k,v in keys.items():
            if key[self._keys.index(k)] != v:
                return False
        return True

    def key(self, key_tuple):
        """transform a key tuple to a dict with key field name"""
        return {k : v for k,v in zip(self._keys, key_tuple)}

    def replace(self, **items):
        """Update item in NamedDict.
        If any keys field is not specified, it is assigned value of None in the key tuple.
        Other fields are updated or added to that key tuple's value in the NamedDict
        """
        key = self._getkey(**items)
        if key not in self:
            self[key] = dict()
        self[key].update(self._getvalue(**items))
        return self[key]
    
    def match(self, **keys):
        """Return the single match, with exact key field values, from NamedDict"""
        return self.get(self._getkey(**keys), None)
    
    def remove(self, **keys):
        """Remove the single match, with exact key field values, from NamedDict"""
        key = self._getkey(**keys)
        return self.pop(key) if key in self else None
    
    def search(self, **keys):
        """Return set of items with key tuple fields that match any specified in argument"""
        return [{**{k:v for k,v in zip(self._keys, key)},
                 **{k:v for k,v in value.items()}}
                for key,value in self.items() if self._matchkey(key, **keys)]



def df_to_numeric(df):
    """Convert all columns of type object to float"""
    for col in df.columns:
        if df[col].dtype == np.dtype('O'):
            df[col] = df[col].astype(float)
    return df

def df_lags(curr, var, key, nlags):
    """Return dataframe with {nlags} of a column {var}, requiring rows have same value of {key}"""
    out = curr[[var]].rename(columns = {var : 0})      # set first column to be unshifted by i=0
    for i in range(1,nlags):
        prev = curr[[key, var]].shift(i, fill_value=0) # set next column to be shifted by i=i+1
        prev.loc[prev[key] != curr[key], :] = np.nan   # require shifted value of {key} has same value
        out.insert(i, i, prev[var])
    return out

def df_hqueue(left, right, width=0, dropna=True):
    """Outer join left and right on index, drop last cols if wider than width, drop empty rows"""
    df = left.join(right, how='outer', sort=True, lsuffix='l', rsuffix='r')
    if width and len(df.columns) > width:          # drop last column if wider than width
        df = df.iloc[:, (len(df.columns)-width):]
    if dropna:                                     # drop empty rows
        df = df.drop(index = df.index[np.sum(df.isnull(),1) == len(df.columns)])
    return df
    
def standardize(x):
    """subtract mean and divide by std, ignore nan's"""
    return (x - np.nanmean(x)) / np.nanstd(x)
    
def winsorize(x, lo = 0.05, hi =0.05):
    """trims bottom {lo} and top {hi} fractions, ignoring NaN's
    Note: sp.stats.mstats.winsorize which handles nan's incorrectly"""
    q = np.nanpercentile(x, [100*lo, 100*(1-hi)], interpolation = 'lower')
    return np.where(x <= q[0], q[0], np.where(x > q[1], q[1], x))

def add_constant(x, n=None):
    if not n:
        n = len(x)
    return np.concatenate((np.array(x).reshape(n, -1), np.ones(n, 1)), axis=1)

def orthogonalize(y, x, fit_intercept = True):
    """orthogonalize vector {y} on {x} and intercept (if {fit_intercep}), and handles NaN's"""
    if fit_intercept:
        xx = add_constant(x, len(y))
    else:
        xx = x.reshape(len(y), -1)
    yy = np.array(y).reshape(-1, 1)
    f = ~np.isnan(yy).all(axis=1) & ~np.isnan(xx).all(axis=1)   # require non-missing samples
    A = xx[f, :]
    b = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(yy[f,:]))
    return y - xx.dot(b).reshape(y.shape)
    
class DataFrame(pd.DataFrame):
    """Subclass extends pandas.DataFrame class with additional useful methods."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def _constructor(self):
        """to fix __finalized__ issue, see
        https://github.com/pandas-dev/pandas/issues/13208#issuecomment-326556232
        """
        def _c(*args, **kwargs):
            return DataFrame(*args, **kwargs).__finalize__(self)
        return _c

    def standardize(self):
        """standardize all columns"""
        return self.apply(standardize)
    
    def winsorize(self, hi=0.05, lo=0.05):
        """trim each column at respective top {hi} and bottom {lo} fractional values"""
        return self.apply(standardize, hi=hi, lo=lo)


    def orthogonalize(self, x, fit_intercept = True):
        """orthogonalize each column on {x} and intercept (if {fit_intercept}), and handles NaN's"""
        if fit_intercept:
            xx = add_constant(x, len(self))
        else:
            xx = x.reshape(len(self), -1)
        return self.apply(orthogonalize, x=xx, fit_intercept = False)


    def to_numeric(self, inplace=False, **kwargs):
        """Convert all columns of type object to numeric."""
        return df_to_numeric(self if inplace else self.copy())
    
    def lags(self, var, key, nlags):
        """Return new dataframe with multiple lags of a column.

        Parameters
        ----------
        var : string
            name of column (values in ascending time sequence) to take lags of 
        key : string
            name of column with key value that divides sub-blocks (None if single block)
        nlags : int
            number of lags

        Returns
        -------
        new : DataFrame
            {nlags} of the column {var}, requiring rows have same value of {key}
        """
        return df_lags(self, var, key, nlags)

    def hqueue(self, right, width=0, dropna=True):
        """Outer join {right} dataframe on indexes, drop first cols if wider than {width}.

        Parameters
        ----------
        right : DataFrame
            horizontally stack columns from this DataFrame
        width: int, optional
            total number of columns to keep (default 0 or None to keep all)
        dropna = bool, optional
            keep rows only if all are notnull (default True)

        Returns
        -------
        new : DataFrame
            outer-joined with {right} dataframe. 
            leftmost columns of self are dropped if exceeds {width} (if width is set)
            rows are dropped if all are null, if dropna is True (default)
        """
        return df_hqueue(self, right, width, dropna)

if __name__ == '__main__':
    print('util:__main__')
    
"""
pd.api.types.is_list_like()
pd.api.types.is_scalar
pd.api.types.is_numeric_dtype(['1'])
"""
