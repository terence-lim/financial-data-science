"""Wrapper class over rpy2 package to interface with R environment

Deconstruct and expose an rpy2 or numpy/pandas object interchangeably.

- rpy2

Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
from pandas import DataFrame, Series
from pandas.api import types
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rpy2.robjects as ro
from rpy2.robjects import FloatVector, ListVector, IntVector, StrVector, NULL
from typing import List, Tuple, Any

def StrListVector(strList: ListVector | StrVector
                  | str | List) -> StrVector | ListVector:
    """Convert nested list input to StrVector or ListVector"""
    try:
        assert(len(strList) > 0)   # NULL, None, '', non-str scalar  etc
    except:
        return NULL
    if isinstance(strList, ListVector):   # already a ListVector
        return ListVector(strList)
    elif isinstance(strList, StrVector):  # already a StrVector
        return StrVector(strList)
    elif isinstance(strList, str):        # str scalar, so apply StrVector
        return StrVector([strList])
    elif any([types.is_list_like(s) for s in strList]):  # not the deepest list
        return ListVector([(None, StrListVector(s)) for s in strList])
    else:  
        return StrVector(list(strList))   # is deepest list(-like) of str types

def _flatten(s):
    """Generator returns each terminal item (by DFS) from nested list"""
    try:
        for t in s:
            if types.is_list_like(t):
                yield from _flatten(t)
            else:
                yield t
    except:
        yield s

        
def _combine(*args):
    """Flatten each arg, and concat all into a flat list"""
    return [item for sublist in args for item in list(_flatten(sublist))] 


class PyR:
    """Store and expose as rpy2 or numpy/pandas objects 

    Args:
        item: input R object, or numpy array, or pandas DataFrame or Series
        names: named labels of object items

    Attributes:
        iloc: dict or numpy array

            - internally, objects are stored as either numpy array, 
              or dict of objects (when input was R ListVector or DataFrame, 
              or Python dict).
            - TODO: should use other safer property getters to view object 
              in target types: e,g, .frame (pandas), .ro (RObject), 
              or .values (python dict or ndarray)

        dim (tuple of int): dimensions of data objects

        names, rownames, colnames (StrVectors): named labels of object items

    Notes:

    - input item can be of either Python or R object type
    - input labels in each dimension can be explicitly provided, and should have
      same dim as object, as error checking is minimal
    - In R, matrices are column-major ordered (aka Fortran-like index order,
      with the first index changing fastest) although the R constructor 
      matrix() accepts a boolean argument byrow that, when true, will build 
      the matrix as if row-major ordered (aka C-like, which is also Python numpy
      default order, where the last axis index changes fastest)
    - A suggested convention is to append '_' to R function names and 
      '_r' to R objects, and capitalize initial letter of PyR instances.
    - r['plot'] may need to explicitliy set xlab='', ylab=''
    - TODO: if hasattri('slots'), esp 'ts' class, e.g. nile.slots.items()

    Examples:

    >>> from rpy2.robjects import r
    >>> from rpy2.robjects.packages import importr
    >>> amen_r = importr('amen')                    # use R library
    >>> c_ = r['c']                                 # link R routines
    >>> Nodevars = PyR(r['IR90s'].rx2['nodevars'])  # retrieve R data
    >>> Gdp = Nodevars[:, 'gdp']                    # getitem subset with slice
    >>> topgdp = Gdp.values > sorted(Gdp.py, reverse=True)[30] # python calcs
    >>> Dyadvars = PyR(r['IR90s'].rx2['dyadvars'])
    >>> Y = Dyadvars[topgdp, topgdp, 'exports']  # getitem with boolean index 
    >>> Y.iloc = np.log(Y.iloc + 1)          # update with python calculations
    """

    def __init__(self, item: Any,
                 names: StrVector | ListVector | str | List[str] | None = None,
                 verbose: int = 0):
        """Make instance from an input python or R (rpy2) object"""
        
        self.verbose = verbose
        self.dim = ()

        # extract names, colnnames, rownames, index, columns as StrVector attr
        if hasattr(item, 'names'):   # R attributes
            self.names = item.names
        if names is not None:
            self.names = StrListVector(names)
        if hasattr(item, 'colnames'):
            self.colnames = item.colnames
        if hasattr(item, 'rownames'):            
            self.rownames = item.rownames

        if isinstance(item, (Series, DataFrame)):  # Pandas attributes
            self.rownames = StrVector(item.index)
        if isinstance(item, DataFrame):
            self.colnames = StrVector(item.columns)
        if isinstance(item, Series) and item.name is not None:
            self.colnames = StrVector([item.name])

        if isinstance(item, (ListVector, ro.vectors.DataFrame, dict)):
            try: # convert to dict if dict-like (i.e. ListVector, R DataFrame)
                names = [self.names[i] if isinstance(self.names, StrVector)
                         else self.names[0][i] for i in range(len(item))]
            except:
                names = [k for k,v in item.items()]
            self.iloc = {n: PyR(v) for n, (k,v) in zip(names, item.items())}
            
            if verbose:
                print(f"PyR: dict (len={len(self.iloc)}){type(item)}")
                
        else:    # not dict-like, so convert to numpy array and apply shape dims
            self.iloc = np.array(item)
            if hasattr(item, 'dim'):
                self.dim = tuple(item.dim)
                if len(self.dim) > 1:
                    self.iloc = self.iloc.reshape(tuple(item.dim), order='F')
            self.dim = self.iloc.shape

            # try to infer rownames if not attribute
            if (not hasattr(self, 'rownames') and len(self.iloc.shape) > 1
                and self.names and isinstance(self.names, ListVector)):
                self.rownames = self.names[0]

            # try to infer colnames if not attribute
            if (not hasattr(self, 'colnames') and len(self.iloc.shape) > 1
                and self.names and isinstance(self.names, ListVector)):
                self.colnames = self.names[1]   # try to infer colnames
                
            if verbose:
                print(f"PyR: ndarray {self.iloc.shape} {type(item)}")

        # TODO: WARNING self.names.shape (if hasattr and not null) via try
        # len(names)==len(dict) else len(names)=len(self.dim) or sum(self.dim)

    def __repr__(self):
        """str representation, preferabaly pretty print as pandas DataFrame"""
        
        if not isinstance(self.iloc, dict):
            if len(self.iloc.shape) <= 2:
                return str(self.frame)
        return str(self.iloc)

    @staticmethod
    def savefig(filename, display=True, ax=None, figsize=(12,12)):
        """Save R graphics to file, or return R command, optionally imshow"""
        
        s = "dev.copy(png, '{}'); dev.off()".format(filename)
        if display is not None:
            ro.r(s)
            if display:
                if not ax:
                    fig, ax = plt.subplots(clear=True, figsize=figsize)
                img = mpimg.imread(filename)
                ax.imshow(img, interpolation='nearest')
                ax.axis('off')
        return s
    
    def assign(self, obj):
        """Directly update internal data object (must be same numpy shape)"""

        if isinstance(obj, dict):
            assert(isinstance(self.iloc, dict) and len(self.iloc)==len(obj))
            self.iloc = obj
        else:
            obj = np.array(obj)
            assert(obj.shape == self.iloc.shape)
            self.iloc = obj
    
    @property
    def ro(self):
        """Expose a view as RObject, so that can pass to R environment"""

        # Convert to R vector of correct data type
        if isinstance(self.iloc, dict):
            out = ListVector([(None, PyR(v).ro) for v in self.iloc])  
        if types.is_float_dtype(self.iloc):
            out = FloatVector(self.iloc.reshape(-1, order='F'))
        elif types.is_integer_dtype(self.iloc):
            out = IntVector(self.iloc.reshape(-1, order='F'))
        else:
            out = StrVector(self.iloc.reshape(-1, order='F'))
        if len(self.dim) > 1:  # reshape to R Array if has non-trivial dim
            out = ro.r.array(out, dim=IntVector(self.dim))

        # Collect R object name attributes            
        if hasattr(self, 'rownames'):
            out.rownames = StrVector(self.rownames)
        if hasattr(self, 'colnames'):
            out.colnames = StrVector(self.colnames)
        if hasattr(self, 'names'):
            out.names = ListVector(self.names) if isinstance(
                self.names, ListVector) else StrVector(self.names)
        return out

    @property
    def frame(self):
        """Expose a view as pandas DataFrame"""

        out = DataFrame(self.values)
        if hasattr(self, 'names') and isinstance(self.names, StrVector):
            if len(self.names) == len(out.columns):
                out.columns = list(self.names)
            if len(self.names) == len(out.index):
                out.index = list(self.names)
        if hasattr(self, 'rownames') and isinstance(self.rownames, StrVector):
            out.index = list(self.rownames)
        if hasattr(self, 'colnames') and isinstance(self.colnames, StrVector):
            out.columns = list(self.colnames)
        return out

    @property
    def values(self):
        """Expose view as python dict (when ListVector) or ndarray (when not)"""
        return ({k:v.values for k,v in self.iloc.items()}
                if isinstance(self.iloc, dict) else self.iloc)

    def __getitem__(self, args):
        """Returns copy of subset of data object from given slice or index"""
        try:
            if isinstance(self.iloc, dict):  # return item of dict
                if isinstance(args, int):
                    try:
                        args = list(self.names).index(args)
                    except:
                        args = list(self.iloc.keys()).index(args)
                return self.iloc[args]

            # replace any str labels in args with its index in self.names
            if isinstance(args, tuple) and self.names is not None:
                args = tuple(self.index(a, i) for i,a in enumerate(args))

            # extract corresponding subset of names
            if self.names:
                names_ = deepcopy(self.names)
                names = ListVector(names_)
                for i in range(len(self.names)):
                    if isinstance(names_[i], StrVector):
                        s = np.array(names_[i])[args[i]]                    
                        names[i] = StrVector([s] if isinstance(s, str) else s)
            else:
                names = NULL

            # finally extract by looping over each dim; enables R-like indexing
            out = deepcopy(self.iloc)
            for i, arg in enumerate(args):
                a = [slice(None)]*len(args)
                a[i] = arg
                dims = len(out.shape)
                out = out[tuple(a)]
                if self.verbose:
                    print(i, out.shape, dims, tuple(a))
                if len(out.shape) < dims:  # if this dimension is flattened out
                    names = names[:i] + names[(i+1):]
            return PyR(out, names=names)
        except:
            raise Exception(f"getitem: {args}")
        
    @property
    def nrow(self):
        """Length of first dimension, as R IntVector type"""
        
        return IntVector([self.iloc.shape[0]])
    
    @property
    def ncol(self):
        """Length of second dimension, as R IntVector type"""
        
        return IntVector([self.iloc.shape[1]])

    def index(self, s: str | List[str], axis: int = -1):
        """Helper method to lookup index/es of (list of) str label in names"""
        
        if isinstance(s, str):
            return list(self.names[axis]).index(s)
        elif types.is_list_like(s):
            return [self.index(t, axis=axis) for t in s]
        else:
            return s


#if __name__ == "__main__":
if False:   # replicate Ch 1 Gaussian AME of Hoff (2018) "Amen" tutorial
    import numpy as np
    import numpy.ma as ma
    from numpy.ma import masked_invalid as valid
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    
    stats_ro = importr('stats')
    base_ro  = importr('base')
    amen_ro  = importr('amen')
    utils_ro = importr('utils')

    matrix_ro   = ro.r['matrix']
    t_ro        = ro.r['t']
    anova_ro    = ro.r['anova']
    lm_ro       = ro.r['lm']
    ame_ro      = ro.r['ame']    # default nscan=10000, odens=25 => 400 samples
    summary_ro  = ro.r['summary']
    plot_ro     = ro.r['plot']
    circplot_ro = ro.r['circplot']
    IR90s_ro    = ro.r['IR90s']
    
    # Load GDP and exports data
    Nodevars = PyR(IR90s_ro.rx2['nodevars'])
    Gdp = Nodevars[:, 'gdp']
    Dyadvars = PyR(IR90s_ro.rx2['dyadvars'])
    topgdp = Gdp.values > sorted(Gdp.values, reverse=True)[30]
    Y = Dyadvars[topgdp, topgdp, 'exports']
    Y.assign(np.log(Y.values + 1))
    Y[:5,:4]

    # Simple ANOVA to show random effects
    rowcountry_ro = matrix_ro(Y.rownames, Y.nrow, Y.ncol)
    colcountry_ro = t_ro(rowcountry_ro)
    formula_ro = ro.Formula("c(Y) ~ c(Rowcountry) + c(Colcountry)")
    formula_ro.environment['Rowcountry'] = rowcountry_ro
    formula_ro.environment['Colcountry'] = colcountry_ro
    formula_ro.environment['Y'] = Y.ro
    fit_anova_ro = anova_ro(lm_ro(formula_ro))
    print(fit_anova_ro)

    # display exporter and imported effects
    muhat = np.nanmean(Y.ro)
    Ahat = PyR(Y.frame.mean(axis=1) - muhat, names=['Ahat'])
    print(Ahat.frame['Ahat'].sort_values(ascending=False)[:6]) 
    Bhat = PyR(Y.frame.mean(axis=0) - muhat, names=['Bhat'])
    print(Bhat.frame['Bhat'].sort_values(ascending=False)[:6])  

    # But ignores corr of random effects, fundamental characteristic of dyads
    print(np.cov(Ahat.values, Bhat.values))
    print(np.corrcoef(Ahat.values, Bhat.values)[0,1])
    
    outer = Y.values - (muhat + np.add.outer(Ahat.values, Bhat.values))
    print(ma.cov(valid(outer.flatten()), valid(outer.T.flatten())).data)
    print(ma.corrcoef(valid(outer.flatten()), valid(outer.T.flatten()))[0,1])

    # Social Relations Model
    fit_SRM_ro = ame_ro(Y.ro, plot=False, print=False, family='nrm')
    Fit_SRM = PyR(fit_SRM_ro)
    _ = summary_ro(fit_SRM_ro)
    plot_ro(fit_SRM_ro)

    # Compare empirical and model estimates
    print(muhat, np.nanmean(Fit_SRM['BETA'].values))  # overall mean
    print(_combine(np.cov(Ahat.values, Bhat.values))[:3])  # mean covariances
    vcmean = Fit_SRM['VC'][:, :4].frame.mean()  # posterior variance parms
    print(vcmean[:3])

    # Residual Dyadic Correlation
    print(vcmean['cab'] / (np.sqrt(vcmean['va']) * np.sqrt(vcmean['vb'])))
    print(ma.corrcoef(valid(_combine(outer)), valid(_combine(outer.T)))[0,1])
    print(np.mean(Fit_SRM['VC'][:, 3].values))
    
    # SRRM
    Xn = PyR(IR90s_ro.rx2('nodevars'))[topgdp, :]
    Xn.iloc[:, :2] = np.log(Xn.values[:, :2])
    Xd = PyR(IR90s_ro.rx2('dyadvars'))[topgdp, topgdp, np.array([0,2,3,4])]
    Xd.iloc[:, :, 2] = np.log(Xd.values[:, :, 2])
    fit_srrm_ro = ame_ro(Y.ro, Xd=Xd.ro, Xr=Xn.ro, Xc=Xn.ro,
                         plot=False, print=False)
    Fit_srrm = PyR(fit_srrm_ro)
    _ = summary_ro(fit_srrm_ro)
    plot_ro(fit_srrm_ro)

    gof = Fit_srrm['GOF'].frame.iloc[:1,:]    # actual in first row of gof
    gof.loc['mean', :] = np.nanmean(Fit_srrm['GOF'].values[1:,:], axis=0)
    gof.loc['std', :] = np.nanstd(Fit_srrm['GOF'].values[1:,:], axis=0)
    print(gof)

    # OLS
    fit_rm_ro = ame_ro(Y.ro, Xd=Xd.ro, Xr=Xn.ro, Xc=Xn.ro, print=False,
                       plot=False, rvar=False, cvar=False, dcor=False)
    _ = summary_ro(fit_rm_ro)
    plot_ro(fit_rm_ro)

    # SRRM with latent factor multiplicative effects
    fit_ame2_ro = ame_ro(Y.ro, Xd=Xd.ro, Xr=Xn.ro, Xc=Xn.ro, R=2,
                         plot=False, print=False)
    Fit_ame2 = PyR(fit_ame2_ro)
    _ = summary_ro(fit_ame2_ro)
    plot_ro(fit_ame2_ro)

    # plots
    circplot_ro(Y.ro, U=fit_ame2_ro.rx2['U'], V=fit_ame2_ro.rx2['V'],
              row_names=Y.rownames, col_names=Y.colnames,
              plotnames=True, pscale=FloatVector([1.5]))
if False:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import Formula, Environment
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector, ListVector, IntVector, StrVector, \
        NULL
    stats = importr('stats')
    base = importr('base')
    
    # Create matrix in R 
    v = ro.FloatVector([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    m = ro.r.matrix(v, nrow = 2)
    m = ro.r['matrix'](v, nrow = 2)    
        
    ctl = FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
    trt = FloatVector([4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69])
    group = base.gl(2, 10, 20, labels = ["Ctl","Trt"])
    weight = ctl + trt
    ro.globalenv["weight"] = weight
    ro.globalenv["group"] = group
    lm_D9 = stats.lm("weight ~ group")
    print(stats.anova(lm_D9))
    lm_D90 = stats.lm("weight ~ group - 1")
    print(base.summary(lm_D90))
    res = ro.StrVector(['abc', 'def'])
    v = ro.FloatVector([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    m = ro.r['matrix'](v, nrow = 2)
    letters = ro.r['letters']
    rcode = 'paste(%s, collapse="-")' % (letters.r_repr())
    res = ro.r(rcode)
