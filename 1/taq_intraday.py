"""
Intraday liquidity analysis of TAQ tick data
Supervised and supervised learning of volume: anomaly detection, regression, neural nets

References:

"""
import dives
import dives.util
import dives.dbengines
import dives.structured
import dives.taq
import dives.custom

import time, re, pickle
import numpy as np
import scipy as sp
import pandas as pd
import sklearn, sklearn.covariance, sklearn.ensemble, sklearn.neighbors, sklearn.svm
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  # for date formatting in plots

import importlib
importlib.reload(dives)
importlib.reload(dives.util)
importlib.reload(dives.structured)
importlib.reload(dives.dbengines)
importlib.reload(dives.taq)
importlib.reload(dives.custom)

from dives.util import DataFrame, NamedDict
from dives.dbengines import SQL, Redis
from dives.structured import BusDates, CRSP
from dives.taq import TAQ, plot_taq, measure_liquidity, clean_trade, clean_nbbo
from dives.custom import CustomRegressor, CustomScorer

import secret
verbose = secret.value('verbose')
sql = SQL(**secret.value('sql'))
rdb = Redis(**secret.value('redis'))
bd = BusDates(sql)    # test BusDates class
crsp = CRSP(sql, bd, rdb)

if False:
    filename = '/media/terence/3E5C-8708/TAQ/EQY_US_ALL_REF_MASTER_20171101.gz'
    master = TAQ(filename).read()
    
    filename = '/home/terence/Downloads/EQY_US_ALL_TRADE_20171101.gz'
    index_file ='/home/terence/Downloads/EQY_US_ALL_TRADE_20171101.gzidx'
    symbol_file = '/home/terence/Downloads/EQY_US_ALL_TRADE_20171101.pkl'
    """
    trade = TAQ(filename, index_file=index_file)
    trade.create_symbols(symbol_file)

    """
    Trade = TAQ(filename, index_file)
    trade = Trade.open(symbol_file)

    filename = '/home/terence/Downloads/EQY_US_ALL_NBBO_20171101.gz'
    index_file ='/home/terence/Downloads/EQY_US_ALL_NBBO_20171101.gzidx'
    symbol_file = '/home/terence/Downloads/EQY_US_ALL_NBBO_20171101.pkl'
    """
    nbbo = TAQ(filename)
    nbbo.create_symbols(symbol_file)
    nbbo.create_index(index_file)
    nbbo = TAQ(filename, index_file)
    with nbbo.open(symbol_file) as f:
        cq = f.get('AABA')
    """    
    Nbbo = TAQ(filename, index_file)
    nbbo = Nbbo.open(symbol_file)

    """
    cq = nbbo.get('AAPL')
    ct = trade.get('AAPL')
    it = nbbo.iter()
    cq = next(nbbo)
    """
    symbol = 'SPY'
    symbol = 'HP'
    plot_taq(symbol, trade, nbbo)

    t = trade.get(symbol)
    q = nbbo.get(symbol)
    ct = clean_trade(t)
    cq = clean_nbbo(q)   
    print(len(t), len(ct))  # 193097 187947
    print(len(q), len(cq))  # 617010 599541

    df = measure_liquidity(ct, cq, minutes=5)
    measures = ['impact','realized','effective','quoted', 'midquote', 'vwap']
    plt.figure(figsize=(10, 12))
    for i in range(len(measures)):
        plt.subplot(3,2,i+1)
        ax = plt.gca()
        sns.lineplot(data=df[measures[i]])
        plt.title(measures[i])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
    #
    # Start here...
    # Keywords: TAQ, Intraday Liquidity, Day Trading, PCA, Anomaly Detection, Supervised Learning
    #
       

    # open TAQ raw daily files for indexed gzip access
    filename = '/media/terence/3E5C-8708/TAQ/EQY_US_ALL_REF_MASTER_20171101.gz'
    master = TAQ(filename).read()   
    filename = '/home/terence/Downloads/EQY_US_ALL_TRADE_20171101.gz'
    index_file ='/home/terence/Downloads/EQY_US_ALL_TRADE_20171101.gzidx'
    symbol_file = '/home/terence/Downloads/EQY_US_ALL_TRADE_20171101.pkl'
    trade = TAQ(filename, index_file).open(symbol_file)
    filename = '/home/terence/Downloads/EQY_US_ALL_NBBO_20171101.gz'
    index_file ='/home/terence/Downloads/EQY_US_ALL_NBBO_20171101.gzidx'
    symbol_file = '/home/terence/Downloads/EQY_US_ALL_NBBO_20171101.pkl'
    nbbo = TAQ(filename, index_file).open(symbol_file)

    # get usual universe of stocks from CRSP as of prior day, and keep their 8-character cusip
    pordate = bd.shift(20171101, -1)
    univ = crsp.get_universe(pordate)
    univ = univ[univ['cap'] > 300000]
    ncusip = crsp.get_section('names',['ncusip','exchcd'],'date',pordate,0).drop_duplicates('ncusip')
    univ = univ.join(ncusip, how='inner').reset_index().set_index('ncusip')

    # loop over each stock in universe, and extract and compute 5-minute liquidity metrics
    cache = dict()
    toc = time.time()
    univ['symbol'] = ''
    for symbol, header in master.iterrows():
        if header['CUSIP'][:8] in univ.index:
            tic = time.time()
            t = trade.get(symbol)
            q = nbbo.get(symbol)
            ct = clean_trade(t)
            cq = clean_nbbo(q)
            if ct is not None and cq is not None and len(ct) and len(cq):
                df = measure_liquidity(ct, cq, minutes=5)
                cache[symbol] = df
                univ.loc[univ.index == header['CUSIP'][:8], 'symbol'] = symbol
                print(symbol, len(df), len(t), len(ct), len(q), len(cq),
                      time.time() - tic, len(cache), time.time()-toc)
            else:
                print('**** {} ****'.format(symbol))
    univ = univ.set_index('symbol', drop=False)
    univ = univ[~univ.index.duplicated()].reindex(list(cache.keys()))
    with open('/home/terence/Downloads/out/liquidity.pkl','wb') as f:
        pickle.dump((cache, univ), f)
    print(time.time() - toc)

if True:
    # derive and accumulate in dict by liquidity measure, and remove extreme obs
    with open('/home/terence/Downloads/out/liquidity.pkl','rb') as f:
        cache, univ = pickle.load(f)

    measures = ['impact','realized','effective','quoted', 'depth','vwap', 'midquote','volume']
    derived = ['relquoted', 'relmidquote', 'relvwap', 'reldepth', 'turnover']
    liquidity = {measure : DataFrame() for measure in measures + derived}
    tic = time.time()
    for symbol in cache:
        if univ.loc[symbol, 'cap'] >= 300000: #2000000:
            df = cache[symbol][measures].copy()
            df['relquoted']   = df['quoted']   / df['midquote']     # df['midquote'].mean() # 
            df['relmidquote'] = df['midquote'] / df['midquote'][0]  # df['midquote'].mean() # 
            df['relvwap']     = df['vwap']     / df['vwap'][0]
            df['reldepth']    = df['depth']    / df['depth'].mean()
            df['turnover']    = df['volume']   / df['volume'].sum()
            if not np.isnan(df.iloc[1:-1]).any().any():   # screen out illiquid stocks
                for measure in liquidity.keys():
                    liquidity[measure][symbol] = df[measure]
    print(time.time() - tic)


if False:
    #
    # Eigen analysis of quotes: which measures of liquidity show more systematic variability
    # 4 spreads, turnover, vwap, midquote, depth
    # df rows are time series, columns are stocks: demean by column so each stock's series has zero mean
    #        nulls = liquidity[measure].isnull().sum()
    #
    # Plot average trend and variance explained
    #
    """
    Comments: impact and realized also affected by price trends, so is noisy: look at averages
    """
    for ifig, measures in enumerate([['relquoted', 'impact','realized', 'effective'],
                                     ['relmidquote','relvwap','turnover','reldepth']]):
        plt.figure(ifig + 1, figsize=(12,12))
        for iplot, measure in enumerate(measures):
            trend = liquidity[measure].mean(axis = 1)   # simple average trend            
            ax = plt.subplot(2, 2, iplot + 1)
            sns.lineplot(data=trend)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xlabel(measure)
            plt.ylabel('average')
    plt.show()

    for ifig, measures in enumerate([['relquoted', 'impact','realized', 'effective'],
                                     ['relmidquote','relvwap','turnover','reldepth']]):
        plt.figure(ifig + 1, figsize=(12,12))
        ax = plt.subplot(2,2,1)   # get ax of first subplot, to be shared by other subplots
        for iplot, measure in enumerate(measures):
            X = liquidity[measure].iloc[1:-1]   # ignore first and last five minute interval
            X = X - X.mean(axis=0)
            U, s, Vh = sp.linalg.svd(X, full_matrices=False)            
            explained = [sum(s[:k]**2)/sum(s**2) for k in range(1, 25)]
            plt.subplot(2, 2, iplot + 1, sharex=ax, sharey=ax)
            sns.lineplot(y=explained, x=np.arange(1, len(explained)+1))
            plt.xlabel(measure)
            plt.ylabel('variance explained by top k eigs')
            print(measure, explained)
    plt.show()

    #
    # Digression on PCA
    #
    measure = 'turnover'
    df = liquidity[measure].iloc[1:-1]
    X = np.array(df - df.mean(axis=0))

    # For plotting in 2-dimensional space, reduce data by PCA
    U, s, Vh = sp.linalg.svd(X, full_matrices=False)
    """
    Explain here: Technical references on PCA etc can clarify the specific definitions, but intuitively...
    """
    # PCA by SVD: columns of U are time-series, i'th column of Vh is stocks' loadings on i'th eigenvector
    # Shows Vh are loadings, by compare to linear regression, hence eigenvalues and R2 components are equal
    sst = np.mean(np.mean(X**2))
    for k in range(5):
        fitted = (s[:k] * U[:, :k]).dot(Vh[:k, :]) # Vh column *sqrt(s) is loadings on U column eigenvector
        sse = np.mean(np.mean((X - fitted)**2))
        print(k, 1-(sse/sst), sum(s[:k]**2)/sum(s**2))

    # plot average, and top 3 eigenvectors
    plt.figure(1, figsize=(9, 9))
    plt.clf()
    trend = X.mean(axis=1)
    ax = plt.subplot(2, 2, 1)
    sns.lineplot(x=df.index, y=trend)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))    
    plt.xlabel(measure)
    ax = plt.subplot(2, 2, 2)
    for i in range(3):
        plt.subplot(2, 2, i+2, sharex=ax, sharey=ax)
        sns.lineplot(x=df.index, y=np.sqrt(s[i])*U[:,i]*np.sign(np.corrcoef(U[:,i], trend)[0,1]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))        
        plt.xlabel('eigenvector #' + str(i+1))
    
    #
    # Anomaly detection of turnover
    # https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html
    # 

    # define outlier/anomaly detection methods to be compared
    y = X.T         # sample rows should now be stocks, feature columns are now the time intervals
    x = Vh[:2,:].T  # use stocks' loadings on top two components for (imperfect) 2D scatter plots
    outliers_fraction = 5/y.shape[0]  # set fraction to extract 5 anomalies    
    anomaly_algorithms = [
        ("Robust covariance",
         sklearn.covariance.EllipticEnvelope(contamination = outliers_fraction)),
        ("One-Class SVM",
         sklearn.svm.OneClassSVM(nu = outliers_fraction, kernel = "rbf", gamma = 0.1)),
        ("Isolation Forest",
         sklearn.ensemble.IsolationForest(contamination = outliers_fraction, random_state = 42)),
        ("Local Outlier Factor",
         sklearn.neighbors.LocalOutlierFactor(n_neighbors = 35, contamination = outliers_fraction))]

    plt.figure(1, figsize=(9,9))
    ax = plt.subplot(2, 2, 1)
    symbols = set()   # to collect stock symbols of detected outliers
    for iplot, (name, algorithm) in enumerate(anomaly_algorithms):
        name, algorithm = anomaly_algorithms[iplot]

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(y)
        else:
            y_pred = algorithm.fit(y).predict(y)

        plt.subplot(2, 2, iplot+1, sharex=ax, sharey=ax)
        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(Vh[0, :], Vh[1, :], s=10, color=colors[(y_pred + 1) // 2])
        plt.title(name)

        # ideally: label the scatter plot.  Also: accumulate to set to plot_taq
        print(name, list(df.columns[y_pred < 0]))
        symbols = symbols.union(df.columns[y_pred < 0])

    for ifig, symbol in enumerate(symbols):
        plot_taq(symbol, trade, nbbo)

        ax = plt.gca()
        plt.plot(df.index, X[:, np.where(df.columns == symbol)[0][0]])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.ylabel(measure)
        plt.xlabel(symbol)
        plt.show()
        
if True:
    # construct y_all = turnover after 3pm less mean before 3pm
    y_all = (liquidity['turnover'].iloc[66:]-liquidity['turnover'].iloc[:66].mean(axis=0)).sum(axis=0)
    sns.distplot(y_all)
    
    # construct X_all = columns of eigens from 65 intervals named 'quoted1', etc.
    X_all = DataFrame()
    for measure in ['turnover', 'midquote', 'quoted']:
        df = liquidity[measure].iloc[1:66]
        X = np.array(df - df.mean(axis=0))
        U, s, Vh = sp.linalg.svd(X, full_matrices=False)
        topK = 20
        eigen_features = [measure + str(c+1) for c in range(len(Vh))]
        time_features = [measure[0] + str(c) for c in range(len(X))]        
        X_all = pd.concat([DataFrame(data=X.T, columns = time_features, index=df.columns),
                           DataFrame(data=Vh[:topK, :].T, columns=eigen_features[:topK],
                                     index=df.columns),
                           X_all], axis=1)
    X_all = X_all.reindex(y_all.index)

    # TODO (Jan 24, 2020)
    # Put in pipeline to standardize, transform
    # Try either eigen or raw
    # cross_validate_score
    # r2, spearman and pearson (to difference in sample and out sample)

    # stratifiedkfold(3) on ~1600 samples for test error/train error

if False:
    out = NamedDict(['name','split'])
    for name in CustomRegressor.keys(): #['xgboost','xgboostCV']:  #
        clf = CustomRegressor[name]
        results = sklearn.model_selection.cross_validate(
            clf, X_all, y_all, verbose=verbose,
            return_train_score=True,
            return_estimator=True,
            scoring = {s : CustomScorer(s) for s in ['r2', 'spearman']},
            cv = sklearn.model_selection.KFold(3, shuffle=True, random_state=42))
        out.replace(name = name,
                    split = 'train',
                    elapsed = results['fit_time'].mean(),
                    r2 = results['train_r2'].mean(),
                    spearman = results['train_spearman'].mean())
        out.replace(name = name,
                    split = 'test',
                    elapsed = results['score_time'].mean(),
                    r2 = results['test_r2'].mean(),
                    spearman = results['test_spearman'].mean())
        pprint(out)

    # train and test r2 only for illustration of regularization and generalization trade-offs of the models
    sns.catplot(x='split', y='r2', hue='name', data=DataFrame(out.search()), kind='bar')

    
#        r2.loc[name, 'test'] = results['test_r2'].mean()
#        r2.loc[name, 'train'] = results['train_r2'].mean()
#        spearman.loc[name, 'test'] = results['test_spearman'].mean()
#        spearman.loc[name, 'train'] = results['train_spearman'].mean()
#        print(r2)
#        print(spearman)
