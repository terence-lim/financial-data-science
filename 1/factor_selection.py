"""
More quant factors/return predicting signals

References:
“The Characteristics that Provide Independent Information about Average U.S. Monthly Stock Returns,” 
Jeremiah Green, John Hand and Frank Zhang. Review of Financial Studies 30:12, 4389-4436 (December 2017). 
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
import time
from pprint import pprint
import dives
import dives.util
import dives.dbengines
import dives.structured
import dives.evaluate
import dives.custom

import importlib
importlib.reload(dives)
importlib.reload(dives.util)
importlib.reload(dives.structured)
importlib.reload(dives.dbengines)
importlib.reload(dives.evaluate)
importlib.reload(dives.custom)

from dives.util import fractiles, DataFrame, NamedDict, winsorize, standardize
from dives.dbengines import SQL, Redis
from dives.structured import BusDates, Benchmarks, CRSP, Signals
from dives.evaluate import BackTest, run_backtest
from dives.custom import CustomRegressor, CustomScorer

import secret
sql = SQL(**secret.value('sql'))       
rdb = Redis(**secret.value('redis'))       
    
bd = BusDates(sql)
bench = Benchmarks(sql, bd)
crsp = CRSP(sql, bd, rdb)
signals = Signals(sql)
backtest = BackTest(sql, bench, 'RF')

RPS = True

if True:  #  Machine Learning RPS part 1

    models = {k: CustomRegressor[k]
              for k in ['plsRegression','elasticNetCV','randomForestRegressionCV','xgboostCV']}
    
    windows = {'chfeps' : 1}  # label and window
    windows = {**{k:1 for k in ['mom12m', 'mom36m', 'mom6m', 'mom1m', 'chmom', 'divyld','indmom']},
               **{k:1 for k in ['ill','maxret','retvol','baspread','std_dolvol','zerotrade',
                                'std_turn','turn', 'dolvol']},
               **{k:1 for k in ['beta','idiovol','pricedelay']},                                
               **{k:12 for k in ['absacc','acc', 'agr', 'bm', 'cashpr', 'cfp', 'chcsho', 'chinv',
                                 'depr', 'dy', 'egr', 'ep', 'gma', 'grcapx', 'grltnoa', 'hire',
                                 'invest', 'lev', 'lgr' ,'pchdepr', 'pchgm_pchsale', 'pchquick',
                                 'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga', 'pchsaleinv',
                                 'pctacc', 'quick', 'salecash', # 'rd_sale', 'rd_mve', 'realestate'
                                 'salerec', 'saleinv', 'secured', 'sgr', 'sp', 'tang', 'bm_ia',
                                 'cfp_ia', 'chatoia' , 'chpmia', 'pchcapx_ia', 'chempia', 'mve_ia']},
               **{k:3 for k in ['stdacc', 'stdcf', 'roavol', 'sgrvol', 'cinvest', 'chtx',
                                'rsup', 'roaq', 'cash', 'nincr']},
               **{k:3 for k in ['chfeps', 'chnanalyst', 'disp', 'fgr5yr', 'sfe', 'sue']},
               **{k:4 for k in ['ear', 'aeavol']}}
    labels = list(windows.keys())

    output = NamedDict(['date','model','split'])  # replaces train and test
    scores = ['pearson', 'spearman', 'r2', 'explained_variance']
    avail  = DataFrame(columns = labels)
    regressors = {m: dict() for m in models}
    out = DataFrame()    # keep predictions
    
if RPS: # Machine Learning RPS part 2
    data = DataFrame()                # accumulate endog and exog variables, for fit and predict
    out = DataFrame()                 # accumulate signal prediction values, for signals.save
    beg, end = 19830701, 20190630     # rebalances date range
    width = 12*9                      # non-overlapping train/test period lengths in months

    for p, pordate in enumerate(bd.endmo_range(beg, end)):
        tic = time.time()
        df = crsp.get_universe(pordate)
        df = df[df['deciles'] > 1]       # drop microcap stocks
        df['sic2'] = df['siccd'] // 100  # 2-digit sic groupby to fill in industry means when missing
        df['counts'] = 0                 # count number of non-missing data items for each stock
        df['period'] = p                 # enumerate current period number
        df['rebaldate'] = pordate
        group = df.groupby(['sic2'])
        for label, window in windows.items():
            start = bd.endmo(pordate,
                             months = -window)
            signal = signals.get_section(label,
                                         pordate,
                                         start = start).reindex(df.index)
            assert(signal[label].notnull().mean() > 0)
            if signal[label].notnull().mean() > 0:  # drop sparse signals
                df[label] = standardize(winsorize(signal[label]))        # standardize winsorize signal
                df.loc[df[label].notnull(), 'counts'] += 1               # counter for non-missing item
                f = df[label].isnull()
                df.loc[f, label] = group[label].transform('mean').loc[f] # industry means when missing
                g = df[label].isnull()
                df.loc[g, label] = 0                                     # 0 when no industry mean
                avail.loc[pordate, label] = sum(~f)
            else:
                print('***', pordate, label, '***')
                avail.loc[pordate, label] = 0
        print(pordate, avail.loc[pordate, label], time.time()-tic)
        df = df[df['counts'] > df['counts'].max() / 10]          # require stocks have 10% non-missing
        df = df.join(crsp.get_ret(bd.shift(pordate, 1), bd.endmo(pordate, 1)),
                     how='left').reindex(df.index)
        data = data.append(df, ignore_index=True, sort=True)     # append new month's data
        data = data[data['period'] > (p - width)]                # keep only training sample width
        
        if (p >= width and df['ret'].notnull().any()):           # predict and compute score every month
            y_test = np.array(df[['ret']])
            y_test[np.isnan(y_test)] = np.nanmean(y_test)
            X_test = np.array(df[labels])
            for name, model in models.items():
                df[name] = model.predict(X_test)                 # get predictions for each algo
                output.replace(date = pordate,
                               model = name,
                               split = 'test',
                               **{s: CustomScorer(s, make_scorer=False)(y_test, df[name]) 
                                  for s in scores})              # order of (y_true, y_pred) is critical!
                print(pordate, name, output.match(date=pordate, model=name, split='test'))
            out = out.append(df[['rebaldate'] + list(models.keys())].reset_index(), ignore_index=True)
        
        if (p < end) and ((p % width) == (width - 1)):           # re-fit model every {width} months
            f = data['ret'].notnull()
            y_train = np.array(data.loc[f, 'ret'])
            X_train = np.array(data.loc[f, labels])
            for name, model in models.items():
                model.fit(X_train, y_train)                      # fit each model
                y_pred = model.predict(X_train)
                output.replace(date = pordate,
                               model = name,
                               split = 'train',
                               **{s : CustomScorer(s, make_scorer=False)(y_train, y_pred)
                                  for s in scores})              # order of (y_true, y_pred) is critical!
                print(pordate, name, output.match(date=pordate, model=name, split='train'))
                
                regressors[name][pordate] = model
    for name in models.keys():                # finally, save signal values
        signals.save(out, name, append=False)

    # save models in pickled
    with open('/home/terence/Downloads/out/models1.pkl','wb') as f:
        pickle.dump((regressors, train, test, avail), f)

if False:
        
    # show feature importances, each period
    for name, periods in models.items():
        for date, model in periods.items():
            importances = np.array(model.get_importances()).flatten()
            features = np.argsort(-abs(importances))
            n = sum(importances != 0)
            print('Method: {:10s}   Date: {}'.format(name, date))
            for feature in features[:min(n, 10)]:
                print('{:12s} {:10.6f}'.format(labels[feature], importances[feature]))

    # compare train and test errors, each period
    # linear model seems competitive -- because of data mining?
    score = 'spearman'
    r = DataFrame(data=DataFrame(train[score]).mean(axis=0), columns=['train']).join(
        DataFrame(data=DataFrame(test[score]).mean(axis=0), columns=['test'])).reset_index()
    result = pd.melt(r.rename(columns = {'index' : 'method'}),
                     id_vars='method', var_name='split', value_name=score)
    sns.catplot(x='split', y=score, hue='method', data=result, kind='bar')
    
    # run portfolio_sorts backtests on algo signals
    benchnames = ['HML(mo)', 'ST_Rev(mo)','Mom(mo)','Mkt-RF(mo)']
    rebalbeg, rebalend = 19790601, 20190630
    for signal in models.keys():
        run_backtest(backtest, sql, signal, 1, benchnames, rebalbeg, rebalend, html=None, outdir=None)
    plt.show()

