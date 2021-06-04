"""Comparing supervised learning models for regression

- subset selection, partial least squares, ridge, lasso regression
- cross validation, feature importances, dimension reduction
- gradient boosting, random boosting, ensembles
- sklearn, statsmodels, St Louis Fed FRED, GDP

Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from pandas.api.types import is_list_like, is_numeric_dtype
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error    
import time
import os
from finds.alfred import Alfred, fred_qd
from settings import settings
ECHO = False
imgdir = os.path.join(settings['images'], 'regression')
alf = Alfred(api_key=settings['fred']['api_key'], echo=ECHO)

# Get FRED-QD data
qdf, qt = fred_qd(202004)
df = qdf  #[list(qt.index[qt['factors']==1])]
t = qt['transform']
#df = df[qt[qt['factors']==1].index]
beg = 19620701
end = 20191231  # ignore 2020
freq = 'Q'   
transformed = []
for col in df.columns:
    transformed.append(alf.transform(df[col], tcode=t[col], freq=freq))
data = pd.concat(transformed, axis=1).iloc[2:]
c = list(data.columns)
data = data.loc[(data.index >= beg) & (data.index <= end)]

# Drop columns with missing data
missing = []
for series_id in df.columns:
    g = data[series_id].notna()
    missing.extend([(date, series_id) for date in data.index[~g]])
missing_per_row = data.isna().sum(axis=1)
missing = DataFrame.from_records(missing, columns=['date', 'series_id'])
print('original:', data.shape, 'dropna:', data.dropna(axis=1).shape)
data = data.dropna(axis=1)   # drop columns where missing values
print(missing['series_id'].value_counts())


# Split time series train and test set
# - train through 2014: test next five years (20 quarters) thru 2019
def ts_split(X, Y, end=20141231):
    return X[Y.index<=end], X[Y.index>end], Y[Y.index<=end], Y[Y.index>end]
Y = data['GDPC1'].iloc[1:]
X = data.shift(1).iloc[1:]
test = Series(name='test', dtype=float)    # collect test and train errors
train = Series(name='train', dtype=float)
final_models = {}                          # collect final fitted models


# Forward Selection
def forward_select(Y, X, selected, by='aic'):
    """helper to forward select next regressor"""
    remaining = [x for x in X.columns if x not in selected]
    results = []
    for x in remaining:
        r = sm.OLS(Y, X[selected + [x]]).fit()
        results.append({'select': x, 'aic': r.aic, 'bic': r.bic,
                        'rsquared': r.rsquared,
                        'rsquared_adj': r.rsquared_adj})
    return DataFrame(results).sort_values(by=by).iloc[0].to_dict()

# split train/test and forward select
X_train, X_test, Y_train, Y_test = ts_split(X, Y)
tic = time.time()
selected = []
models = {}
by = 'bic'
for i in range(1, 32):
    select = forward_select(Y_train, X_train, selected, by=by)
    models.update({i: select})
    selected.append(select['select'])
selected = DataFrame.from_dict(models, orient='index')

# report best bic, and show selection criteria
best = selected[[by]].iloc[selected[by].argmin()]
subset = selected.loc[:best.name].round(3)
subset.index = [alf.header(s) for s in subset['select']]
print('Subset Selected')
print(subset.to_latex())
subset

DataFrame.from_dict({n: {'series_id': s, 'description': alf.header(s)}
                     for n, s in selected.loc[:best.name, 'select'].items()},
                    orient='index').set_index('series_id')

# Plot BIC vs number selected
fig, ax = plt.subplots(num=1, clear=True, figsize=(5,3))
selected['bic'].plot(ax=ax, c='C0')
selected['aic'].plot(ax=ax, c='C1')
ax.plot(best.name, float(best), "or")
ax.legend(['BIC', 'AIC', f"best={best.name}"], loc='upper left')
ax.set_title(f"Forward Subset Selection with {by.upper()}")
bx = ax.twinx()
selected['rsquared'].plot(ax=bx, c='C2')
selected['rsquared_adj'].plot(ax=bx, c='C3')
bx.legend(['rsquared', 'rsquared_adj'], loc='center right')
bx.set_xlabel('# Predictors')
plt.savefig(os.path.join(imgdir, 'forward.jpg'))
plt.show()

# evaluate train and test mse
model = sm.OLS(Y_train, X_train[subset]).fit()
name = f"Forward Subset Regression (k={len(subset)})"
test[name] = mean_squared_error(Y_test, model.predict(X_test[subset]))
train[name] = mean_squared_error(Y_train, model.predict(X_train[subset]))
final_models[name] = model
DataFrame({'name': name, 'train': np.sqrt(train[name]),
           'test': np.sqrt(test[name])}, index=['RMSE'])

# Dimension Reduction: Partial Least Squares
# - split train and test, fit standard scaling using train set
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
X_train, X_test, Y_train, Y_test = ts_split(X, Y)
scale = StandardScaler().fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

## fit with 5-fold CV to choose n_components
n_splits=5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
mse = Series(dtype=float)
for i in np.arange(1, 31):
    pls = PLSRegression(n_components=i)
    score = cross_val_score(pls, scale.transform(X_train), Y_train,
                            cv=kf, scoring='neg_mean_squared_error').mean()
    mse.loc[i] = -score

## show CV results and best model
fig, ax = plt.subplots(clear=True, num=1, figsize=(5,3))
mse.plot(ylabel='Mean Squared Error', xlabel='Number of Components',
         title=f"PLS Regression with {n_splits}-fold CV", ax=ax)
best = mse.index[mse.argmin()]
ax.plot(best, mse.loc[best], "or")
ax.legend(['MSE', f"best={best}"])
plt.savefig(os.path.join(imgdir, 'pls.jpg'))
plt.show()

# evaluate train and test mse
model = PLSRegression(n_components=best).fit(X_train, Y_train)
name = f"PLS Regression"
test[name] = mean_squared_error(Y_test, model.predict(X_test))
train[name] = mean_squared_error(Y_train, model.predict(X_train))
final_models[name] = model
DataFrame({'name': name, 'train': np.sqrt(train[name]),
           'test': np.sqrt(test[name])}, index=['RMSE'])

# Ridge Regression
from sklearn.linear_model import Ridge, RidgeCV
alphas = 10**np.linspace(-2, -9, 100)*0.5  # for parameter tuning
X_train, X_test, Y_train, Y_test = ts_split(X, Y)

np.random.seed(42)
X_subset = X_train #X_train[np.random.choice(X_train.columns, 50)]
# Plot fitted coefficients (of forward selected subset) vs regularization
coefs = [Ridge(normalize=True, alpha=alpha)\
         .fit(X_subset, Y_train).coef_ for alpha in alphas]
fig, ax = plt.subplots(num=1, clear=True, figsize=(10,6))
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlabel('value of alpha regularization parameter')
ax.set_title('Ridge Regression fitted coefficients')
plt.savefig(os.path.join(imgdir, 'ridge.jpg'))
plt.show()
    
# RidgeCV LOOCV
model = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error',
                normalize=True, store_cv_values=True).fit(X_train, Y_train)
name = f"Ridge (alpha={model.alpha_:.3g})"
test[name] = mean_squared_error(Y_test, model.predict(X_test))
train[name] = mean_squared_error(Y_train, model.predict(X_train))
final_models[name] = model
DataFrame({'name': name, 'train': np.sqrt(train[name]),
           'test': np.sqrt(test[name])}, index=['RMSE'])
    
# Lasso Regression
from sklearn.linear_model import Lasso, LassoCV
alphas = 10**np.linspace(-2  yyyyyyyyyy, -9, 100)*0.5  # for parameter tuning
X_train, X_test, Y_train, Y_test = ts_split(X, Y)

# Plot fitted coefficients (of forward selected subset) vs regularization
coefs = [Lasso(max_iter=100000, normalize=True, alpha=alpha)\
         .fit(X_subset, Y_train).coef_  for alpha in alphas]
fig, ax = plt.subplots(num=3, clear=True, figsize=(10,6))
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlabel('value of alpha regularization parameter')
ax.set_title('Lasso fitted coefficients')
plt.savefig(os.path.join(imgdir, 'lasso.jpg'))
plt.show()

# LassoCV 10-Fold CV
model = LassoCV(alphas=None, cv=10, normalize=True, n_jobs=-1,
                max_iter=30000).fit(X_train, Y_train)
name = f"Lasso (alpha={model.alpha_:.3g})"
test[name] = mean_squared_error(Y_test, model.predict(X_test))
train[name] = mean_squared_error(Y_train, model.predict(X_train))
final_models[name] = model
DataFrame({'name': name, 'train': np.sqrt(train[name]),
           'test': np.sqrt(test[name])}, index=['RMSE'])

# Display nonzero coefs
nonzero = np.sum(np.abs(model.coef_) > 0)
argsort = np.flip(np.argsort(np.abs(model.coef_)))[:nonzero]
df = DataFrame({'series_id': X_train.columns[argsort],
                'desc': [alf.header(s) for s in X_train.columns[argsort]],
                'coef': model.coef_[argsort]}).round(6).set_index('series_id')
df
with pd.option_context("max_colwidth", 80):
    print(df.to_latex(columns=['coef', 'desc']))

# Gradient boost
from sklearn.ensemble import GradientBoostingRegressor
X_train, X_test, Y_train, Y_test = ts_split(X, Y)

# tune max_depth with 5-fold CV
n_splits=5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
mse = Series(dtype=float)
for i in range(1, 10): # tune max_depth for best performance
    boosted = GradientBoostingRegressor(max_depth=i, random_state=0)
    score = cross_val_score(boosted, X_train, Y_train, cv=kf,
                            scoring='neg_mean_squared_error').mean()
    mse.loc[i] = -score

fig, ax = plt.subplots(clear=True, num=1, figsize=(5,3))
mse.plot(ax=ax, ylabel='Mean Squared Error', xlabel='max depth',
         title=f"Gradient Boosting Regressor with {n_splits}-fold CV")
best = mse.index[mse.argmin()]
ax.plot(best, mse.loc[best], "or")
ax.legend(['mse', f"best={best}"])
plt.savefig(os.path.join(imgdir, 'boosting.jpg'))
plt.show()

# evaluate train and test MSE
name = f"Boosting (depth={best})"
model = GradientBoostingRegressor(max_depth=best,
                                  random_state=0).fit(X_train, Y_train)
test[name] = mean_squared_error(Y_test, model.predict(X_test))
train[name] = mean_squared_error(Y_train, model.predict(X_train))
final_models[name] = model
DataFrame({'name': name, 'train': np.sqrt(train[name]),
           'test': np.sqrt(test[name])}, index=['RMSE'])

# Show feature importance
top_n = 10
imp = Series(model.feature_importances_, index=X.columns).sort_values()
DataFrame.from_dict({i+1: {'importance': imp[s],
                           'series_id': s,
                           'description': alf.header(s)}
                     for i, s in enumerate(np.flip(imp.index[-top_n:]))},
                    orient='index')

# Random Forest
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, Y_train, Y_test = ts_split(X, Y)

# tune max_depth with 5-fold CV
n_splits=5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
mse = Series(dtype=float)
for i in range(3, 20): #tune for best performance
    model = RandomForestRegressor(max_depth=i, random_state=0)
    score = cross_val_score(model, X_train, Y_train, cv=kf,
                            scoring='neg_mean_squared_error').mean()
    mse.loc[i] = -score
    print(i, np.sqrt(abs(score)))

fig, ax = plt.subplots(clear=True, num=1, figsize=(5,3))
mse.plot(ax=ax, ylabel='MSE', xlabel='max depth',
         title=f"Random Forest Regressor with {n_splits}-fold CV")
best = mse.index[mse.argmin()]
ax.plot(best, mse.loc[best], "or")
ax.legend(['Mean Squared Error', f"best={best}"])
plt.savefig(os.path.join(imgdir, 'randomforest.jpg'))
plt.show()

name = f"RandomForest (depth={best})"
model = RandomForestRegressor(max_depth=best,
                              random_state=0).fit(X_train, Y_train)
test[name] = mean_squared_error(Y_test, model.predict(X_test))
train[name] = mean_squared_error(Y_train, model.predict(X_train))
final_models[name] = model
DataFrame({'name': name, 'train': np.sqrt(train[name]),
           'test': np.sqrt(test[name])}, index=['RMSE'])

# show top feature Importances
top_n = 20
imp = Series(model.feature_importances_, index=X.columns).sort_values()
DataFrame.from_dict({i+1: {'importance': imp[s],
                           'series_id': s,
                           'description': alf.header(s)}
                     for i, s in enumerate(np.flip(imp.index[-top_n:]))},
                    orient='index')

# Plot summary of model RMSE's
fig, ax = plt.subplots(num=1, clear=True, figsize=(10,6))
np.sqrt(train.rename('train').to_frame().join(test.rename('test')))\
  .sort_values('test').plot.barh(ax=ax, width=0.85)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_title('Regression RMSE')
ax.figure.subplots_adjust(left=0.35)
plt.savefig(os.path.join(imgdir, 'rmse.jpg'))
plt.show()
    
