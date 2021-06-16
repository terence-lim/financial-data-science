"""Approximate Factor Models, VAR and TCN from FRED-MD

Extract an approximate factor structure that reduces the number of
dimensions (variables) in a large set of economic time series
(FRED-MD), with number of components automatically selected by an
information criterion and outliers replaced with an EM approach.

Contrast Vector Autoregression and Temporal Convolutional Networks to
model the time-series persistance of the extracted factors, and
compare prediction errors.

- PCA, EM, vector autoregression, temporal convolutional networks
- Bai and Ng (2002), McCracken and Ng (2016), St Louis Fed FRED, and others

Terence Lim
License: MIT

"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import os
import re
import time
from datetime import datetime
from finds.alfred import Alfred, fred_md, fred_qd
from finds.solve import integration_order, is_inlier

from settings import settings
imgdir = os.path.join(settings['images'], 'ts')
alf = Alfred(api_key=settings['fred']['api_key'])

# Transformation Codes, and Stationarity
qdf, qt = fred_qd(202004) # 202004      # from vintage April 2020
mdf, mt = fred_md(201505) # 201505

print('Number of time series and suggested transformations, by tcode:')
tcodes = pd.concat([Series(alf.tcode_[i], name=i).to_frame().T
                    for i in range(1,8)], axis=0).fillna(False)
tcodes = tcodes.join(qt['transform'].value_counts().rename('fred-qd'))\
               .join(mt['transform'].value_counts().rename('fred-md'))\
               .fillna(0).astype({'fred-qd': int, 'fred-md': int})

# Estimate and Compare Integration Order
out = {}
for label, df, t in [['md', mdf, mt], ['qd', qdf, qt]]:
    stationary = dict()
    for series_id, tcode in t['transform'].items():
        if tcode in [1, 2, 3, 4, 5, 6]:
            s = df[series_id] if tcode <= 3 else np.log(df[series_id])
            order = integration_order(s.dropna(), pvalue=0.05)
            stationary[series_id] = {
                'tcode': tcode,
                'I(p)': order,
                'different': '*' if (tcode - 1) % 3 != order else '',
                'title': alf.header(series_id)}
            #print(series_id, order, tcode)
    stationary = DataFrame.from_dict(stationary, orient='index')
    stationary = stationary.sort_values(stationary.columns.to_list())
    c = stationary.groupby(['tcode','I(p)'])['title'].count().reset_index()
    out[label] = c.pivot(index='tcode', columns='I(p)',
                         values='title').fillna(0).astype(int)
    out[label].columns=[f"I({p})" for p in out[label].columns]
print('Series by tcode, transformations and estimated order of integration:')
c = pd.concat([tcodes.drop(columns='fred-md'), out['qd'], tcodes['fred-md'],
    	       out['md']], axis=1).fillna(0).astype(int)
c

# PCA-EM Algorithm for Approximate Factors Model - Bai and Ng 2002
from finds.alfred import pcaEM, BaiNg, marginalR2

for freq, df, t in [['monthly', mdf, mt], ['quarterly', qdf, qt]]:
    df = qdf.copy()
    t = qt['transform']

    # Apply tcode transformations
    transformed = []
    for col in df.columns:
        transformed.append(alf.transform(df[col], tcode=t[col], freq=freq[0]))
    data = pd.concat(transformed, axis=1).iloc[2:]
    cols = list(data.columns)
    sample = data.index[((np.count_nonzero(np.isnan(data), axis=1)==0) |
                         (data.index <= 20141231)) & (data.index >= 19600301)]

    # set missing and outliers in X to NaN
    X = np.array(data.loc[sample])
    X[~is_inlier(X, method='iq10')] = np.nan

    # compute PCA EM and auto select number of components, r
    x, model = pcaEM(X, kmax=None, p=2, tol=1e-21, echo=False)
    r = BaiNg(x, p=2, standardize=True)

    # show marginal R2's of series to each component
    mR2 = marginalR2(x, standardize=True)
    print(f"FRED-{freq[0].upper()}D {freq} series:")
    print(f"Explained by {r} factors: {np.sum(np.mean(mR2[:r,:], axis=1)):.3f}"
          f" ({len(x)} obs: {min(sample)}-{max(sample)})")
    for k in range(r):
        print()
        print(f"Factor:{1+k:2d}  Variance Explained={np.mean(mR2[k, :]):6.3f}")
        args = np.argsort(-mR2[k, :])
        for i, arg in enumerate(args[:10]):
            print(f"{mR2[k, arg]:6.3f} {cols[arg]:16s} {alf.header(cols[arg])}")


## Extract factors

# pipe.fit through 20141231, pipe.transform through 20201231
df, t = fred_md(202104) # 201505
sample_date = 20141231
data = []
for col in df.columns:
    data.append(alf.transform(df[col], tcode=t['transform'][col], freq='m'))
data = pd.concat(data, axis=1).iloc[2:]
cols = list(data.columns)
sample = data.index[((np.count_nonzero(np.isnan(data), axis=1)==0) |
                     (data.index <= sample_date)) & (data.index >= 19600301)]
train_sample = sample[sample <= sample_date]
test_sample = sample[sample <= 20191231]

# replace missing and outliers with PCA EM and fixed number of components r=8
r = 8
X = np.array(data.loc[train_sample])
X[~is_inlier(X, method='iq10')] = np.nan
x, model = pcaEM(X, kmax=r, p=0, echo=False)

# Extract factors with SVD
y = ((x-x.mean(axis=0).reshape(1,-1))/x.std(axis=0,ddof=0).reshape(1,-1))
u, s, vT = np.linalg.svd(y, full_matrices=False)
factors = DataFrame(u[:, :r], columns=np.arange(1, 1+r),
                    index=pd.DatetimeIndex(train_sample.astype(str), freq='M'))
Series(s[:r]**2 / np.sum(s**2), index=np.arange(1, r+1), name='R2').to_frame().T

# Equivalent to sklearn PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(r))])
pipe.fit(x)                        # fit model on training data
X = np.array(data.loc[sample])     # to transform on full sample data
x, model = pcaEM(X, kmax=8, p=0, echo=False)  # replace missing (not outlier)
factors = DataFrame(StandardScaler().fit_transform(pipe.transform(x)),
                    index=pd.DatetimeIndex(sample.astype(str), freq='infer'),
                    columns=np.arange(1, 1+r))
Series(pipe.named_steps['pca'].explained_variance_ratio_,
       index=np.arange(1,r+1), name='R2').to_frame().T   # sanity check

## To indicate recession periods in the plots
usrec = alf('USREC', freq='m')  
usrec.index = pd.DatetimeIndex(usrec.index.astype(str), freq='infer')
g = usrec.astype(bool) | usrec.shift(-1, fill_value=0).astype(bool)
g = (g != g.shift(fill_value=0)).cumsum()[g].to_frame()
g = g.reset_index().groupby('USREC')['date'].agg(['first','last'])
vspans = [(v[0], v[1]) for k, v in g.iterrows()]
print(alf.header('USREC'))
DataFrame(vspans, columns=['Start', 'End'])

## Plot extracted factors
fig = plt.figure(figsize=(9, 10), num=1, clear=True)
for col in factors.columns:
    ax = fig.add_subplot(4, 2, col)
    flip = -np.sign(max(factors[col]) + min(factors[col])) # try match sign
    (flip*factors[col]).plot(ax=ax, color=f"C{col}")
    for a,b in vspans:
        if b >= min(factors.index):
            ax.axvspan(max(a, min(factors.index)), min(b, max(factors.index)),
                       alpha=0.3, color='grey')
    ax.legend([f"Factor {col} Estimate", 'NBER Recession'], fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"Factor Estimates {factors.index[0]:%b-%Y}:"
             f"{factors.index[-1]:%b-%Y}", fontsize=12)
plt.savefig(os.path.join(imgdir, 'approximate.jpg'))
plt.show()


# Vector Autoregression Model
"""
- predicting multiple time series (of the extracted factors)
"""
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from finds.display import plot_bands

maxlags = 16
train_date = '2014-12-31'
train_index = factors.index[factors.index <= train_date]
test1_index = factors.index[(factors.index >= train_index[-maxlags]) &
                            (factors.index <= '2019-12-31')]
test2_index = factors.index[(factors.index >= test1_index[-maxlags])]
train_data = factors.loc[train_index].copy()
test1_data = factors.loc[test1_index].copy()
test2_data = factors.loc[test2_index].copy()
M = train_data.shape[1]
model = VAR(train_data, freq='M')
model

## Selecting lag order
"""
The lagged coefficients estimated from the Vector Autoregression produce a 
multi-period cumulative forecast 
"""
results = {p: model.fit(p) for p in range(1, maxlags+1)}   # VAR(p) models
DataFrame({ic: model.fit(maxlags=maxlags, ic=ic).k_ar
           for ic in ['aic', 'fpe', 'hqic', 'bic']},
          index=['lag order selected by:'])\
          .rename_axis(columns='Information Criterion:')

## Plot impulse response function with confidence bands
"""
- positive autocorrelations
"""
irf = results[2].irf(maxlags)
m = 4  #M
fig, axes = plt.subplots(m, m, figsize=(9,10), num=1, clear=True,
                         sharex=True, sharey=True)
for impulse in range(m):
    for response in range(m):
        plot_bands(mean=irf.cum_effects[:, impulse, response],
                   stderr=irf.cum_effect_stderr()[:, impulse, response],
                   hline=1 if impulse == response else 0,
                   title=None if impulse else f"Response of Factor {response}",
                   ylabel=None if response else f"Impulse from Factor {impulse}",
                   fontsize=3, ax=axes[impulse, response])
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'impulse.jpg'))
plt.show()

## Collect one-period ahead forecasts and errors in train and test sets
from sklearn.metrics import mean_squared_error as mse
mean_error = dict()
var_error= {p: dict() for p in results}
for x in [train_data, test1_data, test2_data]:
    for i in range(maxlags, len(x)):
        # error of unconditional mean forecast
        mean_error[x.index[i]] = mse(x.iloc[i].values, train_data.mean().values)
        
        # error VAR(p) model forecasts
        for p in var_error:
            pred = results[p].forecast(x.iloc[(i-p):i].values, 1)
            var_error[p][x.index[i]] = mse([x.iloc[i].values], pred)

## Compute mean test and train set errors of all VAR(p) models
errors = {0: Series(mean_error, name="TrainSampleMean")}  # VAR(0)
errors.update({p: Series(var_error[p], name=f"VAR({p})") for p in var_error})

out = [Series({'Train Error': e.loc[e.index <= train_date].mean(),
               'Test1 Error': e.loc[test1_index].mean(),
               'Test2 Error': e.loc[test2_index].mean()},
              name=e.name) for p, e in errors.items()]
out = pd.concat(out, axis=1).T.rename_axis(columns="1961-07-31...2019-12-31:")
out

## Plot Train and Test Error of TCN Models
fig, ax = plt.subplots(1, 1, figsize=(9, 5), num=1, clear=True)
ax.plot(np.arange(len(out)), out['Train Error'], color=f"C0")
ax.plot(np.arange(len(out)), out['Test1 Error'], color=f"C1")
ax.plot([], [], color=f"C2")
bx = ax.twinx()
bx.plot(np.arange(len(out)), out['Test2 Error'], color=f"C2")
ax.set_title(f'Train, Test1(2014-2019) and Test2(2020+) Errors of Var(p) Models')
ax.set_ylabel('Mean Squared Error')
bx.set_ylabel('Test2 sample period [2020-01 ... 2021-03]')
ax.set_xlabel('lags (0)')
ax.legend(['Train Error', 'Test1 Error', 'Test2 Error'], loc='center left')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'varerr.jpg'))
plt.show()


## Plot monthly mean squared error
fig, ax = plt.subplots(1, 1, figsize=(9, 5), num=1, clear=True)
ax.set_yscale('log')
for i, p in enumerate([0, 2, 4, 12]):
    errors[p].plot(ax=ax, c=f'C{i+1}', style='-')
for a,b in vspans:
    if b >= min(errors[0].index):
        ax.axvspan(max(a, min(errors[0].index)), min(b, max(errors[0].index)),
                   alpha=0.3, color='grey')
ax.set_title('Mean Squared Error of VAR Forecasts')
ax.axvline(max(train_index), color='black', linestyle='--')
ax.axvline(max(test1_index), color='black', linestyle='--')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'varmse.jpg'))
plt.show()

# Temporal 1D Convolutional Net (TCN)
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TCN(torch.nn.Module):
    class CausalConv1dBlock(torch.nn.Module):
        """Building block Conv1d, ReLU, skip, dropout, dilation and padding"""
        def __init__(self, in_channels, out_channels, kernel_size, dilation,
                     dropout):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size-1)*dilation, 0), 0),
                torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                                dilation=dilation),
                torch.nn.ReLU(),
                torch.nn.ConstantPad1d(((kernel_size-1)*dilation, 0), 0),
                torch.nn.Conv1d(out_channels, out_channels, kernel_size,
                                dilation=dilation),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout))
            self.skip = lambda x: x
            if in_channels != out_channels:     # downsample if necessary
                self.skip = torch.nn.Conv1d(in_channels, out_channels,1)

        def forward(self, x):
            return self.network(x) + self.skip(x)  # with skip connection


    def __init__(self, n_features, layers=[8, 8, 8], kernel_size=3, dropout=0.0):
        """TCN model by connecting multiple convolution layers"""
        super().__init__()
        c = n_features
        L = []
        for total_dilation, l in enumerate(layers):
            L.append(self.CausalConv1dBlock(in_channels=c,
                                            out_channels=l,
                                            kernel_size=kernel_size,
                                            dilation=2*(total_dilation+1),
                                            dropout=dropout))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, n_features, 1)

    def forward(self, x):
        """input is (B, n_features, L)), linear expects (B, * n_features)"""
        return self.classifier(self.network(x))

    def save(self, filename):
        """save model state to filename"""
        return torch.save(self.state_dict(), filename)

    def load(self, filename):
        """load model name from filename"""
        self.load_state_dict(torch.load(filename, map_location='cpu'))
        return self


## Form input data from training set
seq_len = 16    # length of each input sequence for TCN
train_exs = [train_data.iloc[i-(seq_len+1):i].values
             for i in range(seq_len+1, len(train_data))]
n_features = train_data.shape[1]

## Fit TCN models with increasing layers of convolution and dropout rates
layers = [1, 2, 3, 4]                      # range of convolution block layers
dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # range of dropout rates
batch_size = 8
step_size = 300      # learning rate scheduler step size
lr = 0.01            # initial learning rate
num_lr = 6
res = []             # to collect results summaries
tcn_error = dict()   # to store prediction errors
for layer in layers:
    for dropout in dropouts:
        # Set model, optimizer, loss function and learning rate scheduler
        model = TCN(n_features, layers=[16]*layer, kernel_size=2,
                    dropout=dropout).to(device)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1,
                                                    step_size=step_size)
        loss_function = nn.MSELoss()

        # Run training loop over num_epochs with batch_size
        num_epochs = step_size * num_lr
        for epoch in range(num_epochs):
            idxs = np.arange(len(train_exs))   # shuffle indxs into batches
            random.shuffle(idxs)
            batches = [idxs[i:(i+batch_size)]
                       for i in range(0, len(idxs), batch_size)]
    
            total_loss = 0.0               # train by batch
            model.train()
            for batch in batches:
                # input has shape (batch_size=8, n_features=8, seq_len=16)
                train_ex = torch.tensor([[train_exs[idx][seq] for idx in batch]
                                         for seq in range(seq_len+1)])\
                                .permute(1,2,0).float().to(device)
                model.zero_grad()
                X = train_ex[:,:,:-1]
                Y = train_ex[:,:,1:]
                output = model(X)
                loss = loss_function(output, Y)  # calculated over all outputs
                total_loss += float(loss)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch % (step_size//2)) == 0:
                print(epoch, layer, dropout, optimizer.param_groups[0]['lr'],
                      total_loss/len(batches))
            model.eval()

        # Compute MSE of one-period ahead forecast error in train and test sets
        e = dict()
        for x in [train_data, test1_data, test2_data]:
            for i in range(seq_len, len(x)):
                X = torch.tensor(x.iloc[(i-seq_len):i].values.T)\
                         .unsqueeze(0).float().to(device)
                pred = model(X)
                e[x.index[i]] = mse([x.iloc[i].values],
                                    pred[:,:,-1].cpu().detach().numpy())
        name=f"TCN{layer}_{dropout*100:.0f}"
        model.save(os.path.join(imgdir, name + '.pt'))
        e = Series(e, name=name)
        tcn_error[name] = e
        res.append(Series({'depth': layer, 'dropout': dropout,
                           'Train Error': float(e[e.index <= train_date].mean()),
                           'Test1 Error': float(e[test1_index].mean()),
                           'Test2 Error': float(e[test2_index].mean())},
                          name=name))
        #print(pd.concat(res, axis=1).T)
res = pd.concat(res, axis=1).T

## Plot monthly mean squared error
fig, ax = plt.subplots(1, 1, figsize=(9, 5), num=1, clear=True)
ax.set_yscale('log')
legend = []
for i, layer in enumerate([1, max(layers)]):
    for j, dropout in enumerate([0, int(100*max(dropouts))]):
        tcn_error[f"TCN{layer}_{dropout}"].plot(ax=ax, c=f'C{i*2+j}')
        legend.append(f"depth={layer} dropout=0.{dropout}")
for a,b in vspans:
    if a >= train_index[maxlags]:
        ax.axvspan(a, min(b, max(test2_index)), alpha=0.3, color='grey')
ax.set_title('Mean Squared Error of TCN Forecasts')
ax.axvline(max(train_index), color='black', linestyle='--')
ax.axvline(max(test1_index), color='black', linestyle='--')
ax.legend(legend, loc='upper left')
plt.tight_layout()
#plt.savefig(os.path.join(imgdir, 'tcnmse.jpg'))
plt.show()


## Display Train and Test Error of TCN Models by dropout and depth
res.astype({'depth':int})\
   .rename(columns={s + ' Error': s for s in ['Train', 'Test1', 'Test2']})\
   .pivot(index=['dropout'], columns=['depth'])\
   .swaplevel(0, 1, 1).round(4).sort_index(axis=1)


## Plot Train and Test Error of TCN Models
fig, ax = plt.subplots(1, 1, figsize=(9, 5), num=1, clear=True)
bx = ax.twinx()
for layer in np.unique(res.depth):
    select = res['depth'].eq(layer)
    Series(index=res['dropout'][select], data=res['Train Error'][select].values,
           name=f"Layers={layer:.0f} (Train Error)")\
           .plot(ax=ax, color=f"C{layer:.0f}", style='-', marker='o')
    Series(index=res['dropout'][select], data=res['Test1 Error'][select].values,
           name=f"Layers={layer:.0f} (Test1 Error)")\
           .plot(ax=ax, color=f"C{layer:.0f}", style='--', marker='s')
    ax.plot([],[], color=f"C{layer:.0f}", ls=':',
            label=f"Layers={layer:.0f} (Test2 Error)")
    Series(index=res['dropout'][select], data=res['Test2 Error'][select].values)\
           .plot(ax=bx, color=f"C{layer:.0f}", style=':')    
ax.set_title('Train and Test Error of TCN Models')
ax.set_ylabel('Mean Squared Error')
ax.set_xlabel('Dropout')
ax.legend(loc='upper center', fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'tcn.jpg'))
plt.show()


"""
# Filter Outliers and Summarize Missing and Outliers
Check for outliers in the transformed series prior to constructing
the factors: observations that deviate from the sample median by more
than ten interquartile ranges. The outliers are removed and treated as
missing values.
outlier, missing = [], []    # to accumulate outlier and missing obs
for series_id in data.columns:
    f = is_inlier(data[series_id], method='tukey')#'iq10')
    g = data[series_id].notna()
    missing.extend([(date, series_id) for date in data.index[~g]])
    outlier.extend([(date, series_id, data.loc[date, series_id])
                    for date in data.index[~f & g]])
missing_per_row = data.isna().sum(axis=1)
outlier = DataFrame.from_records(outlier,
                                 columns=['date', 'series_id', 'value'])
missing = DataFrame.from_records(missing, columns=['date', 'series_id'])    
outlier['date'].value_counts()[:10]
    
n = 10
out = pd.concat([outlier['date'].value_counts()[:n],
                 outlier['series_id'].value_counts()[:n], 
                 missing['date'].value_counts()[:n],
                 missing['series_id'].value_counts()[:n]])
from finds.printing import print_multicolumn
print_multicolumn(out, rows=n, latex=True)
"""
