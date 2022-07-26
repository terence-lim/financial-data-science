"""Approximate Factor Models, VAR and TCN

- PCA, EM
- Approximate factors and selection: Bai and Ng (2002), McCracken and Ng (2016)
- vector autoregression, temporal convolutional networks

Copyright 2022, Terence Lim

MIT License
"""
import finds.display
def show(df, latex=True, ndigits=4, **kwargs):
    return finds.display.show(df, latex=latex, ndigits=ndigits, **kwargs)
figext = '.jpg'

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import os
import re
import time
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from finds.alfred import Alfred, fred_md, fred_qd, factors_em, remove_outliers,\
    mrsq, select_bai_ng
from finds.display import plot_bands
from finds.recipes import integration_order

from conf import VERBOSE, credentials, paths
imgdir = os.path.join(paths['images'], 'ts')
alf = Alfred(api_key=credentials['fred']['api_key'], verbose=-1)


# FRED-MD, Transformation Codes and Stationarity

## Transformation Codes, and Stationarity
qd_df, qd_codes = fred_qd() # 202004
md_df, md_codes = fred_md() # 201505

print('Number of time series and suggested transformations, by tcode:')
tcodes = pd.concat([Series(alf._tcode[i], name=i).to_frame().T
                    for i in range(1, 8)], axis=0).fillna(False)
tcodes = tcodes.join(qd_codes['transform']\
                     .value_counts()\
                     .rename('fred-qd'))\
               .join(md_codes['transform']\
                     .value_counts()\
                     .rename('fred-md'))\
               .fillna(0).astype({'fred-qd': int, 'fred-md': int})\
               .rename_axis(index='tcode')
tcodes


## Estimate and Compare Integration Order
out = {}
for label, df, transforms in [['md', md_df, md_codes['transform']],
                              ['qd', qd_df, qd_codes['transform']]]:
    stationary = dict()
    for series_id, tcode in transforms.items():
        if tcode in range(1, 8):
            s = np.log(df[series_id]) if tcode in [4, 5, 6] else df[series_id]
            order = integration_order(s.dropna(), pvalue=0.05)
            expected_order = 2 if tcode == 7 else ((tcode - 1) % 3)
            stationary[series_id] = {
                'tcode': tcode,
                'I(p)': order,
                'different': order - expected_order,
                'title': alf.header(series_id)}
            #print(series_id, order, tcode)
    stationary = DataFrame.from_dict(stationary, orient='index')
    stationary = stationary.sort_values(stationary.columns.to_list())
    c = stationary.groupby(['tcode','I(p)'])['title'].count().reset_index()
    out[label] = c.pivot(index='tcode', columns='I(p)',
                         values='title').fillna(0).astype(int)
    out[label].columns=[f"I({p})" for p in out[label].columns]
print('Series by tcode, transformations and estimated order of integration:')
results = pd.concat([tcodes.drop(columns='fred-md'),
                     out['qd'],
                     tcodes['fred-md'],
                     out['md']], axis=1).fillna(0).astype(int)
show(results,
     caption='FRED-MD order of integration, transformations and frequency')
show(stationary[stationary['different'] > 0],
     max_colwidth=60,
     caption='FRED-MD series with unit root after transformations')


# Verify BaiNg implemention on published FRED-MD and FRED-QD reports
qd_df, qd_codes = fred_qd(202004)
md_df, md_codes = fred_md(201505)
for freq, df, transforms in [['monthly', md_df, md_codes['transform']],
                             ['quarterly', qd_df, qd_codes['transform']]]:    
    # Apply tcode transformations
    transformed = []
    for col in df.columns:
        transformed.append(alf.transform(df[col],
                                         tcode=transforms[col],
                                         freq=freq[0]))
    data = pd.concat(transformed, axis=1).iloc[2:]
    cols = list(data.columns)
    sample = data.index[((np.count_nonzero(np.isnan(data), axis=1)==0)
                         | (data.index <= 20141231))
                        & (data.index >= 19600301)]

    # set missing and outliers in X to NaN
    X = data.loc[sample]
    X = remove_outliers(X)

    # compute factors EM and auto select number of components, r
    Z = factors_em(X, p=2, verbose=1)
    r = select_bai_ng(Z, p=2)
    
    # show marginal R2's of series to each component
    mR2 = mrsq(Z, r).to_numpy()
    show(DataFrame({'selected': r,
                    'variance explained': np.sum(np.mean(mR2[:, :r], axis=0)),
                    'start': min(sample),
                    'end': max(sample),
                    'obs': Z.shape[0],
                    'series': Z.shape[1]},
                   index=[f'factors']),
         latex=False,
         caption=f"FRED-{freq[0].upper()}D {freq} series:")

    for k in range(r):
        args = np.argsort(-mR2[:, k])
        show(DataFrame.from_dict({mR2[arg, k].round(4):
                                  {'series': cols[arg],
                                   'description': alf.header(cols[arg])}
                                  for arg in args[:10]},
                                 orient='index'),
             latex=False,
             caption=f"Factor:{1+k} Variance Explained={np.mean(mR2[:,k]):.4f}")


## Sanity check Extract factors: SVD == PCA

## pipe.fit through 20141231, pipe.transform through 20201231
df, t = fred_md()  #202104 # 201505
transforms = t['transform']

sample_date = 20141231
data = []
for col in df.columns:
    data.append(alf.transform(df[col], tcode=transforms[col], freq='m'))
data = pd.concat(data, axis=1).iloc[2:]
cols = list(data.columns)
sample = data.index[((np.count_nonzero(np.isnan(data), axis=1)==0) |
                     (data.index <= sample_date)) & (data.index >= 19600301)]
train_sample = sample[sample <= sample_date]
test_sample = sample[sample <= 20191231]

## replace missing and outliers with PCA EM and fixed number of components r=8
r = 8
X = np.array(data.loc[train_sample])
X[~not_outlier(X, method='iq10')] = np.nan
x, model = pcaEM(X, kmax=r, p=0, verbose=0)

## Extract factors with SVD
y = ((x-x.mean(axis=0).reshape(1,-1))/x.std(axis=0,ddof=0).reshape(1,-1))
u, s, vT = np.linalg.svd(y, full_matrices=False)
#factors = DataFrame(u[:, :r], columns=np.arange(1, 1+r),
#                    index=pd.DatetimeIndex(train_sample.astype(str), freq='M'))
Series(s[:r]**2 / np.sum(s**2), index=np.arange(1, r+1), name='R2').to_frame().T

## Equivalent to sklearn PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(r))])
pipe.fit(x)                        # fit model on training data
X = np.array(data.loc[sample])     # to transform on full sample data
X, model = pcaEM(X, kmax=8, p=0, verbose=0)  # replace missing (not outlier)
factors = DataFrame(StandardScaler().fit_transform(pipe.transform(X)),
                    index=pd.DatetimeIndex(sample.astype(str), freq='infer'),
                    columns=np.arange(1, 1+r))
Series(pipe.named_steps['pca'].explained_variance_ratio_,
       index=np.arange(1,r+1), name='R2').to_frame().T   # sanity check



## Retrieve recession periods from FRED
vspans = alf.date_spans('USREC')
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
plt.savefig(os.path.join(imgdir, 'approximate' + figext))
plt.show()



# Vector Autoregression Model
"""
- predicting multiple time series (of the extracted factors)

Contrast Vector Autoregression and Temporal Convolutional Networks to
model the time-series persistance of the extracted factors, and
compare prediction errors.

"""
maxlags = 16
train_split = '2014-12-31'   # training period up to this date
test_split = '2019-12-31'    # split test period into two (due to COVID)
train_index = factors.index[factors.index <= train_split]
test1_index = factors.index[(factors.index >= train_index[-maxlags]) &
                            (factors.index <= test_split)]
test2_index = factors.index[(factors.index >= test1_index[-maxlags])]
test_end = max(test2_index).strftime('%Y-%m-%d')
train_data = factors.loc[train_index].copy()
test1_data = factors.loc[test1_index].copy()
test2_data = factors.loc[test2_index].copy()
M = train_data.shape[1]      # M is number of vectors to autoregress
model = VAR(train_data, freq='M')

## Auto Selecting lag order
"""
The lagged coefficients estimated from the Vector Autoregression produce a 
multi-period cumulative forecast 
"""
show(DataFrame({ic: model.fit(maxlags=maxlags, ic=ic).k_ar
                for ic in ['aic', 'fpe', 'hqic', 'bic']},
               index=['optimal p:'])\
     .rename_axis(columns='IC:'),
     caption="Optimal number of VAR(p) lags selected by Information Criterion")


## Collect one-period ahead forecasts and errors in train and test sets
results = {p: model.fit(p) for p in range(1, maxlags+1)}   # VAR(p) models
from sklearn.metrics import mean_squared_error as mse
mean_error = dict()   # to collect unconditional mean error each month
var_error= {p: dict() for p in results}  # collect error each month for each p

for sample in [train_data, test1_data, test2_data]:  # loop over 3 subsamples
    
    for i in range(maxlags, len(sample)):  # start at maxlag'th obs
    
        # accumulate to error of unconditional mean forecast
        mean_error[sample.index[i]] = mse(sample.iloc[i].values,
                                          train_data.mean().values)
        
        # accumulate to error of VAR(p) model forecasts
        for p in var_error:

            # include p prior for VAR(p) to forecast i'th obs
            pred = results[p].forecast(sample.iloc[(i-p):i].values, 1)
            var_error[p][sample.index[i]] = mse([sample.iloc[i].values], pred)

## Collect mean test and train set errors of all VAR(p) models
errors = {0: Series(mean_error, name="TrainSampleMean")}  # VAR(0)
errors.update({p: Series(var_error[p], name=f"VAR({p})")
               for p in var_error})
out = [Series({'Train Error': e.loc[e.index <= train_split].mean(),
               'Test1 Error': e.loc[test1_index].mean(),
               'Test2 Error': e.loc[test2_index].mean()},
              name=e.name) for p, e in errors.items()]
out = pd.concat(out, axis=1).T.rename_axis(columns="1961-07-31...2019-12-31:")
out

## Plot Train and Test Error of all VAR(p) Models
fig, ax = plt.subplots(1, 1, figsize=(10, 5), num=1, clear=True)
ax.plot(np.arange(len(out)), out['Train Error'], color="C0")
ax.plot(np.arange(len(out))+.3, out['Test1 Error'], color="C1")
ax.plot([], [], color="C2")   # dummy for legend labels
argmin = out['Test1 Error'].argmin()
ax.plot(argmin, out.iloc[argmin]['Test1 Error'], 'o', color="C1")
bx = ax.twinx()
bx.plot(np.arange(len(out)), out['Test2 Error'], color=f"C2")
ax.set_title(f'Var(p) Forecast Errors in Train, Test1 and Test2 Periods')
ax.set_ylabel(f'MSE of Train and Test1 [{train_split}:{test_split}]')
bx.set_ylabel(f'MSE of Test2 period [{test_split}:{test_end}]')
ax.set_xlabel('lag order of VAR(p)')
ax.legend(['Train Error', 'Test1 Error', 'Test2 Error',
           f'Test1 Min Error (={out.iloc[argmin]["Test1 Error"]:.4f})'],
          loc='center right')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'varerr' + figext))
plt.show()


# Temporal 1D Convolutional Net (TCN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TCN(torch.nn.Module):
    class CausalConv1dBlock(torch.nn.Module):
        """Conv1d block with ReLU, skip, dropout, dilation and padding"""
        def __init__(self, in_channels, out_channels, kernel_size, dilation,
                     dropout):
            super().__init__()

            print('kernel', kernel_size, 'dilation', dilation)
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
            if in_channels != out_channels:   # downsample for skip if necessary
                self.skip = torch.nn.Conv1d(in_channels, out_channels, 1)

        def forward(self, x):
            return self.network(x) + self.skip(x)  # with skip connection


    def __init__(self, n_features, blocks, kernel_size, dropout):
        """TCN model by connecting multiple convolution layers"""
        super().__init__()
        in_channels = n_features
        L = []
        for dilation, hidden in enumerate(blocks):
            L.append(self.CausalConv1dBlock(in_channels=in_channels,
                                            out_channels=hidden,
                                            kernel_size=kernel_size,
                                            dilation=2**dilation,
                                            dropout=dropout))
            in_channels = hidden
        self.network = torch.nn.Sequential(*L) if L else lambda x: x
        if L:
            self.classifier = torch.nn.Conv1d(in_channels, n_features, 1)
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.ConstantPad1d((kernel_size-1, 0), 0),                
                torch.nn.Conv1d(in_channels, n_features, kernel_size))

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
## Fit TCN models with increasing layers of convolution and dropout rates
n_features = train_data.shape[1]    # number of input planes
batch_size = 8
step_size = 100      # learning rate scheduler step size
lr = 0.01            # initial learning rate
num_lr = 6
res = []             # to collect results summaries
tcn_error = dict()   # to store prediction errors

kernel_sizes = [1, 2, 4]
dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
blocks = [0, 1, 2]
for block in blocks:
    for parm in (dropouts if block else kernel_sizes):
        if block:
            dropout = parm
            kernel_size = 3
        else:
            dropout = 0.0
            kernel_size = parm
        modelname = f"TCN{block}_{kernel_size}_{dropout*100:.0f}"

        # Set model, optimizer, loss function and learning rate scheduler
        model = TCN(n_features=n_features, blocks=[n_features]*block,
                    kernel_size=kernel_size, dropout=dropout).to(device)
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
#                train_ex = torch.tensor([[train_exs[idx][seq] for idx in batch]
#                                         for seq in range(seq_len+1)])\
#                                .permute(1,2,0).float().to(device)
                # Creating a tensor from a list of numpy.ndarrays is extremely
                # slow. Please consider converting the list to a single
                # numpy.ndarray with numpy.array() before converting to a tensor
                nparray = np.array([[train_exs[idx][seq] for idx in batch]
                                    for seq in range(seq_len+1)])
                train_ex = torch.tensor(nparray)\
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
                print(epoch, num_epochs, block, dropout,
                      optimizer.param_groups[0]['lr'], total_loss/len(batches))
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
        model.save(os.path.join(imgdir, modelname + '.pt'))
        e = Series(e, name=modelname)
        tcn_error[modelname] = e
        res.append(Series({
            'blocks': block, 'dropout': dropout, 'kernel_size': kernel_size,
            'Train Error': float(e[e.index <= train_split].mean()),
            'Test1 Error': float(e[test1_index].mean()),
            'Test2 Error': float(e[test2_index].mean())}, name=modelname))
        #print(pd.concat(res, axis=1).T)
res = pd.concat(res, axis=1).T.astype({'blocks': int, 'kernel_size': int})
res


## Plot monthly mean squared error
fig, ax = plt.subplots(1, 1, figsize=(10, 5), num=1, clear=True)
ax.set_yscale('log')
legend = []
for col, (modelname, parm) in enumerate(res.iterrows()):
    if parm.dropout in [0.0, 0.5]:
        tcn_error[modelname].plot(ax=ax, c=f'C{col}', lw=.5, ls='-')
        legend.append(f"depth={parm.blocks} kernel={parm.kernel_size}, "
                      f"dropout=.{100*parm.dropout:.0f}")
for a,b in vspans:
    if a >= train_index[maxlags]:
        ax.axvspan(a, min(b, max(test2_index)), alpha=0.3, color='grey')
ax.set_title('Mean Squared Error of TCN Predictions')
ax.axvline(max(train_index), color='black', linestyle='--')
ax.axvline(max(test1_index), color='black', linestyle='--')
ax.legend(legend, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'tcnmse' + figext))
plt.show()

## Display train and test error of TCN models by dropout and depth parameters
res[res['blocks'].gt(0)]\
    .rename(columns={s+' Error': s for s in ['Train','Test1','Test2']})\
    .pivot(index=['dropout'], columns=['blocks'])\
    .drop(columns=['kernel_size'])\
    .swaplevel(0, 1, 1).round(4).sort_index(axis=1)

res[res['blocks'].eq(0)]\
    .rename(columns={s+' Error': s for s in ['Train','Test1','Test2']})\
    .pivot(index=['kernel_size'], columns=['blocks'])\
    .drop(columns=['dropout'])\
    .swaplevel(0, 1, 1).round(4).sort_index(axis=1)

## Plot Train and Test Error of TCN Models
fig, ax = plt.subplots(1, 1, figsize=(10, 5), num=1, clear=True)
bx = ax.twinx()
col = 0
for block in np.unique(res.blocks):
    select_block = res['blocks'].eq(block)
    for kernel_size in np.unique(res['kernel_size'][select_block]):
        select = res['blocks'].eq(block) & res['kernel_size'].eq(kernel_size)

        Series(index=res['dropout'][select], 
               data=res['Train Error'][select].values,
               name=f"Layers={block:.0f} (Train Error)")\
               .plot(ax=ax, color=f"C{col}", style='-', marker='o')
        Series(index=res['dropout'][select], 
               data=res['Test1 Error'][select].values,
               name=f"Layers={block:.0f} (Test1 Error)")\
               .plot(ax=ax, color=f"C{col}", style='--', marker='s')
        ax.plot([],[], color=f"C{col}", ls=':',
                label=f"Layers={block:.0f} (Test2 Error)")
        Series(index=res['dropout'][select], 
               data=res['Test2 Error'][select].values)\
               .plot(ax=bx, color=f"C{col}", style=':')
        col = col + 1    
ax.set_title('Train and Test Error of TCN Models')
ax.set_ylabel('Mean Squared Error')
ax.set_xlabel('Dropout')
ax.legend(loc='upper center', fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'tcn' + figext))
plt.show()
