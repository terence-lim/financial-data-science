"""Approximate Factor Models, VAR and TCN from FRED-MD

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
from finds.solve import integration_order

from settings import settings
imgdir = os.path.join(settings['images'], 'ts')
alf = Alfred(api_key=settings['fred']['api_key'])

# Transformation Codes, and Stationarity
qdf, qt = fred_qd(202004)      # from vintage April 2020
mdf, mt = fred_md()

print('Number of time series and suggested transformations, by tcode:')
tcodes = pd.concat([Series(alf.tcode_[i], name=i).to_frame().T
                    for i in range(1,8)], axis=0).fillna(False)
tcodes.join(qt['transform'].value_counts().rename('fred-qd'))\
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
pd.concat([tc.drop(columns='fred-md'), out['qd'], tc['fred-md'], out['md']],
          axis=1).fillna(0).astype(int)

## helper method to identifier series outliers
def as_inliers(x, method='iq10', bounds=False):
    """Returns boolean index indicating non-outliers in input series"""
    def nancmp(f, a, b):
        valid = ~np.isnan(a)
        valid[valid] = f(a[valid] , b)
        return valid

    x = np.array(x)
    if len(x.shape) == 1:
        lb, median, ub = np.nanpercentile(x, [25, 50, 75])
        if method.lower().startswith('tukey'):
            w = 1.5
            f = [lb - (w * (ub - lb)), ub + (w * (ub - lb))]
            if not bounds:
                f = (nancmp(np.greater_equal, x, f[0]) &
                     nancmp(np.less_equal, x, f[1]))
        elif method.lower() in 'iq10':
            w = float(re.sub('\D', '', method))
            f = [median - (w * (ub - lb)), median + (w * (ub - lb))]
            if not bounds:
                f = (nancmp(np.greater_equal, x,f[0]) &
                     nancmp(np.less_equal, x, f[1]))
        else:
            raise Exception("outliers method not in ['iq10', 'tukey']")
        return np.array(f)
    else:
        return np.hstack([as_inliers(x[:, i], method=method, bounds=bounds)\
                          .reshape(-1, 1) for i in range(x.shape[1])])

"""
We check for outliers in the transformed series prior to constructing
the factors. An outlier is defined as an observation that deviates
from the sample median by more than ten interquartile ranges. The
outliers are removed and treated as missing values.  
"""
## Retrieve FRED-MD and apply transformation codes
mdf, mt = fred_md() #201505
df = mdf.copy()
t = mt['transform']
beg = 19600301
end = 20141231
end = 20991231
freq = 'm'
#df.head().append(df.tail())

# Apply tcode transformations
transformed = []
for col in df.columns:
    transformed.append(alf.transform(df[col], tcode=t[col], freq=freq))
data = pd.concat(transformed, axis=1).iloc[2:]
cols = list(data.columns)
sample = data.index[(data.index >= beg) & (data.index <= end)]

# Filter Outliers and Summarize Missing and Outliers
outlier, missing = [], []    # to accumulate outlier and missing obs
for series_id in df.columns:
    f = as_inliers(data[series_id], method='iq10')
    g = data[series_id].notna()
    missing.extend([(date, series_id) for date in data.index[~g]])
    outlier.extend([(date, series_id, data.loc[date, series_id])
                    for date in data.index[~f & g]])
missing_per_row = data.isna().sum(axis=1)
outlier = DataFrame.from_records(outlier,
                                 columns=['date', 'series_id', 'value'])
missing = DataFrame.from_records(missing, columns=['date', 'series_id'])    
outlier['date'].value_counts()[:10]
    
#r = outlier['series_id'].value_counts()[:10]
#DataFrame(index=r.index,
#         data={'title':[alf.header(s) for s in r.index], 'months':r.values})
n = 10
out = pd.concat([outlier['date'].value_counts()[:n],
                 outlier['series_id'].value_counts()[:n], 
                 missing['date'].value_counts()[:n],
                 missing['series_id'].value_counts()[:n]])
from finds.printing import print_multicolumn
print_multicolumn(out, rows=n, latex=True)

# PCA-EM Approximate Factors
## Bai and Ng 2002
from finds.alfred import pcaEM, BaiNg, marginalR2
X = np.array(data.loc[sample])
X[~as_inliers(X, method='iq10')] = np.nan    # missing and outliers to NaN
x, model = pcaEM(X, kmax=None, p=2, tol=1e-21, echo=False)
r = BaiNg(x, p=2, standardize=True)
mR2 = marginalR2(x, standardize=True)
print(f"Explained by {r} factors: {np.sum(np.mean(mR2[:r,:], axis=1)):.3f}"
      f" ({len(x)} obs: {min(sample)}-{max(sample)}")
econ = []
for k in range(r):
    print()
    print(f"Factor:{1+k:2d}  Variance Explained={np.mean(mR2[k, :]):6.3f}")
    args = np.argsort(-mR2[k, :])
    for i, arg in enumerate(args[:10]):
        print(f"{mR2[k, arg]:6.3f} {cols[arg]:16s} {alf.header(cols[arg])}")
    econ.append(Series(data=x[:,args[0]], name=cols[args[0]],
                       index=pd.DatetimeIndex(sample.astype(str), freq='infer')))
econ = pd.concat(econ, axis=1)
econ = econ.sub(econ.mean()).div(econ.std(ddof=0))  # standardize

# extract factors
y = ((x-x.mean(axis=0).reshape(1,-1))/x.std(axis=0,ddof=0).reshape(1,-1))
u, s, vT = np.linalg.svd(y, full_matrices=False)
factors = DataFrame(u[:, :r], columns=np.arange(1, 1+r),
                    index=pd.DatetimeIndex(sample.astype(str), freq='infer'))
Series(s[:r]**2 / np.sum(s**2), index=np.arange(1, r+1), name='R2').to_frame().T

# equivalent with sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(r))])
pipe.fit(x)   # fit model on training data
Series(pipe.named_steps['pca'].explained_variance_ratio_,
       index=np.arange(1,r+1), name='R2').to_frame().T   # sanity check

# to indicate recession periods in the plots
usrec = alf('USREC', freq=freq)
usrec.index = pd.DatetimeIndex(usrec.index.astype(str), freq='infer')
g = (usrec != usrec.shift()).cumsum()[usrec.gt(0)].to_frame()
g = g.reset_index().groupby('USREC')['date'].agg(['first','last'])
vspans = [(v[0], v[1]) for k, v in g.iterrows()]
print(alf.header('USREC'))
DataFrame(vspans, columns=['Start', 'End'])

fig = plt.figure(figsize=(9, 10), num=1, clear=True)
for col in factors.columns:
    ax = fig.add_subplot(4, 2, col)
    flip = -np.sign(max(factors[col]) + min(factors[col])) # try match sign
    (flip*factors[col]).plot(ax=ax, color=f"C{col}")
    for a,b in vspans:
        ax.axvspan(a, b, alpha=0.3, color='grey')
    ax.legend([f"Factor {col} Estimate", 'NBER Recession'], fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"Factor Estimates {factors.index[0]:%b-%Y}:"
             f"{factors.index[-1]:%b-%Y}", fontsize=12)
plt.savefig(os.path.join(imgdir, 'approximate.jpg'))
plt.show()


# Vector Autoregression Model of the Extracted Factors
## predicting composites of, rather than individual, macroeconomic time series
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from finds.display import plot_bands

test_date = '2014-12-31'
train_data = factors[factors.index <= test_date].copy()
maxlags = 16
test_data = factors[(factors.index >= train_data.index[-maxlags]) &
                    (factors.index <= '2019-12-31')].copy() 
M = train_data.shape[1]
model = VAR(train_data)

## Selecting lag order
results = {p: model.fit(p) for p in [1, 2, 4, 12]}   # VAR(p) models
DataFrame({ic: model.fit(maxlags=maxlags, ic=ic).k_ar
           for ic in ['aic', 'fpe', 'hqic', 'bic']},
          index=['lag order selected by:'])\
          .rename_axis(columns='Information Criterion:')

## Plot impulse response function with confidence bands
irf = results[2].irf(maxlags)
fig, axes = plt.subplots(M, M, figsize=(9,10), num=1, clear=True,
                         sharex=True, sharey=True)
for impulse in range(M):
    for response in range(M):
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
for x in [train_data, test_data]:
    for i in range(maxlags, len(x)):
        # error of unconditional mean forecast
        mean_error[x.index[i]] = mse(x.iloc[i].values, train_data.mean().values)
        
        # error VAR(p) model forecasts
        for p in var_error:
            pred = results[p].forecast(x.iloc[(i-p):i].values, 1)
            var_error[p][x.index[i]] = mse([x.iloc[i].values], pred)

## Compute mean test and train set errors of all models
mean_error = Series(mean_error, name="TrainMean")
out = [Series({'Train Error': mean_error.loc[mean_error.index<=test_date].mean(),
               'Test Error': mean_error.loc[mean_error.index>test_date].mean()},
              name=mean_error.name)]
var_error = {p: Series(var_error[p], name=f"VAR({p})") for p in var_error}
for p, e in var_error.items():
    out.append(Series({'Train Error': e.loc[e.index<=test_date].mean(),
                       'Test Error': e.loc[e.index>test_date].mean()},
                      name=e.name))
out = pd.concat(out, axis=1).T.rename_axis(columns="1961-07-31...2019-12-31:")\
                           .rename_axis(index=f"(Test-split={test_date})")
print(out)

## Plot monthly mean squared error
fig, ax = plt.subplots(1, 1, figsize=(9, 5), num=1, clear=True)
ax.set_yscale('log')
mean_error.plot(ax=ax, c='C0', style='-')
for i, (p, v) in enumerate(var_error.items()):
    v.plot(ax=ax, c=f'C{i+1}', style='-')
for a,b in vspans:
    ax.axvspan(a, b, alpha=0.3, color='grey')
ax.set_title('Mean Squared Error of VAR Forecasts')
ax.axvline(test_date, color='black', linestyle='--')
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


## Create input data from training set
seq_len = 16    # length of each input sequence for TCN
train_exs = [train_data.iloc[i-(seq_len+1):i].values
             for i in range(seq_len+1, len(train_data))]
n_features = train_data.shape[1]

## Fit TCN models with increasing layers of convolution and dropout rates
max_layers = 4       # range of convolution block depths
dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # range of dropout rates
lr = 0.01            # initial learning rate
step_size = 300      # learning rate scheduler step size
res = []             # to collect results summaries
tcn_error = dict()   # to store prediction errors
for layers in range(1, max_layers+1):
    for dropout in dropouts:
        model = TCN(n_features, layers=[16]*layers, kernel_size=2,
                    dropout=dropout).to(device)
        print(model)

        # Set optimizer, loss and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1,
                                                    step_size=step_size)
        loss_function = nn.MSELoss()

        # Run training loop over num_epochs with batch_size
        batch_size = 8
        num_epochs = step_size*6
        for epoch in range(num_epochs):
    
            # shuffle indxs into batches
            idxs = np.arange(len(train_exs))
            random.shuffle(idxs)
            batches = [idxs[i:(i+batch_size)]
                       for i in range(0, len(idxs), batch_size)]
            batch = batches[0]
    
            # train by batch
            total_loss = 0.0
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
                print(epoch, layers, dropout, optimizer.param_groups[0]['lr'],
                      total_loss/len(batches))
            model.eval()

        # Compute MSE of one-period ahead forecast error in train and test sets
        e = dict()
        for x in [train_data, test_data]:
            for i in range(seq_len, len(x)):
                X = torch.tensor(x.iloc[(i-seq_len):i].values.T)\
                         .unsqueeze(0).float().to(device)
                pred = model(X)
                e[x.index[i]] = mse([x.iloc[i].values],
                                    pred[:,:,-1].cpu().detach().numpy())
        name=f"TCN{layers}_{dropout*100:.0f}"
        model.save(os.path.join(imgdir, name + '.pt'))
        e = Series(e, name=name)
        tcn_error[name] = e
        res.append(Series({'depth': layers, 'dropout': dropout,
                           'Train Error': float(e[e.index<=test_date].mean()),
                           'Test Error': float(e[e.index>test_date].mean())},
                          name=name))
        print(pd.concat(res, axis=1).T)
res = pd.concat(res, axis=1).T
print(model)

## Plot monthly mean squared error
fig, ax = plt.subplots(1, 1, figsize=(9, 5), num=1, clear=True)
ax.set_yscale('log')
legend = []
for i, layers in enumerate([1, 4]):
    for j, dropout in enumerate([0, 50]):
        tcn_error[f"TCN{layers}_{dropout}"].plot(ax=ax, c=f'C{i*2+j}')
        legend.append(f"depth={layers} dropout=0.{dropout}")
for a,b in vspans:
    ax.axvspan(a, b, alpha=0.3, color='grey')
ax.set_title('Mean Squared Error of TCN Forecasts')
ax.axvline(test_date, color='black', linestyle='--')
ax.legend(legend, loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'tcnmse.jpg'))
plt.show()


## Display Train and Test Error of TCN Models by dropout and depth
print(res.astype({'depth':int})\
      .pivot(index=['dropout'], columns=['depth'],
             values=['Train Error','Test Error'])\
      .swaplevel(0, 1, 1).sort_index(axis=1).to_string())  


## Plot Train and Test Error of TCN Models
fig, ax = plt.subplots(1, 1, figsize=(9, 5), num=1, clear=True)
for layers in np.unique(res.depth):
    for err, style in zip(['Train Error', 'Test Error'], ['-',':']):
        select = res['depth'].eq(layers)
        Series(index=res['dropout'][select], data=res[err][select].values,
               name=f"Layers={layers:.0f} ({err})")\
               .plot(ax=ax, color=f"C{layers:.0f}", style=style)
ax.set_title('Train and Test Error of TCN Models')
ax.set_ylabel('Mean Squared Error')
ax.set_xlabel('Dropout')
ax.legend(loc='lower right', fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(imgdir, 'tcn.jpg'))
plt.show()
