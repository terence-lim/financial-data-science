"""LSTM, State Space, and Mixture Models on economic series

Apply and compare methods for extracting lower-dimensional states
from a large panel of economic time series: State space models, Hidden
Markov Models, Gaussian mixtures, and LSTM networks.

Use BIC criterion, visualize the extracted hidden states/latent factors 
in recessionary and economic time periods, and compare persistence.

- Long Short-Term Memory network, hidden states, state space model, mixture models
- BIC, log-likelihood, mixed-frequency
- pytorch, hmmlearn, statsmodels, sklearn, FRED-MD
- Chen, Pelger and Zhu (2020) and others

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
import statsmodels.api as sm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from finds.alfred import fred_md, fred_qd, Alfred
from finds.learning import hmm_summary
from settings import settings
imgdir = os.path.join(settings['images'], 'rnn')

# Load and pre-process time series from FRED
alf = Alfred(api_key=settings['fred']['api_key'])

# to indicate recession periods in the plots
usrec = alf('USREC', freq='m')
usrec.index = pd.DatetimeIndex(usrec.index.astype(str), freq='infer')
g = usrec.astype(bool) | usrec.shift(-1, fill_value=0).astype(bool)
g = (g != g.shift(fill_value=0)).cumsum()[g].to_frame()
g = g.reset_index().groupby('USREC')['date'].agg(['first','last'])
vspans = [(v[0], v[1]) for k, v in g.iterrows()]

# Retrieve FRED-MD series and apply tcode transformations
beg = 19600301
end = 20200131

df, t = fred_md(202004)      # from vintage April 2020
data = []
for col in df.columns:
    data.append(alf.transform(df[col], tcode=t['transform'][col], freq='m'))
mdf = pd.concat(data, axis=1).iloc[2:]

mdata = mdf[(mdf.index >= beg) & (mdf.index <= end)].dropna(axis=1)
mdata = (mdata - mdata.mean(axis=0)) / mdata.std(axis=0, ddof=0)
mdata.index = pd.DatetimeIndex(mdata.index.astype(str), freq='m')
mdata

df, t = fred_qd(202004)      # from vintage April 2020
data = []
for col in df.columns:
    data.append(alf.transform(df[col], tcode=t['transform'][col], freq='q'))
df = pd.concat(data, axis=1).iloc[2:]

qdata = df[(df.index >= beg) & (df.index <= end)].dropna(axis=1)
qdata.index = pd.DatetimeIndex(qdata.index.astype(str), freq='q')
qdata


# LSTM pytorch for time series
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=n_features)

    def forward(self, x, hidden_state=None):
        """
        x: shape (seq_len, batch, input_siz)
        h: of shape (num_layers * num_directions, batch, hidden_size)
        c: of shape (num_layers * num_directions, batch, hidden_size)
        output: shape (seq_len, batch, num_directions * hidden_size)
        """
        output, (h, c) = self.lstm(x, hidden_state)
        return self.linear(output), (h.detach(), c.detach())

# create input data for LSTM, with sequence length 12 (months)
seq_len = 16
train_exs = [mdata.iloc[i-(seq_len+1):i].values
             for i in range(seq_len+1, len(mdata))]
n_features = mdata.shape[1]

hidden_factors = dict()
prediction_errors = dict()
for hidden_size in [1,2,3,4]:
    model = LSTM(n_features=n_features, hidden_size=hidden_size).to(device)
    print(model)

    # Set optimizer and learning rate scheduler, with step_size=30
    lr, num_lr, step_size = 0.001, 3, 400
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                gamma=0.1)
    loss_function = nn.MSELoss()

    batch_size, num_epochs = 32, step_size*num_lr
    for i in tqdm(range(num_epochs)):   # Run training loop per epoch
        idx = np.arange(len(train_exs))  # shuffle indxs into batches
        random.shuffle(idx)
        batches = [idx[i:(i+batch_size)] for i in range(0,len(idx),batch_size)]
        total_loss = 0.0   
        model.train() 
        for batch in batches:    # train each batch
            # train_ex input has shape (seq_len, batch_size=16, n_features)
            train_ex = torch.tensor([[train_exs[idx][seq] for idx in batch]
                                     for seq in range(seq_len+1)]).float()
            model.zero_grad()
            y_pred, hidden_state = model.forward(train_ex[:-1].to(device))
            loss = loss_function(y_pred[-1], train_ex[-1].to(device))
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # collect predictions and hidden states, and compute mse
    with torch.no_grad():    # reduce memory consumption for eval
        hidden_state = []
        prediction_error = []
        mse = nn.MSELoss()
        for i in range(seq_len+1, len(mdata)):
            # single test example of shape (seq_len=12, batch_size=1, n_features)
            test_ex = torch.tensor(mdata[i-(seq_len+1):i].values)\
                           .float().unsqueeze(dim=1).to(device)
            y_pred, (h, c) = model.forward(test_ex[:-1], None)
            prediction_error.append(float(mse(y_pred[-1], test_ex[-1])))
            hidden_state.append(h[0][0].cpu().numpy())
    hidden_factors[hidden_size] = DataFrame(
        hidden_state,index=mdata.index[(1+seq_len):len(mdata)])
    prediction_errors[f"Hidden Size {hidden_size}"] = np.mean(prediction_error)
    print(prediction_errors)

# Plot hidden states process
fig, axes = plt.subplots(len(hidden_factors), 1, figsize=(9,10),num=1,clear=True)
for hidden_factor, ax in zip(hidden_factors.values(), axes):
    hidden_factor.plot(ax=ax, style='--', legend=False)
    for a,b in vspans:
        if a >= min(hidden_factor.index):
            ax.axvspan(a, min(b, max(hidden_factor.index)), alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(f"LSTM with hidden_size = {len(hidden_factor.columns)}")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"LSTM Hidden States Process", fontsize=12)
plt.savefig(os.path.join(imgdir, 'lstm.jpg'))
plt.show()

# Dynamic Factor Models
"""
- can be cast into state space form and estimated via Kalman Filter
   https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_dfm_coincident.html

- fit (linear) dynamic factor model with DynamicFactorMQ 
  https://www.statsmodels.org/devel/generated/statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ.html

- allows for
  - mixed frequency data (for nowcasting)
  - EM algorithm by default accomodates larger data sets:  Kalman filter
    and optimizing of likelihood with quasi-Newton methods slow
"""
dynamic_factors = dict()
for i in [1, 2, 3, 4]:
    mod = sm.tsa.DynamicFactorMQ(endog=mdata,
                                 factors=1,                # num factor blocks
                                 factor_multiplicities=i,  # num factors in block
                                 factor_orders=2,          # order of factor VAR
                                 idiosyncratic_ar1=False)  # False=white noise
    fitted = mod.fit_em(disp=20, maxiter=200, full_output=True)
    dynamic_factors[i] = DataFrame(fitted.factors.filtered.iloc[seq_len+1:])
    dynamic_factors[i].columns = list(np.arange(len(dynamic_factors[i].columns)))
    mse = nn.MSELoss()
    prediction_errors[f"Dynamic Factors {i}"] = float(
        mse(torch.tensor(fitted.fittedvalues.iloc[mod.factor_orders+1:].values),
            torch.tensor(mdata.iloc[mod.factor_orders+1:].values)))
#print(fitted.summary(0))

# Plot dynamic factors
fig, axes = plt.subplots(len(dynamic_factors),1,figsize=(9,10),num=1,clear=True)
for dynamic_factor, ax in zip(dynamic_factors.values(), axes):
    dynamic_factor.plot(ax=ax, style='--', legend=False)
    for a,b in vspans:
        if a >= min(dynamic_factor.index):
            ax.axvspan(a, min(b, max(dynamic_factor.index)), alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(f"Dynamic Factors = {len(dynamic_factor.columns)}")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"Fitted Dynamic Factors ", fontsize=12)
plt.savefig(os.path.join(imgdir, 'dynamic.jpg'))
plt.show()


# Correlation of LSTM hidden state process with (linear) dynamic factors
rsq = dict()
for k, hidden_factor in hidden_factors.items():
    rsq[k] = [sm.OLS(y, sm.add_constant(dynamic_factors[len(dynamic_factors)]))\
              .fit().rsquared for _, y in hidden_factor.iteritems()]
print('Average variance of LSTM hidden states explained by dynamic factors')
DataFrame({k: np.mean(r) for k, r in rsq.items()}, index=['R-square'])\
    .rename_axis("# hidden states in LSTM:", axis=1)

# Mixed Frequency Dynamic Factor Model
scaler = StandardScaler().fit(qdata['GDPC1'].values.reshape((-1, 1)))
gdp = DataFrame(scaler.transform(qdata['GDPC1'].values.reshape((-1, 1))),
                index = qdata.index, columns=['GDPC1'])

mod = sm.tsa.DynamicFactorMQ(endog=mdata,
                             endog_quarterly=gdp,
                             factors=1,                # num factor blocks
                             factor_multiplicities=8,  # num factors in block
                             factor_orders=2,          # order of factor VAR
                             idiosyncratic_ar1=False)  # False=white noise
fitted = mod.fit_em(disp=1, maxiter=200, full_output=True)
dynamic_factor = DataFrame(fitted.factors.filtered.iloc[seq_len+1:])
dynamic_factor.columns = list(np.arange(len(dynamic_factor.columns)))

## Plot fitted GDP values in 2007-2010
beg = '2006-12-31'
end = '2010-12-31'
fig, ax = plt.subplots(figsize=(9,5), num=1, clear=True)
y = fitted.fittedvalues['GDPC1']
y = y[(y.index > beg) & (y.index <= end)]
ax.plot_date(y.index, (scaler.inverse_transform(y)/3).cumsum(), fmt='-o', color='C0')
x = gdp.copy()
x.index = pd.DatetimeIndex(x.index.astype(str), freq=None)
x = x[(x.index > beg) & (x.index <= end)]
ax.plot_date(x.index, scaler.inverse_transform(x).cumsum(), fmt='-o',color='C1')
ax.legend(['monthly fitted', 'quarterly actual'], loc='upper left')
ax.set_title('Quarterly GDP and Fitted Monthly Estimates from Dynamic Factor Model')
plt.tight_layout()
plt.savefig(os.path.join(imgdir, f"mixedfreq.jpg"))
plt.show()

# Show "Nowcast" of GDP
Series({'Last Date of Monthly Data:': mdata.index[-1].strftime('%Y-%m-%d'),
        'Last Date of Quarterly Data:': qdata.index[-1].strftime('%Y-%m-%d'),
        'Forecast of Q1 GDP quarterly rate':
        scaler.inverse_transform(fitted.forecast('2020-03')['GDPC1'][[-1]])[0],
        'Forecast of Q2 GDP quarterly rate':
        scaler.inverse_transform(fitted.forecast('2020-06')['GDPC1'][[-1]])[0]},
       name = 'Forecast').to_frame()


# Hidden Markov Model
"""
- HMM with Gaussian emissions
- Vary number of states and cov types to compare BIC
- 'full' and 'tied' have too many parameters of cov matrix => lowest IC at 1 component
- 'spherical' constraints features same variance each state => too many components=9
- 'diag' cov matrix for each states => balanced: min IC at n_components=3
"""
out = []
for covariance_type in ["full", "diag", "tied", "spherical"]:
    for n_components in range(1,16):
        markov = hmm.GaussianHMM(n_components=n_components,
                                 covariance_type=covariance_type,
                                 verbose=False, tol=1e-6, random_state=42, n_iter=100)\
                    .fit(mdata.values, [len(mdata)])
        results = hmm_summary(markov, mdata, [len(mdata)])
        #print(n_components, Series(results, name=covariance_type).to_frame().T)
        result = {'covariance_type': covariance_type,  'n_components': n_components}
        result.update(results)
        out.append(Series(result))
        result = pd.concat(out, axis=1).T.convert_dtypes()
print(result.to_string(float_format='{:.1f}'.format))

# display estimated transition and stationary distributions
n_components = 3
markov = hmm.GaussianHMM(n_components=n_components, covariance_type='diag',
                         verbose=False, tol=1e-6, random_state=42, n_iter=100)\
                         .fit(mdata.values, [len(mdata)])
pred = DataFrame(markov.predict(mdata), columns=['state'], index=mdata.index)
matrix = hmm_summary(markov, mdata, [len(mdata)], matrix=True)['matrix']
matrix

## Plot predicted states by selected economic series
"""
- Recession state, recovery pre-2000 state and recoverty post-2000 state
"""
# helper to plot predicted states
def plot_states(modelname, labels, beg, end):
    n_components = len(np.unique(labels))
    markers = ["o", "s", "d", "X", "P", "8", "H", "*", "x", "+"][:n_components] 
    series_ids = ['IPMANSICS', 'SPASTT01USM661N']
    fig, axes = plt.subplots(len(series_ids),ncols=1,figsize=(9,5),num=1,clear=True)
    axes[0].set_title(f"{modelname.upper()} Predicted States", {'fontsize':12})
    for series_id, ax in zip(series_ids, axes.ravel()):
        df = alf(series_id)
        print(df)
        df.index = pd.DatetimeIndex(df.index.astype(str), freq='infer')
        df = df[(df.index >= beg) & (df.index <= end)]
        for i, marker in zip(range(n_components), markers):
            df.loc[labels==i].plot(ax=ax, style=marker, markersize=2,
                                   color=f"C{i}", rot=0)
            ax.set_xlabel(f"{series_id}: {alf.header(series_id)}", {'fontsize':8})
        for a,b in vspans:
            if (b > min(df.index)) & (a < max(df.index)):
                ax.axvspan(max(a, min(df.index)), min(b, max(df.index)),
                           alpha=0.3, color='grey')
        ax.legend([f"state {i}" for i in range(n_components)], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, f"{modelname.lower()}.jpg"))

plot_states('hmm', pred.values.flatten(), min(pred.index), max(pred.index))
plt.show()

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='diag').fit(mdata)
labels = gmm.predict(mdata)
plot_states('GMM', labels, min(mdata.index), max(mdata.index))
plt.show()

## Compare persistance of HMM and GMM
print("Average Persistance of Hidden Markov states:",
      np.mean(pred[:-1].values == pred[1:].values))
print()
print('Stationary Distributiion of Hidden Markov:')
print(matrix.iloc[:,-1])
print()
print("Average Persistance of Gaussian Mixtures states:",
      np.mean(labels[:-1] == labels[1:]))
print()
print('Stationary Distribution of Gaussian Mixture:')
print(Series(labels).value_counts().sort_index()/len(labels))

