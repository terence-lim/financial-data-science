"""LSTM, State Space Models, and Mixtures from FRED-MD

We apply and compare four methods for extracting lower-dimensional states from a large panel of economic time series: State space models, Hidden Markov Models, Gaussian mixtures, and LSTM networks.

We visualize the extracted hidden states/latent factors to recessionary and economic time periods, and compare the estimated persistence of states.

- Long Short-Term Memory networks, hidden states, state space models, Gaussian mixtures
- pytorch, hmmlearn, statsmodels, sklearn
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
from finds.alfred import fred_md, Alfred
from settings import settings
imgdir = os.path.join(settings['images'], 'rnn')

# Load and pre-process time series from FRED
alf = Alfred(api_key=settings['fred']['api_key'])

# to indicate recession periods in the plots
usrec = alf('USREC', freq='m')
usrec.index = pd.DatetimeIndex(usrec.index.astype(str), freq='infer')
g = (usrec != usrec.shift()).cumsum()[usrec.gt(0)].to_frame()
g = g.reset_index().groupby('USREC')['date'].agg(['first','last'])
vspans = [(v[0], v[1]) for k, v in g.iterrows()]

# Retrieve FRED-MD series
mdf, mt = fred_md(202104)      # from vintage April 2020
beg = 19600301
end = 20191231 # 20191231

# Apply tcode transformations
df = mdf
t = mt['transform']
transformed = []
for col in df.columns:
    transformed.append(alf.transform(df[col], tcode=t[col], freq='m'))
df = pd.concat(transformed, axis=1).iloc[2:]

data = df[(df.index >= beg) & (df.index <= end)].dropna(axis=1)
data = (data-data.mean(axis=0))/data.std(axis=0, ddof=0)
data.index = pd.DatetimeIndex(data.index.astype(str), freq='infer')
cols = list(data.columns)
data

# LSTM pytorch module for time series
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
seq_len = 12
train_exs = [data.iloc[i-(seq_len+1):i].values
             for i in range(seq_len+1, len(data))]
n_features = data.shape[1]

# Initialize LSTM with hidden_size=4, on GPU device
hidden_size = 4
model = LSTM(n_features=n_features, hidden_size=hidden_size).to(device)
print(model)

# Set optimizer and learning rate scheduler, with step_size=30
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
loss_function = nn.MSELoss()

# Run training loop, with num_epochs=300 and batch_size=4 
batch_size = 32
num_epochs = 1000
for i in range(num_epochs):
    
    # shuffle indxs into batches
    idxs = np.arange(len(train_exs))
    random.shuffle(idxs)
    batches = [idxs[i:(i+batch_size)] for i in range(0, len(idxs), batch_size)]

    # train each batch
    total_loss = 0.0
    model.train()
    for batch in batches:
        # train_ex input has shape (seq_len, batch_size=16, n_features)
        train_ex = torch.tensor([[train_exs[idx][seq] for idx in batch]
                                 for seq in range(seq_len+1)])\
                        .permute(1,2,0).float().to(device)
        model.zero_grad()
        y_pred, hidden_state = model.forward(train_ex[:-1])
        loss = loss_function(y_pred[-1], train_ex[-1])
        total_loss += float(loss)
        loss.backward()
        optimizer.step()
    scheduler.step()
    if (i % 200) == 0:
        print(i, optimizer.param_groups[0]['lr'], total_loss/len(batches))

# collect predictions and hidden states, and compute mse
with torch.no_grad():    # reduces memory consumption for eval
    hidden_state = []
    prediction_error = []
    mse = nn.MSELoss()
    for i in range(seq_len+1, len(data)):
        # single test example of shape (seq_len=12, batch_size=1, n_features)
        test_ex = torch.tensor(data[i-(seq_len+1):i].values)\
                       .float().unsqueeze(dim=1).to(device)
        y_pred, (h, c) = model.forward(test_ex[:-1], None)
        prediction_error.append(float(mse(y_pred[-1], test_ex[-1])))
        hidden_state.append(h[0][0].cpu().numpy())
hidden_factors = DataFrame(hidden_state, index=data.index[(1+seq_len):len(data)])
print(np.mean(prediction_error))

# Plot hidden states process
fig = plt.figure(figsize=(9, 10), num=1, clear=True)
for col in hidden_factors.columns:
    ax = fig.add_subplot(4, 1, col+1)
    (np.sign(hidden_factors[col].corr(usrec))*
     hidden_factors[col]).plot(ax=ax, style='o', color=f"C{col}")
    for a,b in vspans:
        ax.axvspan(a, b, alpha=0.3, color='grey')
    ax.legend([f"Hidden State {col}", 'NBER Recession'], fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"LSTM Hidden States {hidden_factors.index[0]:%b-%Y}:"
             f"{hidden_factors.index[-1]:%b-%Y}", fontsize=12)
plt.savefig(os.path.join(imgdir, 'lstm.jpg'))

# Dynamic Factor Models
# - can be cast into state space form and estimated via Kalman Filter
# https://www.statsmodels.org/devel/examples/notebooks/generated/
#  statespace_dfm_coincident.html
# Fit (linear) dynamic factor model with DynamicFactorMQ 
# https://www.statsmodels.org/devel/generated/
#  statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ.html
# allows for
# - mixed frequency data (for nowcasting)
# - EM algorithm by default accomodates larger data sets:  Kalman filter
#   and optimizing of likelihood with quasi-Newton methods slow
import statsmodels.api as sm
mod = sm.tsa.DynamicFactorMQ(endog=data,
                             factors=1,                # num factor blocks
                             factor_multiplicities=4,  # num factors in block
                             factor_orders=12,         # order of factor VAR
                             idiosyncratic_ar1=False)  # False=white noise
fitted = mod.fit_em(disp=5, maxiter=1000, full_output=True)
dynamic_factors = DataFrame(fitted.factors.filtered.iloc[seq_len+1:])
dynamic_factors.columns = list(np.arange(len(dynamic_factors.columns)))

# compute mse of fitted observations
mse = nn.MSELoss()
print(float(mse(torch.tensor(fitted.fittedvalues.iloc[seq_len+1:].values),
                torch.tensor(data.iloc[seq_len+1:].values))))

# Plot estimated dynamic factors
fig = plt.figure(figsize=(9, 10), num=2, clear=True)
for col in dynamic_factors.columns:
    ax = fig.add_subplot(4, 1, col+1)
    (np.sign(dynamic_factors[col].corr(usrec))*
     dynamic_factors[col]).plot(ax=ax, style='o', color=f"C{col}")
    for a,b in vspans:
        ax.axvspan(a, b, alpha=0.3, color='grey')
    ax.legend([f"Dynamic Factor {col} Estimate", 'NBER Recession'], fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"Estimated Factor {dynamic_factors.index[0]:%b-%Y}:"
             f"{dynamic_factors.index[-1]:%b-%Y}", fontsize=12)
plt.savefig(os.path.join(imgdir, 'dynamic.jpg'))
plt.tight_layout()
plt.show()

# Correlation of LSTM hidden state process with (linear) dynamic factors 
for col, y in hidden_factors.iteritems():
    res = sm.OLS(y, sm.add_constant(dynamic_factors)).fit()
    print()
    print('Hidden Factor', col, '  rsquared =', np.round(res.rsquared, 4))
    print(res.summary().tables[1])

    
## helper to plot predicted states
def plot_states(modelname, labels, beg, end):
    n_components = len(np.unique(labels))
    markers = ["o", "s", "d", "X", "P", "8", "H", "*", "x", "+"][:n_components] 
    series_ids = ['IPMANSICS', 'S&P 500', 'AAAFFM', 'GS1']
    df = mdf[series_ids].copy()
    df.index = pd.DatetimeIndex(df.index.astype(str), freq='infer')
    df = df[(df.index >= beg) & (df.index <= end)]
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(9,10), num=1, clear=True)
    axes[0].set_title(f"{modelname} predicted states", {'fontsize':9})
    for s, ax in zip(series_ids, axes.ravel()):
        for i, marker in zip(range(n_components), markers):
            df.loc[labels==i, s].plot(ax=ax, style=marker, markersize=2,
                                      color=f"C{i}", rot=0)
            ax.set_xlabel(f"{s}: {alf.header(s)}", {'fontsize':8})
        for a,b in vspans:
            if (b > min(df.index)) & (a < max(df.index)):
                ax.axvspan(max(a, min(df.index)), min(b, max(df.index)),
                           alpha=0.3, color='grey')
        ax.legend([f"state {i}" for i in range(n_components)], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, f"{modelname.lower()}.jpg"))

Hidden Markov Model from FRED-MD
"""
 - Gaussian HMM
 - with 3 states
 - display estimated transition and stationary distributions
"""
from hmmlearn import hmm
n_components = 3
markov = hmm.GaussianHMM(n_components=n_components,
                         covariance_type="full",
                         verbose=True,
                         tol=1e-6,
                         n_iter=100,
                         algorithm='viterbi')\
            .fit(data.values, [len(data)])
print('Dimension of observations:', markov.n_features)
print(markov.monitor_)
pred = DataFrame(markov.predict(data), columns=['state'], index=data.index)
matrix = DataFrame(markov.transmat_).rename_axis(columns='Transition Matrix:')
matrix['Stationary'] = markov.get_stationary_distribution()
print(matrix.to_latex())

plot_states('hmm', pred.values.flatten(), min(pred.index), max(pred.index))
plt.show()

#
# GMM
#
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, covariance_type='full').fit(data)
labels = gmm.predict(data)
plot_states('GMM', labels, min(data.index), max(data.index))
plt.show()

## Compare persistance of HMM and GMM
print("Persistance of Hidden Markov states:",
      np.mean(pred[:-1].values == pred[1:].values))
print("Persistance of Gaussian Mixtures states:",
      np.mean(labels[:-1] == labels[1:]))
