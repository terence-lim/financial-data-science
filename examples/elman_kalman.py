"""Recurrent Neural Networks and Linear Dynamic Factor Models

- Long Short Term Memory (LSTM) RNN, Kalman Filter, Mixed Frequency

Copyright 2023, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import re
import time
from datetime import datetime
import statsmodels.api as sm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from finds.alfred import fred_md, fred_qd, Alfred
from finds.display import show
from conf import VERBOSE, paths, credentials

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None

imgdir = paths['images'] / 'states'

### Load and pre-process time series from FRED
alf = Alfred(api_key=credentials['fred']['api_key'])
vspans = alf.date_spans('USREC')  # to indicate recession periods in the plots
beg = 19600301
end = 20200131
# TODO: end = train up to MINUS two quarters, forecast through current

### Retrieve FRED-MD series and apply tcode transformations
df, t = fred_md(202004)        # from vintage April 2020?
data = []
for col in df.columns:
    data.append(alf.transform(df[col], tcode=t['transform'][col], freq='m'))
mdf = pd.concat(data, axis=1).iloc[2:]
mdata = mdf[(mdf.index >= beg) & (mdf.index <= end)].dropna(axis=1)
mdata = (mdata - mdata.mean(axis=0)) / mdata.std(axis=0, ddof=0)
mdata.index = pd.DatetimeIndex(mdata.index.astype(str), freq='m')
mdata

### Retrieve FRED-QD series and apply tcode transformations
df, t = fred_qd(202004)      # from vintage April 2020
data = []
for col in df.columns:
    data.append(alf.transform(df[col], tcode=t['transform'][col], freq='q'))
df = pd.concat(data, axis=1).iloc[2:]
qdata = df[(df.index >= beg) & (df.index <= end)].dropna(axis=1)
qdata.index = pd.DatetimeIndex(qdata.index.astype(str), freq='q')
qdata


## LSTM in pytorch for time series
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

### Create input data for LSTM, with sequence length 16 (months)
seq_len = 16
train_exs = [mdata.iloc[i-(seq_len+1):i].values
             for i in range(seq_len+1, len(mdata))]
n_features = mdata.shape[1]


### Train LSTM
hidden_factors = dict()
prediction_errors = dict()
for hidden_size in [1, 2, 3, 4]:
    model = LSTM(n_features=n_features,
                 hidden_size=hidden_size).to(device)
    print(model)

    # Set optimizer and learning rate scheduler, with step_size=30
    lr, num_lr, step_size = 0.001, 3, 400
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=step_size,
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
            nparray = np.array([[train_exs[idx][seq] for idx in batch]
                                for seq in range(seq_len+1)])
            train_ex = torch.tensor(nparray).float()
            model.zero_grad()
            y_pred, hidden_state = model.forward(train_ex[:-1].to(device))
            loss = loss_function(y_pred[-1],
                                 train_ex[-1].to(device))
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
            # single test example shaped (seq_len=12, batch_size=1, n_features)
            test_ex = torch.tensor(mdata[i-(seq_len+1):i].values)\
                           .float()\
                           .unsqueeze(dim=1)\
                           .to(device)
            y_pred, (h, c) = model.forward(test_ex[:-1], None)
            prediction_error.append(float(mse(y_pred[-1], test_ex[-1])))
            hidden_state.append(h[0][0].cpu().numpy())
    hidden_factors[hidden_size] = DataFrame(hidden_state,
                                            index=mdata.index\
                                            [(1+seq_len):len(mdata)])
    prediction_errors[f"Hidden Size {hidden_size}"] = np.mean(prediction_error)
    print(prediction_errors)

### Plot LSTM hidden states sequence
fig, axes = plt.subplots(len(hidden_factors), 1, figsize=(9,10),
                         num=1, clear=True)
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
plt.savefig(imgdir / 'lstm.jpg')


## Dynamic Factor Models
dynamic_factors = dict()
for i in [1, 2, 3, 4]:
    mod = sm.tsa.DynamicFactorMQ(endog=mdata,
                                 factors=1,               # num factor blocks
                                 factor_multiplicities=i, # num factors in block
                                 factor_orders=2,         # order of factor VAR
                                 idiosyncratic_ar1=False) # False=white noise
    fitted = mod.fit_em(disp=20,
                        maxiter=200,
                        full_output=True)
    dynamic_factors[i] = DataFrame(fitted.factors.filtered.iloc[seq_len+1:])
    dynamic_factors[i].columns = list(range(len(dynamic_factors[i].columns)))
    mse = nn.MSELoss()
    prediction_errors[f"Dynamic Factors {i}"] = float(
        mse(torch.tensor(fitted.fittedvalues.iloc[mod.factor_orders+1:].values),
            torch.tensor(mdata.iloc[mod.factor_orders+1:].values)))
#print(fitted.summary(0))

### Plot dynamic factors
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
plt.savefig(imgdir / 'dynamic.jpg')



## Correlation of LSTM hidden state process with (linear) dynamic factors
rsq = dict()
for k, hidden_factor in hidden_factors.items():
    rsq[k] = [sm.OLS(y, sm.add_constant(dynamic_factors[len(dynamic_factors)]))\
              .fit().rsquared for _, y in hidden_factor.iteritems()]
print('Average variance of LSTM hidden states explained by dynamic factors')
DataFrame({k: np.mean(r) for k, r in rsq.items()},
          index=['R-square']).rename_axis("# hidden states in LSTM:", axis=1)


## Mixed Frequency Dynamic Factor Model
scaler = StandardScaler()\
    .fit(qdata['GDPC1'].values.reshape((-1, 1)))
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

### Plot fitted GDP values in 2007-2010

# TODO: fit through current
beg = '2006-12-31'
end = '2010-12-31'
fig, ax = plt.subplots(figsize=(9,5), num=1, clear=True)
y = fitted.fittedvalues['GDPC1']
y = y[(y.index > beg) & (y.index <= end)]
ax.plot_date(y.index,
             (scaler.inverse_transform(y.to_numpy().reshape(-1,1))/3).cumsum(),
             fmt='-o',
             color='C0')
x = gdp.copy()
x.index = pd.DatetimeIndex(x.index.astype(str), freq=None)
x = x[(x.index > beg) & (x.index <= end)]
ax.plot_date(x.index, scaler.inverse_transform(x).cumsum(), fmt='-o',color='C1')
ax.legend(['monthly fitted', 'quarterly actual'], loc='upper left')
ax.set_title('Quarterly GDP and Monthly Estimates from Dynamic Factor Model')
plt.tight_layout()
plt.savefig(imgdir / "mixedfreq.jpg")


# Show "Nowcast" of GDP
#Series({'Last Date of Monthly Data:': mdata.index[-1].strftime('%Y-%m-%d'),
#        'Last Date of Quarterly Data:': qdata.index[-1].strftime('%Y-%m-%d'),
#        'Forecast of Q1 GDP quarterly rate':
#        scaler.inverse_transform(fitted.forecast('2020-03')['GDPC1'][[-1]])[0],
#        'Forecast of Q2 GDP quarterly rate':
#        scaler.inverse_transform(fitted.forecast('2020-06')['GDPC1'][[-1]])[0]},
#       name = 'Forecast').to_frame()
