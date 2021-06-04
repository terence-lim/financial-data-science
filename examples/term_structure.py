"""Term Structure of Interest Rates

- bootstrap, splines, yield curve, duration
- Liu and Wu (2020), St Louis Fed FRED

Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import re
from finds.alfred import Alfred
from finds.busday import to_monthend
from settings import settings
imgdir = os.path.join(settings['images'], 'ts')
alf = Alfred(api_key=settings['fred']['api_key'])

# Term Structure of Interest Rates
# list of monthly Constant Maturity Treasury, excluding inflation-indexed
c = alf.get_category(115)  # Fed H.15 Selected Interest Rates
print(c['id'], c['name'])
t = Series({s['id']: s['title'] for s in c['series']
            if s['frequency']=='Monthly' and 'Inflation' not in s['title']})
print(t.to_latex())
    
# retrieve CMT yields, and infer maturity from label
b = pd.concat([alf(s, freq='M') for s in t.index], axis=1, join='inner')
b.columns = [int(re.sub('\D', '', col)) * (1 if col.endswith('M') else 12)
             for col in b.columns]    # infer maturity in months from label
b = b.sort_index(axis=1)              # sort columns by ascending maturity
b

# helper methods for basic bond math calculations
def pv(flow, n, spot):
    """PV of cash flow after compounding spot rate over n periods"""
    return flow / ((1 + spot) ** n)

def weighted_maturity(flows, spot, first=1, returned=False):
    """Average maturity weighted by PV of flows given effective spot rates"""
    v = [pv(flow=flow, n=n+first, spot=rate)
         for n, (flow, rate) in enumerate(zip(flows, spot))]
    return np.average(np.arange(len(v))+first, weights=v, returned=returned)

def par_duration(nominal, n, face=1, m=1, first=1):
    """Macaulay duration of a coupon par bond"""
    coupon = nominal * face  # par bond
    flows = [coupon/m]*(n*m - 1) + [face + coupon/m] 
    d,v = weighted_maturity(flows, [nominal/m]*(n*m), first=first,returned=True)
    return d / m

def forward(spot, base=0):
    """forward rate implied by spot starting after base periods"""
    return [(((1 + num)**(n + 1 + base) / (1 + den)**(n + base)) - 1)
            for n, (num, den) in enumerate(zip(spot, [0] + list(spot[:-1])))]
    
from pandas.api import types
def dcf(flows, spot, first=1):
    """PV of flows, starting at period first, by compounding associated spot"""
    if not types.is_list_like(flows):   # flows can be different each period
        flows = [flows]                 #  else assume same flow every period
    if not types.is_list_like(spot):    # spot rates can be different each flow
        spot = [spot]                   #  else use same spot rate every period
    if len(flows) == 1:
        flows = list(flows) * len(spot) # repeat flows to be same length as spot
    if len(spot) == 1:
        spot = list(spot) * len(flows)  # repeat spot to be same length as flows
    return np.sum([pv(flow=flow, n=first+n, spot=rate)
                   for n, (flow, rate) in enumerate(zip(flows, spot))])

def bootstrap(ytm, nominal, m):
    """Nominal rate to maturity of par bond from ytm and sequence of nominals"""
    n = len(nominal) + 1     # implicit # of coupons, including last at maturity
    return (((1 + ytm/m)/(1 - dcf(ytm/m, [r/m for r in nominal])))**(1/n) - 1)*m

# plot yields and durations over time
y = pd.concat([b[c*12].rename(c)/100 for c in [2,5,7,10,20,30]], axis=1)
y.index = pd.to_datetime(y.index, format='%Y%m%d')
d = pd.concat([y[c].apply(par_duration, n=c, m=2) for c in y], axis=1)
fig, axes = plt.subplots(2, 3, clear=True, num=1, figsize=(12,7))
for i in range(6):
    ax = axes[i // 3, i % 3]
    y.iloc[:, i].rename('yield').plot(ax=ax, c=f"C{i*2}",
                                      fontsize=8, legend=True,
                                      title=f"{y.columns[i]}-Year")
    bx = ax.twinx()
    d.iloc[:, i].rename('duration').plot(ax=bx, c=f"C{i*2+1}",
                                         fontsize=8, legend=True)
    ax.set_xlabel('')
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'term.jpg'))
plt.show()
    
# interpolate yield curve spline as of a historical date
# https://home.treasury.gov/policy-issues/financing-the-government/
#   interest-rate-statistics/treasury-yield-curve-methodology
curve_date = 20021231
from scipy.interpolate import CubicSpline
yields = b.loc[curve_date].values
maturities = b.columns.to_list()
yield_curve = CubicSpline(maturities, yields, bc_type='clamped')

# Bootstrap nominal annual rates from interpolated ytm semi-annually
ytm = [yield_curve(t) / 100 for t in range(6, 361, 6)]  
m = 2
nominal = np.array([])   # to store annualized rates from bootstrap
for y in ytm:
    nominal = np.append(nominal, bootstrap(y, nominal=nominal, m=m))

# Sanity-check results
for n in range(1, 1+len(ytm)):  # discounted cash flows all close to par=1
    assert(abs(dcf(ytm[n-1]/m, nominal[:n]/m) +
               pv(1, n=n, spot=nominal[n-1]/m) - 1) < 0.001)
f = np.array(forward(spot=nominal/m))  # as single-period rate
for n in range(1, 1+len(ytm)):  # compounded forwards all close to nominal
    assert(abs(np.prod(1 + f[:n]) - (1 + nominal[n-1] / m)**n) < 0.001)
# convert to annual effective rates
spot_rates = (((1 + np.array(nominal/2))**2) - 1) * 100
forward_rates = (((1 + np.array(f))**2) - 1) * 100
semi_annual = np.arange(6, 361, 6)     # enumerate semi-annual months

# Retrieve reconstructed yield curve history
# https://sites.google.com/view/jingcynthiawu/yield-data
def fetch_lw(file_id='1_u9cRxmOSiwp_tFvlaORuhS-zwl935s0'):
    src = "https://drive.google.com/uc?export=download&id={}".format(file_id)
    x = pd.ExcelFile(src)
    df = x.parse()
    dates = np.where(df.iloc[:, 0].astype(str).str[0].str.isdigit())[0]
    return DataFrame(np.exp(df.iloc[dates,1:361].astype(float).values/100) - 1,
                     index=to_monthend(df.iloc[dates, 0].values),
                     columns=np.arange(1, 361))
df = fetch_lw()
reconstructed_rates = df.loc[curve_date, :] * 100   # as of curve_date 20021231

# Plot historical reconstructed rates as 3D surface
from mpl_toolkits.mplot3d import Axes3D
r = df.dropna()
X, Y = np.meshgrid((r.index//10000) + ((((r.index//100)%100)-1)/12),
                   r.columns.astype(float)/12)
Z = r.T.to_numpy()*100
fig = plt.figure(num=1, clear=True, figsize=(10,8))
ax = fig.gca(projection='3d')
f = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)
ax.set_title('Reconstructed Treasury Interest Rates [Liu and Wu (2020)]')
ax.set_xlabel('date')
ax.set_ylabel('maturity (years)')
ax.set_zlabel('annual rate (%)')
fig.colorbar(f, shrink=0.5, aspect=5)
plt.tight_layout(pad=0)
plt.savefig(os.path.join(imgdir, 'reconstructed.jpg'))
plt.show()

# Plot the yield curves
fig, ax = plt.subplots(num=1, clear=True, figsize=(12,7))
ax.plot(maturities, yields, 'o', label='CMT nominal yields')
ax.plot(semi_annual, yield_curve(semi_annual), label="cubic-splined yields")
ax.plot(semi_annual, spot_rates, label="bootstrapped spot rate")
ax.plot(reconstructed_rates, label="reconstructed spot rate (Liu&Wu)")
ax.plot(semi_annual, forward_rates, label="bootstrapped forward 6-mo rate")
ax.set_title(f"Yield Curve {curve_date} from Constant Maturity Treasuries")
ax.set_xlabel('maturity (months)')
ax.set_ylabel('interest rate (annual %)')
ax.legend()
plt.tight_layout(pad=2)
plt.savefig(os.path.join(imgdir, 'curve.jpg'))
plt.show()
    
