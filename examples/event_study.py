"""Event Study Abnormal Returns

- key developments CAR, BHAR and post-event drift

Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import os
import time
from finds.database import SQL
from finds.busday import BusDay
from finds.structured import PSTAT, CRSP, Benchmarks
from finds.backtesting import EventStudy
from settings import settings
LAST_DATE = settings['crsp_date']

ECHO = True
sql = SQL(**settings['sql'], echo=ECHO)
user = SQL(**settings['user'], echo=ECHO)
bd = BusDay(sql)
keydev = PSTAT(sql, bd)
crsp = CRSP(sql, bd, rdb=None)
bench = Benchmarks(sql, bd)
eventstudy = EventStudy(user, bench, LAST_DATE)
outdir = os.path.join(settings['images'], 'events')

# event window parameters
end = 20201201
beg = 19890101  # 20020101
minobs = 250
left, right, post = -1, 1, 21

# str formatter to pretty print event and role description given their id's 
eventformat = lambda e, r: "{event} ({eventid}) {role} [{roleid}]".format(
    event=keydev.event_[e], eventid=e, role=keydev.role_[r], roleid=r)
events = sorted(keydev.event_.keys())   # list of eventid's
roles = sorted(keydev.role_.keys())     # list of roleid's

## Helpers to merge events and crsp, and screen stock universe

# to lookup prevailing exchange and share codes by permno and date
shrcd = crsp.build_lookup('permno', 'shrcd')
exchcd = crsp.build_lookup('permno', 'exchcd')

def event_pipeline(eventstudy, mincap=300000, **arg):
    """helper to merge keydev events and crsp, and screen stock universe"""
    df = keydev.get_linked(
        dataset='keydev',
        date_field='announcedate',
        fields=['keydevid',
                'keydeveventtypeid',
                'keydevtoobjectroletypeid'],
        where=(f"announcedate >= {arg['beg']} and announcedate <= {arg['end']}"
               f" and keydeveventtypeid = {arg['eventid']} "
               f" and keydevtoobjectroletypeid = {arg['roleid']}"))\
               .drop_duplicates(['permno','announcedate'])\
               .set_index(['permno','announcedate'], drop=False)
    
    stk = arg['stocks'].get_many(
        dataset='daily',
        permnos=df['permno'],
        date_field='date',
        dates=arg['stocks'].bd.offset(df['announcedate'], left-1),
        fields=['prc', 'shrout']).fillna(0)
    
    df['cap'] = (stk['prc'].abs() * stk['shrout']).values
    df['exchcd'] = [exchcd(r.permno, r.date) for r in stk.itertuples()]
    df['shrcd'] = [shrcd(r.permno, r.date) for r in stk.itertuples()]
    r = (df['cap'].gt(mincap) &              # require cap > $300M
         df['exchcd'].isin([1,2,3]) &        # primary exchange
         df['shrcd'].isin([10,11])).values   # domestic common stocks
    rows = eventstudy(crsp, df[r], left, right, post, 'announcedate')
    return df.loc[rows.to_records(index=False).tolist()]  # restrict df to rows

## Compute abnormal returns of all events
# %%capture
tic = time.time()
for i, eventid in enumerate(events):
    for roleid in roles:
        # retrieve all observations of this eventid, roleid
        df = event_pipeline(eventstudy, stocks=crsp, beg=beg, end=end,
                            eventid=eventid, roleid=roleid,
                            left=left, right=right, post=post)
        if len(df) < minobs:  # require min number of events
            continue
        
        # retrieve announcement window returns
        r = eventstudy(crsp, df, left, right, post, 'announcedate')
        if r['date'].nunique() < minobs:  # require min number of dates
            continue

        # compute both BHAR and CAR averages, plot and save
        bhar = eventstudy.fit(car=False, name='bhar')
        car = eventstudy.fit(car=True, name='car')
        eventstudy.write(label=f"{eventid}_{roleid}")
        s = pd.concat([bhar, car], axis=1).T
        print(eventformat(eventid, roleid))
        print(s.to_string(float_format='%.4f', index=False))
        print()
        fig, axes = plt.subplots(2, 1, clear=True, num=1, figsize=(10,12))
        eventstudy.plot(title=eventformat(eventid, roleid),
                        vline=right, ax=axes[0], name='bhar')
        eventstudy.plot(title='', vline=right, ax=axes[1], name='car')
        if outdir:
            plt.savefig(os.path.join(outdir, f"{eventid}_{roleid}.jpg"))
    print('Elapsed:', time.time()-tic, 'secs')


## Summarize BHAR's of all events, by 3-day event window abnormal returns

df = eventstudy.read(name='bhar')\
               .set_index('permno').sort_values('window', ascending=False)
dx = DataFrame(df.index.str.split('_').to_list()).astype(int)
df.index = pd.MultiIndex.from_frame(dx).set_names(['eventid','roleid'])

df['event'] = keydev.event_[df.index.get_level_values(0)].values
df['role'] = keydev.role_[df.index.get_level_values(1)].values

mindays = (df['days']>1000).values
print(df[mindays].iloc[:10].drop(columns='name')\
      .to_string(formatters={'effective':'{:.0f}'.format}, float_format='%.4f',
                 index=False))
print(df[mindays].iloc[::-1].iloc[:10].drop(columns='name')\
      .to_string(formatters={'effective':'{:.0f}'.format}, float_format='%.4f',
                 index=False))
print(df[mindays].iloc[:10].drop(columns=['name'])\
      .to_latex(index=False, formatters={'effective':'{:.0f}'.format}))
print(df[mindays].iloc[::-1].iloc[:10].drop(columns=['name'])\
      .to_latex(index=False, formatters={'effective':'{:.0f}'.format}))
print(df.sort_values('post_t').to_string(float_format='%.4f'))
df.drop(columns='name')

## Show single plots for each of three events
eventid, roleid = 80, 1
eventid, roleid = 26, 1
df = event_pipeline(eventstudy, stocks=crsp, eventid=eventid, roleid=roleid,
                    beg=beg, end=end, left=left, right=right, post=post)
bhar = eventstudy.fit(car=False)
fig, ax = plt.subplots(clear=True, num=1, figsize=(10,6))
eventstudy.plot(title=eventformat(eventid, roleid), vline=right, ax=ax)


## show single plot by market cap and half-period
midcap = 20000000
for i, (eventid, roleid) in enumerate([[50,1], [83,1]]):
    #eventid, roleid = 50, 1
    #eventid, roleid = 83, 1
    df = event_pipeline(eventstudy, stocks=crsp, eventid=eventid, roleid=roleid,
                        beg=beg, end=end, left=left, right=right, post=post)
    halfperiod = np.median(df['announcedate'])
    sample = {'[FirstHalf]': df['announcedate'].ge(halfperiod).values,
              '[SecondHalf]': df['announcedate'].lt(halfperiod).values,
              '[Large]': df['cap'].ge(midcap).values,
              '[Small]': df['cap'].lt(midcap).values,
              '': None}
    for ifig, (label, rows) in enumerate(sample.items()):
        fig, ax = plt.subplots(clear=True, num=1+ifig, figsize=(5,6))
        bhar = eventstudy.fit(rows=rows, car=False)
        eventstudy.plot(title=eventformat(eventid, roleid) + ' ' + label,
                        drift=True, ax=ax, c=f"C{i*5+ifig}")
        plt.savefig(os.path.join(outdir, label + f"{eventid}_{roleid}.jpg"))
        
for i, (eventid, roleid) in enumerate([[80,1], [26,1]]):
    #eventid, roleid = 50, 1
    #eventid, roleid = 83, 1
    df = event_pipeline(eventstudy, stocks=crsp, eventid=eventid, roleid=roleid,
                        beg=beg, end=end, left=left, right=right, post=post)
    halfperiod = np.median(df['announcedate'])
    sample = {'[FirstHalf]': df['announcedate'].ge(halfperiod).values,
              '[SecondHalf]': df['announcedate'].lt(halfperiod).values,
              '[Large]': df['cap'].ge(midcap).values,
              '[Small]': df['cap'].lt(midcap).values,
              '': None}
    for ifig, (label, rows) in enumerate(sample.items()):
        fig, ax = plt.subplots(clear=True, num=1+ifig, figsize=(5,6))
        bhar = eventstudy.fit(rows=rows, car=False)
        eventstudy.plot(title=eventformat(eventid, roleid) + ' ' + label,
                        drift=False, ax=ax, c=f"C{i*5+ifig}")
        plt.savefig(os.path.join(outdir, label + f"{eventid}_{roleid}.jpg"))
        
#plt.show()

## Show Max Order Statistic and Bonferroni Adjustment
import statsmodels.api as sm
import scipy
from pandas.api import types
class MaxStat:
    """Max Order Statistic probability distributions"""
    def __init__(self, dist=scipy.stats.norm, n=None, **params):
        self.dist_ = dist
        self.params_ = params
        self.n = n

    def cdf(self, z, n=None):
        """cdf for max order statistic"""
        return [self.cdf(y, n) for y in z] if types.is_list_like(z)\
            else self.dist_.cdf(z, **self.params_)**(n or self.n)

    def pdf(self, z, n=None):
        """cdf for max order statistic"""
        n = n or self.n
        return [self.pdf(y, n) for y in z] if types.is_list_like(z)\
            else self.dist_.pdf(z, **self.params_) * n * self.cdf(z, n=n-1)

    def ppf(self, z, n=None):
        """inverse cdf for max order statistic"""
        return [self.ppf(y, n) for y in z] if types.is_list_like(z)\
            else self.dist_.ppf(z, **self.params_)**(n or self.n)
        
    def pvalue(self, z, n=None):
        """z-value for max order statistic"""
        return [self.pvalue(y, n) for y in z] if types.is_list_like(z)\
            else 1 - (self.dist_.cdf(z, **self.params_)**(n or self.n))
        
    def zvalue(self, p, n=None):
        """z-value for max order statistic"""
        return [self.zvalue(y, n) for y in z] if types.is_list_like(p)\
            else self.dist_.ppf((1-p)**(1/(n or self.n)), **self.params_)

    def bonferroni(self, p, n=None):
        """corrected z-value with with Bonferroni adjustment"""
        return [self.bonferroni(y, n) for y in z] if types.is_list_like(p)\
            else self.dist_.ppf((1-(p/n)), **self.params_)
    
# Display order statistics of folded normal        
y = eventstudy.read(name='bhar')['post_t'].values
Z = max(y)
n = len(y)
print(f"Events tested={n}, Max z-value={Z:.4f}:")
maxstat = MaxStat(scipy.stats.foldnorm, c=0)
    
p = Series({n: maxstat.pvalue(Z, n=n) for n in
            sorted([n] + [1, 2, 30, 60, 120, 250, 1000, 1600])}, name='pvalue')
print(f"\nMax order statistic p-value(z={Z:.2f}) by sample size:")
print(DataFrame(p).T.to_string(float_format='%.4f'))

P=0.05
print(f"\nRejection Region(p-value={P:.2f}) by sample size:")
zb = Series({n: maxstat.bonferroni(P, n=n)
             for n in [1, 2, 20, 100, 1000, 100000, 1000000]},name='max-order')
zc = Series({n: maxstat.zvalue(P, n=n)
             for n in [1, 2, 20, 100, 1000, 100000, 1000000]},name='bonferroni')
print(pd.concat([zc, zb], axis=1).round(3).to_latex())

# Plot CDF of Max Order Statistic
X = np.linspace(0, 6, 600)
df = DataFrame(index=X)
df = pd.concat([Series(data=[maxstat.cdf(x, n) for x in X], index=X, name=n)
                for n in [1, 2, 20, 100, 1000, 100000, 1000000]], axis=1)
fig, ax = plt.subplots(clear=True, num=1, figsize=(10,6))
df.plot(ax=ax, title='CDF of Max-Order Statistic by Sample Size',
        xlabel='z-value', ylabel='Cumulative Distribution Function')
ax.axvline(1.96, c='grey', alpha=0.5)
ax.annotate("1.96", xy=(1.96, 1))
plt.savefig(os.path.join(outdir, 'cdf.jpg'))
plt.show()

## Show distribution of post-event t-values
import scipy
import seaborn as sns
y = eventstudy.read(name='bhar')['post_t']
fig, axes = plt.subplots(1, 2, num=1, clear=True, figsize=(10,6))
    
ax = sns.distplot(y, kde=False, hist=True, ax=axes[0], color='C0')
bx = ax.twinx()
x = np.linspace(*plt.xlim(), 100)
bx.plot(x, scipy.stats.norm.pdf(x), color="C1")
ax.set_title('Post-event drift and Normal Density', fontsize=10)
ax.set_xlabel('Post-event drift t-value')

ax = sns.distplot(abs(y), kde=False, hist=True, ax=axes[1], color='C1')
bx = ax.twinx()
x = np.linspace(*plt.xlim(), 100)
bx.plot(x, scipy.stats.foldnorm.pdf(x, 0), color="C1")
ax.set_title('Abs(post-event drift) and Folded Normal Density', fontsize=10)
ax.set_xlabel('Abs(post-event drift) t-value')
plt.savefig(os.path.join(outdir, 'hist.jpg'))

DataFrame(y.describe()).T
    
    
