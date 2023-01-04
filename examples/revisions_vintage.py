"""Economic time series

- St Louis Fed FRED: popular series, api
- ALFRED: archival, releases, vintages, revisions
- FRED-MD: release dates

Copyright 2023, Terence Lim

MIT License
"""
import time
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from finds.alfred import Alfred, fred_md, fred_qd
from finds.display import show
from datetime import datetime
from conf import credentials, VERBOSE, paths

%matplotlib qt
VERBOSE = 1      # 0
SHOW = dict(ndigits=4, latex=True)  # None

imgdir = paths['images'] / 'ts'
alf = Alfred(api_key=credentials['fred']['api_key'], verbose=VERBOSE)
#savefile=paths['scratch'] + 'fred.md')
today = datetime.today().strftime('%Y-%m-%d')

# Popular FRED series: top two pages
r = {}
for page in [1]:   # scrape first two pages
    popular = Alfred.popular(page)
    for s in popular:
        t = alf.request_series(s)   # requests 'series' FRED api
        if t.empty:
            r.update({s:{}})
        else:
            r.update({s: t.iloc[-1][['title', 'popularity']]})
show(DataFrame.from_dict(r, orient='index'),
     caption=f"Popular Series in FRED, retrieved {today}", **SHOW)
    
# Traversing categories tree
node = 0
while True:
    node = alf.get_category(node)
    print(f"[{node['id']}]", node['name'],
          f"(#children={len(node['children'])})",
          f"(#series={len(node['series'])})")
    if not (node['children']):
        break
    node = np.min([child['id'] for child in node['children']])
for i, row in enumerate(node['series']):
    print(i, row['id'], row['title'])


# INDPRO by latest, vintage, revision number, time lag
series_id = 'INDPRO'  # https://www.bea.gov/gdp-revision-information
#series_id = 'CPIAUCSL'
print(f"Latest revision retrieved {today}:")
print(alf(series_id,
          start=20200401,
          end=20200731,
          realtime=True))

print("First Release:")
print(alf(series_id,
          release=1,
          start=20200401,
          end=20200731,
          realtime=True))

print("Second Release:")
print(alf(series_id,
          release=2,
          start=20200401,
          end=20200731,
          realtime=True))

print("Revised Up to 5-months Later:")
print(alf(series_id,
          release=pd.DateOffset(months=5),
          start=20200401,
          end=20200731,
          realtime=True))

print("Latest as of Vintage date 2020-06-30:")
print(alf(series_id,
          vintage=20200630,
          realtime=True,
          start=20200401))

# INDPRO revisions history up to N-months later
df = pd.concat([alf(series_id,
                    start=20160101,
                    release=pd.DateOffset(months=m))\
                .rename(f"Revised up to {m}-months later")
                for m in [1, 3, 9, 21, 33]], axis=1)
df.index = pd.DatetimeIndex(df.index.astype(str))
ax = df.plot(logy=False)
ax.set_title(f"{series_id}: Revisions up to N-months later")
if imgdir:
    plt.savefig(imgdir / 'release_months.jpg')
print(df)

# INDPRO revisions history by revision number
df = pd.concat([alf(series_id,
                    start=20160101,
                    release=n).rename(f"release {n}")
                for n in range(1, 9)], axis=1)
df.index = pd.DatetimeIndex(df.index.astype(str))
ax = df.plot(logy=False)
ax.set_title(f"Revisions of {series_id} by release number")
if imgdir:
    plt.savefig(imgdir / 'release_revisions.jpg')
show(df[df.index > '2019-01-01'],
     caption=f"{series_id} Revisions, retrieved {today}", **SHOW)


# Release dates of series in FRED-MD collection
md_df, md_transform = fred_md()
end = md_df.index[-1]
out = {}
for i, col in enumerate(md_df.columns):
    out[col] = alf(col, release=1, start=end, end=end, realtime=True)
    if col.startswith('S&P'):  # stock market data available same day close
        out[col] = Series({end: end}, name='realtime_start').to_frame()
    elif col in alf._splice_fredmd:
        if isinstance(alf._splice_fredmd[col], str):
            out[col] = alf(alf._splice_fredmd[col],
                           release=1,
                           start=end,
                           end=end,
                           realtime=True)
        else:  # if FRED-MD series was spliced
            out[col] = pd.concat([alf(c,
                                      release=1,
                                      start=end,
                                      end=end,
                                      realtime=True)
                                  for c in alf._splice_fredmd[col][1:]])

# special case of Consumer Sentiment (date convention)
out['UMCSENT'] = alf('UMCSENT', release=1, realtime=True)
out['UMCSENT'] = out['UMCSENT'][out['UMCSENT']['realtime_start'] > end].iloc[:1]

# special case of Claims (weekly averages)
out['CLAIMS'] = alf('ICNSA', release=1, realtime=True)
out['CLAIMS'] = out['CLAIMS'][out['CLAIMS']['realtime_start'] > end].iloc[:1]

# Plot for a representative monthly cross-section
#release = Series({k: max(v['realtime_start'])
#                  for k,v in out.items() if v is not None and len(v)})\
#                      .sort_values()
release = Series({k: str(min(v['realtime_start'])) if v is not None and len(v)
                  else None
                  for k,v in out.items()}).sort_values()

fig, ax = plt.subplots(clear=True, num=1, figsize=(13, 5))
ax.plot(pd.to_datetime(release, errors='coerce'))
ax.axvline(release[~release.isnull()].index[-1], c='r')
ax.set_title(f"Current ({end}) FRED-MD series, retrieved {today}")
ax.set_ylabel('First Release Date')
ax.set_xticks(np.arange(len(release)))
ax.set_xticklabels(release.index, rotation=90, fontsize='xx-small')
plt.tight_layout(pad=2)
if imgdir:
    plt.savefig(imgdir / 'fredmd_release.jpg')

Series(release.values,
       index=[(s, alf.header(s)) for s in release.index]).tail(20)

