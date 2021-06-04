"""Economic time series and releases: revisions and vintages

- St Louis Fed FRED/ALFRED

Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from finds.alfred import Alfred, fred_md
import time
from datetime import datetime
import os
from settings import settings
ECHO = True
imgdir = os.path.join(settings['images'], 'ts')
alf = Alfred(api_key=settings['fred']['api_key'], echo=ECHO)
#savefile=settings['scratch'] + 'fred.md')

# Popular FRED series: top two pages
r = {}
for page in [1, 2]:   # scrape first two pages
    popular = Alfred.popular(page)
    for s in popular:
        t = alf.series(s)   # calls 'series' FRED api
        r.update({s:{} if t.empty else t.iloc[-1][['title', 'popularity']]})
DataFrame.from_dict(r, orient='index')
    
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
s = 'INDPRO'  # https://www.bea.gov/gdp-revision-information
print(f"Latest revision (as of {datetime.today().strftime('%Y-%m-%d')}:")
print(alf(s, start=20200401, end=20200731, realtime=True))
print("First Release:")
print(alf(s, release=1, start=20200401, end=20200731, realtime=True))
print("Second Release:")
print(alf(s, release=2, start=20200401, end=20200731, realtime=True))
print("Five-Monthly Lag:")
print(alf(s, release=pd.DateOffset(months=5), start=20200401, end=20200731,
          realtime=True))
print("Vintage date = 20200630:")
print(alf(s, vintage=20200630, realtime=True, start=20200401))

# INDPRO revisions history by months lagged
df = pd.concat([alf(s, start=20140101, release=pd.DateOffset(months=m))\
                .rename(f"{m}-month lag") for m in [1,3,9,21,33]], axis=1)
df.index = pd.DatetimeIndex(df.index.astype(str))
ax = df.plot(logy=False)
ax.set_title(f"Releases of {s} by months lag")
if imgdir: plt.savefig(os.path.join(imgdir, 'release_months.jpg'))
plt.show()
print(df)

# INDPRO revisions history by revision number
df = pd.concat([alf(s, start=20140101, release=n).rename(f"release {n}")
                for n in range(1, 6)], axis=1)
df.index = pd.DatetimeIndex(df.index.astype(str))
ax = df.plot(logy=False)
ax.set_title(f"Revisions of {s}")
if imgdir: plt.savefig(os.path.join(imgdir, 'release_revisions.jpg'))
plt.show()
print(df)


# Release dates of series in FRED-MD collection
mdf, mt = fred_md()
end = mdf.index[-3]
out = {}
for i, col in enumerate(mdf.columns):
    out[col] = alf(col, release=1, start=end, end=end, realtime=True)
    if col.startswith('S&P'):  # stock market data available same day close
        out[col] = Series({end: end}, name='realtime_start').to_frame()
    elif col in alf.fred_adjust:
        if isinstance(alf.fred_adjust[col], str):
            out[alf.fred_adjust[col]] = alf(alf.fred_adjust[col], release=1,
                                            start=end, end=end, realtime=True)
        else:  # if FRED-MD series was spliced
            out[col] = pd.concat([alf(c, release=1, start=end, end=end,
                                      realtime=True)
                                  for c in alf.fred_adjust[col][1:]])

# special cases of Claims (averages weekly) and Cons Sentiment (date convention)
out['UMCSENT'] = alf('UMCSENT', release=1, realtime=True)
out['UMCSENT'] = out['UMCSENT'][out['UMCSENT']['realtime_start'] > end].iloc[:1]
out['CLAIMS'] = alf('ICNSA', release=1, realtime=True)
out['CLAIMS'] = out['CLAIMS'][out['CLAIMS']['realtime_start'] > end].iloc[:1]

# Plot for a representative monthly cross-section
release = Series({k: max(v['realtime_start']) for k,v in out.items()
                  if v is not None and len(v)}).sort_values()
fig, ax = plt.subplots(clear=True, num=1, figsize=(10,6))
ax.plot(pd.DatetimeIndex(release.astype(str)))
ax.set_title(f"Release Dates of FRED-MD series ending {end}")
ax.set_xticks(np.arange(len(release)))
ax.set_xticklabels(release.index, rotation=90, fontsize='xx-small')
plt.tight_layout(pad=2)
if imgdir: plt.savefig(os.path.join(imgdir, 'fredmd_release.jpg'))
plt.show()

Series(release.values,
       index=[(s, alf.header(s)) for s in release.index]).tail(20)

