"""Convenience methods for data visualization

- matplotlib, seaborn, statsmodels, pandas

Author: Terence Lim
License: MIT
"""
import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame, Series
import wget
import os
import re
import time
import requests
import calendar
from datetime import datetime
from pandas.api.types import is_list_like, is_datetime64_any_dtype
from pandas.api.types import is_integer_dtype, is_string_dtype, is_numeric_dtype
from pandas.api import types
from numpy.ma import masked_invalid as valid            
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import dates as mdates
from matplotlib import colors, cm
from matplotlib.lines import Line2D
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  # for date formatting in plots

# plt.style.use('ggplot')

def row_formatted(df, formats={}, width=None):
    """Apply display formats by row index, and set row index width

    Examples
    --------
    row_formatted(prices, formats={'vwap': '{:.0f}', 'mid': '{:.3f}'})
    """
    out = df.apply(lambda x: x.map(formats.get(x.name,'{}').format), axis=1)
    if width:
        out.index = out.index.str.slice(0, width)
    return out

def plot_bands(mean, stderr, width=1.96, x=None, ylabel=None, xlabel=None,
               c="b", loc='best', legend=None, ax=None, fontsize=10, 
               title=None, hline=None, vline=None):
    """Line plot a series with confidence bands"""
    ax = ax or plt.gca()
    if x is None:
        x = np.arange(len(mean))      # x-axis is event day number
    if hline is not None:
        if not is_list_like(hline):
            hline = [hline]
        for line in hline:
            ax.axhline(line, linestyle=':', color='g')
    if vline is not None:
        if not is_list_like(vline):
            vline = [vline]
        for line in vline:
            ax.axvline(line, linestyle=':', color='g')
    ax.plot(x, mean, ls='-', c=c)
    ax.fill_between(x, mean-(width*np.array(stderr)),
                    mean+(width*np.array(stderr)), alpha=0.3, color=c)
    if legend:
        ax.legend(legend, loc=loc, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+4)
    ax.set_ylabel(ylabel, fontsize=fontsize+2)
    ax.set_xlabel(xlabel, fontsize=fontsize+2)
        
def plot_scatter(x, y, labels=None, ax=None, xlabel=None, ylabel=None,
                 c=None, cmap=None, alpha=0.75, edgecolor=None, s=10,
                 marker=None, title='', abline=True, fontsize=12):
    """Scatter plot, optionally with abline slope and point labels

    Parameters
    ----------
    x : Series or array-like
        to plot on horizontal axis
    y : Series or array-like
        to plot on horizontal axis
    labels : Series or array-like of str, default is None
        annotate plotted points with text
    ax : matplotlib axes object, optional
        from plt.subplots() or plt.gca(), default is None
    xlabel : str, optional
        horizontal axis label, default is x.name else None
    ylabel : str, optional
        vertical axis label, default is y.name else None
    title : str, optional
        title of plot, default is ''
    abline : bool, default None
        plot abline if True, or 45-degree if False, If None, do not plot slope
    s : numeric, default 10
        marker area size
    """
    if ax is None:
        ax = plt.gca()
        ax.cla()
        ax.clear()
    if c is not None and cmap is not None:
        cmin = min(c)
        cmax = max(c)
        norm = colors.Normalize(cmin - (cmax-cmin)/2, cmax)
        c = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(c)
        cmap = None
    cax = ax.scatter(x, y, marker=marker, s=s, c=c, alpha=alpha,
                     edgecolor=edgecolor, cmap=cmap)
    #cmap=plt.cm.get_cmap('tab10', 3)
    if abline is not None:
        xmin, xmax, ymin, ymax = ax.axis()
        if abline:     # plot fitted slope
            f = ~(np.isnan(x) | np.isnan(y))
            slope, intercept = np.polyfit(list(x[f]), list(y[f]), 1)
            y_pred = [slope * i + intercept for i in list(x[f])]
            ax.plot(x[f], y_pred, 'g-')
        else:          # plot 45-degree line
            bottom_left, top_right = min(xmin, ymin), max(xmax, ymax)
            ax.plot([bottom_left, top_right], [bottom_left, top_right], 'g-')
    
    xlabel = xlabel or (x.name if hasattr(x, 'name') else None)
    ylabel = ylabel or (y.name if hasattr(y, 'name') else None)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if labels is not None:
        for t, xt, yt in zip(labels, x, y):
            plt.text(xt * 1.01, yt * 1.01, t, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+4)
    mfc = cax.get_fc()[0]    
    return Line2D([0], [0], marker=marker, mfc=mfc, ms=10, ls='', c=mfc)

def plot_hist(*args, kde=True, hist=False, bins=None, pdf=scipy.stats.norm.pdf,
              ax=None, title='', xlabel='', ylabel='density', fontsize=12):
    """Histogram bar plot with target density"""
    ax = ax or plt.gca()
    ax=plt.gca()
    for arg in args:
        frame = DataFrame(arg)
        for col in frame.columns:
            y = frame[col].notnull().values
            sns.distplot(frame[col][y], kde=kde, hist=hist,
                         bins=bins, label=col, ax=ax)
    if pdf:
        if not types.is_list_like(pdf):
            pdf = [pdf]
        if isinstance(pdf, dict):
            labels = list(pdf.keys())
            pdf = list(pdf.values())
        else:
            labels = None
            pdf = list(pdf)
        bx = ax.twinx() if args else ax
        bx.yaxis.set_tick_params(rotation=0, labelsize=fontsize)
        x= np.linspace(*plt.xlim(), 100)
        for i, p in enumerate(pdf):
            bx.plot(x, p(x), label=labels[i] if labels else None,
                    color=f"C{len(args)+i}")
        if labels:
            bx.legend(labels, loc='center right')
    ax.legend(loc='center left')
    ax.xaxis.set_tick_params(rotation=0, labelsize=fontsize)
    ax.yaxis.set_tick_params(rotation=0, labelsize=fontsize)
    ax.set_title(title, fontsize=fontsize+4)
    ax.set_ylabel(ylabel, fontsize=fontsize+4)
    ax.set_xlabel(xlabel, fontsize=fontsize+4)

def plot_bar(y, ax=None, labels=None, xlabel=None, ylabel=None, fontsize=12,
             title='', legend=None, loc='best', labelsize=8, rotation=0):
    """Bar plot with annotated points"""
    ax = ax or plt.gca()
    bars = list(np.ravel(y.plot.bar(ax=ax, width=0.8).containers, order='F'))
    ax.set_title(title, fontsize=fontsize+4)
    ax.xaxis.set_tick_params(rotation=0, labelsize=fontsize)
    ax.yaxis.set_tick_params(rotation=0, labelsize=fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize+2)        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize+2)
    if legend is not None:
        ax.legend(legend, loc)
    elif loc is not None:
        ax.legend(loc=loc)
    if labels is not None:
        for pt, freq in zip(bars, np.ravel(labels)):
            ax.annotate(str(freq), fontsize=labelsize,
                        xy=(pt.get_x() + pt.get_width() / 2, pt.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', rotation=rotation)
    
def plot_date(y1, y2=None, ax=None, xmin=0, xmax=99999999, fontsize=12,
              label1=None, label2=None, legend1=None, legend2=None, cn=0,
              loc1='upper left', loc2='upper right', ls1='-', ls2='-',
              hlines=[], vlines=[], vspans=[], marker=None,
              rescale=False, yscale=False, title='', points=None, **kwargs):
    """Line plot with int date on x-axis, and primary and secondary y-dataframes

    Parameters
    ----------
    y1 : DataFrame
       to plot on primary y-axis
    y2 : DataFrame, optional
       to plot on secondary y-axis (default is None)
    ax : matplotlib axes object, optional
       from plt.subplots() or plt.gca(), default is None
    cn : int, default is 0
       to cycle through CN colors starting at N=cn
    xmin : int, optional
       minimum of x-axis date range (default is auto)
    xmax : int, optional
       maximum of x-axis date range (default is auto)
    hlines : list of int (default = [])
        y-axis points where to place horizontal lines
    vlines : list of int (default = [])
        x-axis points where to place vertical lines 
    vspans : list of int tuples (default = [])
        vertical regions to highlight
    """
    ax = ax or plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    if y1 is not None:
        y1 = DataFrame(y1)
        y1 = y1.loc[(y1.index >= xmin) & (y1.index <= xmax)]
        base = y1.loc[max(y1.notna().idxmax()),:] if rescale else 1
        #sns.lineplot(x = pd.to_datetime(y1.index[f], format='%Y%m%d'),
        #y = y1.loc[f], ax=ax)
        for ci, c in enumerate(y1.columns):
            f = y1.loc[:,c].notnull().values
            ax.plot(pd.to_datetime(y1.index[f], format='%Y%m%d'),
                    y1.loc[f,c] / (base[c] if rescale else 1),
                    marker=marker, linestyle=ls1, color=f'C{ci+cn}')
        if points is not None:
            ax.scatter(pd.to_datetime(points.index, format='%Y%m%d'), points,
                       marker='o', color='r')
        if len(y1.columns) > 1 or legend1:
            ax.set_ylabel('')
            ax.legend(legend1 or y1.columns, fontsize=fontsize, loc=loc1)
        if label1:
            ax.set_ylabel(label1, fontsize=fontsize+2)
    if y2 is not None:
        y2 = DataFrame(y2)
        y2 = y2.loc[(y2.index >= xmin) & (y2.index <= xmax)]
        base = y2.loc[max(y2.notna().idxmax()),:] if rescale else 1
        bx = ax.twinx()
        for cj, c in enumerate(y2.columns):
            g = y2.loc[:,c].notnull().values
            bx.plot(pd.to_datetime(y2.index[g], format='%Y%m%d'),
                    y2.loc[g, c] / (base[c] if rescale else 1),
                    marker=marker, linestyle=ls2, color=f"C{ci+cj+cn+1}")
        if yscale:
            amin, amax = ax.get_ylim()
            bmin, bmax = bx.get_ylim()
            ax.set_ylim(min(amin, bmin), max(amax, bmax))
        if len(y2.columns) > 1 or legend2:
            bx.set_ylabel('')
            bx.legend(legend2 or y2.columns, fontsize=fontsize, loc=loc2)
        if label2:
            bx.set_ylabel(label2, fontsize=fontsize+2)
    for hline in hlines:
        plt.axhline(hline, linestyle='-.', color='y')
    for vline in vlines:
        plt.axvline(pd.to_datetime(vline, format='%Y%m%d'), ls='-.', color='y')
    for vspan in vspans:
        plt.axvspan(*([pd.to_datetime(v, format='%Y%m%d') for v in vspan]),
                    alpha=0.5, color='grey')
    ax.xaxis.set_tick_params(rotation=0, labelsize=fontsize)
    ax.yaxis.set_tick_params(rotation=0, labelsize=fontsize)
    plt.title(title, fontsize=fontsize+4)

open_t = pd.to_datetime('1900-01-01T09:30')
close_t = pd.to_datetime('1900-01-01T16:00') 
def plot_time(y1, y2=None, ax= None, xmin=open_t, xmax=close_t, marker=None,
              title='', loc1=None, loc2=None, legend1=None, legend2=None, 
              fontsize=12, **kwargs):
    """Plot lines with time on x-axis, and primary and secondary y-axis

    Parameters
    ----------
    y1 : DataFrame
        to plot on left axis
    y2: DataFrame or None
        to plot on right axis
    ax : axis
        matplotlib axes object to plot in
    xmin : datetime or None, default is '1900-01-01T09:30'
        left-most x-axis time, None to include all 
    xmax : datetime, or None, default is '1900-01-01T16:00'
        right-most x-axis time, None to include all
    marker : str, default is None
        style of market to plot
    title : str, default is ''
        text to display as title
    loc1, loc2 : str, default is None
        locations to place legend/s
    legend1, legend2 : list of str, default is None
        labels to display in legend
    """
    ax = ax or plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    cn = 0   # to cycle through matplotlib 'CN' color palette
    left = DataFrame(y1)
    if xmin:
        left = left.loc[(left.index >= xmin)]
    if xmax:
        left = left.loc[(left.index <= xmax)]
    for cn, c in enumerate(left.columns):
        f = left.loc[:, c].notnull().values
        if cn:    # kludgy hack with time-axis
            ax.plot(left.index[f], left.loc[f, c], color = 'C' + str(cn))
        else:
            sns.lineplot(x=left.index[f], y=left.loc[f, c],
                         color='C' + str(cn),  ax=ax)
    ax.legend(legend1 or left.columns, loc=loc1 or 'upper left',
              fontsize=fontsize)
    if len(left.columns) > 1:
        ax.set_ylabel('', fontsize=fontsize+2)
    if y2 is not None:
        right = DataFrame(y2)
        if xmin:
            right = right.loc[(right.index >= xmin)]
        if xmax:
            right = right.loc[(right.index <= xmax)]
        bx = ax.twinx()
        bx.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        for i, c in enumerate(right.columns):
            g = right.loc[:, c].notnull().values
            bx.plot(right.index[g], right.loc[g, c], color='C' + str(cn+i+1))
        bx.legend(legend2 or right.columns, loc=loc2 or 'lower right',
                  fontsize=fontsize)
        if len(right.columns) > 1:
            bx.set_ylabel('')
    ax.xaxis.set_tick_params(rotation=0, labelsize=fontsize)
    ax.yaxis.set_tick_params(rotation=0, labelsize=fontsize)
    plt.title(title, fontsize=fontsize+4)

def plot_fitted(fitted, resid, n=3, ax=None, title='', fontsize=12,
                strftime='%Y-%m-%d'):
    """Convenience method to plot residuals and identify outliers"""
    ax = ax or plt.gca()
    outliers = np.argpartition(resid.abs().values, -n)[-n:]
    sns.regplot(x=fitted, y=resid, lowess=True, ax=ax,
                scatter_kws={"s": 20, 'alpha': 0.5},
                line_kws={"color": "r", "lw": 1})
    ax.scatter(fitted[outliers], resid[outliers], c='m', alpha=.25)
    for i in outliers:
        ax.annotate(resid.index[i].strftime(strftime) if strftime else
                    str(resid.index[i]), xy=(fitted.iloc[i],resid.iloc[i]),
                    c='m')
    ax.set_title(title or f"Residuals vs Fitted")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    return resid.iloc[outliers].rename('outliers')

from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.api as sm 
def plot_qq(resid, title='', ax=None, z=2.807, strftime='%Y-%m-%d'):
    """Convenience wrapper over QQ Plot"""
    pp = ProbPlot(resid, fit=True)
    outliers = abs(pp.sample_quantiles) > z
    ax = ax or plt.gca()
    pp.qqplot(ax=ax, color='C0', alpha=.5)
    sm.qqline(ax=ax, line='45', fmt='r--', lw=1)
    z = resid.sort_values().index[outliers]
    for x, y, i in zip(pp.theoretical_quantiles[outliers],
                       pp.sample_quantiles[outliers], z):
        ax.annotate(i.strftime(strftime) if strftime else str(i),
                    xy=(x,y),c='m')
    ax.set_title(title or 'Normal Q-Q')
    ax.set_ylabel('Standardized residuals')
    return DataFrame({'residuals': pp.sorted_data[outliers], 'standardized':
                      pp.sample_quantiles[outliers]}, index=z)

def plot_scale(fitted, resid, ax=None, title='', n=3, strftime='%Y-%m-%d'):
    """Convenience method to plot scale of residuals with outliers"""
    ax = ax or plt.gca()
    resid = np.sqrt(np.abs(resid/resid.std()))
    ax.scatter(fitted, resid, alpha=0.5);
    sns.regplot(fitted, resid, scatter=False, ci=False, lowess=True,
                line_kws={'color': 'r', 'lw': 1});
    ax.set_title(title or f"Scale-Location")
    ax.set_ylabel('$\sqrt{|Standardized \ residuals|}$');
    ax.set_xlabel('Fitted values')
    outliers = np.argpartition(resid.values, -n)[-n:]
    ax.scatter(fitted[outliers], resid[outliers], c='m', alpha=.25)
    for i in outliers:
        ax.annotate(resid.index[i].strftime(strftime) if strftime else
                    str(resid.index[i]), xy=(fitted[i], resid[i]), c='m')

def plot_leverage(resid, h, d, ddof, ax=None, strftime='%Y-%m-%d'):
    """Convenience method to plot leverage and identify influential points"""
    ax = ax or plt.gca()
    s = np.sqrt(np.sum(np.array(resid)**2 * (1 - h))/(len(h) - ddof))
    r = resid/s   # studentized residual
    sns.regplot(h, r, scatter=True, ci=False, lowess=True,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'r', 'lw': 1})
    influential = np.where(d > 1)[0]
    ax.scatter(h[influential], r[influential], c='m', alpha=.5)
    for i in influential:
        ax.annotate(r.index[i].strftime(strftime) if strftime else
                    str(r.index[i]), xy=(h[i], r[i]), c='r')
    ax.set_title("Residuals vs Leverage")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residuals")
    legend = None
    x = np.linspace(0.001, ax.get_xlim()[1], 50)    
    for sign in [1, -1]:   # plot Cook's Distance thresholds (both signs)
        for thresh in [.5, 1]:
            y = sign * np.sqrt(thresh * ddof / x) * (1-x)
            g = (y > ax.get_ylim()[0]) & (y < ax.get_ylim()[1])
            if g.any():
                legend, = ax.plot(x[g], y[g], color='m', lw=.5+thresh, ls='--')
    if legend:
        legend.set_label("Cook's Distance")
        ax.legend(loc='best')
    return DataFrame({'influential': resid.iloc[influential],
                      "cook's D": d[influential],
                      "leverage": h[influential]},
                     index=resid.iloc[influential].index)
        
"""
matplotlib examples: https://mode.com/example-gallery/python_horizontal_bar/
  # Despine
  ax.spines['right'].set_visible(False)

  # Switch off ticks
  ax.tick_params(axis="both", which="both", bottom="off", top="off", 
      labelbottom="on", left="off", right="off", labelleft="on")

  # Draw vertical axis lines
  vals = ax.get_xticks()
  for tick in vals:
      ax.axvline(x=tick, linestyle='dashed', alpha=.4, color='#eeeeee', zorder=1)

  # Set x-axis label
  ax.set_xlabel("Average Trip Duration", labelpad=20, weight='bold', size=12)

  # Set y-axis label
  ax.set_ylabel("Start Station", labelpad=20, weight='bold', size=12)

  # Format y-axis label
  ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
"""
