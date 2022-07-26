"""Convenience wrappers and functions for data plotting and display

- matplotlib
- seaborn
- statsmodels
- pandas

Functions for:

- chart types: date axis, time axis, confidence bands, bar, hist, scatter
- regression diagnostics
- formatting DataFrames

Copyright 2022, Terence Lim

MIT License
"""
from typing import Iterable, Mapping, List, Any, Tuple, Callable, Dict
import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame, Series, Timestamp
import os
import re
import time
import requests
import calendar
from datetime import datetime
from pandas.api.types import is_list_like, is_datetime64_any_dtype,\
    is_integer_dtype, is_string_dtype, is_numeric_dtype
from pandas.api import types
from numpy.ma import masked_invalid as valid            
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.api as sm 
from matplotlib import dates as mdates
from matplotlib import colors, cm
from matplotlib.lines import Line2D
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  # for date formatting in plots
from finds.busday import to_date, to_datetime

# plt.style.use('ggplot')

##############################
#
# 1. Chart types: hist, scatter, bar, date, time, bands
#
##############################
def plot_date(y1: DataFrame, y2: DataFrame | None = None, ax: Any = None,
              xmin: int = 0, xmax: int = 99999999, cn: int = 0,
              fontsize: int = 12, rescale: bool = False, yscale: bool = False,
              ls1: str = '-', ls2: str = '-', marker: str = '',
              hlines: List[float] = [], vlines: List[int] = [],
              vspans: List[Tuple[int, int]] = [], xlabel: str = '',
              points: DataFrame | Series | None = None, rotation: float = 0,
              title: str = "", ylabel1: str = "", ylabel2: str = "",
              legend1: List[str] = [], legend2: List[str] = [],
              loc1: str = 'upper left', loc2: str = 'upper right', **kwargs):
    """Line plot with int date on x-axis, and primary and secondary y frames

    Args:
        y1: Plot on primary y-axis
        y2: Plot on secondary y-axis
        ax: Matplotlib axes object from plt.subplots() or plt.gca()
        cn: Starting CN color to cycle through
        xmin: Minimum of x-axis date range
        xmax: Maximum of x-axis date range (default is auto)
        rotation: Rotation of x-axis ticks
        hlines: Y-axis points where to place horizontal lines
        vlines: X-axis points where to place vertical lines 
        vspans: Vertical regions to highlight
        xlabel: X-axis label
        ylabel1: Primary y-axis label
        ylabel2: Secondary y-axis label
        fontsize: Base font size
        points: Points and labels to annotate
        title: Main title
        legend1, legend2: Lists of legend labels
        loc1, loc2: Locations to place legends
    """
    ax = ax or plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    ax.xaxis.set_tick_params(rotation=rotation, labelsize=fontsize)
    ax.yaxis.set_tick_params(rotation=rotation, labelsize=fontsize)
    if y1 is not None:
        y1 = DataFrame(y1)
        y1 = y1.loc[(y1.index >= xmin) & (y1.index <= xmax)]
        base = y1.loc[max(y1.notna().idxmax()),:] if rescale else 1
        #sns.lineplot(x = pd.to_datetime(y1.index[f], format='%Y%m%d'),
        #y = y1.loc[f], ax=ax)
        for ci, c in enumerate(y1.columns):
            f = y1.loc[:,c].notnull().values
            ax.plot(to_datetime(y1.index[f]),
                    y1.loc[f,c] / (base[c] if rescale else 1),
                    marker=marker,
                    linestyle=ls1,
                    color=f'C{ci+cn}')
        if points is not None:
            ax.scatter(to_datetime(points.index),
                       points,
                       marker='o',
                       color='r')
        if len(y1.columns) > 1 or legend1:
            ax.set_ylabel('')
            ax.legend(legend1 or y1.columns,
                      fontsize=fontsize,
                      loc=loc1)
        if ylabel1:
            ax.set_ylabel(ylabel1, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
            
    if y2 is not None:
        y2 = DataFrame(y2)
        y2 = y2.loc[(y2.index >= xmin) & (y2.index <= xmax)]
        base = y2.loc[max(y2.notna().idxmax()),:] if rescale else 1
        bx = ax.twinx()
        for cj, c in enumerate(y2.columns):
            g = y2.loc[:,c].notnull().values
            bx.plot(to_datetime(y2.index[g]),
                    y2.loc[g, c] / (base[c] if rescale else 1),
                    marker=marker,
                    linestyle=ls2,
                    color=f"C{ci+cj+cn+1}")
        if yscale:
            amin, amax = ax.get_ylim()
            bmin, bmax = bx.get_ylim()
            ax.set_ylim(min(amin, bmin), max(amax, bmax))
        if len(y2.columns) > 1 or legend2:
            bx.set_ylabel('')
            bx.legend(legend2 or y2.columns,
                      fontsize=fontsize,
                      loc=loc2)
        if ylabel2:
            bx.set_ylabel(ylabel2, fontsize=fontsize+2)
    for hline in hlines:
        plt.axhline(hline,
                    linestyle='-.',
                    color='y')
    for vline in vlines:
        plt.axvline(to_datetime(vline),
                    ls='-.',
                    color='y')
    for vspan in vspans:
        plt.axvspan(*([to_datetime(v) for v in vspan]),
                    alpha=0.5,
                    color='grey')
    plt.title(title, fontsize=fontsize+4)
    plt.tight_layout()

def plot_bands(mean: Series, stderr: Series, width: float = 1.96,
               x: List[int] = [], ylabel: str = '', xlabel: str = '',
               c: str = "b", loc: str = 'best', legend: List[str] = [],
               ax: Any = None, fontsize: int = 10, title: str = '',
               hline: List[float] = [], vline: List[float] = []):
    """Line plot a series with confidence bands

    Args:
        mean: Mean values to plot
        stderr: Stderr values to plot confidence bands
        width: Multipler on stderr for confidence bands
        c: Color to fill bands
        x: X-axis values
        ylabel, xlabel: Axis labels
        legend: List of legend labels
        loc: Location to display legend
        ax: Axis object
        fontsize: Base font size
        title: Main title string
        hline: List of y-axis values to plot horizontal lines
        vline: List of x-axis values to plot vertical lines
    """
    ax = ax or plt.gca()
    if not x:
        x = np.arange(len(mean))      # x-axis is event day number
    for line in hline:
        ax.axhline(line, linestyle=':', color='g')
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
        
def plot_scatter(x: Series, y: Series, labels: List = [], ax: Any = None,
                 xlabel: str = '', ylabel: str = '', c: Any = None,
                 cmap: Any = None, alpha: float = 0.75, edgecolor: Any = None,
                 s: float = 10, marker: str | None = None, title: str = '',
                 abline: bool | None = True, fontsize: int = 12):
    """Scatter plot, optionally with abline slope and point labels

    Args:
        x: Series to plot on horizontal axis
        y: Series to plot on horizontal axis
        labels: List of annotations for points
        ax: Matplotlib axes object, from plt.subplots() or plt.gca()
        xlabel: Horizontal axis label
        ylabel: Vertical axis label
        title: Title of plot
        abline: To plot abline or 45-degree line. If none, do not plot slope
        labels: List of 3-tuples (text, x, y) to annotate
        alpha: transparency of scatter points
        edgecolor: edge color of scatter points
        marker: marker type of scatter points
        s: Marker area size
        cmap: Color map to use for scatter points
        abline: True for fitted slope, False for 45-degree line, None is no plot
        fontsize: Base font size
        title: Main title string
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
    
    xlabel = xlabel or (x.name if hasattr(x, 'name') else "")
    ylabel = ylabel or (y.name if hasattr(y, 'name') else "")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if len(labels):
        for t, xt, yt in zip(labels, x, y):
            plt.text(xt * 1.01, yt * 1.01, t, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+4)
    mfc = cax.get_fc()[0]    
    return Line2D([0], [0], marker=marker, mfc=mfc, ms=10, ls='', c=mfc)

def plot_hist(*args, kde: bool = True, hist: bool = False,
              bins: List[float] = [],
              pdf: Callable | List | Dict = scipy.stats.norm.pdf,
              ax: Any = None, title: str = '', xlabel: str = '',
              ylabel: str = 'density', fontsize: int = 12):
    """Histogram bar plot with a benchmark probability density
    
    Args:
        ax: Axis object
        bins: List of bin values
        hist: Plots histogram
        kde: Plots kernel density curve
        pdf: Benchmark probability density
        ylabel, xlabel: Axis labels
        fontsize: Base font size
        title: Main title text
    """
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

def plot_bar(y: DataFrame, ax: Any = None, labels: List[str] = [],
             xlabel: str = '', ylabel: str = '', fontsize: int = 12,
             title: str = '', legend: List[str] = [], loc: str = 'best',
             labelsize: int = 8, rotation: float = 0.):
    """Bar plot with annotated points

    Args:
        y: DataFrame of y-values, observations in rows, variables in columns
        ax: Axis object
        labels: List of labels to annotate
        labelsize: Font size of annotation labels text
        rotation: Rotate annotation labels text
        xlabel, ylabel: Axis labels
        fontsize: Base font size
        title: Main title text
        legend: List of legend names
        loc: Location for legend
    """
    ax = ax or plt.gca()
    bars = list(np.ravel(y.plot.bar(ax=ax, width=0.8).containers, order='F'))
    ax.set_title(title, fontsize=fontsize+4)
    ax.xaxis.set_tick_params(rotation=0, labelsize=fontsize)
    ax.yaxis.set_tick_params(rotation=0, labelsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize+2)        
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize+2)
    if not loc:
        if legend:
            ax.legend(legend, loc=loc)
        else:
            ax.legend(loc=loc)
    if labels:
        for pt, label in zip(bars, labels):
            ax.annotate(str(label),
                        fontsize=labelsize,
                        xy=(pt.get_x() + pt.get_width() / 2, pt.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        rotation=rotation)
    
open_t = pd.to_datetime('1900-01-01T09:30')  # usual NYSE open
close_t = pd.to_datetime('1900-01-01T16:00') # usual NYSE close
def plot_time(y1: DataFrame, y2: DataFrame | None = None, ax: Any = None,
              xmin: Timestamp = open_t, xmax: Timestamp = close_t,
              marker: str = '', title: str = '', loc1: str = '',
              loc2: str = '', legend1: List[str] = [],
              legend2: List[str] = [], fontsize: int = 12, **kwargs):
    """Plot lines with Timestamp time on x-axis; primary and secondary y-axis

    Args:
        y1: DataFrame to plot on left axis
        y2: DataFrame (or None) to plot on right axis
        ax : matplotlib axes object to plot in
        xmin: left-most x-axis time, None to include all 
        xmax: right-most x-axis time, None to include all
        marker: style of marker to plot
        title: text to display as title
        loc1, loc2: locations to place legend/s
        legend1, legend2: labels to display in legend
        fontsize: Base font size
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

    
##############################
#
# 2. Regression diagnostics: fitted, qq, leverage, scale
#
##############################
    
def plot_fitted(fitted: Series, resid: Series, n: int = 3, ax: Any = None,
                title: str = "Residuals vs Fitted", fontsize: int = 12,
                strftime: str = '%Y-%m-%d') -> Series:
    """Plot residuals and identify outliers
    
    Args:
        ax: Axis object
        fitted: Fitted Series
        residual: Residual Series
        n: Number of outlier points in each end to identify
        strftime: string to format time display
        title: Main title text
        fontsize: Base font size
    """
    ax = ax or plt.gca()
    outliers = np.argpartition(resid.abs().values, -n)[-n:]
    sns.regplot(x=fitted,
                y=resid,
                lowess=True,
                ax=ax,
                scatter_kws={"s": 20, 'alpha': 0.5},
                line_kws={"color": "r", "lw": 1})
    ax.scatter(fitted[outliers],
               resid[outliers],
               c='m',
               alpha=.25)
    for i in outliers:
        if strftime:
            label = resid.index[i].strftime(strftime)
        else:
            label = str(resid.index[i])        
        ax.annotate(label,
                    xy=(fitted.iloc[i],resid.iloc[i]),
                    c='m')
    ax.set_title(title)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    return resid.iloc[outliers].rename('outliers')

def plot_qq(resid: Series, title: str = 'Normal Q-Q', ax: Any = None,
            z: float = 2.807, strftime: str = '%Y-%m-%d') -> DataFrame:
    """QQ Plot

    Args:
        resid: Residual Series
        strftime: string to format time display
        title: Main title text
        z: Z-value to identify outliers
    """
    pp = ProbPlot(resid, fit=True)
    outliers = abs(pp.sample_quantiles) > z
    ax = ax or plt.gca()
    pp.qqplot(ax=ax,
              color='C0',
              alpha=.5)
    sm.qqline(ax=ax,
              line='45',
              fmt='r--',
              lw=1)
    z = resid.sort_values().index[outliers]
    for x, y, i in zip(pp.theoretical_quantiles[outliers],
                       pp.sample_quantiles[outliers],
                       z):
        if strftime:
            label = i.strftime(strftime)
        else:
            label = str(i)
        ax.annotate(label,
                    xy=(x,y),
                    c='m')
    ax.set_title(title)
    ax.set_ylabel('Standardized residuals')
    return DataFrame({'residuals': pp.sorted_data[outliers],
                      'standardized': pp.sample_quantiles[outliers]}, index=z)

def plot_scale(fitted: Series, resid: Series, ax: Any = None,
               title: str = "Scale-Location", n: int = 3, alpha: float = 0.5,
               strftime: str = '%Y-%m-%d'):
    """Plot scale of residuals with outliers

    Args:
        fitted: Fitted Series
        resid: Residual Series
        ax: Axis object
        title: Main title text
        alpha: Transparency of points
        n: number of outliers in each end to identify
        strftime: string to format time display
    """
    ax = ax or plt.gca()
    resid = np.sqrt(np.abs(resid/resid.std()))
    ax.scatter(fitted,
               resid,
               alpha=alpha);
    sns.regplot(x=fitted,
                y=resid,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'r', 'lw': 1});
    ax.set_title("Scale-Location")
    ax.set_ylabel('$\sqrt{|Standardized \ residuals|}$');
    ax.set_xlabel('Fitted values')
    outliers = np.argpartition(resid.values, -n)[-n:]
    ax.scatter(fitted[outliers], resid[outliers], c='m', alpha=.25)
    for i in outliers:
        if strftime:
            label = resid.index[i].strftime(strftime)
        else:
            label = str(resid.index[i])
        ax.annotate(label,
                    xy=(fitted[i], resid[i]),
                    c='m')

def plot_leverage(resid: Series, hat: np.array, dist: np.array, ddof: int,
                  title: str = "Residuals vs Leverage", ax: Any = None,
                  strftime='%Y-%m-%d') -> DataFrame:
    """Plot leverage and identify influential points

    Args:
        resid: Residual Series
        hat: Hat values
        dist: Distance values
        ddof: Degrees of freedom of model
        ax: Axis object
        title: Main title text
        strftime: string to format time display
    """
    ax = ax or plt.gca()
    s = np.sqrt(np.sum(np.array(resid)**2 * (1 - hat))/(len(hat) - ddof))
    r = resid/s   # studentized residual
    sns.regplot(x=hat,
                y=r,
                scatter=True,
                ci=False,
                lowess=True,
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'r', 'lw': 1})
    influential = np.where(dist > 1)[0]
    ax.scatter(hat[influential], r[influential], c='c', alpha=.5)
    annotate = np.where(dist > 0.5)[0]
    for i in annotate:
        if strftime:
            label = r.index[i].strftime(strftime)
        else:
            label = str(r.index[i])        
        ax.annotate(label,
                    xy=(hat[i], r[i]),
                    c=('r' if dist[i] > 1 else 'c'))
    ax.set_title(title)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residuals")
    legend = None
    x = np.linspace(0.001, ax.get_xlim()[1], 50)    
    for sign in [1, -1]:   # plot Cook's Distance thresholds (both signs)
        for thresh in [.5, 1]:
            y = sign * np.sqrt(thresh * ddof / x) * (1 - x)
            g = (y > ax.get_ylim()[0]) & (y < ax.get_ylim()[1])
            if g.any():
                legend, = ax.plot(x[g], y[g], color='m', lw=.5+thresh, ls='--')
                ax.annotate(str(thresh),
                            xy=(max(x[g]),
                                max(y[g]) if sign < 0 else min(y[g])),
                            c='m')
    if legend:
        legend.set_label("Cook's Distance")
        ax.legend(loc='best')
    return DataFrame({'influential': resid.iloc[influential],
                      "cook's D": dist[influential],
                      "leverage": hat[influential]},
                     index=resid.iloc[influential].index)


##############################
#
# 3. DataFrame format
#
##############################


def row_formatted(df: DataFrame, formats: Dict = {}, default: str = '{}',
                  width: int = 0) -> DataFrame:
    """Apply display formats by row index, and set row index width

    Args:
        df: DataFrame to format by row
        formats: Dictionary of format strings, keyed by index label
        default: Default format string
        width: To truncate index widths

    Examples:

    >>> row_formatted(prices, formats={'vwap': '{:.0f}', 'mid': '{:.3f}'})
    >>> formats = dict.fromkeys(['start date', 'Num Stocks'], '{:.0f}')
    >>> row_formatted(DataFrame(out), formats=formats, default='{:.3f}'))
    """
    
    out = df.apply(lambda x: x.map(formats.get(x.name, default).format), axis=1)
    if width:
        out.index = out.index.str.slice(0, width)
    return out


def show(df: DataFrame, latex: bool | None = True, ndigits: int | None = None,
         max_colwidth: int = 100, caption: str = '',
         **kwargs) -> DataFrame | None:
    """Helper to output a DataFrame in publishing format

    Args:
        df: DataFrame to display
        latex: True is to_latex. False is to_string. None simply returns df
        max_colwidth: Pandas context to allow maximum width of column
        ndigits: Number of digits to round floats
        caption: Caption of latex table, or title with underline to print
        kwargs: Passed on to to_latex, e.g.

        - na_rep (str, default 'NaN'): Missing data representation
        - float_format (default None): Formatter for floating point numbers, 
          e.g. float_format="%.2f" and float_format="{:0.2f}".format
          will both result in 0.1234 being formatted as 0.12.
        - bold_rows (default False): Make the row labels bold in the output.
        - longtable: Use a longtable environment instead of tabular. Requires
          adding a \\usepackage{longtable} to your LaTeX preamble.
        - label: LaTeX label to be placed inside \\label{} in the output.
          This is used with \\ref{} in the main .tex file.
        - position: LaTeX positional argument for tables, to be placed after
          \\begin{} in the output.

    """
    with pd.option_context("max_colwidth", max_colwidth):
        if hasattr(df, 'to_frame'):
            df = df.to_frame()
        if ndigits is not None:
            df = df.round(ndigits)
        if latex:               # display amd return entire in latex
            s = df.to_latex(caption=caption, **kwargs) 
            print(s)
            return s
        elif latex is None:     # return as is (for notebook display)
            if caption:
                return df.rename_axis(caption)
            else:
                return df                       
        else:
            s = df.to_string()  # print and return entire frame as str
            if caption:
                print(caption)
                print('-' * len(caption))
            print(s)
            print()
            return s
        
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
      ax.axvline(x=tick, linestyle='--', alpha=.4, color='#eeeeee', zorder=1)

  # Set x-axis label
  ax.set_xlabel("Average Trip Duration", labelpad=20, weight='bold', size=12)

  # Set y-axis label
  ax.set_ylabel("Start Station", labelpad=20, weight='bold', size=12)

  # Format y-axis label
  ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
"""

