"""Convenience methods for pretty printing

- pandas

Author: Terence Lim
License: MIT
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import os
import re
import time

ECHO = False

def print_all(df, decimals=3, width=120, rows=None, cols=None, colwidth=None,
              noprint=False):
    """Display data frame with temporarily expanded display options"""
    #pd.set_option('display.expand_frame_repr', False)
    options = {'display.max_rows': rows,
               'display.max_columns': cols,
               'display.width': width,
               'display.max_colwidth': colwidth,
               'display.float_format': (None if decimals is None else
                                        f"{{:.{decimals}f}}".format)}
    rollback = {k: pd.get_option(k) for k in options.keys()}  # for rollback
    for k,v in options.items():
        pd.set_option(k, v)
    s = str(df)
    for k,v in rollback.items():   # when done, reset options to rollback
        pd.set_option(k, v)
    if noprint:
        return s
    else:
        print(s)

def print_doc(infile=None, tex=True, verbose=True):
    """Display method and class names and initial doc-string from source file"""
    def match(pattern, input):   # helper to match and extract
        try:
            return "".join(re.match(pattern, input).groups())
        except:
            return None
    def to_tex(x):
        return re.sub('["#]', '', x.replace('_','\_'))\
                 .replace('  ',' {} ').replace('&', '\&') if tex else x

    if infile is None:    # default infile names to print_doc
        f = ['pyR.py', 'graph.py', 'solve.py', 'taq.py', 'alfred.py',
             'edgar.py', 'structured.py', 'unstructured.py', 'sectors.py',
             'backtesting.py', 'readers.py', 'display.py', 'gdrive.py',
             'database.py', 'busday.py', 'learning.py']
        for s in f:
            print_doc('finds/' + s, tex=tex, verbose=verbose)
        return
    tt = "\\texttt{{ {} }} &  " if tex else "{}"   # format strings for output
    it = "\\textit{{class}} \\texttt{{{}}} &  " if tex else "{}"
    endline = "    \\\\\n"  if tex else "\n"
    outfile = os.path.basename(infile).replace('.', '_') \
              + ('.tex' if tex else '.doc')
    with open(infile, 'rt') as f:    # readlines from infile
        lines = f.readlines()
    isclass = False  # if top-level is a class, then match second level defs
    with open(outfile, 'wt') as f:
        for i, line in enumerate(lines):
            x = match(r"^def\s+(\w+)", line) # matches top-level def
            if x is None and isclass:        # matches second-level if not class
                x = match(r"^(\s\s\s\s)def\s+(\w+)", line)
            else:
                isclass = False
            if isclass and x is not None:
                print(x)
            if (x is not None and not re.match("\s*_[a-zA-Z]", x) and
                "_str" not in x and "_repr" not in x and
                "_init" not in x and "_iter" not in x):
                f.write(tt.format(to_tex(x)))
                for j in range(i+1, i+6):
                    y = match(r'^\s*"""(.*)', lines[j])
                    if y is not None or "def " in lines[j]:
                        break
                if y is not None:
                    f.write(to_tex(y))
                f.write(endline)
            x = match(r"^class\s+(\w+)", line)
            if x is not None:    # output class and its first docstring
                isclass = True
                f.write(it.format(to_tex(x)))
                for j in range(i+1, i+6):
                    y = match(r'^\s*"""(.*)', lines[j])
                    if y is not None  or "def " in lines[j]:
                        break
                if y is not None:
                    f.write(to_tex(y))
                f.write(endline)
    return outfile

def print_columns(*args, noprint=False, header=False):
    """Print multiple input series in columns across, ignore indexes"""
    out = DataFrame({k: list(v) for k,v in enumerate(args)})\
          .to_string(index=False, header=header)
    if noprint:
        return out
    else:
        print(out)

def print_multicolumn(series, cols=None, rows=None, noprint=False,
                      header=True, index=True, latex=False,
                      sep='', linesep='', left='', right=''):
    """Display a Series or DataFrame in multiple cols, down rows then across"""
    if latex:     # to use latex decorations
        sep = sep or "&"
        linesep = linesep or "\\\\ \\hline"
        left = left or "\\verb|"
        right = right or "|"
    rows = rows or (len(series)+cols-1) // cols  # infer number of rows from cols
    s = series.to_string(index=index).split('\n')
    if header and len(s) > len(series):   # optionally: print header if available
        t = f" {sep} ".join(f"{left}{s[0]}{right}"
                            for j in range(0, len(series), rows)) + linesep
        if noprint:
            out.append(t)
        else:
            print(t)
    s = s[-len(series):]   # drop header row, if any, from series to print
    out = []
    for i in range(rows):  # print line, each by skipping over rows of series
        t = f" {sep} ".join(f"{left}{s[j]}{right}"
                            for j in range(i, len(series), rows)) + linesep
        if noprint:
            out.append(t)
        else:
            print(t)
    if noprint:
        return "\n".join(out)

def print_verbose(*args, echo=False):
    """Print message if global verbose flag is true, or return verbose value"""
    if not args:
        return ECHO  # return value of global verbose flag is no args
    if ECHO or echo:
        print(*args)
