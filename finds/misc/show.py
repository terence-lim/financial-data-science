"""Dataframe formatting

Copyright 2022, Terence Lim

MIT License
"""
import requests
import time
import random
import pandas as pd
from pandas import DataFrame
from typing import Dict

# with pd.option_context('display.float_format', lambda x: '{:,.0f}'.format(x)):
    
class Show:
    """Helper to format DataFrame for output

    Args:
        latex: True returns to_latex. False returns to_string. None returns df
        ndigits: Number of digits to round floats
    """
    def __init__(self, latex: bool | None = True, ndigits: int | None = None):
        self.latex = latex
        self.ndigits = ndigits

    def __call__(self, df: DataFrame, max_colwidth: int = 100, caption: str = '',
                 max_rows: int | None = 15, **kwargs) -> DataFrame | None:
        """Format and display data frame

        Args:
          df: DataFrame to display
          max_colwidth: Pandas option to allow maximum columns to display
          max_colwidth: Pandas option to allow maximum rows to display
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
        - index (bool, default True): Write row names (index)
        """
        #with pd.option_context("max_colwidth",max_colwidth,'max_rows',max_rows):
        if hasattr(df, 'to_frame'):
            df = df.to_frame()
        if self.ndigits is not None:
            df = df.round(self.ndigits)
        if self.latex:               # display amd return entire in latex
            s = df.to_latex(caption=caption, **kwargs) 
            print(s)
            return s
        elif self.latex is None:     # return as is (for notebook display)
            if caption:
                return df.rename_axis(caption)
            else:
                return df                       
        else:
            s = df.to_string(**kwargs)  # print entire frame as str
            if caption:
                print(caption)
                print('-' * len(caption))
            print(s)
            print()
            return None

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

