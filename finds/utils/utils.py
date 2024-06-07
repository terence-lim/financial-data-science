"""Miscellaneous utilities

Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_list_like, is_integer_dtype
from datetime import datetime
from typing import Dict, List, Tuple, Any

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


def to_type(v: Any, t=str) -> Any:
    """Convert each element in nested input list to target type"""
    return [to_type(u, t) for u in v] if is_list_like(v) else t(v)


