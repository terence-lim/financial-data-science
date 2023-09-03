"""Evaluate daily returns performance of series of portfolio holdings

Copyright 2022, Terence Lim

MIT License
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict
from finds.structured.stocks import Stocks
_VERBOSE = 1

class DailyPerformance:
    """Compute daily realized returns on periodic holdings
    
    Args:
      stocks: Stocks returns dataset
    """
    
    def __init__(self, stocks: Stocks):
        self.stocks = stocks

    def __call__(self, holdings: Dict[int, Series], end: int) -> Series:
        """Return series of daily returns through end date

        Args:
          holdings: dict (key is int date) of holdings Series (index is permno)
          end: Last date of daily returns to compute performance for

        Returns:
          Series of daily realized portfolio returns
        """
        rebals = sorted(holdings.keys())   # portfolio rebalance dates
        dates = self.stocks.bd.date_range(rebals[0], end) # daily rebaldates
        curr = holdings[rebals[0]]         # initial portfolio
        perf = dict()                      # to collect daily performance
        for date in dates[1:]:   # loop over return dates
            ret = self.stocks.get_section(dataset='daily',
                                          fields=['ret','retx'],
                                          date_field='date',
                                          date=date).dropna()
            perf[date] = sum(curr * ret['ret'].reindex(curr.index, fill_value=0))
            if date in rebals:   # update daily portfolio holdings
                curr = holdings[date]
            else:
                curr = curr * (1 + ret['retx'].reindex(curr.index).fillna(0))
        return Series(perf, name='ret')

if __name__=="__main__":
    from env.conf import credentials
    
