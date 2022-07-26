# Financial Data Science python library

These modules support retrieving and manipulating structured and
unstructured financial data sets, and fitting and testing quant finance and
machine learning models.

__NEW__ [See documentation in sphinx-format](https://terence-lim.github.io/financial-data-science-docs/)

[https://github.com/terence-lim/financial-data-science](https://github.com/terence-lim/financial-data-science)

by: [Terence Lim](https://www.linkedin.com/in/terencelim)

&nbsp;


### [taq.py](taq.py)

- Class and methods to process TAQ trade and quotes tick data

  - NYSE Daily TAQ: Master, NBBO, Trades
  - market microstructure: bid-ask spreads, trade conditions, tick test

### [sectors.py](sectors.py)

- Implement industry sectoring, and wrapper over BEA web api and data

  - Bureau of Economic Analysis: Input-Output Use Tables
  - SIC, NAICS crosswalks: https://www.naics.com/
  - Fama-French industry codes

### [graph.py](graph.py)

- Graph network convenience wrapper

  - networkx: link prediction, community detection, centrality

### [alfred.py](alfred.py)

- Class and methods to access ALFRED/FRED apis, and FRED-MD/FRED-QD

  - FRED, ALFRED: St Louis Fed api's, with revision vintages
  - FRED-MD, FRED-QD: McCracken website at St Louis Fed
  - Bai and Ng (2002), McCracken and Ng (2015, 2020) factors-EM algorithm

    - https://research.stlouisfed.org/econ/mccracken/fred-databases/

### [edgar.py](edgar.py)

- Class and methods to retrieve and manipulate EDGAR text data

  - SEC Edgar: 10-K, 10-Q, 8-K
  - MD&A and Business Descriptions items

### [unstructured.py](unstructured.py)

- Classes to implement interface for unstructured and textual datasets

  - FOMC minutes
  - Loughran and McDonald words
  - S&P CapitalIQ key developments situations text

### [structured.py](structured.py)

- Classes to implement interface for structured data sets

  - CRSP (daily, monthly, names, delistings, distributions, shares outstanding)
  - S&P/CapitalIQ Compustat (Annual, Quarterly, Key Development, customers)
  - IBES Summary

- Redis store: SQL query results are (optionally) cached in in Redis

- Signals class to store and retrieve derived signal values

- Subclasses to mimic parent class interfaces with pre-loaded batch in memory

- Lookup identifiers within and across data sets


### [busday.py](busday.py)

- Implement custom trading-day business date calendar 

  - Numpy busdaycalendar
  - Pandas CustomBusinessDay and offsets
  - FamaFrench daily research factors

### [database.py](database.py)

- Wrappers for database engines

  - SQL: sqlalchemy
  - MongoDB: pymongo
  - Redis NoSQL key-value store: redis

- Convenience methods to:

  - Load, store and manipulate pandas DataFrames with SQLAlchemy database schemas
  - Serialize DataFrames to Redis key-value store (for caching SQL query results)

### [backtesting.py](backtesting.py)

- Evaluate backtests, event studies and risk premiums

  - Event studies: cumulative abnormal returns
  - Risk premiums: Fama-MacBeth regressions
  - Walk-forward portfolio rebalances Backtest: Sharpe ratio, appraisal ratio, ...
  - DailyPerformance: Daily returns performance of periodic portfolio holding


### [recipes.py](recipes.py)

- Numerical and data helper functions

- econometrics: unit root, linear regression
- FFT: convolutions and correlations
- data filters
- financial: bonds and risk math

### [pyR.py](pyR.py)

- Wrapper class over rpy2 package to interface with R environment

  - Deconstruct and expose an rpy2 or numpy/pandas object interchangeably.

### [gdrive.py](gdrive.py)

- Convenience class methods to use google drive apis

  - google REST api's


### [display.py](display.py)

- Convenience wrappers for data plotting and display

  - matplotlib
  - seaborn
  - statsmodels
  - pandas

- Functions for:

  - chart types: date axis, time axis, confidence bands, bar, hist, scatter
  - plotting linear regression diagnostics
  - formatting DataFrames
