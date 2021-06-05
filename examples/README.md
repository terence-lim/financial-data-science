# Financial Data Science python code examples

These code examples explore data science and machine learning methods
on large and textual financial data sets.

[https://github.com/terence-lim/financial-data-science](https://github.com/terence-lim/financial-data-science)

by: [Terence Lim](https://www.linkedin.com/in/terencelim)

&nbsp;

## Applications in Unsupervised Learning

### Topic models and FOMC meeting minutes

[fomc_topics.py](fomc_topics.py)

- NMF, LSA, LDA, PLSI matrix decomposition models

### LSTM networks, state space models and mixtures from FRED-MD

[economic_states.py](economic_states.py)

- Long Short-Term Memory networks, hidden states, state space models, Gaussian mixtures
- pytorch, hmmlearn, statsmodels, sklearn

### Unsupervised learning models for clustering economic series

[unsupervised_economics.py](unsupervised_economics.py)

- KMeans, agglomerative, spectral clustering, nearest neighbors, PCA
- sklearn, FRED-MD

## Applications in Supervised Learning

### DAN for text classification

[dan_classifier.py](dan_classifier.py)

- pytorch, deep averaging networks, word embeddings, spacy
- S&P Key Developments, Wharton Research Data Services

### Classification models and events text

[classification_models.py](classification_models.py)

- sklearn, naivebayes, logistic, linearsvc, mlp, decisiontree, wordcloud
- text classification, S&P Key Developments, Wharton Research Data Services

### Binary classification of events text

[keydev_classifier.py](keydev_classifier.py)

- text classification, logistic regression, stochastic gradient descent
- precision, recall, ROC curve, sensitivity, specificity
- S&P Key Developments, Wharton Research Data Services

### Sentiment analysis of Edgar company filings

[sec_sentiment.py](sec_sentiment.py)

- Cohen, Malloy and Nguyen (2020), Loughran and McDonald (2011), and others
- sklearn, nltk, SEC Edgar, Wharton Research Data Services

### Approximate factor models, VAR and TCN from FRED-MD

[approximate_factors.py](approximate_factors.py)

- PCA, EM, vector autoregression, temporal convolutional networks
- Bai and Ng (2002), McCracken and Ng (2016), and others

### Supervised learning models for regression

[regression_models.py](regression_models.py)

- subset selection, partial least squares, ridge, lasso regression
- cross validation, feature importances, dimension reduction
- gradient boosting, random boosting, ensembles
- sklearn, statsmodels, St Louis Fed FRED, GDP

## Applications in Linear Regression

### Forecasting and econometrics

[econometric_forecast.py](econometric_forecast.py)

- seasonality, spectral density, unit root, stationarity
- autocorrelation functions, AR, MA, SARIMAX
- scipy, statsmodels, seaborn, St Louis Fed FRED

### Linear regression diagonostics and residual plots

[linear_diagnostics.py](linear_diagnostics.py)

- linear regression assumptions, residual plots, robust standard errors
- outliers, leverage, multicollinearity
- statsmodels, St Louis Fed FRED

### Economic time series and releases

[revisions_vintage.py](revisions_vintage.py)

- : revisions and vintages, St Louis Fed FRED/ALFRED

## Applications in Risk Modelling

### Market microstructure

[market_microstructure.py](market_microstructure.py)

- intraday liquidity, variance ratio, effective spreads, tick sign test
- tick data, NYSE Daily TAQ 

### Factor and empirical covariance matrix from NYSE TAQ

[taq_covariance.py](taq_covariance.py)

- covariance matrix shrinkage, PCA, minimum variance portfolios
- high frequency tick data, NYSE Daily TAQ

### Conditional volatility models

[conditional_volatility.py](conditional_volatility.py)

- Value at Risk, GARCH, EWMA, Scholes-Williams Beta
- VIX, Bitcoin, St Louis Fed FRED

### Bond market index components and interest rate indicators

[bond_returns.py](bond_returns.py)

- PCA, St Louis Fed FRED

### Term structure of interest rates

[term_structure.py](term_structure.py)

- bootstrap, splines, yield curve, duration
- Liu and Wu (2020), St Louis Fed FRED

## Applications in Network Science

### Social network analysis of BEA industries

[social_iouse.py](social_iouse.py)

- Input-Output Use Tables, Social Relations Regression Model
- igraph, rpy2, Bureau of Economic Analysis

### Graph centrality and BEA input-output use tables

[bea_centrality.py](bea_centrality.py)

- igraph, network, centrality, BEA Input-Output Use Table
- Choi and Foerster (2017), Bureau of Economic Analysis, and others

### Industry sectoring

[industry_community.py](industry_community.py)

- igraph, community detection, modularity
- Text-based Network Industry Classification (Hoberg and Phillips, 2016)

### Principal customers network

[customer_ego.py](customer_ego.py)

- igraph, ego graph, betweenness centrality
- S&P Compustat, Wharton Research Data Services

## Applications in Quantitative Finance

### Event study abnormal returns

[event_study.py](event_study.py)

- CAR, BHAR, post-event drift, order statistics, Bonferroni adjustment
- S&P Key Developments, Wharton Research Data Services

### Weekly reversals strategy

[weekly_reversal.py](weekly_reversal.py)

- information coefficient, slippage, cross-sectional dispersion
- structural breaks, unknown changepoint
- rpy2, CRSP, Wharton Research Data Services

### Factor investing

[quant_factors.py](quant_factors.py)

- return predicting signals, portfolios sorts, backtests
- CRSP, Compustat, IBES, Wharton Research Data Services
- Green, Hand and Zhang (2013) and others

### Risk premiums from Fama-Macbeth cross-sectional regressions

[fama_macbeth.py](fama_macbeth.py)

- pandas datareader, Fama French data library

### Fama-French and momentum research factors

[fama_french.py](fama_french.py)

- CRSP, Compustat, Wharton Research Data Services
