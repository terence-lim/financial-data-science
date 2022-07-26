# Financial Data Science: Exercises in Python

These code examples explore data science and machine learning methods
on large or textual financial data sets (note that the accompanying Jupyter notebooks
in the
[https://github.com/terence-lim/data-science-notebooks](https://github.com/terence-lim/data-science-notebooks) repo
reflect an older version, and will be updated very soon)

[https://github.com/terence-lim/financial-data-science](https://github.com/terence-lim/financial-data-science)

by: [Terence Lim](https://www.linkedin.com/in/terencelim)

&nbsp;


## Applications in Unsupervised Learning


### Topic Models from FOMC meeting minutes

[fomc_topics.py](fomc_topics.py)

- Topic Models: FOMC minutes text
- Matrix Decomposition: NMF, LSA, LDA, PLSI

### Hidden State Models and Economic Time Series

[economic_states.py](economic_states.py)

- Recurrent Neural Networks: LSTM and Elman SRN
- Linear Dynamic Factor Models
- Hidden Markov Models and Gaussian Mixture Model


## Applications in Supervised Learning

### Classification of events text

[keydev_classifier.py](keydev_classifier.py)

- Text classification: S&P Key Developments events
- Logistic regression: Generalized Linear Models, stochastic gradient descent
- nltk: tokenizer, lemmatizer, stemmer
- sklearn metrics: accuracy, precision, recall, confusion_matrix, auc, roc_curve

### Classification models and events text

[classification_models.py](classification_models.py)

- Supervised learning models for classification

  - Text classification, S&P Key Developments
  - gensim: preprocessing, phrases
  - sklearn: naivebayes, logistic, linearsvc, decisiontree

### Deep Averaging Networks for text classification

[dan_classifier.py](dan_classifier.py)

- Feedforward Neural Networks: torch, deep averaging networks
- Word vectors: spacy, GloVe, relativize, frozen, fine-tuning


### Sentiment Analysis of 10K MD&A in Edgar Company Filings

[mda_sentiment.py](mda_sentiment.py)

- Sentiment Analysis: Loughran and McDonald (2011) sentiment word list
- 10-K Company Filings: SEC Edgar, Cohen, Malloy and Nguyen (2020), 


### Approximate factor models, VAR and TCN from FRED-MD

[approximate_factors.py](approximate_factors.py)

- Approximate Factor Models, VAR and TCN

  - PCA, EM
  - Approximate factors and selection: Bai and Ng (2002), McCracken and Ng (2016)
  - vector autoregression, temporal convolutional networks

### Supervised Learning Regression Models

[regression_models.py](regression_models.py)

- subset selection, partial least squares, ridge, gradient boost, random forest
- cross validation, feature importances, dimension reduction
- sklearn, statsmodels

## Applications in Linear Regression

### Econometrics and Forecasting

[econometric_forecast.py](econometric_forecast.py)

- Trends: seasonality
- Autocorrelation Function: AR, MA, SARIMAX
- Unit root: integration order
- Forecasting: single-step, multi-step
- Granger Casuality
- Vector Auto-Regression: impulse response function

### Linear Regression Diagonostics and Residual Plots

[linear_diagnostics.py](linear_diagnostics.py)

- Linear regression diagnostics: HAC robust standard errors
- Outliers: leverage, influential points, residual plots
- Multicollinearity: variance inflation factor
- Interactions and Polynomial Regression

### Economic Time Series

[revisions_vintage.py](revisions_vintage.py)

- St Louis Fed FRED: popular series, api
- ALFRED: archival, releases, vintages, revisions
- FRED-MD: release dates


## Applications in Risk Modelling

### Market Microstructure

[market_microstructure.py](market_microstructure.py)

- Tick data: NYSE Daily TAQ 
- Spreads: quoted, effective, price impact, realized
- Volatility: variance ratio, Parkinsons, Klass-Garman

### Covariance Matrix and Risk Decomposition

[covariance_matrix.py](covariance_matrix.py)

- Covariance Matrix: Principal Components, Shrinkage, EWMA
- Risk Decomposition, Black-Litterma

### Conditional Volatility

[conditional_volatility.py](conditional_volatility.py)

- Value at Risk, Expected Shortfall
- GARCH, EWMA
- VIX, Bitcoin

### Bond market index components and interest rate indicators

[bond_returns.py](bond_returns.py)

- Bond Returns and Interest Rates

- Principal Components Analysis: bond index returns and interest rates


### Term Structure of Interest Rates

[term_structure.py](term_structure.py)

- yield curve, duration, forward rates, spot rates, yield-to-maturity
- bootstrap, splines
- reconstructed yield curve (Liu and Wu, 2020), St Louis Fed FRED


## Applications in Network Science


### Graph centrality and BEA input-output use tables

[bea_centrality.py](bea_centrality.py)

- Graph Centrality and BEA Input-Output Use Tables

  - Centrality: eigenvector, hub, authority, pagerank,
  - BEA: Input-Output Use Table, Choi and Foerster (2017)


### Industry Sectoring and Community Detection

[industry_community.py](industry_community.py)

- Community Detection
- Text-based Network Industry Classification (Hoberg and Phillips, 2016)

### Link Prediction

[link_prediction.py](link_prediction.py)

- Link prediction: resource_allocation, jaccard coefficient, 
  adamic_adar, preferential_attachment
- Accuracy: precision, recall, ROC curve, AUC, confusion matrix, 
- Text-based Network Industry Classification (Hoberg and Phillips, 2016)

### Principal Customers Network

[customer_ego.py](customer_ego.py)

- Graphs: ego network, induced subgraph
- Supply chain: principal customers

## Applications in Quantitative Finance

### Event Study Abnormal Returns

[event_study.py](event_study.py)

- S&P/Capital IQ Key Developments
- event study: CAR, BHAR, post-announcement drift (Kolari et al 2010 and others)
- multiple testing: Holm FWER, Benjmain-Hochberg FDR, Bonferroni p-values

### Weekly Reversals Strategy

[weekly_reversal.py](weekly_reversal.py)

- Weekly reversal contrarian strategy (Lo and Mackinlay 1990, and others)
- Structural change with unknown breakpoint
- Implementation slippage


### Survivorship-bias and low-price stocks strategy

[lowprice_survivors.py](lowprice_survivors.py)

- Low-price portfolio spread returns
- Survivorship-bias
- Autocorrelation-consistent standard errors: Newey-West

### Stock Prices and Adjustments

[stock_prices.py](stock_prices.py)

- Total stock returns: splits and dividend adjustment factors



### Factor Investing

[quant_factors.py](quant_factors.py)

- return predicting signals (Green, Hand and Zhang, 2013, and others)
- CRSP, Compustat, IBES

### Risk Premiums from Fama-Macbeth Cross-sectional Regression

[fama_macbeth.py](fama_macbeth.py)

- Fama-Macbeth cross-sectional regression: risk premiums
- CRSP, Compustat, 
- Ken French Data Library: Fama-French test assets
### Fama-French Research Factors

[fama_french.py](fama_french.py)

- Fama-French monthly research factors: HML, SMB, Mom, STRev
- Portfolio sorts
- CRSP, Compustat, Ken French Data Library
