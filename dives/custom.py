"""
the dives.custom module defines some common function call arguments and useful wrappers
"""
# The MIT License
#
# Copyright (c) 2020 Terence Lim (https://terence-lim.github.io/)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation he rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

try:
    import secret
    verbose = secret.value('verbose')
except:
    verbose = 0

import re
import numpy as np, scipy as sp
import keras, xgboost

import sklearn
import sklearn.model_selection, sklearn.preprocessing, sklearn.metrics
import sklearn.linear_model, sklearn.discriminant_analysis, sklearn.cross_decomposition
import sklearn.naive_bayes, sklearn.svm, sklearn.tree, sklearn.ensemble, sklearn.neighbors

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('punkt', quiet = True)
nltk.download('wordnet', quiet = True)

def tuneClassifier(clf, n_iter=None, random_state=None, verbose=verbose, **kwargs):
    """wrapper to append cross-validation parameter search to pipeline with a classifier
    
    Parameters
    ----------
    clf : classifier instance
        sklearn object
    n_iter : int, optional (default is None)
        number of random parameter draws to evaluate in RandomizedSearchCV. None to GridSearchCV
    random_state: numeric, optional (default is None)
        initialize random seed
    verbose : int, optional (default is preset verbose value)
        to display messages
    **kwargs : search parameters
        passed on as param_grid (to GridSearchCV) or param_distributtions (to RandomizedSearchCV)

    Notes
    -----
    StratifiedKFold(n_splits = 3, shuffle=True) is used to divide training sample for CV search
    """
    cv = sklearn.model_selection.StratifiedKFold(3, shuffle=True, random_state=random_state)
    if n_iter is None:
        return sklearn.model_selection.GridSearchCV(
            clf, cv=cv, refit=True, verbose=verbose, param_grid = kwargs)
    else:
        return sklearn.model_selection.RandomizedSearchCV(
            clf, cv=cv, refit=True, verbose=verbose, random_state=random_state,
            param_distributions = kwargs)
    
def tuneRegressor(clf, n_iter=None, random_state=None, verbose=verbose, **kwargs):
    """wrapper to append cross-validation parameter search to pipeline with a regressor
    
    Parameters
    ----------
    clf : regressor instance
        sklearn object
    n_iter : int, optional (default is None)
        number of random parameter draws to evaluate in RandomizedSearchCV. None to GridSearchCV
    random_state: numeric, optional (default is None)
        initialize random seed
    verbose : int, optional (default is preset verbose value)
        to display messages
    **kwargs : search parameters
        passed on as param_grid (to GridSearchCV) or param_distributtions (to RandomizedSearchCV)

    Notes
    -----
    KFold(n_splits = 3, shuffle=True) is used to divide training sample for CV search
    """
    cv = sklearn.model_selection.KFold(3, shuffle=True, random_state=random_state)
    if n_iter is None:
        return sklearn.model_selection.GridSearchCV(
            clf, cv=cv, refit=True, verbose=verbose, param_grid=kwargs)
    else:
        return sklearn.model_selection.RandomizedSearchCV(
            clf, cv=cv, refit=True, verbose=verbose, param_distributions=kwargs)

class kerasClassifier(keras.wrappers.scikit_learn.KerasClassifier):
    """subclass extends KerasClassifier to build "on-the-fly" a dense multi-layer Sequential model.

    Parameters
    ----------
    build_fn : None
        value of None passed to build_fn => __call__ method, see keras implementation of
        wrapper for scikit-learn API https://keras.io/scikit-learn-api/
    layers : list of int, optional (default [100])
        list of sizes of each layer
    activation : string, optional (default 'relu')
        activation function. e.g. {'tanh','relu','sigmoid'}
    metrics : list of string, optional (default ['accuracy'])
        evaluation metric, e.g. {'accuracy', 'mse'}
    loss : string, optional (default 'categorical_crossentropy')
        loss function for training, 
        e.g. {'mean_squared_error','squared_hinge','huber_loss','sparse_categorical_crossentropy'}
,   **kwargs: dict of parameters
        other parameters to pass to fit()

    Notes
    -----
    1. endogeneously sets sizes of input and output dimensions, 
       by delaying compiling model till data seen by fit()
    2. internalizes transform and inverse_transform of labels in fit() and predict(), but
       all subsample labels cannot comprise unseen classes (e.g. stratify from full sample)
    3. allows number and sizes of layers to be parameters, hence can auto-tune by cross-validation

    Other options for keras:
    optimizer = {'adam', 'sgd'}

    See example of base KerasClassifier: https://keras.io/examples/mnist_sklearn_wrapper/ 

    Examples
    --------
    clf = kerasClassifier(epochs=2, batch_size=512, verbose=verbose, layers=[30], activation='tanh')
    clf.fit(X_all, y_all)
    clf.score(X_all, y_all)
    y_pred = clf.predict(X_all)
    clf.get_params()
    clf.params

    score = sklearn.model_selection.cross_val_score(
        kerasClassifier(verbose=verbose), X_all, y_all,
        fit_params={'layers' : [40], 'batch_size': 128, 'verbose': 3, 'epochs':3, 'activation':'tanh'},
        cv=sklearn.model_selection.StratifiedKFold(3)).mean()
    
    results = sklearn.model_selection.cross_validate(
        kerasClassifier(verbose=verbose), X_all, y_all,
        fit_params={'layers': [300, 100], 'verbose': 3, 'epochs':3, 'batch_size':256},
        cv = sklearn.model_selection.StratifiedKFold(2),
        return_train_score=True, return_estimator=True, verbose=verbose)
    """
    def __init__(self, build_fn = None, layers = [100], activation = 'relu', metrics = ['accuracy'],
                 loss = 'categorical_crossentropy', **kwargs):
        super().__init__(build_fn=build_fn, **kwargs)
        self.params = {'layers': layers,          # user parameters, to be used to build model
                       'activation': activation,
                       'loss': loss,
                       'metrics': metrics}
        # print('init', self.get_params())
        
    def __call__(self, **kwargs):
        """treated as the default build_fn when None, to compile and return the model"""
        # print('__call__', self.get_params(), kwargs)
        model = keras.Sequential()
        model.add(keras.layers.Dense(               # dense sequential model with user parameters:
            self.params['layers'][0],                   # size of first layer per parameters
            input_dim = self.n_features,                # input dim per data seen in fit()
            activation = self.params['activation']))    # activation function per parameters
        for layer in self.params['layers'][1:]:         # size of other layers per parameters
            model.add(keras.layers.Dense(layer, activation = self.params['activation']))
        model.add(keras.layers.Dense(self.n_classes,    # output dim per data seen in fit()
                                     activation = 'softmax'))
        model.compile(optimizer = 'adam',             
                      loss = self.params['loss'],       # loss function per parameters
                      metrics = self.params['metrics'], # metrics per parameters
                      **kwargs)
        return model   # as required by KerasClassifier: this method to compile and return the model

    def fit(self, x, y, layers=None, activation=None, loss=None, metrics=None, **kwargs):
        """override fit method, to implement encoder and changeable model structure

        Notes
        -----
        args recognized by super().fit(): batch_size=32, epochs=1, verbose = {0,1,2}, 
        args not recognized by super().fit(): layers, activation, loss, metrics
        hence must explicitly mention in self.fit() args list (and stored in self.params)
          but not passed on to super().fit()
        """
        self.encoder = sklearn.preprocessing.LabelEncoder().fit(y)  # internalize encoding of class labels
        self.n_classes = len(self.encoder.classes_)                 # infer for output dimension
        self.n_features = x.shape[1]                                # infer for input dimension
        # print('fit', layers, self.params['layers'], self.get_params(), kwargs)
        # print(self.encoder.classes_)
        for p in self.params:  # update user parameters via self.fit(), but not recognized by super.fit()
            if eval(p):
                self.params[p] = eval(p)
        return super().fit(x, self.encoder.transform(y), **kwargs)  # transform class labels before fit()

    def score(self, x, y, **kwargs):
        """override score(), to transform encoded class labels first"""
        return super().score(x, self.encoder.transform(y), **kwargs)

    def predict(self, x, **kwargs):
        """override predict(), to inverse_transform encoded class labels first"""
        return self.encoder.inverse_transform(super().predict(x, **kwargs))

class kerasClassifierCV(kerasClassifier):
    """subclass extends kerasClassifier to allow (simple) cross-validation auto-tuning

    Parameters
    ----------
    layers : list of int lists, optional (default is [[50], [100, 25]])
        layers to grid search over

    Returns
    -------
    self.best_estimator_ : best kerasClassifier model
    self.best_score_  : validation score of best model
    self.best_params_ : parameters of best model
    self.model : best Keras model

    Notes
    -----
    this is a kludge, as KerasClassifier conflicts with sklearn.model_selection.GridSearchCV,
    when build_fn is None (i.e. class approach), possible due to sklearn clone() issue (?), 
    see https://github.com/keras-team/keras/issues/13586
    
    Examples
    --------
    clf = kerasClassifierCV() 
    clf.fit(X_all, y_all)
    clf.best_params_                  # {'layers': [100, 25], 'epochs': 2, 'batch_size': 256}
    clf.best_estimator_.params        # {'layers': [100, 25], 'activation': 'relu',  
                                         'loss': 'categorical_crossentropy', 'metrics': ['accuracy']}
    clf.best_estimator_.get_params()  # {'epochs': 2, 'batch_size': 256, 'build_fn': None}
    """
    def __init__(self, layers=[[50], [100, 25]], **kwargs):
        self.params = {'layers' : layers}
    
    def fit(self, x, y, **kwargs):
        """override fit(), to choose best parameters from stratified KFold cross-validation"""
        self.best_score_ = 0
        for layers in self.params['layers']:  # can also vary epochs, batch_size, etc ,..
            params = {'layers' : layers, 'epochs':2, 'batch_size':256}
            results = sklearn.model_selection.cross_validate(
                kerasClassifier(verbose=verbose),
                x, y,
                fit_params = params,
                verbose=verbose,
                cv = sklearn.model_selection.StratifiedKFold(2))
            score = np.mean(results['test_score'])
            if score >= self.best_score_:
                self.best_params_ = params
                self.best_score_ = score
        self.best_estimator_ = kerasClassifier(**self.best_params_)
        self.best_estimator_.fit(x, y, **self.best_params_)
        self.model = self.best_estimator_.model
        return self
    
    def score(self, x, y, **kwargs):
        """override score() to return score of best_estimator_"""
        return self.best_estimator_.score(x, y, **kwargs)
    
    def predict(self, x, **kwargs):
        """override score() to predict from best_estimator_"""
        return self.best_estimator_.predict(x, **kwargs)

CustomClassifier = {   # Some pre-canned classifiers, in a dict()
    'multinomialNB': tuneClassifier(sklearn.naive_bayes.MultinomialNB(), alpha=[1]),
    
    'logistic': tuneClassifier(
        sklearn.linear_model.LogisticRegression(multi_class='auto', max_iter=300),
        random_state=42, **{'C': [1]}),
    
    'logisticCV': tuneClassifier(
        sklearn.linear_model.LogisticRegression(multi_class='auto', max_iter=300),
        random_state=42, **{'C': [0.001, 0.1, 10, 1000]}),

    'svclinearCV': tuneClassifier(
        sklearn.svm.LinearSVC(),random_state=42, **{'C': np.logspace(-4.5, -2, 10)}),

    'kerasClassifierCV': kerasClassifierCV(),
    
    'ldaSVD': tuneClassifier(
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis(), solver=['svd']),
    
    'ldaLSQR': tuneClassifier(
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
        solver=['lsqr'], shrinkage=['auto']),
    
    'svcrbf': tuneClassifier(
        sklearn.svm.SVC(decision_function_shape='ovo'), kernel=['rbf']),
    
    'svclinear': tuneClassifier(sklearn.svm.LinearSVC(), C=[1]),

    'decisionTreeClassifier':tuneClassifier(sklearn.tree.DecisionTreeClassifier(), ccp_alpha=[0.015]),
    
    'randomForestClassifier': tuneClassifier(
        sklearn.ensemble.RandomForestClassifier(random_state=42), n_estimators=[50], max_depth=[3]),
    
    'randomForestClassifierCV': tuneClassifier(
        sklearn.ensemble.RandomForestClassifier(), random_state=42, 
        **{'max_depth': [3,5,7],
           'max_features': ['sqrt','log2',0.2],
           'n_estimators': [20,50,100]}),
    
    'adaBoostClassifierCV': tuneClassifier(
        sklearn.ensemble.AdaBoostClassifier(), random_state=42,
        **{'n_estimators': [25, 50, 100],
           'learning_rate': [0.1, 1, 10]}),
    
    'xgBoostClassifierCV': tuneClassifier(
        xgboost.XGBClassifier(verbose=verbose),
        n_iter=20,
        random_state=42,
        **{"colsample_bytree": sp.stats.uniform(0.7, 0.3),
           "gamma": sp.stats.uniform(0, 0.5),
           "learning_rate": sp.stats.uniform(0.03, 0.3), # default 0.1
           "max_depth": sp.stats.randint(2, 6),          # default 3
           "n_estimators": sp.stats.randint(100, 150),   # default 100
           "subsample": sp.stats.uniform(0.6, 0.4)}),
    }


CustomRegressor = {   # some pre-canned regressor, in a dict()
    'linearRegression': sklearn.linear_model.LinearRegression(),
    
    'lassoCV': sklearn.linear_model.LassoCV(cv=3, max_iter=10000, verbose=verbose),
    
    'ridgeCV' : sklearn.linear_model.RidgeCV(cv=3),
    
    'elasticNetCV': sklearn.linear_model.ElasticNetCV(cv=3, max_iter=-1, verbose=verbose),
    
    'plsRegression': tuneRegressor(
        sklearn.cross_decomposition.PLSRegression(), **{'n_components': np.arange(2,10)}),
    
    'svmrbfCV': tuneRegressor(
        sklearn.svm.SVR(kernel='rbf'), **{'gamma': [1e-3,1e-4], 'C': [1,10,100,1000]}),
    
    'svmlinearCV': tuneRegressor(sklearn.svm.SVR(kernel='linear'), **{'C': [1,10,100]}),
    
    'nearestNeighborRegression': sklearn.neighbors.KNeighborsRegressor(
        n_neighbors=5, weights='uniform'),
    
    'nearestNeighborRegressionCV' : tuneRegressor(
        sklearn.neighbors.KNeighborsRegressor(),
        **{'n_neighbors': np.arange(3,8),
           'weights': ['uniform']}),
    
    'decisionTreeRegression' : sklearn.tree.DecisionTreeRegressor(),
    
    'decisionTreeRegressionCV' : tuneRegressor(
        sklearn.tree.DecisionTreeRegressor(),
        **{'max_features': ['sqrt','log2', 0.2],
           'min_samples_leaf' : [2,5]}),
    
    'randomForestRegression': sklearn.ensemble.RandomForestRegressor(verbose=verbose),
    
    'randomForestRegressionCV': tuneRegressor(
        sklearn.ensemble.RandomForestRegressor(verbose=verbose),
        **{'max_features': ['sqrt','log2', 0.2],
           'min_samples_leaf' : [2,5]}),
    
    'adaboost': sklearn.ensemble.AdaBoostRegressor(random_state=42),
    
    'adaboostCV': tuneRegressor(
        sklearn.ensemble.AdaBoostRegressor(), random_state=42,
        **{'n_estimators': [25, 50, 100],
           'learning_rate': [0.1, 1, 10]}),
    
    'xgboost': xgboost.XGBRegressor(
        objective='reg:squarederror',random_state=42, verbose=verbose),
#        tree_method='gpu_hist', gpu_id=0, max_depth=6),
    
    'xgboostCV': tuneRegressor(   # see https://xgboost.readthedocs.io/en/latest/gpu/index.html
        xgboost.XGBRegressor(objective='reg:squarederror',verbose=verbose),
#                             tree_method='gpu_hist', gpu_id=0, max_depth=6),    # use GPU with scikit-Learn
        n_iter=50, random_state=42, verbose=verbose,
        **{"colsample_bytree": sp.stats.uniform(0.3, 1),
           "gamma": sp.stats.uniform(0, 0.5),
           "learning_rate": sp.stats.uniform(0.03, 0.3), # default 0.1
           "max_depth": sp.stats.randint(2, 6), # default 3
           "n_estimators": sp.stats.randint(100, 150), # default 100
           "subsample": sp.stats.uniform(0.6, 0.4)}),
}

# could not get nvndia gpu for xgboost -- memory full between runs, see notes:
"""xgboost notes

update:
https://xgboost.readthedocs.io/en/latest/build.html
https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/list.html
pip install https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/xgboost-*-py2.py3-none-manylinux1_x86_64.whl 

gpu:
https://xgboost.readthedocs.io/en/latest/gpu/index.html
xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0)

out of memory:
https://github.com/dmlc/xgboost/issues/4286
https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/86848
    # after training, release GPU memory.
    model.save_model('tmp/xgb.model')
    model.__del__()
    model = xgb.Booster()
    model.load_model('tmp/xgb.model')

https://stackoverflow.com/questions/56298728/how-do-i-free-all-memory-on-gpu-in-xgboost
from multiprocessing import Process
def fitting(args):
    clf = xgb.XGBClassifier(tree_method = 'gpu_hist',gpu_id = 0,n_gpus = 4, random_state = 55,n_jobs = -1)
    clf.set_params(**params)
    clf.fit(X_train, y_train, **fit_params)
    #save the model here on the disk
fitting_process = Process(target=fitting, args=(args))
fitting process.start()
fitting_process.join()
# load the model from the disk here
"""

def get_importances(model):
    """feature_importances_"""
    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_
    if hasattr(model, 'get_booster'):
        a = model.get_booster().get_score(importance_type='weight')
        return np.array([k[1] for k in sorted(a.items(), key=lambda v: int(v[0][1:]))]).flatten()
    elif hasattr(model, 'feature_importances_'):
        return np.array(model.feature_importances_).flatten()
    elif hasattr(model, 'coef_'):
        return abs(np.array(model.coef_).flatten())
    else:
        return []


# Notes for custom regressor/classifier:
#   sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None,
#   .best_estimator_ .best_params .best_score
#   sklearn.model_selection.RandomizedSearchCV(
#     param_distributions=params, n_iter=25, return_train_score=True)
#   'rbfSVCCV' :        sklearn.model_selection.GridSearchCV(
#      sklearn.svm.SVC(decision_function_shape='ovo', kernel='rbf', gamma='scale', verbose=verbose),
#      cv=3, param_grid = {'C' : np.logspace(-2, 10, 5)}),   # 'gamma':np.logspace(-9, 3, 5)


def CustomScorer(name=None, make_scorer=True):
    """wrapper for accuracy scorer functions
    
    Parameters
    ----------
    name : string, optional (default is None)
      name of scorer.  if name is None, then return list of scorer names
    make_scorer: boolean, optional (default is True)
      whether to transform with metrics.make_scorer(), which is required for cross_val_score()

    Notes
    -----
    - when passing (y_true, y_pred) to scorers, that order is crucial as metrics may not be symmetric
    - incorporate spearman and pearson scorer functions to be robust to nan's.
    """
    scorers = {
        'accuracy' : sklearn.metrics.accuracy_score,
        'f1' : lambda y, pred: sklearn.metrics.f1_score(y, pred, average='weighted'),
        'precision' : lambda y, pred: sklearn.metrics.precision_score(y, pred, average='weighted'),
        'recall' : lambda y, pred: sklearn.metrics.recall_score(y, pred, average='weighted'),
        'explained_variance' : sklearn.metrics.explained_variance_score,
        'neg_median_absolute_error' : lambda y, pred: -sklearn.metrics.median_absolute_error(y, pred),
        'r2' : sklearn.metrics.r2_score,
        'spearman' : lambda y, pred: sp.stats.spearmanr(y, pred, nan_policy='omit')[0],
        'pearson' : lambda y, pred: sp.stats.pearsonr(np.array(y).reshape((-1,)),
                                                      np.array(pred).reshape((-1,)))[0],
    }
    if name is None:
        return list(scorers.keys())
    scorer = scorers[name]
    return sklearn.metrics.make_scorer(scorer) if make_scorer else scorer

class CustomTokenizer:
    """class to tokenize as required by sklearn vectorizers

    Examples
    --------
    vect = CountVectorizer(tokenizer=CustomTokenizer())
    """
    _rex = re.compile(r"\b[^\d\W][^\d\W][^\d\W]+\b")  # use this regexp to tokenize
    
    def tokenize(self, doc):
        return RegexpTokenizer(self._rex).tokenize(doc)
    def match(self, word):
        return re.match(self._rex, word)
    def __init__(self):
        pass
    def __call__(self, doc):
        return self.tokenize(doc)

class CustomLemmaTokenizer(CustomTokenizer):
    """class to lemmatize and tokenize as required by sklearn vectorizers

    Examples
    --------
    vect = CountVectorizer(tokenizer=CustomLemmaTokenizer())
    """
    def __init__(self, lemmatize=WordNetLemmatizer()):
        self.lemmatize = lemmatize
    def __call__(self, doc):
        if self.lemmatize is None:
            return self.tokenize(doc)
        else:
            return [self.lemmatize(t) for t in self.tokenize(doc)]
        
