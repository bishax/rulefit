import pdb
import warnings

from collections import Counter
import numpy as np
import re
import itertools as it
from scipy.sparse import issparse, hstack
from pandas import DataFrame

from sklearn.utils import check_random_state, check_array
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from custom_transformers import LabelOneHotEncoder
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression


class FriedScaler(BaseEstimator, TransformerMixin):
    """FriedScaler class: Scale linear features within rule ensemble
    
    Scales linear features within a rule ensemble
    to have the same weighting as a rule according to
    Friedman et al. 2005 Section 5.
    
    Each column, $x_i$ is winsorised at `quantile` -> $x_i'$, then 
    standardised by multiplying by $0.4 \text{std}(x_i')$
    
    Attributes
    ----------
    
    scale: numpy.ndarray 
        scale factor for each variable
        
    lower: numpy.ndarray
        lower winsorisation threshold
        
    upper: numpy.ndarray
        upper winsorisation threshold
    
    """
    
    def __init__(self, quantile=0.0):
        """
        Parameters
        ----------
        
        quantile: float
            float in [0, 0.5) signifying the quantiles at which to winsorise
            (`quantile` and `1-quantile`)
            WARNING: If data has small variance then this may need to be 
            very small to avoid blowing up of scale factors
        """
        self.quantile = quantile
        
    def fit(self, X, y=None):
        """ Fit scaler and return self
        
        Winsorise `X` at `quantile` and `1-quantile`.
        Scale each variable (as long as they aren't binary in
        which case they are already rules).
        
        Parameters
        ----------
        
        X: numpy.ndarray
            Co-variates
            
        y: dummy arguement, optional
        """
        self.fit_transform(X, y)
        return self
    
    def fit_transform(self, X, y=None):
        """ Fit scaler and transform input data
        
        Winsorise `X` at `quantile` and `1-quantile`.
        Scale each variable (as long as they aren't binary in
        which case they are already rules).
        
        Parameters
        ----------
        
        X: numpy.ndarray
            Co-variates
            
        y: dummy arguement, optional
        """
        self.scale = np.ones(X.shape[1])
        self.lower = np.percentile(X, self.quantile*100, axis=0)
        self.upper = np.percentile(X, (1-self.quantile)*100, axis=0)
        
        # Winsorize at `self.quantile`
        winX = X.copy()
        is_lower = (winX < self.lower)
        is_higher = (winX > self.upper)
        for col in range(X.shape[1]):
            winX[is_lower[:, col], col] = self.lower[col]
            winX[is_higher[:, col], col] = self.upper[col]
            
            num_uniq = np.unique(X[:, col]).size
            if num_uniq > 2:  # Don't scale binary vars
                self.scale[col] = 0.4/(1e-12 + np.std(winX[:, col]))
        
        large_scale = np.where(self.scale > 1e3)[0]
        if large_scale.size > 0:
            warnings.warn('Scales of {} are larger than 1e3!'.format(large_scale))
        
        return winX*self.scale
        
    def transform(self, X):
        """ Transform input data
        
        Winsorise `X` at pre-fitted `quantile` and `1-quantile`.
        Scale each variable (as long as they aren't binary in
        which case they are already rules) accorded to the already
        fitted scale factors.
        
        Parameters
        ----------
        
        X: numpy.ndarray
            Co-variates
            
        y: dummy arguement, optional
        """
        winX = X.copy()
        is_lower = (winX <= self.lower)
        is_higher = (winX >= self.upper)
        for col in range(X.shape[1]):
            winX[is_lower[:, col], col] = self.lower[col]
            winX[is_higher[:, col], col] = self.upper[col]
        return winX*self.scale
    
class RuleFitClassifier(BaseEstimator, ClassifierMixin):
    """Rule-Fit for binary classification
    
    Generate an ensemble of rules using XGBoost or a sklearn
    tree ensemble method, and use these (optionally with linear
    features) in a L1 (or other penalised) Logistic Regression to 
    build a classifier.
    
    Attributes
    ----------
    
    LR: sklearn.linear_model.LogisticRegression
        Regularised linear regression on ensemble of rules
    
    feature_mask_: np.ndarray
        Array of non-zero feature values
    
    coef_: np.ndarray
        LogisticRegression (`LR`) co-efficients for features in `feature_mask_`
    
    intercept_: np.ndarray
        LogisticRegression (`LR`) intercept
    
    features: np.ndarray of str
        Input feature names
        
    features_: np.ndarray of str
        Output feature names of rule ensembles (and linear features if `linear_features=True`)
        
    """
    
    def __init__(self, 
                 base_estimator=XGBClassifier(),
                 linear_features=True,
                 linear_feature_quantile=0.025,
                 C=1e-1,
                 penalty='l1',
                 n_estimators=10,
                 max_depth=5,
                 rand_tree_size=False,
                 sparse_output=True,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ext_scaler=RobustScaler()):
        """
        Parameters
        ----------
        
        base_estimator: sklearn estimator, default: xgboost.sklearn.XGBClassifier
            Estimator to generate rule ensemble with
            
        linear_features: bool, default: True
            If `True`: Use linear features as well as rules
            
        linear_feature_quantile: float, default: 0.025
            float in [0, 0.5) signifying the quantiles at which to winsorise
            (`quantile` and `1-quantile`).
            WARNING: If data has small variance then this may need to be 
            very small to avoid blowing up of scale factors
            
        C: float, default: 0.1
            Inverse of regularization strength; must be a positive float.
            Like in support vector machines, smaller values specify stronger
            regularization.
        
        
        penalty: {'l1', 'l2'}, default: 'l1'
            Norm used in the regularisation for LogisticRegression
        
        n_estimators: int, default: 10
            Number of trees within `base_estimator`
        
        max_depth: int, optional
            Maximum tree depth of `base_estimator`
        
        rand_tree_size: bool, optional
            NOT YET IMPLEMENTED!
            If `True`, randomise `max_depth` to get rules of varying lengths.
            
        n_jobs: int, optional
            The number of CPUs to use. -1 means 'all CPUs'.
        
        verbose: int, optional
            Increasing verbosity with number.
        
        warm_start: int, optional
            When set to True, reuse the solution of the previous call to fit as
            initialization, otherwise, just erase the previous solution.
            
        class_weight : dict or 'balanced', default: 'balanced'
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one.

            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``.

        ext_scaler: sklearn Transformer, optional
            Scaling transformation to apply to linear features (before Friedman scaling)
        
        """
            
        self.base_estimator = base_estimator
        self.linear_features = linear_features
        self.linear_feature_quantile = linear_feature_quantile
        self.C = C
        self.penalty = penalty
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rand_tree_size = rand_tree_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ext_scaler = ext_scaler
        
    def fit(self, X, y, sample_weight=None):
        """ Fit model to data
        
        X: pandas.DataFrame or numpy.ndarray
            Features
            
        y: pandas.Series or numpy.ndarray
            Target
            
        Returns
        -------
        
        self
        """
            
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self
        
    def transform(self, X, y=None):
        """ Transform data into modified features
        (before being passed to penalised regression step).
        If `linear_features=True` then this will be scaled linear features
        followed by the one-hot-encoding signifying which rules are "on".
        Otherwise this is just the one-hot-encoding signifying which rules are "on".
        
        X: pandas.DataFrame or numpy.ndarray
            Features
            
        y: dummy, optional
            
        Returns
        -------
        
        sparse array
        """
        if isinstance(X, DataFrame):
            is_df = True  # Serves no purpose 
            
        X = check_array(X)  # Validate input data
            
        X = self.ext_scaler.transform(X)  # Scale and centre features
        if self.linear_features:
            X_scale = self._scaler.transform(X)  # Scale linear features to give same a priori weight as rules
            return hstack([X_scale, self._one_hot_encoder.transform(self.base_estimator.apply(X).reshape(-1, self.n_estimators))])
        else:
            return self._one_hot_encoder.transform(self.base_estimator.apply(X).reshape(-1, self.n_estimators))

    def fit_transform(self, X, y, sample_weight=None):
        """ Fit and Transform data into modified features
        (before being passed to penalised regression step).
        If `linear_features=True` then this will be scaled linear features
        followed by the one-hot-encoding signifying which rules are "on".
        Otherwise this is just the one-hot-encoding signifying which rules are "on".
        
        Fitting process involves fitted bagged/boosted tree model to generate rules
        and then using these in a penalised logistic regression.
        
        X: pandas.DataFrame or numpy.ndarray
            Features
            
        y: pandas.Series or numpy.ndarray
            Target
            
        Returns
        -------
        
        sparse array
        """
        # Instantiate rule ensemble generator and set parameters
        if isinstance(self.base_estimator, XGBClassifier):
            self.base_estimator.set_params(n_estimators=self.n_estimators, silent=(self.verbose>0),
                                          max_depth=self.max_depth, n_jobs=self.n_jobs)
        elif isinstance(self.base_estimator, RandomForestClassifier):
            warnings.warn('This base_estimator implementation has not been tested in a while!')
            self.base_estimator.set_params(n_estimators=self.n_estimators, verbose=self.verbose,
                                          max_depth=self.max_depth, n_jobs=self.n_jobs)
        elif isinstance(self.base_estimator, GradientBoostingClassifier):
            warnings.warn('This base_estimator implementation has not been tested in a while!')
            self.base_estimator.set_params(n_estimators=self.n_estimators, verbose=self.verbose,
                                          max_depth=self.max_depth, n_jobs=self.n_jobs)
        else:
            raise NotImplementedError
            
        # Name features
        if isinstance(X, DataFrame):
            self.features = X.columns.values
        else:
            self.features = ['f'+str(i) for i in range(X.shape[1])]
            
        # Check input
        X = check_array(X)
            
        # Generate and extract rules
        if not self.rand_tree_size:
            self.base_estimator.fit(X, y, sample_weight=sample_weight)
            if isinstance(self.base_estimator, XGBClassifier):
                self._rule_dump = self.base_estimator._Booster.get_dump()
        else:
            NotImplementedError()  # TODO: work out how to incrementally train XGB
            
        if self.verbose > 0:
            print('fitting trees')
        
        # For each tree: get leaf numbers and map them to [0, num leaves]
        # before one-hot encoding them
        n_values = "auto"
        leaves_l = []
        for tree_i in self._rule_dump:
            leaves = [int(i) for i in re.findall(r'([0-9]+):leaf=', tree_i)]
            leaves_l.append(leaves)
        self._one_hot_encoder = LabelOneHotEncoder(leaves_l)
        
        if self.verbose > 0:
            print('setup encoding')
        
        # Scale and centre linear features
        X = self.ext_scaler.fit_transform(X)
        
        if self.linear_features:
            # Linear features must be scaled to have same weighting as an average rule
            self._scaler = FriedScaler(quantile=self.linear_feature_quantile)
            X_scale = self._scaler.fit_transform(X)
            X_transform = hstack([X_scale, self._one_hot_encoder.fit_transform(self.base_estimator.apply(X).reshape(-1, self.n_estimators))])
        else:
            X_transform = self._one_hot_encoder.fit_transform(self.base_estimator.apply(X).reshape(-1, self.n_estimators))
            
        if self.verbose > 0:
            print('encoded')
        
        # Fit sparse linear model to rules (and optionally linear features)
        self.LR = LogisticRegression(C=self.C, penalty=self.penalty, class_weight=self.class_weight,
                          warm_start=self.warm_start, solver='saga', verbose=self.verbose)
        self.LR.fit(X_transform, y, sample_weight=sample_weight)
        
        if self.verbose > 0:
            print('fitted')
        
        # Mask features with zero co-efficients
        # self.feature_mask_ = np.arange(self.LR.coef_.size)
        self.feature_mask_ = self.LR.coef_.nonzero()[1]
        
        self.coef_ = self.LR.coef_[0, self.feature_mask_]
        self.intercept_ = self.LR.intercept_
        self.get_feature_names()
        assert self.features_.size == self.feature_mask_.size
        return X_transform
    
    def get_feature_names(self):
        """ Get names of features in the model
        
        Returns
        -------
        
        numpy.ndarray
        """
        if self.linear_features:
            self.features_ = np.concatenate([self.features, np.array(self.extract_rules(labels=self.features))], 0)[self.feature_mask_]
        else:
            self.features_ = np.array(self.extract_rules(labels=self.features))[self.feature_mask_]
        return self.features_
    
    def predict(self, X):
        """ Output model prediction
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        Returns
        -------
        
        np.ndarray
            Bool predictions
        """
        
        return self.LR.predict(self.transform(X))
        
    def predict_proba(self, X):
        """ Output model prediction probability
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        Returns
        -------
        
        np.ndarray
            Probabilistic predictions
        """
        return self.LR.predict_proba(self.transform(X))
    
    def __extract_xgb_dt_rules__(self, dt):
        """ Extract rule set from single decision tree according
        to `XGBClassifier` format
        
        Parameters
        ----------
        
        dt: string
        
        Returns
        -------
        
        list of numpy.ndarray
            Each array is of length three. 
            First indicates feature number,
            Second indicates operator (1 if $>$ otherwise $\leq$),
            Third indicates threshold value
            
        """ 
        md = self.max_depth + 1  # upper limit of max_depth?
        rules = []
        levels = np.zeros((md, 3))  # Stores: (feature name, threshold, next node id)
        path = []

        # Extract feature numbers and thresholds for all nodes
        feat_thresh_l = re.findall(r'\[f([0-9]+)<([-]?[0-9]+\.?[0-9]*)\]', dt)

        _id = 0
        prune = -1
        for line in dt.split('\n')[:-1]:
            # Separate node id and rest of line
            _id, rest = line.split(':')

            # Count number of tabs at start of line to get level (and then remove)
            level = Counter(_id)['\t']
            _id = _id.lstrip()

            if prune > 0:
                # If we were last at a leaf, prune the path
                path = path[:-1+(level-prune)]
            # Add current node to path
            path.append(int(_id))

            if 'leaf' in rest:
                prune = level  # Store where we are so we can prune when we backtrack
                rules.append(levels[:level, (0, 2, 1)].copy())  # Add rules
                rules[-1][:, 1] = rules[-1][:, 1] == np.array(path[1:])  # Convert path to geq/leq operators
            else:
                # Extract (feature name, threshold, next node id)
                levels[level, :] = re.findall(r'\[f([0-9]+)<([-]?[0-9]+\.?[0-9]*)\].*yes=([0-9]+)', line)[0]
                # Don't prune
                prune = -1

        return rules


    def __extract_dt_rules__(self, dt):
        """ Extract rule set from single decision tree according
        to sklearn binary-tree format
        
        Parameters
        ----------
        
        dt: string
        
        Returns
        -------
        
        list of numpy.ndarray
            Each array is of length three. 
            First indicates feature number,
            Second indicates operator (1 if $>$ otherwise $\leq$),
            Third indicates threshold value
            
        """ 
        t = dt.tree_  # Get tree object
        rules = []

        stack = [(0, -1, -1)]  # (node id, parent depth, true[<=thresh]/false[>thresh] arm)
        path = [(0, -1, -1)]  # Begin path at root
        while len(stack) > 0:  # While nodes to visit is not empty
            nid, pd, op = stack.pop()  # Get next node id, path depth, operator

            if (pd > path[-1][1]):  # Going deeper
                path.append((nid, pd, op))
            elif pd == -1:  # ROOT
                pass
            else:  # Back-track
                [path.pop() for _ in range(path[-1][1]-pd+1)]
                path.append((nid, pd, op))

            if t.children_left[nid] > 0:  # If not leaf, add children onto stack
                stack.append((t.children_left[nid], pd + 1, 1))
                stack.append((t.children_right[nid], pd + 1, 0))
            else:  # If leaf append rule
                rules.append(np.array([(t.feature[path[i][0]], path[i+1][2], t.threshold[path[i][0]]) for i in range(len(path)-1)]))

        return rules

    def __convert_rule__(self, x, labels=None, scaler=None):
        """Convert rule represented by an array to readable format
        
        Parameters
        ----------
        
        x: numpy.ndarray
            Input array where each row represents a feature in a rule.
            3 columns:
            First indicates feature number,
            Second indicates operator (1 if $>$ otherwise $\leq$),
            Third indicates threshold value
        
        labels: list of str, optional
            Names of features to replace feature numbers with
        
        scaler:
            Scaler to reverse scaling done in fitting so interpretable
            feature values can be used.
        
        Returns
        -------
        
        list of str
            List containing each stage of input rule
        
        """
        strop = ['>', '<=']

        if scaler is None:
            # If no scaler, do not shift or scale
            nf = x[:, 0].astype(int).max()+1
            scale = np.ones(nf)
            center = np.zeros(nf)
        else:
            scale = scaler.scale_
            center = scaler.center_

        if labels is None:
            return [(str(int(f)) + str(strop[int(op)]) + str(thresh*scale[int(f)]+center[int(f)])) for f, op, thresh in x]
        else:
            return [(labels[int(f)] + str(strop[int(op)]) + str(thresh*scale[int(f)]+center[int(f)])) for f, op, thresh in x]
    
    def extract_rules(self, labels=None):
        """Extract rules from `base_estimator`
        
        Parameters
        ----------
        
        labels: list of str, optional
            Feature names
            
        Returns
        -------
        
        numpy.ndarray
            Containing `str` representing rules in ensembles
            
        """
        # Extract flat list of rules in array form
        if isinstance(self.base_estimator, RandomForestClassifier):
            rules = list(it.chain(*[self.__extract_dt_rules__(dt) for dt in self.base_estimator.estimators_]))
        elif isinstance(self.base_estimator, GradientBoostingClassifier):
            rules = list(it.chain(*[self.__extract_dt_rules(__dt) for dt in self.base_estimator.estimators_.ravel()]))
        elif isinstance(self.base_estimator, XGBClassifier):
            rules = list(it.chain(*[self.__extract_xgb_dt_rules__(dt) for dt in self._rule_dump]))
            
        # Convert each sub-rule into text, join together with '&' and then add to rules
        self.rules = np.array([' & '.join(self.__convert_rule__(r, labels=labels, scaler=self.ext_scaler)) for r in rules])
        
        return self.rules
