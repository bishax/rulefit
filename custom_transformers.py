import pdb
from sklearn.base import TransformerMixin, BaseEstimator
from pandas import DataFrame, concat, get_dummies
from numpy import ndarray, array, arange, sort, argsort, vectorize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import check_array

def binarise(x):
    """ Convert array to array of bools
    
    Convert any non-zero value in `x` to 1, 
    otherwise it is 0
    
    Parameters
    ----------
    
    x: numpy.ndarray
    
    Returns
    -------
    
    numpy.ndarray
    """
    x[x > 0] = 1
    x[x != 1] = 0
    assert (x < 0).sum() == 0
    return x

def dict_transform(x, d):
    """Numpy vectorised dictionary mapper
    
    Vectorised method to use values of `x` as keys to map to values
    in `d`. 
    
    Parameters
    ----------
    
    x: numpy.ndarray
    
    d: dict
    
    
    Notes:
    ------
    Used in `LabelOneHotEncoder.fit` but defined here because
    of list comprehension scoping.
    """
    return vectorize(d.__getitem__)(x)

class pdInit(BaseEstimator, TransformerMixin):
    """ sklearn transformer: ensures input feature order is consistent

    Attributes
    ----------

    column_order_: pandas.Index
        Column order of fitted data
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        
        Parameters
        ----------
        
        X: pandas.DataFrame
            Design matrix
        
        y: dummy, optional
        
        
        Returns
        -------
        
        self
        """
        if not isinstance(X, DataFrame):
            raise TypeError('pdIinit expected type pandas.DataFrame')
        self.column_order_ = X.columns
        return self

    def transform(self, X, y=None):
        """
        
        Parameters
        ----------
        
        X: pandas.DataFrame
            Design matrix
        
        y: dummy, optional
        
        
        Returns
        -------
        
        DataFrame
            Column order same as fitted data
        """
        if not isinstance(X, DataFrame):
            raise TypeError('pdIinit expected type pandas.DataFrame')
        return X.loc[:, self.column_order_]


class LabelOneHotEncoder(BaseEstimator, TransformerMixin):
    """ Sklearn transformer: Label input to range of integers and then one-hot-encode
      
    Attributes
    ----------
    
    label_encoder_l_: list of dicts
        Gives mapping between `classes` and their integer labels
    
    n_values: list of ints
        Number of labels within each column
    
    one_hot_encoder_: sklearn.preprocessing.OneHotEncoder
        OneHotEncoder acting on labels
    """

    def __init__(self, classes=None):
        """
        Parameters
        ----------
        
        classes: list, optional
            List of classes (numbers or strings) to be mapped
            to `int` starting from zero in order.
        """
        
        self.classes = classes

    def fit(self, X, y=None):
        """
        
        Parameters
        ----------
        
        X: numpy.ndarray
            Input to be encoded as label and subsequently one-hotted
        
        y: dummy, optional
        
        
        Returns
        -------
        
        self
        """
        X = check_array(X)
        
        
        self.label_encoder_l_ = [{v:k for k, v in enumerate(c)} for c in self.classes]
        self.n_values_ = [len(enc.keys()) for enc in self.label_encoder_l_]
        
        X_new = array([dict_transform(x, d) for x, d in zip(X.T, self.label_encoder_l_)]).T
        
        self.one_hot_encoder_ = OneHotEncoder(n_values=self.n_values_)
        self.one_hot_encoder_.fit(X_new)
        
        return self

    def transform(self, X, y=None):
        """
        
        Parameters
        ----------
        
        X: numpy.ndarray
            Input to be encoded as label and subsequently one-hotted
        
        y: dummy, optional
        
        
        Returns
        -------
        
        sparse matrix
            1 in i-th position implies i-th rule applies
            
        """
        X = check_array(X)
        return self.one_hot_encoder_.transform(array([dict_transform(x, d) for x, d in zip(X.T, self.label_encoder_l_)]).T)

class pdCategoricalTransformer(BaseEstimator, TransformerMixin):
    """ sklearn transformer: Process categorical variables into dummy variables
    
    Take all columns with dtype `object` (cast as `str`) not in `ignore_cols`
    and convert to dummy variables. If binary, column name 
    remains the same. If categorical, prefix 'columnname__'
    is added to dummy column.
    
    Attributes
    ----------
    
    bin_cols_: list of str
        column names that are binary
    
    cat_cols_: list of str
        column names that are categorical
        
    bin_refs_: dict
        Keys are entries in `bin_cols_`, values are a list of two stored values.
        First corresponds to zero, second to one.
        
    cat_refs_: dict
        Keys are entries in `cat_cols_`, values are a list of unique values
        
    cat_drops_: dict
        Keys are entries in `cat_cols_`, values are the reference category to drop 
        (If `drop_first=True`)
        
    column_order_: list of str
        Order of columns of fitted data to ensure transformed data retains feature ordering
    """
    
    def __init__(self, max_cat=25, ignore_cols=None, drop_first=True):
        """
        Parameters
        ----------

        max_cat : int, optional
            Maximum unique variables allowed to restrict
            feature space getting too large

        ignore_cols : list or str, optional
            Columns to not convert to dummy variables
            
        drop_first: bool, default: True
            If True delete most prevalent indicator for each category
        """
        
        self.max_cat = max_cat
        self.ignore_cols = ignore_cols
        self.drop_first = drop_first
        
    def fit(self, X, y=None):
        """
        Parameters
        ----------
        
        X: DataFrame
            Input data
            
        y: dummy, optional
        
        Returns
        -------
        
        self
        """
        
        X = X.copy()
        # List of columns to consider
        cat_cols = set(X.columns[X.dtypes == object])
        if self.ignore_cols is not None:
            cat_cols = cat_cols - set(self.ignore_cols)
        
        # See attributes
        self.bin_cols_ = []
        self.cat_cols_ = []
        self.bin_refs_ = {}
        self.cat_refs_ = {}
        self.cat_drops_ = {}

        for cn in cat_cols:
            uniq = X[cn].astype(str).unique()  # Unique column entries
            if uniq.size == 1:
                raise ValueError('Column {} only has one value, suggest dropping'.format(cn))
            elif uniq.size == 2:  # Binary
                self.bin_cols_.append(cn)
                uniq.sort()
                self.bin_refs_[cn] = uniq.tolist()  # Store what value corresponds to one
                X.loc[:, cn] = X[cn] == uniq[1]
            elif uniq.size <= self.max_cat:  # Categorical
                self.cat_cols_.append(cn)
                self.cat_refs_[cn] = uniq.tolist()  # Store dummy columns
                dummies = get_dummies(X[cn], prefix='{}_'.format(cn), drop_first=False)
                
                if self.drop_first:
                    drop = dummies.mean().argmax()  # Store most prevalent category to drop
                    self.cat_drops_[cn] = drop
                    dummies = dummies.drop(drop, 1)
                else:
                    pass
                X = X.join(dummies).drop(cn, 1)
            else:
                raise ValueError('Column {} has {}>{} values. Increase `max_cat`.'.format(cn, uniq.size, self.max_cat))            
                
        self.column_order_ = X.columns.tolist()  # Store column order

        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        
        X: DataFrame
            Input data
            
        y: dummy, optional
        
        Returns
        -------
        
        DataFrame
        """
        
        for cn in self.bin_cols_:  # Binary
            refs = self.bin_refs_[cn]
            
            # Raise ValueError if unseen values
            new = list(set(X[cn].astype(str).unique())-set(refs))
            if len(new) > 0:
                raise ValueError('Unseen values in {} for column {}'.format(X[cn].astype(str).unique(), cn))
            
            X.loc[:, cn] = X[cn] == refs[1]
                
        for cn in self.cat_cols_:  # Categorical
            refs = self.cat_refs_[cn]
            dummies = get_dummies(X[cn], prefix='{}_'.format(cn), drop_first=False)
            
            # Set values in train but not test to zero
            missing = list(set(refs)-set(X[cn].astype(str).unique()))
            for v in missing:
                dummies['{}__{}'.format(cn, v)] = 0
            
            # Raise ValueError if unseen values
            new = list(set(X[cn].astype(str).unique())-set(refs))
            if len(new) > 0:
                raise ValueError('Unseen values in {} for column {}'.format(X[cn].astype(str).unique(), cn))

            # Drop reference category
            if self.drop_first:
                drop = self.cat_drops_[cn]
                dummies = dummies.drop(drop, 1)
            else:
                pass
            
            X = X.join(dummies).drop(cn, 1)
            
        assert X.shape[1] == len(self.column_order_), 'Size of fit {} and transform {} do not match'.format(X.shape[1], len(self.column_order_))

        return X.loc[:, self.column_order_]
    
    
class ColumnExtractor(BaseEstimator, TransformerMixin):
    """ Sklearn transformer: Extract column(s) (or complement of) within a pipeline.
    
    """

    def __init__(self, column, complement=False):
        """
        Parameters
        ----------
        
        column: list-like or string
            column(s) to extract from data
        
        complement: bool, default: False
            If True take all columns except those in `column`
        """
        self.column = column
        self.complement = complement

    def transform(self, X, y=None):
        """Extract columns from input data
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
            DataFrame with features to be extracted by name,
            or array with features to be extracted by number
        
        y: dummy, optional
        
        
        Returns
        -------
        
        pandas.DataFrame
            If `X` is a DataFrame
            
        numpy.ndarray
            If `X` is an array
        """
        
        # Ensure `self.column` is list-like
        if not isinstance(self.column, (list, tuple, ndarray)):
            self.column = [self.column]
            
        if isinstance(X, DataFrame):
            if self.complement:  # Take complement by name
                return X.drop(self.column, 1)
            else:  # Take requested columns by name
                return X.loc[:, self.column]
        else:
            if self.complement:  # Take complement by number (preserve order)
                return X[:, sort(list(set(arange(X.shape[1]))- set(self.column)))]
            else:  # Take requested columns by number
                return X[:, self.column]

    def fit(self, X, y=None):
        """ Dummy method for sklearn API compatibility
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        y: dummy, optional
        
        
        Returns
        -------
        
        self
        """
        return self


class X_flatten(BaseEstimator, TransformerMixin):
    """Sklearn transformer: flatten data within a pipeline
    
    Flatten and convert data to a numpy array.
    
    Notes
    -----
    
    Primary use case is to format data into the right format
    for something like `sklearn.feature_extraction.text.TfidfVectorizer`
    """
    
    def fit(self, X, y=None):
        """ Dummy method for sklearn API compatibility
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        y: dummy, optional
        
        
        Returns
        -------
        
        self
        """
        return self
    
    def transform(self, X, y=None):
        """Apply flatten to data
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
            Data to flatten
        
        y: dummy, optional
        
        
        Returns
        -------
        
        numpy.ndarray
        """
        if isinstance(X, DataFrame):
            return X.values.flatten()
        else:
            return X.flatten()
    
class pdFunctionTransformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer: apply a function to dataframe
    
    Wrapper function to apply a function to data within a sklearn
    pipeline.
    
    Examples
    --------
    
    >>> transformer = pdFunctionTransformer(binarise)
    >>> transformer.transform(X)  # Binarise data in X
    """
    
    def __init__(self, fun, keep_cols=True):
        """
        Parameters
        ----------
        
        fun: function
            Function to apply to data
        
        keep_cols: bool, default: true
            If True return a DataFrame rather than array
        """
        self.fun = fun
        self.keep_cols = keep_cols
        
    def fit(self, X, y=None):
        """ Dummy method for sklearn API compatibility
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        y: dummy, optional
        
        
        Returns
        -------
        
        self
        """
        return self
    
    def transform(self, X, y=None):
        """ Apply `fun` to `X`
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
            Data to apply function to
        
        y: dummy, optional
        
        
        Returns
        -------
        
        pandas.DataFrame
            If `keep_cols=True`
            
        numpy.ndarray
            If `keep_cols=False`
        """
        if isinstance(X, DataFrame):
            if self.keep_cols:
                return DataFrame(self.fun(X.values), columns=X.columns)
            else:
                return self.fun(X.values)
        else:
            return self.fun(X)
    
class pdVectorizer(BaseEstimator, TransformerMixin):
    """Sklearn transformer: vectorise data
    
    Wrapper transformer to vectorise a DataFrame
    and output a DataFrame with columns as new feature names.
    
    Examples
    --------
    
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> tf = pdVectorizer(TfidfVectorizer(max_features=100))
    >>> tf.fit_transform(X)
    """
    
    def __init__(self, tf):
        """
        Parameters
        ----------
        
        tf: sklearn Vectorizer transformer
        """
        self.tf = tf
        
    def fit(self, X, y=None):
        """ Fit `tf`
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        y: dummy, optional
        
        
        Returns
        -------
        
        self
        """
        self.tf.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        """ Transform `X` with `tf` and output into DataFrame
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        y: dummy, optional
        
        
        Returns
        -------
        
        pandas.DataFrame
            DataFrame of vectorized counts corresponding to features created by `tf`
        """
        return DataFrame(self.tf.transform(X).todense(),
                         columns=map(lambda x: 'W:'+x, self.tf.get_feature_names()))
                         # columns=map(lambda x: x, self.tf.get_feature_names()))
    
class pdTransformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer: apply transformer to data
    
    Pandas wrapper to output DataFrame from a transformer.
    """
    
    def __init__(self, tf):
        """
        Parameters
        ----------
        
        tf: sklearn transformer
        """
        self.tf = tf
        
    def fit(self, X, y=None):
        """ Dummy method for sklearn API compatibility
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        y: dummy, optional
        
        
        Returns
        -------
        
        self
        """
        self.tf.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        """ Apply `tf` to data and output DataFrame
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        y: dummy, optional
        
        
        Returns
        -------
        
        DataFrame
        """
        return DataFrame(self.tf.transform(X), columns=X.columns)
    
class pdFeatureUnion(BaseEstimator, TransformerMixin):
    """ Pandas wrapper around `sklearn.pipeline.FeatureUnion`
    
    Extend `sklearn.pipeline.FeatureUnion` to work with DataFrames.
    """
    
    def __init__(self, union):
        """
        union: sklearn.pipeline.FeatureUnion
            FeatureUnion object to put into DataFrame
        """
        self.union = union
        
    def fit(self, X, y=None):
        """ Fit `union`
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        y: pandas.Series or numpy.ndarray, optional
        
        
        Returns
        -------
        
        self
        """
        self.union.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        """ Take transformers from `union` and concatenate
        into DataFrame
        
        Parameters
        ----------
        
        X: pandas.DataFrame or numpy.ndarray
        
        y: pandas.Series or numpy.ndarray, optional
        
        
        Returns
        -------
        
        pandas.DataFrame
            Unionised features
        """
        df = DataFrame([])
        for _, transformer in self.union.transformer_list:
            df = concat([df, transformer.transform(X).reset_index(drop=True)], 1)
        return df
