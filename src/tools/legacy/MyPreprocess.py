# https://gist.github.com/thomasjpfan/9bc1149810307c66d5e90eec04c769be

from this import d
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest

def _get_input_features(input_features, n_features_in):
    if input_features is not None:
        return np.array(input_features, dtype=object)
    return np.array([f"X{i}" for i in range(n_features_in)], dtype=object)

class OneToOneMixin:
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return super().fit(X, y)
        
    def get_feature_names(self, input_features=None):
        return _get_input_features(input_features, self.n_features_in_)
    
class ColumnNameRecorder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X, y=None):
        return X
    
    def get_feature_names(self, input_features=None):
        # uses input_features if self.feature_names_in_ is None
        if self.feature_names_in_ is None:
            return _get_input_features(input_features, self.n_features_in_)
        return self.feature_names_in_
    
from sklearn.compose import ColumnTransformer

class MyColumnTransformer(OneToOneMixin, ColumnTransformer):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self
    
    def fit_transform(self, X, y=None):
        output = super().fit_transform(X, y)
        self.n_features_in_ = X.shape[1]
        return output
    
    def get_feature_names(self, input_features=None):
        input_features = _get_input_features(input_features,
                                             self.n_features_in_)
        feature_names = []
        for name, trans, column, _ in self._iter(fitted=True):
            if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
                continue
            # if column is a mask or indicies use `input_features` to extract column names
            # lets assume column is a name for now
            feature_names.extend([name + "__" + f for f in 
                                  trans.get_feature_names(
                                      input_features=column
                                  )])
            
        return np.array(feature_names, dtype=object)
    
class MyPipeline(Pipeline):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self
    
    def fit_transform(self, X, y=None):
        output = super().fit_transform(X, y)
        self.n_features_in_ = X.shape[1]
        return output
    
    def get_feature_names(self, input_features=None):
        features = _get_input_features(input_features, self.n_features_in_)
        for _, est in self.steps:
            features = est.get_feature_names(input_features=features)
            
        return features
    
    def __getitem__(self, ind):
        # quick hack to get string slice from end to work
        if isinstance(ind.stop, str):
            # find name in pipeline
            for i, (name, _) in enumerate(self.steps):
                if name == ind.stop:
                    ind = slice(ind.start, i, ind.step)
                    break
            else: # no break
                raise IndexError(f"{ind} not in steps")
            
        output = super().__getitem__(ind)
        output.n_features_in_ = self.n_features_in_
        return output
    
    
class NameColumnTransformer(ColumnTransformer):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self
    
    def fit_transform(self, X, y=None):
        output = super().fit_transform(X, y)
        self.n_features_in_ = X.shape[1]
        return output
    
    def get_feature_names(self, input_features=None):
        input_features = _get_input_features(input_features,
                                             self.n_features_in_)
        feature_names = []
        for name, trans, column, _ in self._iter(fitted=True):
            if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
                continue
            # if column is a mask or indicies use `input_features` to extract column names
            # lets assume column is a name for now
            feature_names.extend([name + "__" + f for f in 
                                trans.get_feature_names(
                                    input_features=column
                                )])


def _get_input_features(input_features, n_features_in):
    if input_features is not None:
        return np.array(input_features, dtype=object)
    return np.array([f"X{i}" for i in range(n_features_in)], dtype=object)

class MySimpleImputer(SimpleImputer):
    def __init__(self):
        self.feature_names_in_ = self.get_feature_names(input_features=None)
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return super().fit(X, y)
        
    def get_feature_names(self, input_features=None):
        feature_names_in_ = _get_input_features(input_features,self.n_features_in_)
        return feature_names_in_
    


from skimpy import clean_columns
from sklearn.utils.validation import check_is_fitted

def get_column_names_from_ColumnTransformer(column_transformer, clean_column_names=True, verbose=True):  

    """
    Reference: Kyle Gilde: https://github.com/kylegilde/Kaggle-Notebooks/blob/master/Extracting-and-Plotting-Scikit-Feature-Names-and-Importances/feature_importance.py
    Description: Get the column names from the a ColumnTransformer containing transformers & pipelines
    Parameters
    ----------
    verbose: Bool indicating whether to print summaries. Default set to True.
    Returns
    -------
    a list of the correct feature names
    Note:
    If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns,
    it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns
    to the dataset that didn't exist before, so there should come last in the Pipeline.
    Inspiration: https://github.com/scikit-learn/scikit-learn/issues/12525
    """

    assert isinstance(column_transformer, ColumnTransformer), "Input isn't a ColumnTransformer"
    
    check_is_fitted(column_transformer)

    new_feature_names, transformer_list = [], []

    for i, transformer_item in enumerate(column_transformer.transformers_): 
        transformer_name, transformer, orig_feature_names = transformer_item
        orig_feature_names = list(orig_feature_names)

        if len(orig_feature_names) == 0:
            continue

        if verbose: 
            print(f"\n\n{i}.Transformer/Pipeline: {transformer_name} {transformer.__class__.__name__}\n")
            print(f"\tn_orig_feature_names:{len(orig_feature_names)}")

        if transformer == 'drop':
            continue

        if isinstance(transformer, Pipeline):
            # if pipeline, get the last transformer in the Pipeline
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, 'get_feature_names_out'):
            if 'input_features' in transformer.get_feature_names_out.__code__.co_varnames:
                names = list(transformer.get_feature_names_out(orig_feature_names))
            else:
                names = list(transformer.get_feature_names_out())
        elif hasattr(transformer, 'get_feature_names'):
            if 'input_features' in transformer.get_feature_names.__code__.co_varnames:
                names = list(transformer.get_feature_names(orig_feature_names))
            else:
                names = list(transformer.get_feature_names())

        elif hasattr(transformer,'indicator_') and transformer.add_indicator:
            # is this transformer one of the imputers & did it call the MissingIndicator?

            missing_indicator_indices = transformer.indicator_.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag'\
                                  for idx in missing_indicator_indices]
            names = orig_feature_names + missing_indicators

        elif hasattr(transformer,'features_'):
            # is this a MissingIndicator class? 
            missing_indicator_indices = transformer.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag'\
                                  for idx in missing_indicator_indices]

        else:

            names = orig_feature_names

        if verbose: 
            print(f"\tn_new_features:{len(names)}")
            print(f"\tnew_features: {names}\n")

        new_feature_names.extend(names)
        transformer_list.extend([transformer_name] * len(names))

    transformer_list, column_transformer_features = transformer_list, new_feature_names

    if clean_column_names:
        new_feature_names = list(clean_columns(pd.DataFrame(columns=new_feature_names)).columns)
    
    return new_feature_names