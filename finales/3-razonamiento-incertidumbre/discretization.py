# discretization.py
"""
Discretización de variables continuas
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple, Literal, Optional, List, Union


class Discretizer:
    def __init__(
        self, 
        n_bins: int = 5, 
        strategy: Literal['uniform', 'quantile', 'kmeans'] = 'quantile',
        encode: str = 'ordinal',
        columns_to_discretize: Optional[Union[List[str], str]] = None
    ):
        """
        Initialize discretizer
        
        Parameters:
        -----------
        n_bins : int
            Number of bins to produce
        strategy : str
            Strategy for defining bin widths:
            - 'uniform': All bins have identical widths
            - 'quantile': All bins have the same number of points
            - 'kmeans': Values in each bin have the same nearest center of a 1D k-means cluster
        encode : str
            Method to encode the transformed result ('ordinal' or 'onehot')
        columns_to_discretize : list of str, str, or None
            Names of columns to discretize. If None, discretizes all columns.
            Can be a single column name (str) or a list of column names.
            Use this to keep binary/categorical columns unchanged.
            Example: ['age', 'income', 'score'] will discretize only those columns
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        
        # Handle single string or list of strings
        if isinstance(columns_to_discretize, str):
            self.columns_to_discretize = [columns_to_discretize]
        else:
            self.columns_to_discretize = columns_to_discretize
            
        self.discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            subsample=None
        )
        self._fitted = False
        self._all_columns = None
        self._keep_columns = None
    
    def fit(self, X: pd.DataFrame) -> 'Discretizer':
        """
        Fit the discretizer on training data
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data to fit
            
        Returns:
        --------
        self : Discretizer
            Fitted discretizer
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        self._all_columns = list(X.columns)
        
        # Determine which columns to discretize
        if self.columns_to_discretize is None:
            # Discretize all columns
            self.columns_to_discretize = self._all_columns.copy()
            self._keep_columns = []
        else:
            # Verify all specified columns exist
            missing_cols = set(self.columns_to_discretize) - set(self._all_columns)
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
            # Keep track of columns to preserve
            self._keep_columns = [col for col in self._all_columns 
                                 if col not in self.columns_to_discretize]
        
        # Fit only on columns to discretize
        if len(self.columns_to_discretize) > 0:
            X_to_fit = X[self.columns_to_discretize]
            self.discretizer.fit(X_to_fit)
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted discretizer
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data to transform
            
        Returns:
        --------
        X_discrete : pd.DataFrame
            Discretized data (with specified columns discretized, others unchanged)
        """
        if not self._fitted:
            raise ValueError("Discretizer must be fitted before transform")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # Verify columns match
        if list(X.columns) != self._all_columns:
            raise ValueError("Column names in transform data don't match fit data")
        
        # Create result DataFrame
        X_result = X.copy()
        
        # If discretizing all columns
        if len(self._keep_columns) == 0:
            X_discretized = self.discretizer.transform(X[self.columns_to_discretize])
            X_result[self.columns_to_discretize] = X_discretized.astype(int)
        else:
            # Discretize specified columns
            if len(self.columns_to_discretize) > 0:
                X_discretized = self.discretizer.transform(X[self.columns_to_discretize])
                X_result[self.columns_to_discretize] = X_discretized.astype(int)
            
            # Keep other columns as integers (for binary/categorical)
            X_result[self._keep_columns] = X[self._keep_columns].astype(int)
        
        return X_result.astype(int)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data in one step
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data to fit and transform
            
        Returns:
        --------
        X_discrete : pd.DataFrame
            Discretized data
        """
        return self.fit(X).transform(X)
    
    def get_params(self) -> dict:
        """Get discretizer parameters"""
        return {
            'n_bins': self.n_bins,
            'strategy': self.strategy,
            'encode': self.encode,
            'columns_to_discretize': self.columns_to_discretize
        }
    
    def get_discretized_columns(self) -> List[str]:
        """Get list of columns that were discretized"""
        return self.columns_to_discretize if self.columns_to_discretize else []
    
    def get_kept_columns(self) -> List[str]:
        """Get list of columns that were kept unchanged"""
        return self._keep_columns if self._keep_columns else []
    
    def get_bin_edges(self) -> dict:
        """
        Get bin edges for each discretized column
        
        Returns:
        --------
        bin_edges : dict
            Dictionary mapping column names to their bin edges
        """
        if not self._fitted:
            raise ValueError("Discretizer must be fitted first")
        
        bin_edges = {}
        for i, col in enumerate(self.columns_to_discretize):
            bin_edges[col] = self.discretizer.bin_edges_[i]
        
        return bin_edges


def discretize_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_bins: int = 5,
    strategy: str = 'quantile',
    columns_to_discretize: Optional[Union[List[str], str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Discretizer]:
    """
    Discretize training and test data
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training data
    X_test : pd.DataFrame
        Test data
    n_bins : int
        Number of bins
    strategy : str
        Discretization strategy ('uniform', 'quantile', 'kmeans')
    columns_to_discretize : list of str, str, or None
        Names of columns to discretize. If None, discretizes all columns.
        Example: ['age', 'income'] will discretize only those columns
        
    Returns:
    --------
    X_train_discrete : pd.DataFrame
        Discretized training data
    X_test_discrete : pd.DataFrame
        Discretized test data
    discretizer : Discretizer
        Fitted discretizer object
    """
    discretizer = Discretizer(
        n_bins=n_bins, 
        strategy=strategy,
        columns_to_discretize=columns_to_discretize
    )
    X_train_discrete = discretizer.fit_transform(X_train)
    X_test_discrete = discretizer.transform(X_test)
    
    return X_train_discrete, X_test_discrete, discretizer


def discretize_cv_fold(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_bins: int = 5,
    strategy: str = 'quantile',
    columns_to_discretize: Optional[Union[List[str], str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discretize a single CV fold
    
    Parameters:
    -----------
    X : pd.DataFrame
        Full dataset
    train_idx : np.ndarray
        Indices for training set
    test_idx : np.ndarray
        Indices for test set
    n_bins : int
        Number of bins
    strategy : str
        Discretization strategy
    columns_to_discretize : list of str, str, or None
        Names of columns to discretize. If None, discretizes all columns.
        
    Returns:
    --------
    X_train_discrete : pd.DataFrame
        Discretized training data
    X_test_discrete : pd.DataFrame
        Discretized test data
    """
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    
    X_train_discrete, X_test_discrete, _ = discretize_data(
        X_train, X_test, n_bins, strategy, columns_to_discretize
    )
    
    return X_train_discrete, X_test_discrete
