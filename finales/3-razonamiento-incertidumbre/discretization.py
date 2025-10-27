"""
Discretización de variables continuas
"""
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple, Literal, Optional, List, Union


class Discretizer:
    """
    Wrapper for discretizing continuous features using different strategies.
    Supports selective discretization of specific columns while keeping others unchanged.
    """
    
    def __init__(
        self, 
        n_bins: int = 5, 
        strategy: Literal['uniform', 'quantile', 'kmeans'] = 'quantile',
        encode: str = 'ordinal',
        columns_to_discretize: Optional[Union[List[int], np.ndarray]] = None
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
        columns_to_discretize : list or array-like, optional
            Indices of columns to discretize. If None, discretizes all columns.
            Use this to keep binary/categorical columns unchanged.
            Example: [0, 2, 5] will discretize only columns 0, 2, and 5
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
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
    
    def fit(self, X: np.ndarray) -> 'Discretizer':
        """
        Fit the discretizer on training data
        
        Parameters:
        -----------
        X : np.ndarray
            Training data to fit
            
        Returns:
        --------
        self : Discretizer
            Fitted discretizer
        """
        X = np.asarray(X)
        n_features = X.shape[1]
        
        # Determine which columns to discretize
        if self.columns_to_discretize is None:
            # Discretize all columns
            self.columns_to_discretize = list(range(n_features))
            self._keep_columns = []
        else:
            # Discretize only specified columns
            self.columns_to_discretize = list(self.columns_to_discretize)
            # Keep track of columns to preserve
            self._keep_columns = [i for i in range(n_features) 
                                 if i not in self.columns_to_discretize]
        
        self._all_columns = list(range(n_features))
        
        # Fit only on columns to discretize
        if len(self.columns_to_discretize) > 0:
            X_to_fit = X[:, self.columns_to_discretize]
            self.discretizer.fit(X_to_fit)
        
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted discretizer
        
        Parameters:
        -----------
        X : np.ndarray
            Data to transform
            
        Returns:
        --------
        X_discrete : np.ndarray
            Discretized data (with specified columns discretized, others unchanged)
        """
        if not self._fitted:
            raise ValueError("Discretizer must be fitted before transform")
        
        X = np.asarray(X)
        
        # If discretizing all columns, use standard approach
        if len(self._keep_columns) == 0:
            return self.discretizer.transform(X).astype(int)
        
        # Otherwise, selectively discretize
        X_result = X.copy()
        
        # Discretize specified columns
        if len(self.columns_to_discretize) > 0:
            X_to_transform = X[:, self.columns_to_discretize]
            X_discretized = self.discretizer.transform(X_to_transform).astype(int)
            X_result[:, self.columns_to_discretize] = X_discretized
        
        # Keep other columns as integers (for binary/categorical)
        X_result[:, self._keep_columns] = X[:, self._keep_columns].astype(int)
        
        return X_result.astype(int)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data in one step
        
        Parameters:
        -----------
        X : np.ndarray
            Data to fit and transform
            
        Returns:
        --------
        X_discrete : np.ndarray
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
    
    def get_discretized_columns(self) -> List[int]:
        """Get list of columns that were discretized"""
        return self.columns_to_discretize if self.columns_to_discretize else []
    
    def get_kept_columns(self) -> List[int]:
        """Get list of columns that were kept unchanged"""
        return self._keep_columns if self._keep_columns else []


def discretize_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_bins: int = 5,
    strategy: str = 'quantile',
    columns_to_discretize: Optional[Union[List[int], np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, Discretizer]:
    """
    Discretize training and test data
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data
    X_test : np.ndarray
        Test data
    n_bins : int
        Number of bins
    strategy : str
        Discretization strategy ('uniform', 'quantile', 'kmeans')
    columns_to_discretize : list or array-like, optional
        Indices of columns to discretize. If None, discretizes all columns.
        Example: [0, 2, 5] will discretize only columns 0, 2, and 5
        
    Returns:
    --------
    X_train_discrete : np.ndarray
        Discretized training data
    X_test_discrete : np.ndarray
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
    X: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_bins: int = 5,
    strategy: str = 'quantile',
    columns_to_discretize: Optional[Union[List[int], np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize a single CV fold
    
    Parameters:
    -----------
    X : np.ndarray
        Full dataset
    train_idx : np.ndarray
        Indices for training set
    test_idx : np.ndarray
        Indices for test set
    n_bins : int
        Number of bins
    strategy : str
        Discretization strategy
    columns_to_discretize : list or array-like, optional
        Indices of columns to discretize. If None, discretizes all columns.
        
    Returns:
    --------
    X_train_discrete : np.ndarray
        Discretized training data
    X_test_discrete : np.ndarray
        Discretized test data
    """
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    X_train_discrete, X_test_discrete, _ = discretize_data(
        X_train, X_test, n_bins, strategy, columns_to_discretize
    )
    
    return X_train_discrete, X_test_discrete


def identify_continuous_columns(
    X: np.ndarray,
    max_unique_values: int = 10,
    binary_threshold: int = 2
) -> Tuple[List[int], List[int]]:
    """
    Automatically identify continuous vs binary/categorical columns
    
    Parameters:
    -----------
    X : np.ndarray
        Data to analyze
    max_unique_values : int
        Maximum number of unique values to consider a column as categorical
    binary_threshold : int
        If unique values <= this, consider as binary/categorical
        
    Returns:
    --------
    continuous_cols : list
        Indices of columns identified as continuous
    categorical_cols : list
        Indices of columns identified as binary/categorical
    """
    n_features = X.shape[1]
    continuous_cols = []
    categorical_cols = []
    
    for i in range(n_features):
        unique_values = np.unique(X[:, i])
        n_unique = len(unique_values)
        
        # Check if binary (0/1 or similar)
        if n_unique <= binary_threshold:
            categorical_cols.append(i)
        # Check if clearly categorical (few unique values)
        elif n_unique <= max_unique_values:
            categorical_cols.append(i)
        # Otherwise, treat as continuous
        else:
            continuous_cols.append(i)
    
    return continuous_cols, categorical_cols


def auto_discretize_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_bins: int = 5,
    strategy: str = 'quantile',
    max_unique_values: int = 10
) -> Tuple[np.ndarray, np.ndarray, Discretizer, List[int], List[int]]:
    """
    Automatically identify and discretize only continuous columns
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data
    X_test : np.ndarray
        Test data
    n_bins : int
        Number of bins for discretization
    strategy : str
        Discretization strategy
    max_unique_values : int
        Maximum unique values to consider as categorical
        
    Returns:
    --------
    X_train_discrete : np.ndarray
        Discretized training data
    X_test_discrete : np.ndarray
        Discretized test data
    discretizer : Discretizer
        Fitted discretizer
    continuous_cols : list
        Indices of continuous columns that were discretized
    categorical_cols : list
        Indices of categorical columns that were kept
    """
    # Identify column types
    continuous_cols, categorical_cols = identify_continuous_columns(
        X_train, max_unique_values=max_unique_values
    )
    
    print(f"Identified {len(continuous_cols)} continuous columns: {continuous_cols}")
    print(f"Identified {len(categorical_cols)} categorical/binary columns: {categorical_cols}")
    
    # Discretize only continuous columns
    X_train_discrete, X_test_discrete, discretizer = discretize_data(
        X_train, X_test, n_bins, strategy, 
        columns_to_discretize=continuous_cols
    )
    
    return X_train_discrete, X_test_discrete, discretizer, continuous_cols, categorical_cols
