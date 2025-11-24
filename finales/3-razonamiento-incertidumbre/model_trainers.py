import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.naive_bayes import CategoricalNB
import warnings
warnings.filterwarnings('ignore')

# Import from existing discretization module
from discretization import discretize_data


def train_naive_bayes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    discretization_cols: list,
    n_bins: int = 4,
    discretization_method: str = 'quantile'
) -> Tuple[CategoricalNB, pd.DataFrame]:
    """
    Train a Naive Bayes model with specified discretization parameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        discretization_cols: Columns to discretize
        n_bins: Number of bins for discretization (default: 4)
        discretization_method: 'quantile' or 'uniform' (default: 'quantile')
    
    Returns:
        Tuple of (trained_model, X_test_discretized)
    
    Example:
        >>> model, X_test_disc = train_naive_bayes(
        ...     X_train=X_train_fs,
        ...     y_train=y_train,
        ...     X_test=X_test_fs,
        ...     discretization_cols=['age', 'energy_level'],
        ...     n_bins=4,
        ...     discretization_method='quantile'
        ... )
        >>> y_pred = model.predict(X_test_disc)
        >>> y_proba = model.predict_proba(X_test_disc)[:, 1]
    """
    # Discretize the data
    X_train_disc, X_test_disc, _ = discretize_data(
        X_train=X_train,
        X_test=X_test,
        n_bins=n_bins,
        strategy=discretization_method,
        columns_to_discretize=discretization_cols
    )
    
    # Train the model
    model = CategoricalNB()
    model.fit(X_train_disc, y_train)
    
    return model, X_test_disc


def train_tan(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_col: str,
    discretization_cols: list,
    n_bins: int = 4,
    discretization_method: str = 'quantile'
) -> Tuple[object, pd.DataFrame]:
    """
    Train a Tree Augmented Naive Bayes (TAN) model with specified discretization parameters.
    
    Args:
        train_data: Training data (includes target column)
        test_data: Test data (includes target column)
        target_col: Name of the target column
        discretization_cols: Columns to discretize
        n_bins: Number of bins for discretization (default: 4)
        discretization_method: 'quantile' or 'uniform' (default: 'quantile')
    
    Returns:
        Tuple of (trained_model, test_data_discretized)
    
    Example:
        >>> model, test_data_disc = train_tan(
        ...     train_data=train_data_fs,
        ...     test_data=test_data_fs,
        ...     target_col='pulmonary_disease',
        ...     discretization_cols=['age', 'energy_level'],
        ...     n_bins=4,
        ...     discretization_method='quantile'
        ... )
        >>> X_test = test_data_disc.drop(columns=[target_col])
        >>> y_pred = model.predict(X_test)
        >>> y_proba_df = model.predict_probability(X_test)
        >>> y_proba = y_proba_df[f'{target_col}_1'].values
    """
    from pgmpy.estimators import TreeSearch, MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    
    # Split features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Discretize the data
    X_train_disc, X_test_disc, _ = discretize_data(
        X_train=X_train,
        X_test=X_test,
        n_bins=n_bins,
        strategy=discretization_method,
        columns_to_discretize=discretization_cols
    )
    
    # Reconstruct full dataframes - reset indices to align properly
    X_train_disc = X_train_disc.reset_index(drop=True)
    X_test_disc = X_test_disc.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    train_data_disc = pd.concat([X_train_disc, y_train_reset], axis=1)
    test_data_disc = pd.concat([X_test_disc, y_test_reset], axis=1)
    
    # Learn structure using TAN
    estimator = TreeSearch(train_data_disc)
    dag = estimator.estimate(estimator_type='tan', class_node=target_col)
    
    # Create and train the model
    model = DiscreteBayesianNetwork(dag.edges())
    model.fit(train_data_disc, estimator=MaximumLikelihoodEstimator)
    
    return model, test_data_disc


def train_bayesian_network(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_col: str,
    discretization_cols: list,
    structure_method: str = 'hillclimb_k2',
    n_bins: int = 4,
    discretization_method: str = 'quantile',
    max_indegree: int = 5,
    max_iter: int = 100000
) -> Tuple[object, pd.DataFrame]:
    """
    Train a Bayesian Network model with specified structure learning and discretization parameters.
    
    Args:
        train_data: Training data (includes target column)
        test_data: Test data (includes target column)
        target_col: Name of the target column
        discretization_cols: Columns to discretize
        structure_method: Structure learning method ('hillclimb_k2', 'hillclimb_bic', 'hillclimb_bdeu', 'pc')
        n_bins: Number of bins for discretization (default: 4)
        discretization_method: 'quantile' or 'uniform' (default: 'quantile')
        max_indegree: Maximum number of parents per node (default: 5)
        max_iter: Maximum iterations for hill climbing (default: 100000)
    
    Returns:
        Tuple of (trained_model, test_data_discretized)
    
    Example:
        >>> model, test_data_disc = train_bayesian_network(
        ...     train_data=train_data_fs,
        ...     test_data=test_data_fs,
        ...     target_col='pulmonary_disease',
        ...     discretization_cols=['age', 'energy_level'],
        ...     structure_method='hillclimb_k2',
        ...     n_bins=4,
        ...     discretization_method='quantile'
        ... )
        >>> X_test = test_data_disc.drop(columns=[target_col])
        >>> y_pred = model.predict(X_test)
        >>> y_proba_df = model.predict_probability(X_test)
        >>> y_proba = y_proba_df[f'{target_col}_1'].values
    """
    from pgmpy.estimators import (
        PC, HillClimbSearch, BIC, BDeu,
        BayesianEstimator
    )
    from pgmpy.models import DiscreteBayesianNetwork
    
    # Split features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Discretize the data
    X_train_disc, X_test_disc, _ = discretize_data(
        X_train=X_train,
        X_test=X_test,
        n_bins=n_bins,
        strategy=discretization_method,
        columns_to_discretize=discretization_cols
    )
    
    # Reconstruct full dataframes - reset indices to align properly
    X_train_disc = X_train_disc.reset_index(drop=True)
    X_test_disc = X_test_disc.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    train_data_disc = pd.concat([X_train_disc, y_train_reset], axis=1)
    test_data_disc = pd.concat([X_test_disc, y_test_reset], axis=1)
    
    # Learn structure based on method
    if structure_method.lower() == 'pc':
        estimator = PC(train_data_disc)
        dag = estimator.estimate(
            variant="stable",
            significance_level=0.05
        )
    
    elif structure_method.lower() == 'hillclimb_k2':
        hc = HillClimbSearch(train_data_disc)
        dag = hc.estimate(
            scoring_method='k2',
            max_indegree=max_indegree,
            max_iter=max_iter
        )
    
    elif structure_method.lower() == 'hillclimb_bic':
        bic = BIC(train_data_disc)
        hc = HillClimbSearch(train_data_disc)
        dag = hc.estimate(
            scoring_method='bic-d',
            max_indegree=max_indegree,
            max_iter=max_iter
        )
    
    elif structure_method.lower() == 'hillclimb_bdeu':
        bdeu = BDeu(train_data_disc, equivalent_sample_size=10)
        hc = HillClimbSearch(train_data_disc)
        dag = hc.estimate(
            scoring_method='bdeu',
            max_indegree=max_indegree,
            max_iter=max_iter
        )
    
    else:
        raise ValueError(f"Unknown structure method: {structure_method}")
    
    # Create and train the model
    model = DiscreteBayesianNetwork(dag.edges())
    model.fit(train_data_disc, estimator=BayesianEstimator)
    
    return model, test_data_disc


def predict_naive_bayes(
    model: CategoricalNB,
    X_test_disc: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Get predictions from a trained Naive Bayes model.
    
    Args:
        model: Trained CategoricalNB model
        X_test_disc: Discretized test features
    
    Returns:
        Dictionary with 'predictions' and 'probabilities'
    """
    y_pred = model.predict(X_test_disc)
    y_proba = model.predict_proba(X_test_disc)[:, 1]
    
    return {
        'predictions': y_pred,
        'probabilities': y_proba
    }


def predict_tan(
    model: object,
    test_data_disc: pd.DataFrame,
    target_col: str
) -> Dict[str, np.ndarray]:
    """
    Get predictions from a trained TAN model.
    
    Args:
        model: Trained TAN model
        test_data_disc: Discretized test data (includes target)
        target_col: Name of target column
    
    Returns:
        Dictionary with 'predictions' and 'probabilities'
    """
    X_test = test_data_disc.drop(columns=[target_col])
    y_pred = model.predict(X_test)
    y_proba_df = model.predict_probability(X_test)
    
    # Extract probability for positive class
    proba_col = f'{target_col}_1'
    if proba_col in y_proba_df.columns:
        y_proba = y_proba_df[proba_col].fillna(0.0).values
    else:
        # Fallback: try to find the correct column
        proba_cols = [col for col in y_proba_df.columns if col.startswith(target_col)]
        if len(proba_cols) > 0:
            y_proba = y_proba_df[proba_cols[-1]].fillna(0.0).values
        else:
            raise ValueError(f"Could not find probability column for {target_col}")
    
    return {
        'predictions': y_pred.values if hasattr(y_pred, 'values') else y_pred,
        'probabilities': y_proba
    }

def predict_bayesian_network(
    model: object,
    test_data_disc: pd.DataFrame,
    target_col: str
) -> Dict[str, np.ndarray]:
    """
    Get predictions from a trained Bayesian Network model.
    
    Args:
        model: Trained Bayesian Network model
        test_data_disc: Discretized test data (includes target)
        target_col: Name of target column
    
    Returns:
        Dictionary with 'predictions' and 'probabilities'
    """
    # Only use features that exist in the learned model
    model_variables = set(model.nodes())
    available_features = list(model_variables - {target_col})
    
    if len(available_features) == 0:
        # Model didn't learn any structure, return uniform probabilities
        raise ValueError("Model has no learned structure - cannot make predictions")
    
    X_test = test_data_disc[available_features]
    y_pred = model.predict(X_test)
    y_proba_df = model.predict_probability(X_test)
    
    # Extract probability for positive class
    proba_col = f'{target_col}_1'
    if proba_col in y_proba_df.columns:
        y_proba = y_proba_df[proba_col].fillna(0.0).values
    else:
        # Fallback: try to find the correct column
        proba_cols = [col for col in y_proba_df.columns if col.startswith(target_col)]
        if len(proba_cols) > 0:
            y_proba = y_proba_df[proba_cols[-1]].fillna(0.0).values
        else:
            # Last resort: use predictions as probabilities
            y_proba = (y_pred == 1).astype(float)
    
    return {
        'predictions': y_pred.values if hasattr(y_pred, 'values') else y_pred,
        'probabilities': y_proba
    }
