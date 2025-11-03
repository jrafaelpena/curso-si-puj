"""
Bayesian Network and Naive Bayes model implementations
"""
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import (
    TreeSearch, HillClimbSearch, PC, MmhcEstimator,
    BicScore, BDeuScore, MaximumLikelihoodEstimator
)
from pgmpy.inference import VariableElimination
import pandas as pd

class NaiveBayesModel:
    """Naive Bayes classifier wrapper"""
    
    def __init__(self):
        self.model = CategoricalNB()
        self.model_type = 'naive_bayes'
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesModel':
        """Fit Naive Bayes model"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Naive Bayes model"""
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Get accuracy score"""
        return self.model.score(X, y)


class BayesianNetworkModel:
    """
    Bayesian Network model wrapper using pgmpy
    """
    
    def __init__(
        self,
        structure_learning: str = 'hillclimb_bic',
        max_parents: int = 3,
        scoring_method: str = 'bic'
    ):
        """
        Initialize Bayesian Network model
        
        Parameters:
        -----------
        structure_learning : str
            Structure learning algorithm:
            - 'tan': Tree Augmented Naive Bayes
            - 'pc': PC algorithm
            - 'hillclimb_bic': Hill Climbing with BIC score
            - 'hillclimb_bdeu': Hill Climbing with BDeu score
            - 'tree_search': Tree Search algorithm
            - 'mmhc': Max-Min Hill Climbing (hybrid)
        max_parents : int
            Maximum number of parents for each node
        scoring_method : str
            Scoring method for structure learning ('bic' or 'bdeu')
        """
        self.structure_learning = structure_learning
        self.max_parents = max_parents
        self.scoring_method = scoring_method
        self.model = None
        self.inference = None
        self.feature_names = None
        self.target_name = 'target'
    
    def _prepare_data(
        self, 
        X, 
        y = None
    ) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame for pgmpy
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Feature data
        y : np.ndarray, pd.Series, or None
            Target labels
            
        Returns:
        --------
        data : pd.DataFrame
            Data as pandas DataFrame
        """
        # If X is already a DataFrame, use it directly
        if isinstance(X, pd.DataFrame):
            data = X.copy()
            self.feature_names = list(X.columns)
        else:
            # Convert numpy array to DataFrame
            n_features = X.shape[1]
            self.feature_names = [f'X{i}' for i in range(n_features)]
            data = pd.DataFrame(X, columns=self.feature_names)
        
        # Add target variable if provided
        if y is not None:
            if isinstance(y, pd.Series):
                data[self.target_name] = y.values
            else:
                data[self.target_name] = y
        
        return data
    
    def _learn_structure(self, data: pd.DataFrame) -> BayesianNetwork:
        """Learn Bayesian Network structure"""
        
        if self.structure_learning == 'tan':
            return self._learn_tan_structure(data)
        elif self.structure_learning == 'pc':
            return self._learn_pc_structure(data)
        elif self.structure_learning in ['hillclimb_bic', 'hillclimb_bdeu']:
            return self._learn_hillclimb_structure(data)
        elif self.structure_learning == 'tree_search':
            return self._learn_tree_search_structure(data)
        elif self.structure_learning == 'mmhc':
            return self._learn_mmhc_structure(data)
        else:
            raise ValueError(f"Unknown structure learning method: {self.structure_learning}")
    
    def _learn_tan_structure(self, data: pd.DataFrame) -> BayesianNetwork:
        """Learn Tree Augmented Naive Bayes structure"""
        estimator = TreeSearch(data, root_node=self.target_name)
        model = estimator.estimate(estimator_type='tan')
        return model
    
    def _learn_pc_structure(self, data: pd.DataFrame) -> BayesianNetwork:
        """Learn structure using PC algorithm"""
        est = PC(data)
        model = est.estimate(max_cond_vars=self.max_parents)
        return model
    
    def _learn_hillclimb_structure(self, data: pd.DataFrame) -> BayesianNetwork:
        """Learn structure using Hill Climbing"""
        if self.scoring_method == 'bic' or self.structure_learning == 'hillclimb_bic':
            scoring_method = BicScore(data)
        else:
            scoring_method = BDeuScore(data, equivalent_sample_size=10)
        
        est = HillClimbSearch(data)
        model = est.estimate(
            scoring_method=scoring_method,
            max_indegree=self.max_parents
        )
        return model
    
    def _learn_tree_search_structure(self, data: pd.DataFrame) -> BayesianNetwork:
        """Learn structure using Tree Search"""
        est = TreeSearch(data)
        model = est.estimate()
        return model
    
    def _learn_mmhc_structure(self, data: pd.DataFrame) -> BayesianNetwork:
        """Learn structure using MMHC (Max-Min Hill Climbing)"""
        est = MmhcEstimator(data)
        model = est.estimate(max_indegree=self.max_parents)
        return model
    
    def fit(self, X, y) -> 'BayesianNetworkModel':
        """
        Fit Bayesian Network model
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Training features
        y : np.ndarray or pd.Series
            Training labels
            
        Returns:
        --------
        self : BayesianNetworkModel
            Fitted model
        """
        # Prepare data
        data = self._prepare_data(X, y)
        
        # Learn structure
        self.model = self._learn_structure(data)
        
        # Fit parameters (CPDs) using Maximum Likelihood
        self.model.fit(data, estimator=MaximumLikelihoodEstimator)
        
        # Initialize inference engine
        self.inference = VariableElimination(self.model)
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """
        Predict using Bayesian Network
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Test features
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted labels
        """
        predictions = []
        data = self._prepare_data(X)
        
        for idx in range(len(data)):
            evidence = data.iloc[idx].to_dict()
            
            # Query for target variable
            result = self.inference.map_query(
                variables=[self.target_name],
                evidence=evidence
            )
            predictions.append(result[self.target_name])
        
        return np.array(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : np.ndarray
            Test features
        y : np.ndarray
            True labels
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


def create_model(
    model_type: str,
    max_parents: int = 3,
    **kwargs
) -> Any:
    """
    Factory function to create appropriate model
    
    Parameters:
    -----------
    model_type : str
        Type of model to create
    max_parents : int
        Maximum parents for BN models
    **kwargs : dict
        Additional parameters
        
    Returns:
    --------
    model : Model object
        Instantiated model
    """
    if model_type == 'naive_bayes':
        return NaiveBayesModel()
    elif model_type in ['tan', 'pc', 'hillclimb_bic', 'hillclimb_bdeu', 
                        'tree_search', 'mmhc']:
        return BayesianNetworkModel(
            structure_learning=model_type,
            max_parents=max_parents,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Train and evaluate a model
    
    Parameters:
    -----------
    model : Model object
        Model to train and evaluate
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
        
    Returns:
    --------
    results : dict
        Dictionary with train/test accuracy and times
    """
    # Training
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluation on training set
    train_accuracy = model.score(X_train, y_train)
    
    # Evaluation on test set
    start_time = time.time()
    test_accuracy = model.score(X_test, y_test)
    test_time = time.time() - start_time
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'test_time': test_time
    }