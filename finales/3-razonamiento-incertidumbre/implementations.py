# implementations.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from discretization import discretize_cv_fold


def run_naive_bayes_rskf_experiment(
    inputs_config: Dict,
    discretization_configs: List[Dict],
    cv_config: Dict,
    output_path: Path,
    custom_metrics_from_proba=None  # Pass your metrics function here
) -> pd.DataFrame:
    """
    Run Naive Bayes experiment with RSKF for different configurations.
    """
    
    # Initialize RSKF
    rskf = RepeatedStratifiedKFold(
        n_splits=cv_config['n_splits'],
        n_repeats=cv_config['n_repeats'],
        random_state=cv_config['random_state']
    )
    
    total_iterations = cv_config['n_splits'] * cv_config['n_repeats']
    results = []
    
    for input_name, input_config in inputs_config.items():
        X = input_config['X']
        y = input_config['y']
        discretization_cols = input_config['discretization_cols']
        
        for disc_config in discretization_configs:
            print(f"\nRunning: Input={input_name}, Discretization={disc_config}")
            model = CategoricalNB()
            
            iteration_metrics = {
                'accuracy': [],
                'roc_auc': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            }
            
            fold_idx = 0
            for train_idx, test_idx in rskf.split(X, y):
                fold_idx += 1
                
                X_train_disc, X_test_disc = discretize_cv_fold(
                    X=X,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    n_bins=disc_config['n_bins'],
                    strategy=disc_config['method'],
                    columns_to_discretize=discretization_cols
                )
                
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                model.fit(X_train_disc, y_train)
                y_pred = model.predict(X_test_disc)
                y_proba = model.predict_proba(X_test_disc)[:, 1]
                
                if custom_metrics_from_proba is not None:
                    metrics = custom_metrics_from_proba(y_test, y_proba)
                else:
                    from sklearn.metrics import (
                        accuracy_score, roc_auc_score, precision_score,
                        recall_score, f1_score
                    )
                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "roc_auc": roc_auc_score(y_test, y_proba),
                        "precision": precision_score(y_test, y_pred, zero_division=0),
                        "recall": recall_score(y_test, y_pred, zero_division=0),
                        "f1_score": f1_score(y_test, y_pred, zero_division=0)
                    }
                
                for metric_name, metric_value in metrics.items():
                    iteration_metrics[metric_name].append(metric_value)
            
            result_row = {
                'input_config': input_name,
                'discretization_method': disc_config['method'],
                'n_bins': disc_config['n_bins'],
                'discretization_cols': ', '.join(discretization_cols)
            }
            
            for i in range(total_iterations):
                result_row[f'accuracy_iter_{i+1}'] = iteration_metrics['accuracy'][i]
                result_row[f'auroc_iter_{i+1}'] = iteration_metrics['roc_auc'][i]
            
            result_row['accuracy_mean'] = np.mean(iteration_metrics['accuracy'])
            result_row['accuracy_std'] = np.std(iteration_metrics['accuracy'])
            result_row['auroc_mean'] = np.mean(iteration_metrics['roc_auc'])
            result_row['auroc_std'] = np.std(iteration_metrics['roc_auc'])
            result_row['f1_score_mean'] = np.mean(iteration_metrics['f1_score'])
            result_row['f1_score_std'] = np.std(iteration_metrics['f1_score'])
            result_row['precision_mean'] = np.mean(iteration_metrics['precision'])
            result_row['precision_std'] = np.std(iteration_metrics['precision'])
            result_row['recall_mean'] = np.mean(iteration_metrics['recall'])
            result_row['recall_std'] = np.std(iteration_metrics['recall'])
            
            results.append(result_row)
            
            print(f"  Accuracy: {result_row['accuracy_mean']:.4f} ± {result_row['accuracy_std']:.4f}")
            print(f"  AUROC:    {result_row['auroc_mean']:.4f} ± {result_row['auroc_std']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    config_cols = ['input_config', 'discretization_method', 'n_bins', 'discretization_cols']
    
    # Updated order: F1 before precision/recall
    stat_cols = [
        'accuracy_mean', 'accuracy_std',
        'auroc_mean', 'auroc_std',
        'f1_score_mean', 'f1_score_std',
        'precision_mean', 'precision_std',
        'recall_mean', 'recall_std'
    ]
    
    accuracy_iter_cols = [col for col in results_df.columns if col.startswith('accuracy_iter_')]
    auroc_iter_cols = [col for col in results_df.columns if col.startswith('auroc_iter_')]
    
    ordered_cols = config_cols + stat_cols + accuracy_iter_cols + auroc_iter_cols
    results_df = results_df[ordered_cols]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n✓ Results saved to: {output_path}")
    
    return results_df

def run_tan_rskf_experiment(
    inputs_config: Dict,
    discretization_configs: List[Dict],
    cv_config: Dict,
    output_path: Path,
    custom_metrics_from_proba=None
) -> pd.DataFrame:
    """
    Run Tree Augmented Naive Bayes (TAN) experiment with RSKF for different configurations.
    """
    from pgmpy.estimators import TreeSearch, MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    
    # Initialize RSKF
    rskf = RepeatedStratifiedKFold(
        n_splits=cv_config['n_splits'],
        n_repeats=cv_config['n_repeats'],
        random_state=cv_config['random_state']
    )
    
    total_iterations = cv_config['n_splits'] * cv_config['n_repeats']
    results = []
    
    for input_name, input_config in inputs_config.items():
        data = input_config['data']
        target_col = input_config['target_col']
        discretization_cols = input_config['discretization_cols']
        
        # Extract X and y for cross-validation splitting
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        for disc_config in discretization_configs:
            print(f"\nRunning: Input={input_name}, Discretization={disc_config}")
            
            iteration_metrics = {
                'accuracy': [],
                'roc_auc': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            }
            
            fold_idx = 0
            for train_idx, test_idx in rskf.split(X, y):
                fold_idx += 1
                
                # Discretize the fold
                X_train_disc, X_test_disc = discretize_cv_fold(
                    X=X,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    n_bins=disc_config['n_bins'],
                    strategy=disc_config['method'],
                    columns_to_discretize=discretization_cols
                )
                
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                train_data_fold = pd.concat([X_train_disc, y_train], axis=1)
                test_data_fold = pd.concat([X_test_disc, y_test], axis=1)
                
                estimator = TreeSearch(train_data_fold)
                dag = estimator.estimate(estimator_type='tan', class_node=target_col)
                
                model = DiscreteBayesianNetwork(dag.edges())
                
                model.fit(train_data_fold, estimator=MaximumLikelihoodEstimator)
                
                # 4️⃣ Prediction
                X_test_fold = test_data_fold.drop(columns=[target_col])
                y_pred = model.predict(X_test_fold)
                y_proba_df = model.predict_probability(X_test_fold)
                
                # Extract probability for positive class (assuming binary classification)
                # Adjust column name based on your target encoding
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
                
                # Calculate metrics
                if custom_metrics_from_proba is not None:
                    metrics = custom_metrics_from_proba(y_test, y_proba)
                else:
                    from sklearn.metrics import (
                        accuracy_score, roc_auc_score, precision_score,
                        recall_score, f1_score
                    )
                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "roc_auc": roc_auc_score(y_test, y_proba),
                        "precision": precision_score(y_test, y_pred, zero_division=0),
                        "recall": recall_score(y_test, y_pred, zero_division=0),
                        "f1_score": f1_score(y_test, y_pred, zero_division=0)
                    }
                
                for metric_name, metric_value in metrics.items():
                    iteration_metrics[metric_name].append(metric_value)
            
            # Aggregate results
            result_row = {
                'input_config': input_name,
                'discretization_method': disc_config['method'],
                'n_bins': disc_config['n_bins'],
                'discretization_cols': ', '.join(discretization_cols)
            }
            
            for i in range(total_iterations):
                result_row[f'accuracy_iter_{i+1}'] = iteration_metrics['accuracy'][i]
                result_row[f'auroc_iter_{i+1}'] = iteration_metrics['roc_auc'][i]
            
            result_row['accuracy_mean'] = np.mean(iteration_metrics['accuracy'])
            result_row['accuracy_std'] = np.std(iteration_metrics['accuracy'])
            result_row['auroc_mean'] = np.mean(iteration_metrics['roc_auc'])
            result_row['auroc_std'] = np.std(iteration_metrics['roc_auc'])
            result_row['f1_score_mean'] = np.mean(iteration_metrics['f1_score'])
            result_row['f1_score_std'] = np.std(iteration_metrics['f1_score'])
            result_row['precision_mean'] = np.mean(iteration_metrics['precision'])
            result_row['precision_std'] = np.std(iteration_metrics['precision'])
            result_row['recall_mean'] = np.mean(iteration_metrics['recall'])
            result_row['recall_std'] = np.std(iteration_metrics['recall'])
            
            results.append(result_row)
            
            print(f"  Accuracy: {result_row['accuracy_mean']:.4f} ± {result_row['accuracy_std']:.4f}")
            print(f"  AUROC:    {result_row['auroc_mean']:.4f} ± {result_row['auroc_std']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    config_cols = ['input_config', 'discretization_method', 'n_bins', 'discretization_cols']
    
    stat_cols = [
        'accuracy_mean', 'accuracy_std',
        'auroc_mean', 'auroc_std',
        'f1_score_mean', 'f1_score_std',
        'precision_mean', 'precision_std',
        'recall_mean', 'recall_std'
    ]
    
    accuracy_iter_cols = [col for col in results_df.columns if col.startswith('accuracy_iter_')]
    auroc_iter_cols = [col for col in results_df.columns if col.startswith('auroc_iter_')]
    
    ordered_cols = config_cols + stat_cols + accuracy_iter_cols + auroc_iter_cols
    results_df = results_df[ordered_cols]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n✓ Results saved to: {output_path}")
    
    return results_df

def run_bn_rskf_experiment(
    inputs_config: Dict,
    discretization_configs: List[Dict],
    structure_methods: List[str],
    cv_config: Dict,
    output_path: Path,
    custom_metrics_from_proba=None
) -> pd.DataFrame:
    """
    Run Bayesian Network experiment with RSKF for different configurations and structure learning methods.
    
    Args:
        inputs_config: Dictionary of input configurations
        discretization_configs: List of discretization configurations
        structure_methods: List of structure learning methods ('pc', 'hillclimb_bic', 'hillclimb_bdeu')
        cv_config: Cross-validation configuration
        output_path: Path to save results
        custom_metrics_from_proba: Optional custom metrics function
    
    Returns:
        DataFrame with results
    """
    from pgmpy.estimators import (
        PC, HillClimbSearch, BIC, BDeu,
        MaximumLikelihoodEstimator
    )
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    
    # Initialize RSKF
    rskf = RepeatedStratifiedKFold(
        n_splits=cv_config['n_splits'],
        n_repeats=cv_config['n_repeats'],
        random_state=cv_config['random_state']
    )
    
    total_iterations = cv_config['n_splits'] * cv_config['n_repeats']
    results = []
    
    for input_name, input_config in inputs_config.items():
        data = input_config['data']
        target_col = input_config['target_col']
        discretization_cols = input_config['discretization_cols']
        
        # Extract X and y for cross-validation splitting
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        for disc_config in discretization_configs:
            for structure_method in structure_methods:
                print(f"\nRunning: Input={input_name}, Discretization={disc_config}, Structure={structure_method}")
                
                iteration_metrics = {
                    'accuracy': [],
                    'roc_auc': [],
                    'precision': [],
                    'recall': [],
                    'f1_score': []
                }
                
                fold_idx = 0
                for train_idx, test_idx in rskf.split(X, y):
                    fold_idx += 1
                    
                    # Discretize the fold
                    X_train_disc, X_test_disc = discretize_cv_fold(
                        X=X,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        n_bins=disc_config['n_bins'],
                        strategy=disc_config['method'],
                        columns_to_discretize=discretization_cols
                    )
                    
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]
                    
                    # Reconstruct full DataFrames for BN
                    train_data_fold = pd.concat([X_train_disc, y_train], axis=1)
                    test_data_fold = pd.concat([X_test_disc, y_test], axis=1)
                    
                    # 1️⃣ Structure learning based on method
                    try:
                        if structure_method.lower() == 'pc':
                            estimator = PC(train_data_fold)
                            dag = estimator.estimate(
                                variant="stable",
                                significance_level=0.05
                            )
                        
                        if structure_method.lower() == 'hillclimb_k2':
                            hc = HillClimbSearch(train_data_fold)
                            dag = hc.estimate(
                                scoring_method='k2',
                                max_indegree=5,
                                max_iter=100000
                            )
                        
                        elif structure_method.lower() == 'hillclimb_bic':
                            bic = BIC(train_data_fold)
                            hc = HillClimbSearch(train_data_fold)
                            dag = hc.estimate(
                                scoring_method='bic-d',
                                max_indegree=5,
                                max_iter=100000
                            )
                        
                        elif structure_method.lower() == 'hillclimb_bdeu':
                            bdeu = BDeu(train_data_fold, equivalent_sample_size=10)
                            hc = HillClimbSearch(train_data_fold)
                            dag = hc.estimate(
                                scoring_method='bdeu',
                                max_indegree=5,
                                max_iter=100000
                            )
                        
                        else:
                            raise ValueError(f"Unknown structure method: {structure_method}")
                        
                        # 2️⃣ Wrap in DiscreteBayesianNetwork
                        model = DiscreteBayesianNetwork(dag.edges())
                        
                        # 3️⃣ Parameter learning
                        model.fit(train_data_fold, estimator=BayesianEstimator)
                        
                        # 4️⃣ Prediction - only use features that exist in the learned model
                        model_variables = set(model.nodes())
                        available_features = list(model_variables - {target_col})
                        
                        if len(available_features) == 0:
                            # Model didn't learn any structure, use prior probabilities
                            print(f"  Warning: No features in model for fold {fold_idx}, using priors")
                            # Use class prior as prediction
                            class_probs = y_train.value_counts(normalize=True)
                            y_pred = np.full(len(y_test), class_probs.idxmax())
                            y_proba = np.full(len(y_test), class_probs.max())
                        else:
                            X_test_fold = test_data_fold[available_features]
                            y_pred = model.predict(X_test_fold)
                            y_proba_df = model.predict_probability(X_test_fold)
                            
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
                        
                        # Calculate metrics
                        if custom_metrics_from_proba is not None:
                            metrics = custom_metrics_from_proba(y_test, y_proba)
                        else:
                            from sklearn.metrics import (
                                accuracy_score, roc_auc_score, precision_score,
                                recall_score, f1_score
                            )
                            metrics = {
                                "accuracy": accuracy_score(y_test, y_pred),
                                "roc_auc": roc_auc_score(y_test, y_proba),
                                "precision": precision_score(y_test, y_pred, zero_division=0),
                                "recall": recall_score(y_test, y_pred, zero_division=0),
                                "f1_score": f1_score(y_test, y_pred, zero_division=0)
                            }
                        
                        for metric_name, metric_value in metrics.items():
                            iteration_metrics[metric_name].append(metric_value)
                    
                    except Exception as e:
                        print(f"  Error in fold {fold_idx}: {str(e)}")
                        # Append NaN for this fold
                        for metric_name in iteration_metrics.keys():
                            iteration_metrics[metric_name].append(np.nan)
                
                # Aggregate results
                result_row = {
                    'input_config': input_name,
                    'structure_method': structure_method,
                    'discretization_method': disc_config['method'],
                    'n_bins': disc_config['n_bins'],
                    'discretization_cols': ', '.join(discretization_cols)
                }
                
                for i in range(total_iterations):
                    result_row[f'accuracy_iter_{i+1}'] = iteration_metrics['accuracy'][i]
                    result_row[f'auroc_iter_{i+1}'] = iteration_metrics['roc_auc'][i]
                
                result_row['accuracy_mean'] = np.nanmean(iteration_metrics['accuracy'])
                result_row['accuracy_std'] = np.nanstd(iteration_metrics['accuracy'])
                result_row['auroc_mean'] = np.nanmean(iteration_metrics['roc_auc'])
                result_row['auroc_std'] = np.nanstd(iteration_metrics['roc_auc'])
                result_row['f1_score_mean'] = np.nanmean(iteration_metrics['f1_score'])
                result_row['f1_score_std'] = np.nanstd(iteration_metrics['f1_score'])
                result_row['precision_mean'] = np.nanmean(iteration_metrics['precision'])
                result_row['precision_std'] = np.nanstd(iteration_metrics['precision'])
                result_row['recall_mean'] = np.nanmean(iteration_metrics['recall'])
                result_row['recall_std'] = np.nanstd(iteration_metrics['recall'])
                
                results.append(result_row)
                
                print(f"  Accuracy: {result_row['accuracy_mean']:.4f} ± {result_row['accuracy_std']:.4f}")
                print(f"  AUROC:    {result_row['auroc_mean']:.4f} ± {result_row['auroc_std']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Updated column order with structure_method
    config_cols = ['input_config', 'structure_method', 'discretization_method', 'n_bins', 'discretization_cols']
    
    stat_cols = [
        'accuracy_mean', 'accuracy_std',
        'auroc_mean', 'auroc_std',
        'f1_score_mean', 'f1_score_std',
        'precision_mean', 'precision_std',
        'recall_mean', 'recall_std'
    ]
    
    accuracy_iter_cols = [col for col in results_df.columns if col.startswith('accuracy_iter_')]
    auroc_iter_cols = [col for col in results_df.columns if col.startswith('auroc_iter_')]
    
    ordered_cols = config_cols + stat_cols + accuracy_iter_cols + auroc_iter_cols
    results_df = results_df[ordered_cols]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n✓ Results saved to: {output_path}")
    
    return results_df
