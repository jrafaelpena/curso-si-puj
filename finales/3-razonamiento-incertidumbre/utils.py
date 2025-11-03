# utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

def custom_reports_from_proba(y_true, y_proba):
    """
    Generate a classification report and return metrics as a DataFrame (binary classification).

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_proba : array-like
        Predicted probabilities for the positive class (shape: (n_samples,) or (n_samples, 2)).

    Returns
    -------
    metrics_df : pandas.DataFrame
        DataFrame with columns: accuracy, roc_auc, precision, recall, f1_score
    """
    
    # Handle case where y_proba has 2 columns
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]
    
    # Convert probabilities to class predictions
    y_pred = (y_proba >= 0.5).astype(int)
    
    print('* Classification Report:')
    print(classification_report(y_true, y_pred))
    
    print('* Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_proba)
    precision_value = precision_score(y_true, y_pred)
    recall_value = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f'\nAUROC: {auroc:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall_value:.4f}')
    print(f'Precision: {precision_value:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Create DataFrame
    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "roc_auc": auroc,
        "precision": precision_value,
        "recall": recall_value,
        "f1_score": f1
    }])
    
    return metrics_df

def custom_metrics_from_proba(y_true, y_proba):
    """
    Compute key classification metrics (binary classification) from predicted probabilities.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_proba : array-like
        Predicted probabilities for the positive class (shape: (n_samples,) or (n_samples, 2)).

    Returns
    -------
    metrics_dict : dict
        Dictionary with keys: accuracy, roc_auc, precision, recall, f1_score
    """
    
    # Handle case where y_proba has 2 columns
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]
    
    # Convert probabilities to class predictions
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Compute metrics
    metrics_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    
    return metrics_dict
