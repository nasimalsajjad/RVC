"""
Model evaluation utilities.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Union

def evaluate_model(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def get_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> pd.DataFrame:
    """
    Get confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        pd.DataFrame: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=['True'],
        columns=['Predicted']
    ) 