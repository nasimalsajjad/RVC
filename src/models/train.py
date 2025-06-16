"""
Model training pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from typing import Tuple, Dict, Any

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a machine learning model.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        model: Model instance
        test_size (float): Test set size
        random_state (int): Random seed
        
    Returns:
        Tuple[Any, Dict[str, float]]: Trained model and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return model, metrics

def save_model(model: Any, path: str):
    """
    Save model to disk.
    
    Args:
        model: Trained model
        path (str): Path to save model
    """
    joblib.dump(model, path) 