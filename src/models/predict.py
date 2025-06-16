"""
Model prediction utilities.
"""
import pandas as pd
import joblib
from typing import Union, List, Dict, Any

def load_model(path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        path (str): Path to the model file
        
    Returns:
        Any: Loaded model
    """
    return joblib.load(path)

def predict(model: Any, data: pd.DataFrame) -> Union[List, pd.Series]:
    """
    Make predictions using the model.
    
    Args:
        model: Trained model
        data (pd.DataFrame): Input data
        
    Returns:
        Union[List, pd.Series]: Model predictions
    """
    return model.predict(data)

def predict_proba(model: Any, data: pd.DataFrame) -> pd.DataFrame:
    """
    Get prediction probabilities.
    
    Args:
        model: Trained model
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Prediction probabilities
    """
    return pd.DataFrame(
        model.predict_proba(data),
        columns=model.classes_
    ) 