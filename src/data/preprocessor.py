"""
Data preprocessing pipeline.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class DataPreprocessor:
    def __init__(self, scaler: Optional[StandardScaler] = None):
        """
        Initialize the preprocessor.
        
        Args:
            scaler: Optional scaler instance
        """
        self.scaler = scaler or StandardScaler()
        
    def preprocess(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data.
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Processed features and target
        """
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        return X_scaled, y 