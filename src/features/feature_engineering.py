"""
Feature engineering pipeline.
"""
import pandas as pd
import numpy as np
from typing import List, Optional

class FeatureEngineer:
    def __init__(self, categorical_cols: Optional[List[str]] = None):
        """
        Initialize the feature engineer.
        
        Args:
            categorical_cols: List of categorical column names
        """
        self.categorical_cols = categorical_cols or []
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from the input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Engineered features
        """
        df = data.copy()
        
        # Handle categorical features
        for col in self.categorical_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col)
        
        return df 