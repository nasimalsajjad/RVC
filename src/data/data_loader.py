"""
Data loading utilities for the RCV1 dataset.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_rcv1
from typing import Tuple, Dict, Any, Union
from scipy.sparse import csr_matrix, save_npz
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class RCV1DataLoader:
    """Loader for the RCV1 dataset."""
    
    def __init__(self, cache_dir: str = None, convert_to_dense: bool = False, save_dir: str = "data/raw", use_mock: bool = False):
        """
        Initialize the RCV1 data loader.
        
        Args:
            cache_dir (str, optional): Directory to cache the dataset. Defaults to None.
            convert_to_dense (bool, optional): Whether to convert sparse matrix to dense array. Defaults to False.
            save_dir (str, optional): Directory to save the data. Defaults to "data/raw".
            use_mock (bool, optional): Whether to use mock data for testing. Defaults to False.
        """
        self.cache_dir = cache_dir
        self.convert_to_dense = convert_to_dense
        self.save_dir = save_dir
        self.use_mock = use_mock
        self.data = None
        self.target = None
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _create_mock_data(self) -> Tuple[csr_matrix, csr_matrix]:
        """Create mock data for testing."""
        n_samples = 100
        n_features = 100
        n_classes = 10
        
        # Create sparse matrices
        data = csr_matrix((n_samples, n_features))
        target = csr_matrix((n_samples, n_classes))
        
        # Add some random non-zero elements
        data[0, 0] = 1
        target[0, 0] = 1
        
        return data, target
        
    def load_data(self) -> Tuple[Union[np.ndarray, csr_matrix], Union[np.ndarray, csr_matrix]]:
        """
        Load the RCV1 dataset and save to files.
        
        Returns:
            Tuple[Union[np.ndarray, csr_matrix], Union[np.ndarray, csr_matrix]]: Features and target data
        """
        try:
            if self.use_mock:
                logger.info("Using mock data for testing...")
                self.data, self.target = self._create_mock_data()
            else:
                logger.info("Loading RCV1 dataset...")
                rcv1 = fetch_rcv1(data_home=self.cache_dir)
                self.data = rcv1.data
                self.target = rcv1.target
            
            if self.convert_to_dense:
                logger.info("Converting sparse matrix to dense array...")
                self.data = self.data.toarray()
                self.target = self.target.toarray()
            
            # Save data to files
            self._save_data()
                
            logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
            return self.data, self.target
        except Exception as e:
            logger.error(f"Error loading RCV1 dataset: {str(e)}")
            raise
            
    def _save_data(self):
        """Save data and target to files."""
        try:
            data_path = os.path.join(self.save_dir, "rcv1_data.npz")
            target_path = os.path.join(self.save_dir, "rcv1_target.npz")
            
            if isinstance(self.data, csr_matrix):
                # Save sparse matrices using scipy's save_npz
                save_npz(data_path, self.data)
                save_npz(target_path, self.target)
            else:
                # Save dense arrays using numpy's save
                np.save(data_path, self.data)
                np.save(target_path, self.target)
            
            logger.info(f"Data saved to {data_path}")
            logger.info(f"Target saved to {target_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
            
    def get_sample(self, n_samples: int = 1000) -> Tuple[Union[np.ndarray, csr_matrix], Union[np.ndarray, csr_matrix]]:
        """
        Get a random sample from the dataset.
        
        Args:
            n_samples (int): Number of samples to return
            
        Returns:
            Tuple[Union[np.ndarray, csr_matrix], Union[np.ndarray, csr_matrix]]: Sampled features and target data
        """
        if self.data is None:
            self.load_data()
            
        indices = np.random.choice(self.data.shape[0], n_samples, replace=False)
        return self.data[indices], self.target[indices]
        
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        if self.data is None:
            self.load_data()
            
        return {
            'n_samples': self.data.shape[0],
            'n_features': self.data.shape[1],
            'n_classes': self.target.shape[1],
            'feature_names': [f'feature_{i}' for i in range(self.data.shape[1])],
            'target_names': [f'class_{i}' for i in range(self.target.shape[1])],
            'is_sparse': isinstance(self.data, csr_matrix),
            'data_path': os.path.join(self.save_dir, "rcv1_data.npz"),
            'target_path': os.path.join(self.save_dir, "rcv1_target.npz")
        } 