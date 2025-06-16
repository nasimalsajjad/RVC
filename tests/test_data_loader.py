"""
Tests for the RCV1 data loader.
"""
import pytest
import numpy as np
from scipy.sparse import csr_matrix
import os
import pandas as pd
from src.data.data_loader import RCV1DataLoader

@pytest.fixture
def data_loader():
    """Fixture to create a data loader instance."""
    return RCV1DataLoader()

@pytest.fixture
def dense_data_loader():
    """Fixture to create a data loader instance that converts to dense arrays."""
    return RCV1DataLoader(convert_to_dense=True)

def test_data_loader_initialization():
    """Test data loader initialization."""
    loader = RCV1DataLoader()
    assert loader.data is None
    assert loader.target is None
    assert not loader.convert_to_dense
    assert os.path.exists(loader.save_dir)

def test_load_data_sparse(data_loader):
    """Test loading the RCV1 dataset in sparse format."""
    data, target = data_loader.load_data()
    
    # Check data types
    assert isinstance(data, csr_matrix)
    assert isinstance(target, csr_matrix)
    
    # Check shapes
    assert data.shape[0] > 0
    assert target.shape[0] > 0
    assert data.shape[0] == target.shape[0]
    
    # Check if files were saved
    data_path = os.path.join(data_loader.save_dir, "rcv1_data.csv")
    target_path = os.path.join(data_loader.save_dir, "rcv1_target.csv")
    assert os.path.exists(data_path)
    assert os.path.exists(target_path)
    
    # Check saved data
    saved_data = pd.read_csv(data_path)
    saved_target = pd.read_csv(target_path)
    assert saved_data.shape[0] == data.shape[0]
    assert saved_target.shape[0] == target.shape[0]

def test_load_data_dense(dense_data_loader):
    """Test loading the RCV1 dataset in dense format."""
    data, target = dense_data_loader.load_data()
    
    # Check data types
    assert isinstance(data, np.ndarray)
    assert isinstance(target, np.ndarray)
    
    # Check shapes
    assert data.shape[0] > 0
    assert target.shape[0] > 0
    assert data.shape[0] == target.shape[0]
    
    # Check if files were saved
    data_path = os.path.join(dense_data_loader.save_dir, "rcv1_data.csv")
    target_path = os.path.join(dense_data_loader.save_dir, "rcv1_target.csv")
    assert os.path.exists(data_path)
    assert os.path.exists(target_path)

def test_get_sample(data_loader):
    """Test getting a sample from the dataset."""
    n_samples = 100
    data, target = data_loader.get_sample(n_samples=n_samples)
    
    # Check sample size
    assert data.shape[0] == n_samples
    assert target.shape[0] == n_samples
    
    # Check data type
    assert isinstance(data, csr_matrix)
    assert isinstance(target, csr_matrix)

def test_get_data_info(data_loader):
    """Test getting dataset information."""
    info = data_loader.get_data_info()
    
    # Check info structure
    assert 'n_samples' in info
    assert 'n_features' in info
    assert 'n_classes' in info
    assert 'feature_names' in info
    assert 'target_names' in info
    assert 'is_sparse' in info
    assert 'data_path' in info
    assert 'target_path' in info
    
    # Check values
    assert info['n_samples'] > 0
    assert info['n_features'] > 0
    assert info['n_classes'] > 0
    assert len(info['feature_names']) == info['n_features']
    assert len(info['target_names']) == info['n_classes']
    assert info['is_sparse'] is True
    
    # Check if paths exist
    assert os.path.exists(info['data_path'])
    assert os.path.exists(info['target_path']) 