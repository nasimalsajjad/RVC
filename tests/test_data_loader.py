"""
Tests for the RCV1 data loader.
"""
import pytest
import os
from src.data.data_loader import RCV1DataLoader

@pytest.fixture
def data_loader():
    """Fixture to create a data loader instance with mock data."""
    return RCV1DataLoader(use_mock=True)

def test_data_download(data_loader):
    """Test if RCV1 dataset is loaded successfully."""
    data, target = data_loader.load_data()
    
    # Check if data is loaded
    assert data is not None
    assert target is not None
    
    # Check if data files exist
    data_path = os.path.join(data_loader.save_dir, "rcv1_data.npz")
    target_path = os.path.join(data_loader.save_dir, "rcv1_target.npz")
    assert os.path.exists(data_path)
    assert os.path.exists(target_path) 