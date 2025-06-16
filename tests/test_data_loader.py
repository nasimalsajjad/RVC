"""
Tests for the RCV1 data loader.
"""
import pytest
import os
from src.data.data_loader import RCV1DataLoader

@pytest.fixture
def data_loader():
    """Fixture to create a data loader instance."""
    return RCV1DataLoader()

def test_data_download():
    """Test if RCV1 dataset is downloaded successfully."""
    loader = RCV1DataLoader()
    data, target = loader.load_data()
    
    # Check if data is loaded
    assert data is not None
    assert target is not None
    
    # Check if data files exist
    data_path = os.path.join(loader.save_dir, "rcv1_data.npz")
    target_path = os.path.join(loader.save_dir, "rcv1_target.npz")
    assert os.path.exists(data_path)
    assert os.path.exists(target_path) 