"""
Helper utilities for the project.
"""
import os
import yaml
import logging
from typing import Dict, Any

def setup_logging(log_file: str = 'pipeline.log'):
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(directory: str):
    """
    Ensure directory exists.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory) 