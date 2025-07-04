#!/usr/bin/env python3
"""
Simple test script to verify the predictive maintenance system is working.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_data_files():
    """Test that data files exist and are readable."""
    print("Testing data files...")
    
    # Check if processed data exists
    required_files = ['processed_train.csv', 'processed_val.csv', 'processed_test.csv']
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
            # Try to read a small sample
            try:
                df = pd.read_csv(file, nrows=5)
                print(f"  - Shape: {df.shape}")
                print(f"  - Columns: {list(df.columns)}")
            except Exception as e:
                print(f"  ✗ Error reading {file}: {e}")
        else:
            print(f"✗ {file} missing")
    
    # Check if transformer data exists
    transformer_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
    transformer_dir = 'transformer_data'
    
    if os.path.exists(transformer_dir):
        print(f"✓ {transformer_dir} directory exists")
        for file in transformer_files:
            file_path = os.path.join(transformer_dir, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file} exists")
                try:
                    data = np.load(file_path, mmap_mode='r')
                    print(f"    - Shape: {data.shape}")
                except Exception as e:
                    print(f"    ✗ Error loading {file}: {e}")
            else:
                print(f"  ✗ {file} missing")
    else:
        print(f"✗ {transformer_dir} directory missing")

def test_imports():
    """Test that all modules can be imported."""
    print("\nTesting imports...")
    
    try:
        from preprocess_data import load_dataset, calculate_rul, normalize_data
        print("✓ preprocess_data module imported successfully")
    except Exception as e:
        print(f"✗ Error importing preprocess_data: {e}")
    
    try:
        from utils.logging_config import get_logger, setup_logging
        print("✓ logging_config module imported successfully")
    except Exception as e:
        print(f"✗ Error importing logging_config: {e}")
    
    try:
        from utils.config_manager import ConfigManager
        print("✓ config_manager module imported successfully")
    except Exception as e:
        print(f"✗ Error importing config_manager: {e}")
    
    try:
        from models.tft_data_module import CMAPSSDataModule
        print("✓ tft_data_module imported successfully")
    except Exception as e:
        print(f"✗ Error importing tft_data_module: {e}")
    
    try:
        from models.simple_lstm_model import SimpleLSTM
        print("✓ simple_lstm_model imported successfully")
    except Exception as e:
        print(f"✗ Error importing simple_lstm_model: {e}")

def test_logging():
    """Test logging functionality."""
    print("\nTesting logging...")
    
    try:
        from utils.logging_config import setup_logging, get_logger
        logger = setup_logging(log_level='INFO', console_output=True, file_output=False)
        logger.info("Test log message")
        print("✓ Logging system working")
    except Exception as e:
        print(f"✗ Error with logging: {e}")

def test_config():
    """Test configuration management."""
    print("\nTesting configuration...")
    
    try:
        from utils.config_manager import ConfigManager
        config = ConfigManager()
        
        # Test getting config values
        dataset_ids = config.get('data.dataset_ids')
        if dataset_ids:
            print(f"✓ Configuration loaded: {len(dataset_ids)} datasets")
        else:
            print("✗ No dataset IDs found in config")
        
        # Test validation
        if config.validate_config():
            print("✓ Configuration validation passed")
        else:
            print("✗ Configuration validation failed")
            
    except Exception as e:
        print(f"✗ Error with configuration: {e}")

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from models.simple_lstm_model import SimpleLSTM
        model = SimpleLSTM(input_size=24, hidden_size=32, learning_rate=0.001)
        print("✓ Simple LSTM model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error creating Simple LSTM model: {e}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE SYSTEM - PROJECT TEST")
    print("=" * 60)
    
    test_data_files()
    test_imports()
    test_logging()
    test_config()
    test_model_creation()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("If you see mostly ✓ marks above, the project is working correctly!")
    print("If you see ✗ marks, there may be issues that need to be addressed.")
    print("\nNext steps:")
    print("1. Run: python models/run_pipeline.py --max_epochs 10 --batch_size 32")
    print("2. Check checkpoints/ directory for saved models")
    print("3. Check evaluation_plots/ directory for results")
    print("4. Run: python -m pytest tests/ -v (for unit tests)")

if __name__ == "__main__":
    main() 