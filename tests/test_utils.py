import unittest
import tempfile
import os
import shutil
import logging
import yaml
from unittest.mock import patch, MagicMock

# Import the utilities to test
import sys
sys.path.append('..')
from utils.logging_config import setup_logging, get_logger, PerformanceLogger, log_function_call, log_data_info
from utils.config_manager import ConfigManager


class TestLoggingConfig(unittest.TestCase):
    """Test cases for logging configuration."""
    
    def setUp(self):
        """Set up test directory."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test directory."""
        try:
            import gc
            gc.collect()
            import time
            time.sleep(0.1)
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_setup_logging_console_only(self):
        """Test logging setup with console output only."""
        logger = setup_logging(
            log_level=logging.INFO,
            log_dir=self.temp_dir,
            console_output=True,
            file_output=False
        )
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.INFO)
        
        # Test that we can log messages
        logger.info("Test message")
    
    def test_setup_logging_file_output(self):
        """Test logging setup with file output."""
        logger = setup_logging(
            log_level=logging.DEBUG,
            log_dir=self.temp_dir,
            console_output=False,
            file_output=True
        )
        
        self.assertIsInstance(logger, logging.Logger)
        
        # Test file creation
        log_file = os.path.join(self.temp_dir, 'predictive_maintenance.log')
        logger.info("Test message")
        
        # Check if log file was created
        self.assertTrue(os.path.exists(log_file))
        
        # Check if message was written
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test message", content)
    
    def test_get_logger(self):
        """Test getting logger instances."""
        # Test main logger
        main_logger = get_logger()
        self.assertIsInstance(main_logger, logging.Logger)
        
        # Test named logger
        named_logger = get_logger('test_module')
        self.assertIsInstance(named_logger, logging.Logger)
        self.assertIn('test_module', named_logger.name)
    
    def test_performance_logger(self):
        """Test performance logger context manager."""
        logger = setup_logging(
            log_level=logging.INFO,
            log_dir=self.temp_dir,
            console_output=False,
            file_output=True
        )
        
        with PerformanceLogger("test_operation", logger):
            # Simulate some work
            import time
            time.sleep(0.1)
        
        # Check if performance log was created
        perf_log_file = os.path.join(self.temp_dir, 'predictive_maintenance_performance.log')
        self.assertTrue(os.path.exists(perf_log_file))
        
        with open(perf_log_file, 'r') as f:
            content = f.read()
            self.assertIn("test_operation", content)
            self.assertIn("Completed operation", content)
    
    def test_log_function_call_decorator(self):
        """Test function call logging decorator."""
        logger = setup_logging(
            log_level=logging.DEBUG,
            log_dir=self.temp_dir,
            console_output=False,
            file_output=True
        )
        
        @log_function_call
        def test_function(x, y=10):
            return x + y
        
        result = test_function(5, y=15)
        self.assertEqual(result, 20)
        
        # Check if function call was logged
        log_file = os.path.join(self.temp_dir, 'predictive_maintenance.log')
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertIn("test_function", content)
    
    def test_log_data_info(self):
        """Test data information logging."""
        logger = setup_logging(
            log_level=logging.INFO,
            log_dir=self.temp_dir,
            console_output=False,
            file_output=True
        )
        
        import pandas as pd
        import numpy as np
        
        # Test with DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3, np.nan],
            'B': [4, 5, 6, 7]
        })
        
        log_data_info(df, "test_data", logger)
        
        # Check if data info was logged
        log_file = os.path.join(self.temp_dir, 'predictive_maintenance.log')
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertIn("test_data shape", content)
            self.assertIn("test_data missing values", content)


class TestConfigManager(unittest.TestCase):
    """Test cases for configuration manager."""
    
    def setUp(self):
        """Set up test directory and config file."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create test config
        self.test_config = {
            'data': {
                'dataset_ids': ['FD001', 'FD002'],
                'data_dir': 'test_data',
                'validation_size': 0.2
            },
            'model': {
                'type': 'tft',
                'hidden_size': 64
            },
            'training': {
                'batch_size': 32,
                'max_epochs': 10
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
                'console_output': True,
                'file_output': True
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test directory."""
        try:
            import gc
            gc.collect()
            import time
            time.sleep(0.1)
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_config_loading(self):
        """Test configuration loading from file."""
        config = ConfigManager(self.config_path)
        
        # Test getting values
        dataset_ids = config.get('data.dataset_ids')
        self.assertEqual(dataset_ids, ['FD001', 'FD002'])
        
        hidden_size = config.get('model.hidden_size')
        self.assertEqual(hidden_size, 64)
        
        batch_size = config.get('training.batch_size')
        self.assertEqual(batch_size, 32)
    
    def test_config_defaults(self):
        """Test configuration with default values."""
        # Test with non-existent file
        config = ConfigManager('non_existent_file.yaml')
        
        # Should load default config
        dataset_ids = config.get('data.dataset_ids')
        self.assertIsNotNone(dataset_ids)
        self.assertIsInstance(dataset_ids, list)
    
    def test_config_setting(self):
        """Test setting configuration values."""
        config = ConfigManager(self.config_path)
        
        # Set a new value
        config.set('data.new_param', 'test_value')
        
        # Get the value back
        value = config.get('data.new_param')
        self.assertEqual(value, 'test_value')
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ConfigManager(self.config_path)
        
        # Should pass validation
        self.assertTrue(config.validate_config())
    
    def test_config_saving(self):
        """Test saving configuration."""
        config = ConfigManager(self.config_path)
        
        # Modify config
        config.set('data.new_param', 'test_value')
        
        # Save to new file
        new_config_path = os.path.join(self.temp_dir, 'new_config.yaml')
        config.save_config(new_config_path)
        
        # Load the new config
        new_config = ConfigManager(new_config_path)
        value = new_config.get('data.new_param')
        self.assertEqual(value, 'test_value')
    
    def test_get_model_config(self):
        """Test getting model-specific configuration."""
        config = ConfigManager(self.config_path)
        
        model_config = config.get_model_config('tft')
        self.assertIn('model_type', model_config)
        self.assertEqual(model_config['model_type'], 'tft')
    
    def test_get_training_config(self):
        """Test getting training configuration."""
        config = ConfigManager(self.config_path)
        
        training_config = config.get_training_config()
        self.assertIn('batch_size', training_config)
        self.assertEqual(training_config['batch_size'], 32)
    
    def test_get_data_config(self):
        """Test getting data configuration."""
        config = ConfigManager(self.config_path)
        
        data_config = config.get_data_config()
        self.assertIn('dataset_ids', data_config)
        self.assertEqual(data_config['dataset_ids'], ['FD001', 'FD002'])
    
    def test_create_directories(self):
        """Test directory creation."""
        config = ConfigManager(self.config_path)
        
        # Set some directory paths
        config.set('logging.log_dir', os.path.join(self.temp_dir, 'logs'))
        config.set('evaluation.plot_dir', os.path.join(self.temp_dir, 'plots'))
        
        # Create directories
        config.create_directories()
        
        # Check if directories were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'logs')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'plots')))


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration and logging."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config
        self.config_data = {
            'logging': {
                'level': 'INFO',
                'log_dir': self.temp_dir,
                'console_output': True,
                'file_output': True
            },
            'data': {
                'dataset_ids': ['FD001'],
                'validation_size': 0.2
            }
        }
        
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        try:
            import gc
            gc.collect()
            import time
            time.sleep(0.1)
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_config_logging_integration(self):
        """Test integration between config and logging."""
        # Load config
        config = ConfigManager(self.config_path)
        
        # Setup logging using config
        log_config = config.get('logging')
        logger = setup_logging(
            log_level=getattr(logging, log_config['level']),
            log_dir=log_config['log_dir'],
            console_output=log_config['console_output'],
            file_output=log_config['file_output']
        )
        
        # Test logging
        logger.info("Integration test message")
        
        # Check if log file was created
        log_file = os.path.join(self.temp_dir, 'predictive_maintenance.log')
        self.assertTrue(os.path.exists(log_file))


if __name__ == '__main__':
    unittest.main() 