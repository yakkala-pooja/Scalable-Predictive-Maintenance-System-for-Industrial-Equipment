import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from logging_config import get_logger


class ConfigManager:
    """Configuration manager for the predictive maintenance system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.logger = get_logger('config')
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                self.logger.warning(f"Config file not found at {self.config_path}, using defaults")
                self.config = self.get_default_config()
                return
            
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.logger.info("Using default configuration")
            self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'dataset_ids': ['FD001', 'FD002', 'FD003', 'FD004'],
                'data_dir': 'data',
                'validation_size': 0.2,
                'random_seed': 42,
                'window_size': 30,
                'horizon': 1
            },
            'preprocessing': {
                'normalization_method': 'minmax',
                'handle_missing_values': True,
                'missing_value_strategy': 'interpolate',
                'add_engineered_features': True,
                'feature_engineering': {
                    'rolling_stats': True,
                    'rolling_window': 5,
                    'lag_features': True,
                    'lag_periods': [1, 2, 3],
                    'diff_features': True
                }
            },
            'model': {
                'type': 'tft',
                'tft': {
                    'hidden_size': 64,
                    'attention_head_size': 4,
                    'dropout': 0.1,
                    'num_lstm_layers': 2,
                    'num_attention_heads': 4,
                    'learning_rate': 0.001,
                    'context_size': 64
                },
                'lstm': {
                    'hidden_size': 64,
                    'num_layers': 1,
                    'dropout': 0.1,
                    'learning_rate': 0.001
                }
            },
            'training': {
                'batch_size': 64,
                'max_epochs': 50,
                'patience': 10,
                'lr_scheduler': {
                    'type': 'ReduceLROnPlateau',
                    'factor': 0.5,
                    'patience': 3,
                    'min_lr': 1e-6
                },
                'gradient_clip_val': 1.0,
                'use_mixed_precision': True,
                'num_workers': 4,
                'pin_memory': True
            },
            'hardware': {
                'num_gpus': None,
                'strategy': 'ddp',
                'deterministic': False
            },
            'parallel': {
                'n_workers': None,
                'use_dask': True,
                'dask': {
                    'memory_limit': '2GB',
                    'threads_per_worker': 1,
                    'local_cluster': True
                }
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
                'log_file_prefix': 'predictive_maintenance',
                'console_output': True,
                'file_output': True,
                'max_file_size': 10485760,
                'backup_count': 5
            },
            'evaluation': {
                'metrics': ['mae', 'rmse', 'mape', 'r2', 'precision_25'],
                'generate_plots': True,
                'plot_dir': 'evaluation_plots',
                'save_predictions': True,
                'prediction_format': 'csv'
            },
            'feature_importance': {
                'enabled': True,
                'method': 'attention',
                'n_samples': 1000,
                'generate_plots': True
            },
            'persistence': {
                'checkpoint_dir': 'checkpoints',
                'save_top_k': 3,
                'save_artifacts': True,
                'artifacts_dir': 'artifacts',
                'artifacts': ['model', 'scaler', 'config', 'metadata']
            },
            'monitoring': {
                'enabled': True,
                'metrics': ['train_loss', 'val_loss', 'train_mae', 'val_mae', 'learning_rate', 'epoch'],
                'interval': 1,
                'tensorboard': True,
                'tensorboard_dir': 'lightning_logs'
            },
            'performance': {
                'profiling': False,
                'profiling_dir': 'profiling',
                'memory_profiling': False,
                'gpu_memory_profiling': False
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.dataset_ids')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.dataset_ids')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        self.logger.debug(f"Configuration updated: {key} = {value}")
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration (uses original path if None)
        """
        save_path = path or self.config_path
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
    
    def validate_config(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate required sections
            required_sections = ['data', 'model', 'training', 'logging']
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate data configuration
            data_config = self.config.get('data', {})
            if not data_config.get('dataset_ids'):
                self.logger.error("No dataset IDs specified")
                return False
            
            # Validate model configuration
            model_config = self.config.get('model', {})
            if not model_config.get('type'):
                self.logger.error("No model type specified")
                return False
            
            # Validate training configuration
            training_config = self.config.get('training', {})
            if training_config.get('batch_size', 0) <= 0:
                self.logger.error("Invalid batch size")
                return False
            
            if training_config.get('max_epochs', 0) <= 0:
                self.logger.error("Invalid max epochs")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def get_model_config(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific model type.
        
        Args:
            model_type: Model type (uses config default if None)
        
        Returns:
            Model configuration dictionary
        """
        model_type = model_type or self.get('model.type', 'tft')
        model_config = self.get(f'model.{model_type}', {})
        
        # Add common model parameters
        model_config.update({
            'model_type': model_type,
            'learning_rate': self.get(f'model.{model_type}.learning_rate', 0.001)
        })
        
        return model_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration.
        
        Returns:
            Training configuration dictionary
        """
        return self.config.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration.
        
        Returns:
            Data configuration dictionary
        """
        return self.config.get('data', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration.
        
        Returns:
            Preprocessing configuration dictionary
        """
        return self.config.get('preprocessing', {})
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        directories = [
            self.get('logging.log_dir', 'logs'),
            self.get('evaluation.plot_dir', 'evaluation_plots'),
            self.get('persistence.checkpoint_dir', 'checkpoints'),
            self.get('persistence.artifacts_dir', 'artifacts'),
            self.get('monitoring.tensorboard_dir', 'lightning_logs'),
            self.get('performance.profiling_dir', 'profiling')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")


# Global configuration instance
config_manager = ConfigManager() 