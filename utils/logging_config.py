import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_level=logging.INFO,
    log_dir="logs",
    log_file_prefix="predictive_maintenance",
    console_output=True,
    file_output=True,
    max_file_size=10*1024*1024,  # 10MB
    backup_count=5
):
    """
    Set up comprehensive logging configuration for the predictive maintenance system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        log_file_prefix: Prefix for log file names
        console_output: Whether to output logs to console
        file_output: Whether to output logs to files
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if file_output:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('predictive_maintenance')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handlers
    if file_output:
        # Main log file with rotation
        main_log_file = os.path.join(log_dir, f"{log_file_prefix}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error log file (only errors and critical)
        error_log_file = os.path.join(log_dir, f"{log_file_prefix}_errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # Performance log file (for timing and performance metrics)
        perf_log_file = os.path.join(log_dir, f"{log_file_prefix}_performance.log")
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(detailed_formatter)
        logger.addHandler(perf_handler)
    
    return logger


def get_logger(name=None):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (if None, returns the main logger)
    
    Returns:
        logger: Logger instance
    """
    if name is None:
        return logging.getLogger('predictive_maintenance')
    else:
        return logging.getLogger(f'predictive_maintenance.{name}')


class PerformanceLogger:
    """Context manager for logging performance metrics."""
    
    def __init__(self, operation_name, logger=None):
        self.operation_name = operation_name
        self.logger = logger or get_logger('performance')
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.operation_name} in {duration:.2f} seconds")
        else:
            self.logger.error(f"Failed operation: {self.operation_name} after {duration:.2f} seconds - {exc_type}: {exc_val}")
        
        return False  # Don't suppress exceptions


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time."""
    def wrapper(*args, **kwargs):
        logger = get_logger('function_calls')
        
        # Log function call
        func_name = func.__name__
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
        
        # Execute function with timing
        with PerformanceLogger(func_name, logger):
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func_name} returned successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func_name} failed with error: {str(e)}")
                raise
    
    return wrapper


def log_data_info(data, name="data", logger=None):
    """
    Log information about a dataset.
    
    Args:
        data: Dataset to log information about (DataFrame, array, etc.)
        name: Name of the dataset
        logger: Logger instance to use
    """
    if logger is None:
        logger = get_logger('data_info')
    
    if hasattr(data, 'shape'):
        logger.info(f"{name} shape: {data.shape}")
    
    if hasattr(data, 'dtype'):
        logger.info(f"{name} dtype: {data.dtype}")
    
    if hasattr(data, 'columns'):
        logger.info(f"{name} columns: {list(data.columns)}")
    
    if hasattr(data, 'isnull') and hasattr(data.isnull(), 'sum'):
        missing_values = data.isnull().sum().sum()
        logger.info(f"{name} missing values: {missing_values}")
    
    if hasattr(data, 'memory_usage'):
        memory_usage = data.memory_usage(deep=True).sum()
        logger.info(f"{name} memory usage: {memory_usage / 1024 / 1024:.2f} MB")


# Initialize default logging configuration
if not logging.getLogger('predictive_maintenance').handlers:
    setup_logging() 