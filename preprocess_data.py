import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('utils')
from logging_config import get_logger, PerformanceLogger, log_function_call, log_data_info

@log_function_call
def load_dataset(dataset_id, data_dir='data'):
    """
    Load a single dataset (train, test, and RUL files)
    
    Args:
        dataset_id (str): Dataset ID (e.g., 'FD001')
        data_dir (str): Directory containing the data files
    
    Returns:
        train_df, test_df, rul_df: Pandas DataFrames for train, test, and RUL data
    
    Raises:
        FileNotFoundError: If any of the required data files are not found
        ValueError: If the data files are empty or corrupted
    """
    logger = get_logger('data_loading')
    
    # Column names based on the readme file
    cols = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
           [f'sensor_{i}' for i in range(1, 22)]
    
    try:
        # Load training data
        train_path = os.path.join(data_dir, f'train_{dataset_id}.txt')
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data file not found: {train_path}")
        
        train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=cols)
        if train_df.empty:
            raise ValueError(f"Training data file is empty: {train_path}")
        
        # Load test data
        test_path = os.path.join(data_dir, f'test_{dataset_id}.txt')
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data file not found: {test_path}")
        
        test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=cols)
        if test_df.empty:
            raise ValueError(f"Test data file is empty: {test_path}")
        
        # Load RUL data for test set
        rul_path = os.path.join(data_dir, f'RUL_{dataset_id}.txt')
        if not os.path.exists(rul_path):
            raise FileNotFoundError(f"RUL data file not found: {rul_path}")
        
        rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
        if rul_df.empty:
            raise ValueError(f"RUL data file is empty: {rul_path}")
        
        # Log data information
        log_data_info(train_df, f"train_{dataset_id}", logger)
        log_data_info(test_df, f"test_{dataset_id}", logger)
        log_data_info(rul_df, f"rul_{dataset_id}", logger)
        
        logger.info(f"Successfully loaded dataset {dataset_id}")
        return train_df, test_df, rul_df
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        raise

def calculate_rul(df):
    """
    Calculate Remaining Useful Life for each unit in the dataset
    
    Args:
        df (pd.DataFrame): DataFrame with 'unit' and 'cycle' columns
    
    Returns:
        pd.DataFrame: DataFrame with added 'RUL' column
    """
    # Group by unit and calculate max cycle for each unit
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    
    # Merge with original dataframe
    df = df.merge(max_cycles, on='unit', how='left')
    
    # Calculate RUL as the difference between max cycle and current cycle
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    # Drop the max_cycle column as it's no longer needed
    df = df.drop('max_cycle', axis=1)
    
    return df

def normalize_data(train_df, test_df):
    """
    Normalize sensor readings using MinMaxScaler
    
    Args:
        train_df (pd.DataFrame): Training DataFrame
        test_df (pd.DataFrame): Testing DataFrame
    
    Returns:
        train_df, test_df: Normalized DataFrames
        scaler: Fitted scaler for future use
    """
    # Select sensor columns and operational settings for normalization
    sensor_cols = [col for col in train_df.columns if 'sensor' in col or 'op_setting' in col]
    
    # Initialize and fit scaler on training data
    scaler = MinMaxScaler()
    scaler.fit(train_df[sensor_cols])
    
    # Transform training and test data
    train_df[sensor_cols] = scaler.transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
    
    return train_df, test_df, scaler

def prepare_cmapss_data(dataset_ids=None, val_size=0.2, random_state=42):
    """
    Load and preprocess the NASA CMAPSS dataset
    
    Args:
        dataset_ids (list): List of dataset IDs to load (e.g., ['FD001', 'FD002'])
                           If None, all datasets are loaded
        val_size (float): Validation set size as a fraction of training data
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing train, validation, and test DataFrames
    """
    if dataset_ids is None:
        dataset_ids = ['FD001', 'FD002', 'FD003', 'FD004']
    
    all_train_dfs = []
    all_test_dfs = []
    all_rul_dfs = []
    
    # Load and combine all specified datasets
    for dataset_id in dataset_ids:
        print(f"Loading dataset {dataset_id}...")
        train_df, test_df, rul_df = load_dataset(dataset_id)
        
        # Add dataset_id as a feature
        train_df['dataset_id'] = dataset_id
        test_df['dataset_id'] = dataset_id
        
        all_train_dfs.append(train_df)
        all_test_dfs.append(test_df)
        all_rul_dfs.append(rul_df)
    
    # Concatenate all datasets
    train_df = pd.concat(all_train_dfs, ignore_index=True)
    test_df = pd.concat(all_test_dfs, ignore_index=True)
    rul_df = pd.concat(all_rul_dfs, ignore_index=True)
    
    # Calculate RUL for training data
    print("Calculating RUL for training data...")
    train_df = calculate_rul(train_df)
    
    # Add RUL to test data
    # For test data, we need to use the provided RUL values
    # First, get the max cycle for each unit in test data
    test_max_cycles = test_df.groupby('unit')['cycle'].max().reset_index()
    test_max_cycles.columns = ['unit', 'max_cycle']
    
    # Create a DataFrame with unit and RUL from the provided RUL values
    test_rul_df = pd.DataFrame({
        'unit': range(1, len(rul_df) + 1),
        'RUL_at_end': rul_df['RUL'].values
    })
    
    # Merge with max cycles
    test_max_cycles = test_max_cycles.merge(test_rul_df, on='unit', how='left')
    
    # Merge with test data
    test_df = test_df.merge(test_max_cycles, on='unit', how='left')
    
    # Calculate RUL for each row in test data
    test_df['RUL'] = test_df['RUL_at_end'] + (test_df['max_cycle'] - test_df['cycle'])
    
    # Drop temporary columns
    test_df = test_df.drop(['max_cycle', 'RUL_at_end'], axis=1)
    
    # Normalize data
    print("Normalizing data...")
    train_df, test_df, scaler = normalize_data(train_df, test_df)
    
    # Split training data into train and validation sets
    print(f"Splitting training data with validation size {val_size}...")
    train_units = train_df['unit'].unique()
    val_units = np.random.choice(train_units, size=int(len(train_units) * val_size), replace=False)
    
    val_df = train_df[train_df['unit'].isin(val_units)]
    train_df = train_df[~train_df['unit'].isin(val_units)]
    
    # Check for missing values
    print("Checking for missing values...")
    print(f"Train missing values: {train_df.isnull().sum().sum()}")
    print(f"Validation missing values: {val_df.isnull().sum().sum()}")
    print(f"Test missing values: {test_df.isnull().sum().sum()}")
    
    # Return processed data
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'scaler': scaler
    }

if __name__ == "__main__":
    # Example usage
    data = prepare_cmapss_data()
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Training set shape: {data['train'].shape}")
    print(f"Validation set shape: {data['val'].shape}")
    print(f"Test set shape: {data['test'].shape}")
    
    # Save processed data to CSV files
    data['train'].to_csv('processed_train.csv', index=False)
    data['val'].to_csv('processed_val.csv', index=False)
    data['test'].to_csv('processed_test.csv', index=False)
    
    print("\nProcessed data saved to CSV files.")
    
    # Display sample of the processed data
    print("\nSample of training data:")
    print(data['train'].head()) 