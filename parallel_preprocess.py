import os
import argparse
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client, progress
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import json

def load_dataset_dask(dataset_id, data_dir='data'):
    """
    Load a single dataset (train, test, and RUL files) using Dask
    
    Args:
        dataset_id (str): Dataset ID (e.g., 'FD001')
        data_dir (str): Directory containing the data files
    
    Returns:
        train_df, test_df, rul_df: Dask DataFrames for train, test, and RUL data
    """
    # Column names based on the readme file
    cols = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
           [f'sensor_{i}' for i in range(1, 22)]
    
    # Load training data
    train_path = os.path.join(data_dir, f'train_{dataset_id}.txt')
    train_df = dd.read_csv(train_path, sep=r'\s+', header=None, names=cols)
    
    # Load test data
    test_path = os.path.join(data_dir, f'test_{dataset_id}.txt')
    test_df = dd.read_csv(test_path, sep=r'\s+', header=None, names=cols)
    
    # Load RUL data for test set (small enough to use pandas)
    rul_path = os.path.join(data_dir, f'RUL_{dataset_id}.txt')
    rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    
    return train_df, test_df, rul_df

def calculate_rul(df):
    """
    Calculate Remaining Useful Life for each unit in the dataset
    
    Args:
        df (dd.DataFrame): Dask DataFrame with 'unit' and 'cycle' columns
    
    Returns:
        dd.DataFrame: Dask DataFrame with added 'RUL' column
    """
    # Group by unit and calculate max cycle for each unit
    # Convert to pandas for this operation as it's more efficient for this specific task
    df_pd = df.compute()
    max_cycles = df_pd.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    
    # Convert back to dask dataframe
    max_cycles_dd = dd.from_pandas(max_cycles, npartitions=min(os.cpu_count(), len(max_cycles)))
    
    # Merge with original dataframe
    df = df.merge(max_cycles_dd, on='unit', how='left')
    
    # Calculate RUL as the difference between max cycle and current cycle
    df = df.assign(RUL=df['max_cycle'] - df['cycle'])
    
    # Drop the max_cycle column as it's no longer needed
    df = df.drop('max_cycle', axis=1)
    
    return df

def normalize_data_partition(partition, scaler=None, sensor_cols=None, fit=False):
    """
    Normalize a partition of data using MinMaxScaler
    
    Args:
        partition (pd.DataFrame): Pandas DataFrame partition
        scaler: Fitted scaler or None if we need to fit
        sensor_cols: List of sensor columns
        fit: Whether to fit the scaler on this partition
    
    Returns:
        pd.DataFrame, scaler: Normalized DataFrame and fitted scaler
    """
    if scaler is None:
        scaler = MinMaxScaler()
        if fit:
            scaler.fit(partition[sensor_cols])
    
    partition[sensor_cols] = scaler.transform(partition[sensor_cols])
    return partition, scaler

def normalize_data(train_df, test_df):
    """
    Normalize sensor readings using MinMaxScaler with Dask
    
    Args:
        train_df (dd.DataFrame): Training Dask DataFrame
        test_df (dd.DataFrame): Testing Dask DataFrame
    
    Returns:
        train_df, test_df: Normalized DataFrames
        scaler: Fitted scaler for future use
    """
    # Select sensor columns and operational settings for normalization
    sensor_cols = [col for col in train_df.columns if 'sensor' in col or 'op_setting' in col]
    
    # Compute train dataframe to fit scaler
    train_pd = train_df.compute()
    
    # Initialize and fit scaler on training data
    scaler = MinMaxScaler()
    scaler.fit(train_pd[sensor_cols])
    
    # Transform training data using map_partitions
    train_df = train_df.map_partitions(
        normalize_data_partition, 
        scaler=scaler, 
        sensor_cols=sensor_cols,
        fit=False
    )
    
    # Transform test data using map_partitions
    test_df = test_df.map_partitions(
        normalize_data_partition,
        scaler=scaler,
        sensor_cols=sensor_cols,
        fit=False
    )
    
    return train_df, test_df, scaler

def prepare_cmapss_data_parallel(dataset_ids=None, val_size=0.2, random_state=42, data_dir='data', n_workers=None):
    """
    Load and preprocess the NASA CMAPSS dataset in parallel using Dask
    
    Args:
        dataset_ids (list): List of dataset IDs to load (e.g., ['FD001', 'FD002'])
                           If None, all datasets are loaded
        val_size (float): Validation set size as a fraction of training data
        random_state (int): Random seed for reproducibility
        data_dir (str): Directory containing the data files
        n_workers (int): Number of Dask workers to use (None for auto)
    
    Returns:
        dict: Dictionary containing train, validation, and test DataFrames
    """
    # Start Dask client
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)  # Leave one CPU free
    
    print(f"Starting Dask client with {n_workers} workers...")
    client = Client(n_workers=n_workers, threads_per_worker=1)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    
    try:
        if dataset_ids is None:
            dataset_ids = ['FD001', 'FD002', 'FD003', 'FD004']
        
        # Create a list of delayed tasks for loading datasets
        print(f"Loading datasets {dataset_ids} in parallel...")
        load_tasks = []
        for dataset_id in dataset_ids:
            load_task = dask.delayed(load_dataset_dask)(dataset_id, data_dir)
            load_tasks.append((dataset_id, load_task))
        
        # Execute loading tasks in parallel
        results = dask.compute(*[task for _, task in load_tasks])
        
        all_train_dfs = []
        all_test_dfs = []
        all_rul_dfs = []
        
        # Process results and add dataset_id
        for i, dataset_id in enumerate([id for id, _ in load_tasks]):
            train_df, test_df, rul_df = results[i]
            
            # Add dataset_id as a feature
            train_df = train_df.assign(dataset_id=dataset_id)
            test_df = test_df.assign(dataset_id=dataset_id)
            
            all_train_dfs.append(train_df)
            all_test_dfs.append(test_df)
            all_rul_dfs.append(rul_df)
        
        # Concatenate all datasets
        print("Concatenating datasets...")
        train_df = dd.concat(all_train_dfs)
        test_df = dd.concat(all_test_dfs)
        rul_df = pd.concat(all_rul_dfs, ignore_index=True)
        
        # Calculate RUL for training data
        print("Calculating RUL for training data...")
        train_df = calculate_rul(train_df)
        
        # Add RUL to test data
        print("Adding RUL to test data...")
        # For test data, we need to use the provided RUL values
        # First, get the max cycle for each unit in test data
        test_pd = test_df.compute()  # Convert to pandas for these operations
        test_max_cycles = test_pd.groupby('unit')['cycle'].max().reset_index()
        test_max_cycles.columns = ['unit', 'max_cycle']
        
        # Create a DataFrame with unit and RUL from the provided RUL values
        test_rul_df = pd.DataFrame({
            'unit': range(1, len(rul_df) + 1),
            'RUL_at_end': rul_df['RUL'].values
        })
        
        # Merge with max cycles
        test_max_cycles = test_max_cycles.merge(test_rul_df, on='unit', how='left')
        
        # Convert back to Dask DataFrame
        test_max_cycles_dd = dd.from_pandas(test_max_cycles, npartitions=min(n_workers, len(test_max_cycles)))
        
        # Merge with test data
        test_df = test_df.merge(test_max_cycles_dd, on='unit', how='left')
        
        # Calculate RUL for each row in test data
        test_df = test_df.assign(RUL=test_df['RUL_at_end'] + (test_df['max_cycle'] - test_df['cycle']))
        
        # Drop temporary columns
        test_df = test_df.drop(['max_cycle', 'RUL_at_end'], axis=1)
        
        # Normalize data
        print("Normalizing data...")
        train_df, test_df, scaler = normalize_data(train_df, test_df)
        
        # Split training data into train and validation sets
        print(f"Splitting training data with validation size {val_size}...")
        
        # Convert to pandas for splitting
        train_pd = train_df.compute()
        
        # Get unique units
        train_units = train_pd['unit'].unique()
        np.random.seed(random_state)
        val_units = np.random.choice(train_units, size=int(len(train_units) * val_size), replace=False)
        
        # Split into train and validation
        val_pd = train_pd[train_pd['unit'].isin(val_units)]
        train_pd = train_pd[~train_pd['unit'].isin(val_units)]
        
        # Convert test to pandas as well for consistency
        test_pd = test_df.compute()
        
        # Check for missing values
        print("Checking for missing values...")
        print(f"Train missing values: {train_pd.isnull().sum().sum()}")
        print(f"Validation missing values: {val_pd.isnull().sum().sum()}")
        print(f"Test missing values: {test_pd.isnull().sum().sum()}")
        
        # Return processed data
        return {
            'train': train_pd,
            'val': val_pd,
            'test': test_pd,
            'scaler': scaler
        }
    
    finally:
        # Close the Dask client
        client.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Parallel preprocessing of CMAPSS dataset using Dask')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the raw data files')
    parser.add_argument('--dataset_ids', type=str, nargs='+', default=None,
                        help='Dataset IDs to process (e.g., FD001 FD002)')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation set size as a fraction of training data')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of Dask workers to use (default: auto)')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Process data in parallel
    print(f"Starting parallel preprocessing with Dask...")
    data = prepare_cmapss_data_parallel(
        dataset_ids=args.dataset_ids,
        val_size=args.val_size,
        random_state=args.random_state,
        data_dir=args.data_dir,
        n_workers=args.n_workers
    )
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Training set shape: {data['train'].shape}")
    print(f"Validation set shape: {data['val'].shape}")
    print(f"Test set shape: {data['test'].shape}")
    
    # Save processed data to CSV files
    data['train'].to_csv('processed_train.csv', index=False)
    data['val'].to_csv('processed_val.csv', index=False)
    data['test'].to_csv('processed_test.csv', index=False)
    
    # Calculate and print processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    
    print("\nProcessed data saved to CSV files.")
    
    # Display sample of the processed data
    print("\nSample of training data:")
    print(data['train'].head()) 