import os
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client, progress
import argparse
import time
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def create_sequence_data_for_unit(unit_data, window_size=30, horizon=1, feature_cols=None, time_col='cycle', target_col='RUL'):
    """
    Create sliding window sequences for a single engine unit.
    
    Args:
        unit_data (pd.DataFrame): DataFrame containing time series data for a single unit
        window_size (int): Number of past time steps to use as input features
        horizon (int): Number of steps ahead to predict
        feature_cols (list): List of feature column names
        time_col (str): Column name for time/cycle information
        target_col (str): Column name for the target variable (e.g., 'RUL')
    
    Returns:
        tuple: (sequences, targets) for this unit
    """
    # Sort by time/cycle
    unit_data = unit_data.sort_values(by=time_col)
    
    # Extract features and target
    features = unit_data[feature_cols].values
    target = unit_data[target_col].values
    
    sequences = []
    targets = []
    
    # Create sequences
    for i in range(len(unit_data) - window_size - horizon + 1):
        # Input sequence
        seq = features[i:i+window_size]
        # Target value (RUL at the end of the sequence + horizon)
        tgt = target[i+window_size+horizon-1]
        
        sequences.append(seq)
        targets.append(tgt)
    
    return np.array(sequences), np.array(targets)

def process_dataframe_parallel(df, window_size=30, horizon=1, n_workers=None, group_col='unit', time_col='cycle', target_col='RUL'):
    """
    Process a DataFrame to create sliding window sequences in parallel.
    
    Args:
        df (pd.DataFrame): DataFrame containing time series data
        window_size (int): Number of past time steps to use as input features
        horizon (int): Number of steps ahead to predict
        n_workers (int): Number of workers to use (None for auto)
        group_col (str): Column name to group by (e.g., 'unit' for engine ID)
        time_col (str): Column name for time/cycle information
        target_col (str): Column name for the target variable (e.g., 'RUL')
    
    Returns:
        tuple: (X, y, feature_cols) where X is a 3D array of shape (num_samples, window_size, num_features)
               and y is a 1D array of shape (num_samples,)
    """
    # Get feature columns (exclude group_col, time_col, and target_col)
    feature_cols = [col for col in df.columns if col not in [group_col, time_col, target_col, 'dataset_id']]
    
    # Get unique units
    units = df[group_col].unique()
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(os.cpu_count(), len(units))
    
    print(f"Processing {len(units)} units with {n_workers} workers...")
    
    all_sequences = []
    all_targets = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a dictionary to store futures
        future_to_unit = {}
        
        # Submit tasks for each unit
        for unit in units:
            unit_data = df[df[group_col] == unit]
            future = executor.submit(
                create_sequence_data_for_unit,
                unit_data,
                window_size,
                horizon,
                feature_cols,
                time_col,
                target_col
            )
            future_to_unit[future] = unit
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_unit), total=len(future_to_unit), desc="Processing units"):
            unit = future_to_unit[future]
            try:
                sequences, targets = future.result()
                all_sequences.append(sequences)
                all_targets.append(targets)
            except Exception as e:
                print(f"Error processing unit {unit}: {e}")
    
    # Combine results
    if all_sequences:
        X = np.vstack(all_sequences)
        y = np.concatenate(all_targets)
        return X, y, feature_cols
    else:
        raise ValueError("No sequences were created. Check your data and parameters.")

def prepare_transformer_data_parallel(data_path='processed_train.csv', val_path='processed_val.csv', 
                                     test_path='processed_test.csv', window_size=30, horizon=1,
                                     save_path='transformer_data', n_workers=None):
    """
    Prepare sliding window data for Transformer model training using parallel processing.
    
    Args:
        data_path (str): Path to the processed training data CSV
        val_path (str): Path to the processed validation data CSV
        test_path (str): Path to the processed test data CSV
        window_size (int): Number of past time steps to use as input features
        horizon (int): Number of steps ahead to predict
        save_path (str): Directory to save the prepared data
        n_workers (int): Number of workers to use (None for auto)
    
    Returns:
        None (saves data to files)
    """
    print(f"Loading data from {data_path}, {val_path}, and {test_path}...")
    
    # Load data with progress reporting
    print("Loading training data...")
    train_df = pd.read_csv(data_path)
    
    print("Loading validation data...")
    val_df = pd.read_csv(val_path)
    
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    
    print(f"Creating sequences with window_size={window_size}, horizon={horizon}...")
    
    # Process each dataset in parallel
    print("Processing training data...")
    X_train, y_train, feature_cols = process_dataframe_parallel(
        train_df, window_size, horizon, n_workers
    )
    
    print("Processing validation data...")
    X_val, y_val, _ = process_dataframe_parallel(
        val_df, window_size, horizon, n_workers
    )
    
    print("Processing test data...")
    X_test, y_test, _ = process_dataframe_parallel(
        test_df, window_size, horizon, n_workers
    )
    
    print(f"Generated sequences:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the data
    print(f"Saving data to {save_path}...")
    np.save(os.path.join(save_path, 'X_train.npy'), X_train)
    np.save(os.path.join(save_path, 'y_train.npy'), y_train)
    np.save(os.path.join(save_path, 'X_val.npy'), X_val)
    np.save(os.path.join(save_path, 'y_val.npy'), y_val)
    np.save(os.path.join(save_path, 'X_test.npy'), X_test)
    np.save(os.path.join(save_path, 'y_test.npy'), y_test)
    
    # Save feature column names
    with open(os.path.join(save_path, 'feature_cols.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))
    
    # Save metadata
    metadata = {
        'window_size': window_size,
        'horizon': horizon,
        'num_features': len(feature_cols),
        'feature_columns': feature_cols
    }
    
    # Save as JSON
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Data preparation complete!")
    print(f"Files saved to {save_path}/")

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare sliding window data for Transformer model training using parallel processing')
    
    parser.add_argument('--window_size', type=int, default=30,
                        help='Window size for sequence data')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon')
    parser.add_argument('--train_path', type=str, default='processed_train.csv',
                        help='Path to the processed training data CSV')
    parser.add_argument('--val_path', type=str, default='processed_val.csv',
                        help='Path to the processed validation data CSV')
    parser.add_argument('--test_path', type=str, default='processed_test.csv',
                        help='Path to the processed test data CSV')
    parser.add_argument('--save_path', type=str, default='transformer_data',
                        help='Directory to save the prepared data')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of workers to use (default: auto)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Prepare data with sliding windows
    prepare_transformer_data_parallel(
        data_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        window_size=args.window_size,
        horizon=args.horizon,
        save_path=args.save_path,
        n_workers=args.n_workers
    )
    
    # Calculate and print processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    
    # Example of how to load the data
    print("\nExample of loading the prepared data:")
    X_train = np.load(os.path.join(args.save_path, 'X_train.npy'))
    y_train = np.load(os.path.join(args.save_path, 'y_train.npy'))
    
    print(f"Loaded X_train shape: {X_train.shape}")
    print(f"Loaded y_train shape: {y_train.shape}")
    
    # Display a sample sequence
    print("\nSample sequence (first 5 time steps, first 5 features):")
    print(X_train[0, :5, :5])
    
    # Display corresponding target
    print(f"\nCorresponding target RUL: {y_train[0]}")
