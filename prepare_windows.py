import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

def create_sequence_data(df, window_size=30, horizon=1, group_col='unit', time_col='cycle', target_col='RUL'):
    """
    Create sliding window sequences from time series data.
    
    Args:
        df (pd.DataFrame): DataFrame containing time series data
        window_size (int): Number of past time steps to use as input features
        horizon (int): Number of steps ahead to predict
        group_col (str): Column name to group by (e.g., 'unit' for engine ID)
        time_col (str): Column name for time/cycle information
        target_col (str): Column name for the target variable (e.g., 'RUL')
    
    Returns:
        X (np.array): 3D array of shape (num_samples, window_size, num_features)
        y (np.array): 1D array of shape (num_samples,) containing target values
    """
    # Get feature columns (exclude group_col, time_col, and target_col)
    feature_cols = [col for col in df.columns if col not in [group_col, time_col, target_col, 'dataset_id']]
    
    sequences = []
    targets = []
    
    # Group by the specified column (e.g., by engine unit)
    for _, group in df.groupby(group_col):
        # Sort by time/cycle
        group = group.sort_values(by=time_col)
        
        # Extract features and target
        features = group[feature_cols].values
        target = group[target_col].values
        
        # Create sequences
        for i in range(len(group) - window_size - horizon + 1):
            # Input sequence
            seq = features[i:i+window_size]
            # Target value (RUL at the end of the sequence + horizon)
            tgt = target[i+window_size+horizon-1]
            
            sequences.append(seq)
            targets.append(tgt)
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)
    
    return X, y, feature_cols

def prepare_transformer_data(data_path='processed_train.csv', val_path='processed_val.csv', 
                            test_path='processed_test.csv', window_size=30, horizon=1,
                            save_path='transformer_data'):
    """
    Prepare sliding window data for Transformer model training.
    
    Args:
        data_path (str): Path to the processed training data CSV
        val_path (str): Path to the processed validation data CSV
        test_path (str): Path to the processed test data CSV
        window_size (int): Number of past time steps to use as input features
        horizon (int): Number of steps ahead to predict
        save_path (str): Directory to save the prepared data
    
    Returns:
        None (saves data to files)
    """
    print(f"Loading data from {data_path}, {val_path}, and {test_path}...")
    train_df = pd.read_csv(data_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Creating sequences with window_size={window_size}, horizon={horizon}...")
    X_train, y_train, feature_cols = create_sequence_data(train_df, window_size, horizon)
    X_val, y_val, _ = create_sequence_data(val_df, window_size, horizon)
    X_test, y_test, _ = create_sequence_data(test_df, window_size, horizon)
    
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
    import json
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Data preparation complete!")
    print(f"Files saved to {save_path}/")

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare sliding window data for Transformer model training')
    
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
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Prepare data with sliding windows
    prepare_transformer_data(
        data_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        window_size=args.window_size,
        horizon=args.horizon,
        save_path=args.save_path
    )
    
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