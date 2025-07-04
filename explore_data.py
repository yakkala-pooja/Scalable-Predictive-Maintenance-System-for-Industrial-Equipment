import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_data import load_dataset, calculate_rul

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

def explore_dataset(dataset_id='FD001'):
    """
    Explore and visualize a CMAPSS dataset
    
    Args:
        dataset_id (str): Dataset ID (e.g., 'FD001')
    """
    print(f"\n{'='*50}")
    print(f"Exploring dataset {dataset_id}")
    print(f"{'='*50}")
    
    # Load dataset
    train_df, test_df, rul_df = load_dataset(dataset_id)
    
    # Calculate RUL for training data
    train_df = calculate_rul(train_df)
    
    # Basic statistics
    print("\nTraining Data Statistics:")
    print(f"Number of engines: {train_df['unit'].nunique()}")
    print(f"Number of cycles: {train_df['cycle'].sum()}")
    print(f"Average cycles per engine: {train_df.groupby('unit')['cycle'].max().mean():.2f}")
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Distribution of engine lifetimes
    plt.figure()
    sns.histplot(train_df.groupby('unit')['cycle'].max(), bins=20, kde=True)
    plt.title(f'Distribution of Engine Lifetimes - {dataset_id}')
    plt.xlabel('Number of Cycles')
    plt.ylabel('Count')
    plt.savefig(f'plots/{dataset_id}_engine_lifetimes.png')
    
    # Plot 2: RUL distribution
    plt.figure()
    sns.histplot(train_df['RUL'], bins=30, kde=True)
    plt.title(f'Distribution of RUL Values - {dataset_id}')
    plt.xlabel('Remaining Useful Life (cycles)')
    plt.ylabel('Count')
    plt.savefig(f'plots/{dataset_id}_rul_distribution.png')
    
    # Plot 3: Sensor readings over time for a sample engine
    sample_unit = train_df['unit'].iloc[0]
    sample_data = train_df[train_df['unit'] == sample_unit]
    
    # Select a subset of sensors to plot
    sensors_to_plot = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12']
    
    plt.figure()
    for sensor in sensors_to_plot:
        plt.plot(sample_data['cycle'], sample_data[sensor], label=sensor)
    
    plt.title(f'Sensor Readings Over Time for Engine {sample_unit} - {dataset_id}')
    plt.xlabel('Cycle')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.savefig(f'plots/{dataset_id}_sensor_readings.png')
    
    # Plot 4: Correlation matrix of sensor readings
    plt.figure(figsize=(14, 12))
    sensor_cols = [col for col in train_df.columns if 'sensor' in col]
    corr_matrix = train_df[sensor_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Matrix of Sensor Readings - {dataset_id}')
    plt.tight_layout()
    plt.savefig(f'plots/{dataset_id}_correlation_matrix.png')
    
    # Plot 5: RUL vs Cycle for a few sample engines
    plt.figure()
    sample_units = train_df['unit'].unique()[:5]  # Take first 5 engines
    
    for unit in sample_units:
        unit_data = train_df[train_df['unit'] == unit]
        plt.plot(unit_data['cycle'], unit_data['RUL'], label=f'Engine {unit}')
    
    plt.title(f'RUL vs Cycle for Sample Engines - {dataset_id}')
    plt.xlabel('Cycle')
    plt.ylabel('Remaining Useful Life')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{dataset_id}_rul_vs_cycle.png')
    
    print(f"\nPlots saved to 'plots' directory for dataset {dataset_id}")

if __name__ == "__main__":
    # Explore all datasets
    for dataset_id in ['FD001', 'FD002', 'FD003', 'FD004']:
        explore_dataset(dataset_id)
    
    print("\nData exploration completed for all datasets.") 