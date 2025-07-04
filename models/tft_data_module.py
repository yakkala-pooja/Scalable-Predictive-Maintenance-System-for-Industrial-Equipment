import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import json


class CMAPSSDataset(Dataset):
    """
    Dataset class for CMAPSS data.
    """
    def __init__(self, X, y):
        """
        Initialize the dataset.
        
        Args:
            X: Input features of shape [num_samples, window_size, num_features]
            y: Target values of shape [num_samples]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CMAPSSDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for CMAPSS dataset.
    """
    def __init__(
        self,
        data_dir: str = 'transformer_data',
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize the data module.
        
        Args:
            data_dir: Directory containing the preprocessed data
            batch_size: Batch size for training and validation
            num_workers: Number of workers for data loading
            pin_memory: Whether to use pin_memory for faster GPU transfer
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Set the number of workers based on CPU cores if not specified
        if self.num_workers is None:
            self.num_workers = min(os.cpu_count(), 8)  # Use up to 8 workers
        
        # Attributes to be set in setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_features = None
        
    def worker_init_fn(self, worker_id):
        """
        Worker initialization function to ensure each worker has a different random seed.
        """
        np.random.seed(np.random.get_state()[1][0] + worker_id)
            
    def prepare_data(self):
        """
        Check if the data files exist.
        """
        required_files = [
            'X_train.npy', 'y_train.npy',
            'X_val.npy', 'y_val.npy',
            'X_test.npy', 'y_test.npy',
            'metadata.json'
        ]
        
        for file in required_files:
            file_path = os.path.join(self.data_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file {file_path} not found.")
    
    def setup(self, stage=None):
        """
        Load the data.
        
        Args:
            stage: Stage of the pipeline ('fit', 'validate', 'test')
        """
        # Load metadata
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
            self.num_features = self.metadata['num_features']
        
        # Load data only if needed based on stage
        if stage == 'fit' or stage is None:
            # Load training data
            X_train = np.load(os.path.join(self.data_dir, 'X_train.npy'), mmap_mode='r')
            y_train = np.load(os.path.join(self.data_dir, 'y_train.npy'), mmap_mode='r')
            
            # Load validation data
            X_val = np.load(os.path.join(self.data_dir, 'X_val.npy'), mmap_mode='r')
            y_val = np.load(os.path.join(self.data_dir, 'y_val.npy'), mmap_mode='r')
            
            # Create datasets
            self.train_dataset = CMAPSSDataset(X_train, y_train)
            self.val_dataset = CMAPSSDataset(X_val, y_val)
            
            print(f"Training set size: {len(self.train_dataset)}")
            print(f"Validation set size: {len(self.val_dataset)}")
        
        if stage == 'test' or stage is None:
            # Load test data
            X_test = np.load(os.path.join(self.data_dir, 'X_test.npy'), mmap_mode='r')
            y_test = np.load(os.path.join(self.data_dir, 'y_test.npy'), mmap_mode='r')
            
            # Create dataset
            self.test_dataset = CMAPSSDataset(X_test, y_test)
            
            print(f"Test set size: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """
        Create training data loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self.worker_init_fn
        )
    
    def val_dataloader(self):
        """
        Create validation data loader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self.worker_init_fn
        )
    
    def test_dataloader(self):
        """
        Create test data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self.worker_init_fn
        ) 