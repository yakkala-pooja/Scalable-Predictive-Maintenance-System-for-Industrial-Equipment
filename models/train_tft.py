import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from tft_model import TemporalFusionTransformer
from tft_data_module import CMAPSSDataModule


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a Temporal Fusion Transformer model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='transformer_data',
                       help='Directory containing the preprocessed data')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Hidden size for LSTM and attention layers')
    parser.add_argument('--lstm_layers', type=int, default=1,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--attention_heads', type=int, default=4,
                       help='Number of attention heads')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic behavior for reproducibility')
    
    # Hardware parameters
    parser.add_argument('--devices', type=int, default=None,
                       help='Number of GPUs to use (None = use all available)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true',
                       help='Enable pin_memory for data loading (faster GPU transfer)')
    
    # Directory parameters
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    
    return parser.parse_args()


def plot_loss_curves(trainer, save_dir='plots'):
    """
    Plot training and validation loss curves.
    
    Args:
        trainer: PyTorch Lightning trainer
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get metrics from the trainer
    metrics = trainer.callback_metrics
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    
    # Training loss
    train_loss = np.array(trainer.fit_loop.epoch_loop.train_epoch_output['loss'])
    plt.plot(train_loss, label='Training Loss')
    
    # Validation loss
    val_loss = np.array([x['val_loss'] for x in trainer.fit_loop.epoch_loop.val_epoch_output])
    plt.plot(val_loss, label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()


def train_model(args):
    """Train the TFT model"""
    # Set up seed for reproducibility
    pl.seed_everything(args.random_seed)
    
    # Create data module
    data_module = CMAPSSDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # Setup data to get metadata
    data_module.setup()
    
    # Create model
    model = TemporalFusionTransformer(
        num_time_varying_real_vars=data_module.num_features,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        num_attention_heads=args.attention_heads,
        learning_rate=args.learning_rate
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='tft-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # Learning rate scheduler
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode='min'
    )
    
    # Configure trainer arguments based on available hardware
    trainer_kwargs = {
        'callbacks': [checkpoint_callback, lr_monitor, early_stop_callback],
        'max_epochs': args.max_epochs,
        'logger': True,
        'enable_progress_bar': True,
        'deterministic': args.deterministic
    }
    
    # Use hardware manager for optimal device configuration
    try:
        from utils.hardware_manager import optimize_for_hardware
        hw_config = optimize_for_hardware(model_size_mb=200, base_batch_size=args.batch_size)
        
        # Override with command line arguments if provided
        if args.devices:
            hw_config['pl_config']['devices'] = args.devices
        
        trainer_kwargs.update(hw_config['pl_config'])
        
        # Update batch size if hardware manager suggests different
        if hw_config['batch_size'] != args.batch_size:
            print(f"Hardware optimization suggests batch size {hw_config['batch_size']} instead of {args.batch_size}")
            args.batch_size = hw_config['batch_size']
        
        # Add memory optimization settings
        trainer_kwargs.update(hw_config['memory_settings'])
        
    except ImportError:
        # Fallback to original logic if hardware manager not available
        print("Hardware manager not available, using basic device detection")
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"Found {num_gpus} GPUs, using distributed data parallel strategy")
                trainer_kwargs['strategy'] = 'ddp'
                trainer_kwargs['accelerator'] = 'gpu'
                trainer_kwargs['devices'] = args.devices if args.devices else num_gpus
            else:
                print(f"Found 1 GPU, using single GPU training")
                trainer_kwargs['accelerator'] = 'gpu'
                trainer_kwargs['devices'] = 1
        else:
            print("No GPU found, using CPU training")
            trainer_kwargs['accelerator'] = 'cpu'
            # Use all available CPU cores with a high-performance strategy
            if hasattr(pl.strategies, 'SingleDeviceStrategy'):
                trainer_kwargs['strategy'] = pl.strategies.SingleDeviceStrategy(device='cpu')
    
    # Create trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    return model, trainer


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set up directories
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Train model
    print("Training TFT model...")
    model, trainer = train_model(args)
    
    print("Training complete!")
    
    # Get the best checkpoint path
    if hasattr(trainer.checkpoint_callback, 'best_model_path'):
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"Best checkpoint path: {best_model_path}")
    
    return model


if __name__ == '__main__':
    main() 