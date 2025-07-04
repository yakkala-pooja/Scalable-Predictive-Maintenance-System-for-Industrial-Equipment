import os
import argparse
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Run the full TFT pipeline')
    
    # Data preparation parameters
    parser.add_argument('--window_size', type=int, default=30,
                        help='Window size for sequence data')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size for the model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    
    # Pipeline control
    parser.add_argument('--skip_data_prep', action='store_true',
                        help='Skip data preparation step')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training step')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip model evaluation step')
    parser.add_argument('--use_simple_model', action='store_true',
                        help='Use the simple LSTM model instead of TFT')
    
    # Directories
    parser.add_argument('--data_dir', type=str, default='transformer_data',
                        help='Directory for preprocessed data')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='Directory for model checkpoints')
    parser.add_argument('--eval_dir', type=str, default='evaluation_plots',
                        help='Directory for evaluation plots')
    
    return parser.parse_args()


def run_data_preparation(args):
    """Run data preparation script"""
    print("\n" + "="*80)
    print("STEP 1: Data Preparation")
    print("="*80)
    
    cmd = [
        "python", "prepare_windows.py",
        "--window_size", str(args.window_size),
        "--horizon", str(args.horizon),
        "--train_path", "processed_train.csv",
        "--val_path", "processed_val.csv",
        "--test_path", "processed_test.csv",
        "--save_path", args.data_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_training(args):
    """Run model training script"""
    print("\n" + "="*80)
    print("STEP 2: Model Training")
    print("="*80)
    
    if args.use_simple_model:
        # Use the simple LSTM model if requested
        print("Using simple LSTM model as requested...")
        cmd = [
            "python", "models/train_simple_lstm.py",
            "--data_dir", args.data_dir,
            "--batch_size", str(args.batch_size),
            "--hidden_size", str(args.hidden_size),
            "--max_epochs", str(args.max_epochs),
            "--learning_rate", str(args.learning_rate),
            "--ckpt_dir", args.ckpt_dir
        ]
    else:
        # Use the TFT model by default
        print("Using Temporal Fusion Transformer model...")
        cmd = [
            "python", "models/train_tft.py",
            "--data_dir", args.data_dir,
            "--batch_size", str(args.batch_size),
            "--hidden_size", str(args.hidden_size),
            "--max_epochs", str(args.max_epochs),
            "--learning_rate", str(args.learning_rate),
            "--ckpt_dir", args.ckpt_dir
        ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_evaluation(args):
    """Run model evaluation script"""
    print("\n" + "="*80)
    print("STEP 3: Model Evaluation")
    print("="*80)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    if args.use_simple_model:
        # Find the best LSTM checkpoint
        print("Evaluating simple LSTM model...")
        checkpoints = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.ckpt') and f.startswith('lstm-')]
        if not checkpoints:
            print(f"No LSTM checkpoints found in {args.ckpt_dir}. Skipping evaluation.")
            return
        
        # Sort checkpoints by validation loss (extract val_loss from filename)
        try:
            # Extract validation loss from checkpoint filename
            def get_val_loss(filename):
                # Format: lstm-epoch=02-val_loss=5108.4141.ckpt
                val_loss_str = filename.split('val_loss=')[1].split('.ckpt')[0]
                return float(val_loss_str)
            
            checkpoints.sort(key=get_val_loss)
            best_checkpoint = os.path.join(args.ckpt_dir, checkpoints[0])
            
            # Create a simple evaluation script command
            cmd = [
                "python", "-c",
                f"""
import torch
import pytorch_lightning as pl
from models.simple_lstm_model import SimpleLSTM
from models.tft_data_module import CMAPSSDataModule

# Load model
model = SimpleLSTM.load_from_checkpoint('{best_checkpoint}')
model.eval()

# Create data module
data_module = CMAPSSDataModule(data_dir='{args.data_dir}', batch_size={args.batch_size})
data_module.setup()

# Create trainer
trainer = pl.Trainer(accelerator='cpu', devices=1)

# Test model
print('\\nTesting model...')
test_results = trainer.test(model, data_module)
print(f'\\nTest results: {{test_results}}')
                """
            ]
            
            print(f"Using best checkpoint: {best_checkpoint}")
            print("Running evaluation...")
            subprocess.run(cmd, check=True)
        except (IndexError, ValueError) as e:
            print(f"Error processing checkpoints: {e}")
            print("Skipping evaluation.")
    else:
        # Find the best TFT checkpoint
        print("Evaluating Temporal Fusion Transformer model...")
        checkpoints = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.ckpt') and f.startswith('tft-')]
        if not checkpoints:
            print(f"No TFT checkpoints found in {args.ckpt_dir}. Skipping evaluation.")
            return
        
        # Sort checkpoints by validation loss (extract val_loss from filename)
        try:
            # Extract validation loss from checkpoint filename
            def get_val_loss(filename):
                # Format: tft-epoch=02-val_loss=5108.4141.ckpt
                val_loss_str = filename.split('val_loss=')[1].split('.ckpt')[0]
                return float(val_loss_str)
            
            checkpoints.sort(key=get_val_loss)
            best_checkpoint = os.path.join(args.ckpt_dir, checkpoints[0])
            
            # Use the evaluate_tft.py script
            cmd = [
                "python", "models/evaluate_tft.py",
                "--data_dir", args.data_dir,
                "--checkpoint_path", best_checkpoint,
                "--plot_dir", args.eval_dir
            ]
            
            print(f"Using best checkpoint: {best_checkpoint}")
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except (IndexError, ValueError) as e:
            print(f"Error processing checkpoints: {e}")
            print("Skipping evaluation.")


def main():
    """Main function to run the pipeline"""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    
    # Step 1: Data preparation
    if not args.skip_data_prep:
        run_data_preparation(args)
    else:
        print("\nSkipping data preparation step.")
    
    # Step 2: Model training
    if not args.skip_training:
        run_training(args)
    else:
        print("\nSkipping model training step.")
    
    # Step 3: Model evaluation
    if not args.skip_evaluation:
        run_evaluation(args)
    else:
        print("\nSkipping model evaluation step.")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)
    print("\nResults:")
    print(f"- Processed data: {args.data_dir}/")
    print(f"- Model checkpoints: {args.ckpt_dir}/")
    print(f"- Evaluation plots: {args.eval_dir}/")
    
    # Check if feature importance analysis was performed
    feature_importance_file = os.path.join(args.eval_dir, 'feature_importance_ranking.txt')
    if os.path.exists(feature_importance_file):
        print("\nFeature Importance Analysis:")
        print(f"- Global feature importance plot: {args.eval_dir}/global_feature_importance.png")
        print(f"- Sample-specific feature importance plots: {args.eval_dir}/sample_*_feature_importance.png")
        print(f"- Ranked feature importance: {feature_importance_file}")
        
        # Display top 5 most important features
        try:
            with open(feature_importance_file, 'r') as f:
                lines = f.readlines()
                print("\nTop 5 Most Important Features:")
                for i in range(5):
                    if i + 6 < len(lines):  # Skip header lines (6 lines)
                        print(f"  {lines[i + 6].strip()}")
        except Exception as e:
            print(f"Could not read feature importance file: {e}")
    
    print("\nTo view TensorBoard logs, run:")
    print(f"tensorboard --logdir=lightning_logs/")


if __name__ == '__main__':
    main() 