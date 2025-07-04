import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from typing import Dict, List, Optional

from tft_model import TemporalFusionTransformer
from tft_data_module import CMAPSSDataModule


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TFT model for RUL prediction')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='transformer_data',
                        help='Directory containing the preprocessed data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    # Model parameters
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--plot_dir', type=str, default='evaluation_plots',
                        help='Directory to save evaluation plots')
    parser.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0,
                        help='Number of GPUs to use')
    
    return parser.parse_args()


def plot_predictions(y_true, y_pred, save_path):
    """
    Plot true vs predicted RUL values.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Sort by true RUL for better visualization
    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    # Plot true vs predicted
    plt.scatter(y_true_sorted, y_pred_sorted, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.title('True vs Predicted RUL')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_error_distribution(errors, save_path):
    """
    Plot distribution of prediction errors.
    
    Args:
        errors: Prediction errors
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of errors
    plt.hist(errors, bins=50, alpha=0.7)
    
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    
    # Add statistics to the plot
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    
    plt.axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f}')
    plt.axvline(median_error, color='g', linestyle='--', label=f'Median: {median_error:.2f}')
    
    plt.legend()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def extract_attention_weights(model, dataloader, feature_names, num_samples=5):
    """
    Extract attention weights from the TFT model.
    
    Args:
        model: Trained TFT model
        dataloader: Test dataloader
        feature_names: List of feature names
        num_samples: Number of samples to extract attention weights for
        
    Returns:
        Dictionary of attention weights
    """
    # Modify the forward method to return attention weights
    original_forward = model.forward
    
    attention_weights = {
        'variable_selection': [],
        'samples': []
    }
    
    def forward_with_attention(x):
        batch_size, seq_len, num_features = x.shape
        
        # Process time-varying real variables
        time_varying_real = x.unsqueeze(-1)
        
        # Create dictionary of inputs for the variable selection network
        real_inputs = {f'real_{i}': time_varying_real[..., i, :] for i in range(model.num_time_varying_real_vars)}
        
        # Apply variable selection network across time
        time_varying_embeddings = []
        variable_weights = []
        
        for t in range(seq_len):
            # Get inputs at current time step
            real_inputs_t = {k: v[:, t] for k, v in real_inputs.items()}
            
            # Apply variable selection
            embedding, weight = model.time_varying_real_vsn(real_inputs_t)
            
            time_varying_embeddings.append(embedding)
            variable_weights.append(weight)
            
        # Stack across time dimension
        time_varying_embeddings = torch.stack(time_varying_embeddings, dim=1)
        variable_weights = torch.stack(variable_weights, dim=1)
        
        # Store variable selection weights for this batch
        attention_weights['variable_selection'].append(variable_weights.detach().cpu().numpy())
        
        # LSTM encoding
        lstm_output, _ = model.lstm_encoder(time_varying_embeddings)
        
        # Apply gated residual network
        lstm_output = model.post_lstm_gate(lstm_output)
        lstm_output = model.norm1(lstm_output)
        
        # Self-attention layer
        attn_output, attn_weights = model.multihead_attn(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            need_weights=True
        )
        
        # Apply gated residual network to attention output
        attn_output = model.post_attn_gate(attn_output)
        attn_output = model.norm2(attn_output)
        
        # Take the last time step for prediction
        output = attn_output[:, -1]
        
        # Final output layer
        predictions = model.output_layer(output)
        
        # Store sample data
        if len(attention_weights['samples']) < num_samples:
            for i in range(min(batch_size, num_samples - len(attention_weights['samples']))):
                sample_data = {
                    'input': x[i].detach().cpu().numpy(),
                    'variable_weights': variable_weights[i].detach().cpu().numpy(),
                    'attention_weights': attn_weights[i].detach().cpu().numpy() if attn_weights.dim() > 2 else attn_weights.detach().cpu().numpy(),
                    'prediction': predictions[i].item()
                }
                attention_weights['samples'].append(sample_data)
        
        return predictions
    
    # Replace forward method temporarily
    model.forward = forward_with_attention
    
    # Process a few batches to collect attention weights
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            _ = model(x)
            if len(attention_weights['samples']) >= num_samples:
                break
    
    # Restore original forward method
    model.forward = original_forward
    
    return attention_weights


def plot_feature_importance(attention_weights, feature_names, save_path):
    """
    Plot global feature importance based on variable selection attention weights.
    
    Args:
        attention_weights: Dictionary of attention weights
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    # Calculate average variable selection weights across all samples and time steps
    all_weights = np.concatenate(attention_weights['variable_selection'], axis=0)
    avg_weights = np.mean(all_weights, axis=(0, 1))
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    
    # Sort features by importance
    sorted_indices = np.argsort(avg_weights)
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_weights = avg_weights[sorted_indices]
    
    # Plot horizontal bar chart
    plt.barh(range(len(sorted_features)), sorted_weights, align='center')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance Score')
    plt.title('Global Feature Importance')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_sample_attention(attention_weights, feature_names, sample_idx, save_path):
    """
    Plot feature importance for a single sample.
    
    Args:
        attention_weights: Dictionary of attention weights
        feature_names: List of feature names
        sample_idx: Index of the sample to plot
        save_path: Path to save the plot
    """
    sample = attention_weights['samples'][sample_idx]
    
    # Get variable selection weights for this sample
    var_weights = sample['variable_weights']
    
    # Calculate average weights across time steps
    avg_weights = np.mean(var_weights, axis=0)
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    
    # Sort features by importance
    sorted_indices = np.argsort(avg_weights)
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_weights = avg_weights[sorted_indices]
    
    # Plot horizontal bar chart
    plt.barh(range(len(sorted_features)), sorted_weights, align='center')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance Score')
    plt.title(f'Feature Importance for Sample {sample_idx+1} (Predicted RUL: {sample["prediction"]:.2f})')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create plot directory
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Load metadata
    with open(os.path.join(args.data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load feature names
    feature_names = metadata['feature_columns']
    
    # Create data module
    data_module = CMAPSSDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup data
    data_module.setup()
    
    # Load model from checkpoint
    model = TemporalFusionTransformer.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    # Create trainer
    trainer_kwargs = {}
    
    # Configure device settings based on availability
    if args.gpus > 0 and torch.cuda.is_available():
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = args.gpus
    else:
        trainer_kwargs['accelerator'] = 'cpu'
        trainer_kwargs['devices'] = 1  # Use 1 CPU core
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Test model
    print("Testing model...")
    test_results = trainer.test(model, data_module)
    
    # Collect predictions and true values
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            x, y = batch
            predictions = model(x)
            
            y_true.append(y.cpu().numpy())
            y_pred.append(predictions.squeeze().cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    # Calculate errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Calculate metrics
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    precision_25 = np.mean(abs_errors <= 25)
    
    print(f"\nTest Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Precision@25_cycles: {precision_25:.4f}")
    
    # Plot predictions
    print("\nGenerating plots...")
    plot_predictions(
        y_true, y_pred, 
        os.path.join(args.plot_dir, 'true_vs_predicted.png')
    )
    
    # Plot error distribution
    plot_error_distribution(
        errors, 
        os.path.join(args.plot_dir, 'error_distribution.png')
    )
    
    # Extract attention weights
    print("\nExtracting attention weights for feature importance analysis...")
    attention_weights = extract_attention_weights(
        model, 
        data_module.test_dataloader(), 
        feature_names
    )
    
    # Plot global feature importance
    print("Generating feature importance plots...")
    plot_feature_importance(
        attention_weights,
        feature_names,
        os.path.join(args.plot_dir, 'global_feature_importance.png')
    )
    
    # Plot sample-specific feature importance
    for i in range(min(5, len(attention_weights['samples']))):
        plot_sample_attention(
            attention_weights,
            feature_names,
            i,
            os.path.join(args.plot_dir, f'sample_{i+1}_feature_importance.png')
        )
    
    print(f"\nEvaluation complete. Plots saved to {args.plot_dir}/")
    
    # Save ranked feature importance to a text file
    all_weights = np.concatenate(attention_weights['variable_selection'], axis=0)
    avg_weights = np.mean(all_weights, axis=(0, 1))
    
    # Sort features by importance
    sorted_indices = np.argsort(avg_weights)[::-1]  # Reverse to get descending order
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_weights = avg_weights[sorted_indices]
    
    # Save to text file
    with open(os.path.join(args.plot_dir, 'feature_importance_ranking.txt'), 'w') as f:
        f.write("Feature Importance Ranking:\n")
        f.write("==========================\n\n")
        f.write("Rank | Feature Name | Importance Score\n")
        f.write("-" * 40 + "\n")
        
        for i, (feature, weight) in enumerate(zip(sorted_features, sorted_weights)):
            f.write(f"{i+1:4d} | {feature:12s} | {weight:.6f}\n")
    
    print(f"Feature importance ranking saved to {args.plot_dir}/feature_importance_ranking.txt")


if __name__ == '__main__':
    main() 