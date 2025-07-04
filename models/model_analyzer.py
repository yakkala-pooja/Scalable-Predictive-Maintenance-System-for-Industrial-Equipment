import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import logging
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Comprehensive model analysis and quality assessment."""
    
    def __init__(self, model: nn.Module, model_name: str = "Unknown"):
        """
        Initialize model analyzer.
        
        Args:
            model: PyTorch model to analyze
            model_name: Name of the model for reporting
        """
        self.model = model
        self.model_name = model_name
        self.analysis_results = {}
        self.logger = logging.getLogger('model_analyzer')
        
        # Move model to CPU for analysis
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture and complexity."""
        self.logger.info(f"Analyzing architecture for {self.model_name}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Calculate model size in MB
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # Analyze layer types
        layer_types = {}
        layer_sizes = []
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_type = type(module).__name__
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
                
                # Get layer size for weight matrices
                if hasattr(module, 'weight') and module.weight is not None:
                    layer_sizes.append(module.weight.numel())
        
        # Calculate complexity metrics
        complexity_metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'model_size_mb': size_mb,
            'layer_types': layer_types,
            'max_layer_size': max(layer_sizes) if layer_sizes else 0,
            'avg_layer_size': np.mean(layer_sizes) if layer_sizes else 0,
            'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0
        }
        
        # Determine model complexity category
        if total_params < 100000:
            complexity_metrics['complexity_category'] = 'Small'
        elif total_params < 1000000:
            complexity_metrics['complexity_category'] = 'Medium'
        elif total_params < 10000000:
            complexity_metrics['complexity_category'] = 'Large'
        else:
            complexity_metrics['complexity_category'] = 'Very Large'
        
        self.analysis_results['architecture'] = complexity_metrics
        return complexity_metrics
    
    def analyze_training_curves(self, train_losses: List[float], val_losses: List[float], 
                               train_metrics: Optional[List[float]] = None, 
                               val_metrics: Optional[List[float]] = None) -> Dict[str, Any]:
        """Analyze training curves for overfitting and convergence."""
        self.logger.info("Analyzing training curves")
        
        if len(train_losses) < 2:
            return {'error': 'Insufficient training data for analysis'}
        
        epochs = list(range(1, len(train_losses) + 1))
        
        # Calculate convergence metrics
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        best_val_loss = min(val_losses)
        best_epoch = val_losses.index(best_val_loss) + 1
        
        # Check for overfitting
        overfitting_score = 0
        if len(train_losses) > 5:
            # Calculate trend in last 20% of training
            recent_start = max(0, len(train_losses) - len(train_losses) // 5)
            train_trend = np.polyfit(epochs[recent_start:], train_losses[recent_start:], 1)[0]
            val_trend = np.polyfit(epochs[recent_start:], val_losses[recent_start:], 1)[0]
            
            # Overfitting if train loss decreases while val loss increases
            if train_trend < -0.001 and val_trend > 0.001:
                overfitting_score = min(1.0, abs(val_trend) / abs(train_trend))
        
        # Calculate convergence rate
        initial_loss = train_losses[0]
        convergence_rate = (initial_loss - final_train_loss) / initial_loss if initial_loss > 0 else 0
        
        # Analyze stability
        train_variance = np.var(train_losses[-10:]) if len(train_losses) >= 10 else np.var(train_losses)
        val_variance = np.var(val_losses[-10:]) if len(val_losses) >= 10 else np.var(val_losses)
        
        training_analysis = {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'overfitting_score': overfitting_score,
            'convergence_rate': convergence_rate,
            'train_stability': 1.0 / (1.0 + train_variance),
            'val_stability': 1.0 / (1.0 + val_variance),
            'epochs_to_best': best_epoch,
            'early_stopping_opportunity': best_epoch < len(train_losses) * 0.8
        }
        
        # Determine training quality
        if overfitting_score > 0.5:
            training_analysis['training_quality'] = 'Poor - Significant overfitting'
        elif convergence_rate < 0.3:
            training_analysis['training_quality'] = 'Poor - Poor convergence'
        elif final_val_loss > best_val_loss * 1.1:
            training_analysis['training_quality'] = 'Fair - Could benefit from early stopping'
        else:
            training_analysis['training_quality'] = 'Good - Well converged'
        
        self.analysis_results['training_curves'] = training_analysis
        return training_analysis
    
    def analyze_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction quality and errors."""
        self.logger.info("Analyzing prediction quality")
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Analyze error distribution
        errors = y_true - y_pred
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        error_skew = self._calculate_skewness(errors)
        error_kurtosis = self._calculate_kurtosis(errors)
        
        # Check for bias
        bias_score = abs(error_mean) / (np.std(y_true) + 1e-8)
        
        # Analyze prediction range
        pred_range = np.max(y_pred) - np.min(y_pred)
        true_range = np.max(y_true) - np.min(y_true)
        range_ratio = pred_range / (true_range + 1e-8)
        
        # Calculate confidence intervals
        sorted_errors = np.sort(np.abs(errors))
        error_95 = np.percentile(sorted_errors, 95)
        error_99 = np.percentile(sorted_errors, 99)
        
        prediction_analysis = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'error_mean': error_mean,
            'error_std': error_std,
            'error_skewness': error_skew,
            'error_kurtosis': error_kurtosis,
            'bias_score': bias_score,
            'prediction_range': pred_range,
            'true_range': true_range,
            'range_ratio': range_ratio,
            'error_95th_percentile': error_95,
            'error_99th_percentile': error_99
        }
        
        # Determine prediction quality
        if r2 > 0.9:
            prediction_analysis['prediction_quality'] = 'Excellent'
        elif r2 > 0.8:
            prediction_analysis['prediction_quality'] = 'Good'
        elif r2 > 0.6:
            prediction_analysis['prediction_quality'] = 'Fair'
        else:
            prediction_analysis['prediction_quality'] = 'Poor'
        
        # Check for systematic errors
        if bias_score > 0.1:
            prediction_analysis['bias_warning'] = f'Model shows bias (score: {bias_score:.3f})'
        
        if range_ratio < 0.5 or range_ratio > 2.0:
            prediction_analysis['range_warning'] = f'Prediction range differs significantly from true range (ratio: {range_ratio:.3f})'
        
        self.analysis_results['predictions'] = prediction_analysis
        return prediction_analysis
    
    def analyze_feature_importance(self, feature_names: List[str], 
                                  sample_input: torch.Tensor) -> Dict[str, Any]:
        """Analyze feature importance using gradient-based methods."""
        self.logger.info("Analyzing feature importance")
        
        if not feature_names or len(feature_names) != sample_input.shape[-1]:
            return {'error': 'Feature names do not match input dimensions'}
        
        # Use gradient-based feature importance
        sample_input.requires_grad_(True)
        
        # Forward pass
        output = self.model(sample_input)
        
        # Backward pass
        output.backward()
        
        # Get gradients
        gradients = sample_input.grad.abs().mean(dim=0).detach().numpy()
        
        # Normalize gradients
        feature_importance = gradients / (np.sum(gradients) + 1e-8)
        
        # Sort features by importance
        importance_pairs = list(zip(feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top features
        top_features = importance_pairs[:10]
        bottom_features = importance_pairs[-10:]
        
        feature_analysis = {
            'feature_importance': dict(zip(feature_names, feature_importance)),
            'top_features': top_features,
            'bottom_features': bottom_features,
            'importance_entropy': self._calculate_entropy(feature_importance),
            'max_importance': np.max(feature_importance),
            'min_importance': np.min(feature_importance),
            'importance_std': np.std(feature_importance)
        }
        
        # Determine feature utilization
        non_zero_importance = np.sum(feature_importance > 0.01)
        utilization_ratio = non_zero_importance / len(feature_names)
        
        if utilization_ratio < 0.3:
            feature_analysis['utilization_warning'] = f'Low feature utilization ({utilization_ratio:.1%})'
        elif utilization_ratio > 0.9:
            feature_analysis['utilization_warning'] = f'Very high feature utilization ({utilization_ratio:.1%})'
        
        self.analysis_results['feature_importance'] = feature_analysis
        return feature_analysis
    
    def generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate actionable recommendations based on analysis."""
        self.logger.info("Generating recommendations")
        
        recommendations = {
            'architecture': [],
            'training': [],
            'prediction': [],
            'general': []
        }
        
        # Architecture recommendations
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']
            
            if arch['complexity_category'] == 'Very Large':
                recommendations['architecture'].append(
                    "Consider model compression or distillation for deployment"
                )
            
            if arch['parameter_efficiency'] < 0.8:
                recommendations['architecture'].append(
                    "High number of non-trainable parameters - consider architecture optimization"
                )
        
        # Training recommendations
        if 'training_curves' in self.analysis_results:
            training = self.analysis_results['training_curves']
            
            if training['overfitting_score'] > 0.3:
                recommendations['training'].append(
                    f"Significant overfitting detected (score: {training['overfitting_score']:.3f}) - "
                    "consider regularization, early stopping, or data augmentation"
                )
            
            if training['early_stopping_opportunity']:
                recommendations['training'].append(
                    "Early stopping could improve efficiency - best performance at epoch "
                    f"{training['best_epoch']} out of {training.get('total_epochs', 'unknown')}"
                )
            
            if training['val_stability'] < 0.5:
                recommendations['training'].append(
                    "Unstable validation performance - consider learning rate scheduling or "
                    "different optimizer"
                )
        
        # Prediction recommendations
        if 'predictions' in self.analysis_results:
            pred = self.analysis_results['predictions']
            
            if pred['r2_score'] < 0.7:
                recommendations['prediction'].append(
                    f"Low R² score ({pred['r2_score']:.3f}) - consider feature engineering, "
                    "model architecture changes, or more training data"
                )
            
            if pred.get('bias_warning'):
                recommendations['prediction'].append(
                    "Model shows bias - consider balanced sampling or bias correction techniques"
                )
            
            if pred.get('range_warning'):
                recommendations['prediction'].append(
                    "Prediction range issues - consider output scaling or different loss function"
                )
        
        # Feature importance recommendations
        if 'feature_importance' in self.analysis_results:
            feat = self.analysis_results['feature_importance']
            
            if feat.get('utilization_warning'):
                if 'Low feature utilization' in feat['utilization_warning']:
                    recommendations['architecture'].append(
                        "Many features unused - consider feature selection or dimensionality reduction"
                    )
                else:
                    recommendations['architecture'].append(
                        "Very high feature utilization - model might be overfitting to noise"
                    )
        
        # General recommendations
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']
            if arch['model_size_mb'] > 100:
                recommendations['general'].append(
                    "Large model size - consider quantization for deployment"
                )
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive analysis report."""
        self.logger.info("Generating comprehensive analysis report")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"MODEL ANALYSIS REPORT - {self.model_name}")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Architecture Summary
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']
            report_lines.append("ARCHITECTURE ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Parameters: {arch['total_parameters']:,}")
            report_lines.append(f"Trainable Parameters: {arch['trainable_parameters']:,}")
            report_lines.append(f"Model Size: {arch['model_size_mb']:.2f} MB")
            report_lines.append(f"Complexity Category: {arch['complexity_category']}")
            report_lines.append(f"Parameter Efficiency: {arch['parameter_efficiency']:.3f}")
            report_lines.append("")
        
        # Training Analysis
        if 'training_curves' in self.analysis_results:
            training = self.analysis_results['training_curves']
            report_lines.append("TRAINING ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Training Quality: {training['training_quality']}")
            report_lines.append(f"Overfitting Score: {training['overfitting_score']:.3f}")
            report_lines.append(f"Convergence Rate: {training['convergence_rate']:.3f}")
            report_lines.append(f"Best Epoch: {training['best_epoch']}")
            report_lines.append(f"Validation Stability: {training['val_stability']:.3f}")
            report_lines.append("")
        
        # Prediction Analysis
        if 'predictions' in self.analysis_results:
            pred = self.analysis_results['predictions']
            report_lines.append("PREDICTION ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Prediction Quality: {pred['prediction_quality']}")
            report_lines.append(f"R² Score: {pred['r2_score']:.4f}")
            report_lines.append(f"RMSE: {pred['rmse']:.4f}")
            report_lines.append(f"MAE: {pred['mae']:.4f}")
            report_lines.append(f"MAPE: {pred['mape']:.2f}%")
            report_lines.append(f"Bias Score: {pred['bias_score']:.3f}")
            
            if pred.get('bias_warning'):
                report_lines.append(f"⚠️  {pred['bias_warning']}")
            if pred.get('range_warning'):
                report_lines.append(f"⚠️  {pred['range_warning']}")
            report_lines.append("")
        
        # Feature Importance
        if 'feature_importance' in self.analysis_results:
            feat = self.analysis_results['feature_importance']
            report_lines.append("FEATURE IMPORTANCE")
            report_lines.append("-" * 40)
            report_lines.append("Top 5 Features:")
            for i, (feature, importance) in enumerate(feat['top_features'][:5]):
                report_lines.append(f"  {i+1}. {feature}: {importance:.4f}")
            
            if feat.get('utilization_warning'):
                report_lines.append(f"⚠️  {feat['utilization_warning']}")
            report_lines.append("")
        
        # Recommendations
        if 'recommendations' in self.analysis_results:
            recs = self.analysis_results['recommendations']
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            
            for category, recommendations in recs.items():
                if recommendations:
                    report_lines.append(f"{category.upper()}:")
                    for rec in recommendations:
                        report_lines.append(f"  • {rec}")
                    report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate entropy of probability distribution."""
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log(probabilities))


def analyze_model_quality(model: nn.Module, model_name: str, 
                         y_true: Optional[np.ndarray] = None,
                         y_pred: Optional[np.ndarray] = None,
                         train_losses: Optional[List[float]] = None,
                         val_losses: Optional[List[float]] = None,
                         feature_names: Optional[List[str]] = None,
                         sample_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Comprehensive model quality analysis.
    
    Args:
        model: PyTorch model to analyze
        model_name: Name of the model
        y_true: True target values
        y_pred: Predicted target values
        train_losses: Training loss history
        val_losses: Validation loss history
        feature_names: Names of input features
        sample_input: Sample input tensor for feature importance analysis
    
    Returns:
        Dictionary containing all analysis results
    """
    analyzer = ModelAnalyzer(model, model_name)
    
    # Run all analyses
    analyzer.analyze_model_architecture()
    
    if train_losses and val_losses:
        analyzer.analyze_training_curves(train_losses, val_losses)
    
    if y_true is not None and y_pred is not None:
        analyzer.analyze_predictions(y_true, y_pred)
    
    if feature_names and sample_input is not None:
        analyzer.analyze_feature_importance(feature_names, sample_input)
    
    # Generate recommendations
    analyzer.generate_recommendations()
    
    return analyzer.analysis_results


if __name__ == "__main__":
    # Example usage
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(24, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
    
    # Create dummy model and data
    model = DummyModel()
    
    # Generate dummy predictions
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    
    # Analyze model
    results = analyze_model_quality(
        model=model,
        model_name="Dummy LSTM",
        y_true=y_true,
        y_pred=y_pred,
        train_losses=[1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.33, 0.32, 0.31, 0.30],
        val_losses=[1.1, 0.9, 0.7, 0.6, 0.5, 0.45, 0.44, 0.46, 0.48, 0.50],
        feature_names=[f'feature_{i}' for i in range(24)],
        sample_input=torch.randn(1, 30, 24)
    )
    
    # Generate report
    analyzer = ModelAnalyzer(model, "Dummy LSTM")
    analyzer.analysis_results = results
    report = analyzer.generate_report()
    print(report) 