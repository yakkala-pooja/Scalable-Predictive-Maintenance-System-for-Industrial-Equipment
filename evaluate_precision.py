import numpy as np
import pandas as pd
import torch
import json
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from models.tft_model import TemporalFusionTransformer
from models.tft_data_module import CMAPSSDataModule
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrecisionEvaluator:
    """Evaluates model precision for critical failure prediction within 25-cycle horizon."""
    
    def __init__(self, model_path: str, data_dir: str = 'transformer_data', 
                 critical_threshold: int = 25, horizon: int = 25):
        """
        Initialize the precision evaluator.
        
        Args:
            model_path: Path to the trained model checkpoint
            data_dir: Directory containing test data
            critical_threshold: RUL threshold for critical failure (default: 25 cycles)
            horizon: Prediction horizon for evaluation (default: 25 cycles)
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.critical_threshold = critical_threshold
        self.horizon = horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load metadata
        with open(f'{data_dir}/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load model
        self.model = self._load_model()
        
        # Load test data
        self.test_data = self._load_test_data()
        
        logger.info(f"Precision evaluator initialized")
        logger.info(f"Critical threshold: {critical_threshold} cycles")
        logger.info(f"Prediction horizon: {horizon} cycles")
        logger.info(f"Device: {self.device}")
    
    def _load_model(self) -> TemporalFusionTransformer:
        """Load the trained TFT model."""
        model = TemporalFusionTransformer(
            num_time_varying_real_vars=self.metadata['num_features'],
            hidden_size=64  # Should match training
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {self.model_path}")
        return model
    
    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data."""
        X_test = np.load(f'{self.data_dir}/X_test.npy')
        y_test = np.load(f'{self.data_dir}/y_test.npy')
        
        logger.info(f"Test data loaded: {X_test.shape}")
        return X_test, y_test
    
    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL for given input data."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), 32):  # Batch processing
                batch = X[i:i+32]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                
                pred = self.model(batch_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                predictions.extend(pred.cpu().numpy().flatten())
        
        return np.array(predictions)
    
    def evaluate_critical_failures(self) -> Dict:
        """
        Evaluate precision for critical failure prediction within the horizon.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting critical failure evaluation...")
        
        # Get predictions
        y_pred = self.predict_rul(self.test_data[0])
        y_true = self.test_data[1]
        
        # Create binary labels for critical failures
        # True critical failure: RUL <= critical_threshold
        # Predicted critical failure: Predicted RUL <= critical_threshold
        y_true_critical = (y_true <= self.critical_threshold).astype(int)
        y_pred_critical = (y_pred <= self.critical_threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true_critical, y_pred_critical, zero_division=0)
        recall = recall_score(y_true_critical, y_pred_critical, zero_division=0)
        f1 = f1_score(y_true_critical, y_pred_critical, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_critical, y_pred_critical)
        
        # Additional metrics
        total_samples = len(y_true)
        actual_critical = np.sum(y_true_critical)
        predicted_critical = np.sum(y_pred_critical)
        true_positives = np.sum((y_true_critical == 1) & (y_pred_critical == 1))
        false_positives = np.sum((y_true_critical == 0) & (y_pred_critical == 1))
        false_negatives = np.sum((y_true_critical == 1) & (y_pred_critical == 0))
        
        # Calculate accuracy within horizon
        horizon_accuracy = self._calculate_horizon_accuracy(y_true, y_pred)
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'total_samples': total_samples,
            'actual_critical_failures': int(actual_critical),
            'predicted_critical_failures': int(predicted_critical),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'horizon_accuracy': horizon_accuracy,
            'critical_threshold': self.critical_threshold,
            'prediction_horizon': self.horizon
        }
        
        logger.info(f"Evaluation completed")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Horizon Accuracy: {horizon_accuracy:.4f}")
        
        return results
    
    def _calculate_horizon_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy for predictions within the horizon.
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            
        Returns:
            Accuracy within the horizon
        """
        # Consider a prediction correct if it's within the horizon
        within_horizon = np.abs(y_true - y_pred) <= self.horizon
        return np.mean(within_horizon)
    
    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            save_path: Optional path to save the report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CRITICAL FAILURE PREDICTION EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Model: {self.model_path}")
        report_lines.append(f"Critical Threshold: {results['critical_threshold']} cycles")
        report_lines.append(f"Prediction Horizon: {results['prediction_horizon']} cycles")
        report_lines.append("")
        
        # Key Metrics
        report_lines.append("KEY METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
        report_lines.append(f"Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)")
        report_lines.append(f"F1-Score: {results['f1_score']:.4f}")
        report_lines.append(f"Horizon Accuracy: {results['horizon_accuracy']:.4f} ({results['horizon_accuracy']*100:.2f}%)")
        report_lines.append("")
        
        # Sample Statistics
        report_lines.append("SAMPLE STATISTICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Samples: {results['total_samples']:,}")
        report_lines.append(f"Actual Critical Failures: {results['actual_critical_failures']:,}")
        report_lines.append(f"Predicted Critical Failures: {results['predicted_critical_failures']:,}")
        report_lines.append("")
        
        # Confusion Matrix
        report_lines.append("CONFUSION MATRIX:")
        report_lines.append("-" * 40)
        cm = results['confusion_matrix']
        report_lines.append("                Predicted")
        report_lines.append("              Critical  Normal")
        report_lines.append(f"Actual Critical  {cm[1,1]:6d}  {cm[1,0]:6d}")
        report_lines.append(f"      Normal     {cm[0,1]:6d}  {cm[0,0]:6d}")
        report_lines.append("")
        
        # Performance Assessment
        report_lines.append("PERFORMANCE ASSESSMENT:")
        report_lines.append("-" * 40)
        if results['precision'] >= 0.90:
            report_lines.append("✅ PRECISION TARGET ACHIEVED: >90% precision for critical failures")
        else:
            report_lines.append("❌ PRECISION TARGET NOT MET: <90% precision for critical failures")
        
        if results['recall'] >= 0.80:
            report_lines.append("✅ RECALL TARGET ACHIEVED: >80% recall for critical failures")
        else:
            report_lines.append("⚠️  RECALL TARGET NOT MET: <80% recall for critical failures")
        
        if results['horizon_accuracy'] >= 0.85:
            report_lines.append("✅ HORIZON ACCURACY TARGET ACHIEVED: >85% accuracy within horizon")
        else:
            report_lines.append("⚠️  HORIZON ACCURACY TARGET NOT MET: <85% accuracy within horizon")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        Create visualization plots for the evaluation results.
        
        Args:
            results: Evaluation results dictionary
            save_path: Optional path to save the plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix Heatmap
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Critical'],
                   yticklabels=['Normal', 'Critical'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Metrics Bar Chart
        metrics = ['Precision', 'Recall', 'F1-Score', 'Horizon Accuracy']
        values = [results['precision'], results['recall'], 
                 results['f1_score'], results['horizon_accuracy']]
        
        bars = axes[0, 1].bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        axes[0, 1].set_title('Evaluation Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Sample Distribution
        labels = ['Normal', 'Critical']
        sizes = [results['total_samples'] - results['actual_critical_failures'], 
                results['actual_critical_failures']]
        colors = ['#2E86AB', '#A23B72']
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Sample Distribution')
        
        # 4. Performance Target Comparison
        targets = [0.90, 0.80, 0.85]  # Precision, Recall, Horizon Accuracy targets
        achieved = [results['precision'], results['recall'], results['horizon_accuracy']]
        metric_names = ['Precision', 'Recall', 'Horizon Acc.']
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, targets, width, label='Target', color='lightgray', alpha=0.7)
        axes[1, 1].bar(x + width/2, achieved, width, label='Achieved', color=['green' if a >= t else 'red' for a, t in zip(achieved, targets)])
        
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Target vs Achieved Performance')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metric_names)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        plt.show()


def main():
    """Main function to run the precision evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Critical Failure Prediction Precision')
    parser.add_argument('--model-path', required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data-dir', default='transformer_data',
                       help='Directory containing test data')
    parser.add_argument('--critical-threshold', type=int, default=25,
                       help='RUL threshold for critical failure')
    parser.add_argument('--horizon', type=int, default=25,
                       help='Prediction horizon for evaluation')
    parser.add_argument('--report-path', default='precision_evaluation_report.txt',
                       help='Path to save the evaluation report')
    parser.add_argument('--plot-path', default='precision_evaluation_plots.png',
                       help='Path to save the evaluation plots')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PrecisionEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        critical_threshold=args.critical_threshold,
        horizon=args.horizon
    )
    
    # Run evaluation
    results = evaluator.evaluate_critical_failures()
    
    # Generate report
    report = evaluator.generate_report(results, args.report_path)
    print(report)
    
    # Create plots
    evaluator.plot_results(results, args.plot_path)
    
    # Final assessment
    if results['precision'] >= 0.90:
        print("\n🎉 SUCCESS: Model achieves >90% precision for critical failure prediction!")
    else:
        print(f"\n⚠️  WARNING: Model precision ({results['precision']*100:.2f}%) is below 90% target.")


if __name__ == "__main__":
    main() 