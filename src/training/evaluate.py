"""
Model evaluation utilities for predictive maintenance.
Comprehensive metrics and analysis for different model types.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from ..utils.helpers import MetricsCalculator, Visualizer

class ModelEvaluator:
    """Comprehensive model evaluation for predictive maintenance."""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # Results storage
        self.results = {}
        self.predictions = {}
        self.attention_weights = {}
        
    def evaluate_model(self, data_loader, device: str = 'cuda') -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                sequences = batch['sequence'].to(device)
                targets = batch['target'].to(device)
                
                # Model predictions
                outputs = self.model(sequences)
                
                if isinstance(outputs, tuple):
                    predictions, classification_outputs = outputs
                    probabilities = torch.sigmoid(classification_outputs)
                    all_probabilities.extend(probabilities.cpu().numpy())
                else:
                    predictions = outputs
                
                # Store predictions and targets
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Store attention weights if available
                if hasattr(self.model, 'get_attention_weights'):
                    attention_weights = self.model.get_attention_weights()
                    if attention_weights is not None:
                        all_attention_weights.append(attention_weights.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities) if all_probabilities else None
        
        # Store results
        self.predictions = {
            'targets': all_targets,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
        
        if all_attention_weights:
            self.attention_weights = np.concatenate(all_attention_weights, axis=0)
        
        # Calculate metrics
        results = self._calculate_comprehensive_metrics(
            all_targets, all_predictions, all_probabilities
        )
        
        self.results = results
        return results
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        results = {}
        
        # Regression metrics
        reg_metrics = self.metrics_calculator.regression_metrics(y_true, y_pred)
        results['regression'] = reg_metrics
        
        # Classification metrics (if dual-head model)
        if y_prob is not None:
            # Convert predictions to binary classification
            threshold = self.config.get('classification_threshold', 0.5)
            y_pred_binary = (y_pred > threshold).astype(int)
            y_true_binary = (y_true > threshold).astype(int)
            
            class_metrics = self.metrics_calculator.classification_metrics(
                y_true_binary, y_pred_binary, y_prob.flatten()
            )
            results['classification'] = class_metrics
        
        # Time series specific metrics
        ts_metrics = self.metrics_calculator.time_series_metrics(y_true, y_pred)
        results['time_series'] = ts_metrics
        
        # Maintenance specific metrics
        maintenance_metrics = self.metrics_calculator.maintenance_specific_metrics(
            y_true, y_pred
        )
        results['maintenance'] = maintenance_metrics
        
        # Summary statistics
        results['summary'] = {
            'n_samples': len(y_true),
            'target_mean': float(np.mean(y_true)),
            'target_std': float(np.std(y_true)),
            'prediction_mean': float(np.mean(y_pred)),
            'prediction_std': float(np.std(y_pred)),
            'correlation': float(np.corrcoef(y_true, y_pred)[0, 1])
        }
        
        return results
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze prediction errors in detail."""
        if not self.predictions:
            raise ValueError("No predictions available. Run evaluate_model first.")
        
        y_true = self.predictions['targets']
        y_pred = self.predictions['predictions']
        
        # Calculate errors
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        # Error statistics
        error_stats = {
            'mean_error': float(np.mean(errors)),
            'mean_absolute_error': float(np.mean(abs_errors)),
            'root_mean_squared_error': float(np.sqrt(np.mean(squared_errors))),
            'median_absolute_error': float(np.median(abs_errors)),
            'error_std': float(np.std(errors)),
            'max_error': float(np.max(abs_errors)),
            'min_error': float(np.min(abs_errors))
        }
        
        # Error percentiles
        error_percentiles = {
            f'error_p{p}': float(np.percentile(abs_errors, p))
            for p in [10, 25, 50, 75, 90, 95, 99]
        }
        
        error_stats.update(error_percentiles)
        
        # Identify worst predictions
        worst_indices = np.argsort(abs_errors)[-10:]  # Top 10 worst predictions
        worst_predictions = {
            'indices': worst_indices.tolist(),
            'true_values': y_true[worst_indices].tolist(),
            'predicted_values': y_pred[worst_indices].tolist(),
            'errors': errors[worst_indices].tolist()
        }
        
        return {
            'error_statistics': error_stats,
            'worst_predictions': worst_predictions
        }
    
    def analyze_attention_patterns(self) -> Optional[Dict[str, Any]]:
        """Analyze attention patterns for interpretability."""
        if not hasattr(self, 'attention_weights') or len(self.attention_weights) == 0:
            return None
        
        attention_weights = self.attention_weights
        
        # Average attention across samples
        avg_attention = np.mean(attention_weights, axis=0)
        
        # Attention statistics
        attention_stats = {
            'shape': attention_weights.shape,
            'mean_attention': float(np.mean(avg_attention)),
            'std_attention': float(np.std(avg_attention)),
            'max_attention': float(np.max(avg_attention)),
            'min_attention': float(np.min(avg_attention))
        }
        
        # Find most attended positions
        if len(avg_attention.shape) >= 2:  # Multi-head attention
            # Average across heads
            if len(avg_attention.shape) == 4:  # (heads, seq_len, seq_len)
                position_attention = np.mean(avg_attention, axis=(0, 2))  # Average across heads and target positions
            else:
                position_attention = np.mean(avg_attention, axis=0)
            
            most_attended_positions = np.argsort(position_attention)[-10:].tolist()
            
            attention_stats.update({
                'most_attended_positions': most_attended_positions,
                'position_attention_scores': position_attention[most_attended_positions].tolist()
            })
        
        return {
            'attention_statistics': attention_stats,
            'average_attention_pattern': avg_attention.tolist()
        }
    
    def generate_visualizations(self, output_dir: str = 'evaluation_plots') -> Dict[str, str]:
        """Generate comprehensive evaluation visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.predictions:
            raise ValueError("No predictions available. Run evaluate_model first.")
        
        y_true = self.predictions['targets']
        y_pred = self.predictions['predictions']
        y_prob = self.predictions.get('probabilities')
        
        plot_paths = {}
        
        # 1. Prediction vs True values scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs True Values')
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        scatter_path = output_path / 'predictions_scatter.png'
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['scatter'] = str(scatter_path)
        
        # 2. Residuals plot
        residuals = y_pred - y_true
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        residuals_path = output_path / 'residuals_plot.png'
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['residuals'] = str(residuals_path)
        
        # 3. Error distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        abs_errors = np.abs(residuals)
        plt.hist(abs_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Absolute Errors')
        plt.ylabel('Frequency')
        plt.title('Absolute Errors Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        error_dist_path = output_path / 'error_distribution.png'
        plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['error_distribution'] = str(error_dist_path)
        
        # 4. Classification metrics (if available)
        if y_prob is not None:
            threshold = self.config.get('classification_threshold', 0.5)
            y_true_binary = (y_true > threshold).astype(int)
            y_pred_binary = (y_pred > threshold).astype(int)
            
            # Confusion matrix
            fig = self.visualizer.plot_confusion_matrix(
                y_true_binary, y_pred_binary, 
                labels=['Normal', 'Failure']
            )
            confusion_path = output_path / 'confusion_matrix.png'
            fig.savefig(confusion_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths['confusion_matrix'] = str(confusion_path)
            
            # ROC curve
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob.flatten())
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            roc_path = output_path / 'roc_curve.png'
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['roc_curve'] = str(roc_path)
        
        # 5. Attention visualization (if available)
        if hasattr(self, 'attention_weights') and len(self.attention_weights) > 0:
            # Plot average attention pattern
            avg_attention = np.mean(self.attention_weights, axis=0)
            
            if len(avg_attention.shape) >= 2:
                plt.figure(figsize=(12, 8))
                
                if len(avg_attention.shape) == 4:  # Multi-head attention
                    # Show attention for first head
                    attention_to_plot = avg_attention[0]
                else:
                    attention_to_plot = avg_attention
                
                sns.heatmap(attention_to_plot, cmap='Blues', cbar=True)
                plt.title('Average Attention Pattern')
                plt.xlabel('Source Position')
                plt.ylabel('Target Position')
                
                attention_path = output_path / 'attention_pattern.png'
                plt.savefig(attention_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['attention'] = str(attention_path)
        
        return plot_paths
    
    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        results_to_save = {
            'metrics': self.results,
            'error_analysis': self.analyze_errors() if self.predictions else None,
            'attention_analysis': self.analyze_attention_patterns(),
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        print(f"Evaluation results saved to {filepath}")
    
    def generate_report(self, output_dir: str = 'evaluation_report') -> str:
        """Generate comprehensive evaluation report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate visualizations
        plot_paths = self.generate_visualizations(str(output_path / 'plots'))
        
        # Save detailed results
        results_path = output_path / 'results.json'
        self.save_results(str(results_path))
        
        # Generate markdown report
        report_path = output_path / 'evaluation_report.md'
        self._generate_markdown_report(str(report_path), plot_paths)
        
        return str(report_path)
    
    def _generate_markdown_report(self, filepath: str, plot_paths: Dict[str, str]):
        """Generate markdown evaluation report."""
        with open(filepath, 'w') as f:
            f.write("# Predictive Maintenance Model Evaluation Report\n\n")
            
            # Model information
            f.write("## Model Information\n")
            f.write(f"- Model Type: {self.config.get('model', {}).get('name', 'Unknown')}\n")
            f.write(f"- Architecture: {self.config.get('model', {}).get('architecture', 'Unknown')}\n\n")
            
            # Summary metrics
            if self.results:
                f.write("## Performance Summary\n\n")
                
                # Regression metrics
                if 'regression' in self.results:
                    reg_metrics = self.results['regression']
                    f.write("### Regression Metrics\n")
                    f.write(f"- RMSE: {reg_metrics.get('rmse', 0):.4f}\n")
                    f.write(f"- MAE: {reg_metrics.get('mae', 0):.4f}\n")
                    f.write(f"- R² Score: {reg_metrics.get('r2_score', 0):.4f}\n")
                    f.write(f"- MAPE: {reg_metrics.get('mape', 0):.2f}%\n\n")
                
                # Classification metrics
                if 'classification' in self.results:
                    class_metrics = self.results['classification']
                    f.write("### Classification Metrics\n")
                    f.write(f"- Accuracy: {class_metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"- Precision: {class_metrics.get('precision', 0):.4f}\n")
                    f.write(f"- Recall: {class_metrics.get('recall', 0):.4f}\n")
                    f.write(f"- F1-Score: {class_metrics.get('f1_score', 0):.4f}\n")
                    f.write(f"- ROC-AUC: {class_metrics.get('roc_auc', 0):.4f}\n\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            for plot_type, plot_path in plot_paths.items():
                f.write(f"### {plot_type.replace('_', ' ').title()}\n")
                f.write(f"![{plot_type}]({Path(plot_path).name})\n\n")
            
            f.write("---\n")
            f.write("*Report generated automatically by Predictive Maintenance ML System*\n")
        
        print(f"Markdown report generated: {filepath}")

def evaluate_model_from_checkpoint(checkpoint_path: str, config: Dict, 
                                 test_data_loader, device: str = 'cuda') -> ModelEvaluator:
    """Evaluate model from checkpoint."""
    # Load model based on config
    model_name = config['model'].get('name', 'tft')
    
    if model_name == 'tft':
        from ..models.tft_model import TemporalFusionTransformer
        model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, config=config)
    elif model_name == 'hybrid_cnn_bilstm':
        from ..models.hybrid_model import HybridCNNBiLSTM
        model = HybridCNNBiLSTM.load_from_checkpoint(checkpoint_path, config=config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.to(device)
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(model, config)
    evaluator.evaluate_model(test_data_loader, device)
    
    return evaluator

if __name__ == "__main__":
    print("Model evaluation utilities loaded successfully!")