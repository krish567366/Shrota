"""
Utility functions for predictive maintenance ML system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate comprehensive metrics for predictive maintenance models."""
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            })
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))) * 100
        }
        
        return metrics
    
    @staticmethod
    def time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate time series specific metrics."""
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        
        # MASE (Mean Absolute Scaled Error) - simplified version
        naive_forecast = np.roll(y_true, 1)
        naive_forecast[0] = y_true[0]
        mae_naive = np.mean(np.abs(y_true - naive_forecast))
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        mase = mae_forecast / (mae_naive + 1e-8)
        
        return {
            'smape': smape,
            'mase': mase
        }
    
    @staticmethod
    def maintenance_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   time_to_failure: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate maintenance-specific metrics."""
        metrics = {}
        
        # Early detection rate (predictions before actual failure)
        if time_to_failure is not None:
            early_detections = np.sum((y_pred == 1) & (time_to_failure > 0))
            total_failures = np.sum(y_true == 1)
            metrics['early_detection_rate'] = early_detections / total_failures if total_failures > 0 else 0
        
        # False alarm rate
        false_alarms = np.sum((y_pred == 1) & (y_true == 0))
        total_predictions = len(y_pred)
        metrics['false_alarm_rate'] = false_alarms / total_predictions
        
        # Missed failure rate
        missed_failures = np.sum((y_pred == 0) & (y_true == 1))
        total_failures = np.sum(y_true == 1)
        metrics['missed_failure_rate'] = missed_failures / total_failures if total_failures > 0 else 0
        
        return metrics

class Visualizer:
    """Visualization utilities for predictive maintenance."""
    
    @staticmethod
    def plot_time_series(data: pd.DataFrame, 
                        feature_cols: List[str],
                        target_col: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 10)):
        """Plot multiple time series with failure indicators."""
        n_features = len(feature_cols)
        fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
        
        if n_features == 1:
            axes = [axes]
        
        for i, col in enumerate(feature_cols):
            axes[i].plot(data.index, data[col], alpha=0.7, linewidth=1)
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
            
            # Highlight failure points
            if target_col and target_col in data.columns:
                failure_points = data[data[target_col] == 1]
                if not failure_points.empty:
                    axes[i].scatter(failure_points.index, failure_points[col], 
                                  color='red', s=50, alpha=0.8, label='Failure')
        
        if target_col:
            axes[0].legend()
        
        plt.xlabel('Time')
        plt.title('Time Series Data with Failure Indicators')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels or ['Normal', 'Failure'],
                   yticklabels=labels or ['Normal', 'Failure'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str], 
                              importance_scores: np.ndarray,
                              top_k: int = 20,
                              figsize: Tuple[int, int] = (10, 8)):
        """Plot feature importance."""
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[-top_k:]
        
        plt.figure(figsize=figsize)
        plt.barh(range(top_k), importance_scores[sorted_idx])
        plt.yticks(range(top_k), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importance')
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_prediction_timeline(timestamps: np.ndarray,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None,
                               figsize: Tuple[int, int] = (15, 8)):
        """Plot prediction timeline."""
        fig, axes = plt.subplots(2 if y_prob is not None else 1, 1, 
                               figsize=figsize, sharex=True)
        
        if y_prob is None:
            axes = [axes]
        
        # Plot predictions vs actual
        axes[0].plot(timestamps, y_true, 'o-', label='Actual', alpha=0.7)
        axes[0].plot(timestamps, y_pred, 's-', label='Predicted', alpha=0.7)
        axes[0].set_ylabel('Failure Status')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Failure Predictions Timeline')
        
        # Plot probability if available
        if y_prob is not None:
            axes[1].plot(timestamps, y_prob, 'g-', label='Failure Probability', alpha=0.8)
            axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            axes[1].set_ylabel('Probability')
            axes[1].set_xlabel('Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class ConfigManager:
    """Configuration management utilities."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    @staticmethod
    def save_config(config: Dict, output_path: str):
        """Save configuration to YAML file."""
        import yaml
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {output_path}")
    
    @staticmethod
    def merge_configs(*configs: Dict) -> Dict:
        """Merge multiple configuration dictionaries."""
        merged = {}
        
        for config in configs:
            for key, value in config.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = ConfigManager.merge_configs(merged[key], value)
                else:
                    merged[key] = value
        
        return merged

class ExperimentTracker:
    """Track experiments and model performance."""
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.current_experiment = None
        self.experiment_data = {}
    
    def start_experiment(self, experiment_name: str, config: Dict):
        """Start a new experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_experiment = f"{experiment_name}_{timestamp}"
        
        experiment_path = self.experiment_dir / self.current_experiment
        experiment_path.mkdir(exist_ok=True)
        
        # Save experiment config
        self.save_experiment_config(config)
        
        logger.info(f"Started experiment: {self.current_experiment}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        }
        
        if 'metrics' not in self.experiment_data:
            self.experiment_data['metrics'] = []
        
        self.experiment_data['metrics'].append(log_entry)
        
        # Save metrics to file
        metrics_file = self.experiment_dir / self.current_experiment / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.experiment_data['metrics'], f, indent=2)
    
    def log_artifact(self, artifact_name: str, artifact_data: Any):
        """Log artifact (model, plot, etc.) for current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        artifact_path = self.experiment_dir / self.current_experiment / "artifacts"
        artifact_path.mkdir(exist_ok=True)
        
        # Save artifact based on type
        if isinstance(artifact_data, plt.Figure):
            artifact_data.savefig(artifact_path / f"{artifact_name}.png", dpi=300, bbox_inches='tight')
        elif isinstance(artifact_data, (dict, list)):
            with open(artifact_path / f"{artifact_name}.json", 'w') as f:
                json.dump(artifact_data, f, indent=2, default=str)
        else:
            # Generic pickle save
            import pickle
            with open(artifact_path / f"{artifact_name}.pkl", 'wb') as f:
                pickle.dump(artifact_data, f)
        
        logger.info(f"Saved artifact: {artifact_name}")
    
    def save_experiment_config(self, config: Dict):
        """Save experiment configuration."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        config_file = self.experiment_dir / self.current_experiment / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def finish_experiment(self, final_metrics: Optional[Dict] = None):
        """Finish current experiment."""
        if self.current_experiment is None:
            return
        
        if final_metrics:
            self.log_metrics(final_metrics)
        
        # Save experiment summary
        summary = {
            'experiment_name': self.current_experiment,
            'start_time': self.experiment_data.get('start_time'),
            'end_time': datetime.now().isoformat(),
            'final_metrics': final_metrics
        }
        
        summary_file = self.experiment_dir / self.current_experiment / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Finished experiment: {self.current_experiment}")
        self.current_experiment = None
        self.experiment_data = {}

class GPUMonitor:
    """Monitor GPU usage during training."""
    
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_available = True
            self.device_count = pynvml.nvmlDeviceGetCount()
        except ImportError:
            logger.warning("pynvml not available. GPU monitoring disabled.")
            self.gpu_available = False
    
    def get_gpu_info(self) -> Dict:
        """Get current GPU information."""
        if not self.gpu_available:
            return {}
        
        gpu_info = {}
        
        for i in range(self.device_count):
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Memory info
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Utilization
            util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Temperature
            temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
            
            gpu_info[f'gpu_{i}'] = {
                'memory_used': mem_info.used / 1024**3,  # GB
                'memory_total': mem_info.total / 1024**3,  # GB
                'memory_utilization': (mem_info.used / mem_info.total) * 100,
                'gpu_utilization': util.gpu,
                'temperature': temp
            }
        
        return gpu_info
    
    def log_gpu_stats(self) -> Dict:
        """Log GPU statistics."""
        gpu_info = self.get_gpu_info()
        
        if gpu_info:
            for gpu_id, stats in gpu_info.items():
                logger.info(f"{gpu_id}: Memory: {stats['memory_used']:.1f}GB/{stats['memory_total']:.1f}GB "
                          f"({stats['memory_utilization']:.1f}%), GPU: {stats['gpu_utilization']}%, "
                          f"Temp: {stats['temperature']}°C")
        
        return gpu_info

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def ensure_reproducibility(seed: int = 42):
    """Ensure reproducible results."""
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning("PyTorch not available. Skipping PyTorch seed setting.")

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def get_model_size(model) -> Dict[str, Any]:
    """Get model size information."""
    try:
        import torch
        
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Model size in MB
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / 1024**2
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb,
                'model_size_str': f"{model_size_mb:.2f} MB"
            }
    except ImportError:
        pass
    
    return {}

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test metrics calculator
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.4, 0.2, 0.9])
    
    metrics = MetricsCalculator.classification_metrics(y_true, y_pred, y_prob)
    print("Classification metrics:", metrics)
    
    # Test experiment tracker
    tracker = ExperimentTracker()
    config = {"model": "test", "lr": 0.001}
    tracker.start_experiment("test_experiment", config)
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.8})
    tracker.finish_experiment({"final_loss": 0.3})
    
    print("Utility functions test completed!")