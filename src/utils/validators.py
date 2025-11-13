"""
Comprehensive validation system for predictive maintenance ML pipeline.

This module provides validation for:
- Configuration files and parameters
- Data quality and integrity
- Model inputs and outputs
- API requests and responses
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    """Result of a validation check."""
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None

class ValidationReport:
    """Container for validation results."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def add_error(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        """Add an error to the report."""
        self.results.append(ValidationResult(ValidationLevel.ERROR, message, field, value))
    
    def add_warning(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        """Add a warning to the report."""
        self.results.append(ValidationResult(ValidationLevel.WARNING, message, field, value))
    
    def add_info(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        """Add an info to the report."""
        self.results.append(ValidationResult(ValidationLevel.INFO, message, field, value))
    
    @property
    def has_errors(self) -> bool:
        """Check if report contains errors."""
        return any(r.level == ValidationLevel.ERROR for r in self.results)
    
    @property 
    def has_warnings(self) -> bool:
        """Check if report contains warnings."""
        return any(r.level == ValidationLevel.WARNING for r in self.results)
    
    def get_errors(self) -> List[ValidationResult]:
        """Get all errors."""
        return [r for r in self.results if r.level == ValidationLevel.ERROR]
    
    def get_warnings(self) -> List[ValidationResult]:
        """Get all warnings."""
        return [r for r in self.results if r.level == ValidationLevel.WARNING]
    
    def print_summary(self):
        """Print validation summary."""
        errors = self.get_errors()
        warnings = self.get_warnings()
        
        if errors:
            logger.error(f"❌ {len(errors)} validation errors found:")
            for error in errors:
                logger.error(f"  - {error.message}")
        
        if warnings:
            logger.warning(f"⚠️  {len(warnings)} validation warnings found:")
            for warning in warnings:
                logger.warning(f"  - {warning.message}")
        
        if not errors and not warnings:
            logger.info("✅ All validations passed")

class ConfigValidator:
    """Validator for configuration files."""
    
    REQUIRED_SECTIONS = ['model', 'data', 'training']
    
    MODEL_REQUIRED_FIELDS = {
        'tft': ['input_dim', 'hidden_dim', 'num_heads', 'num_layers', 'num_classes'],
        'hybrid': ['input_dim', 'cnn_channels', 'lstm_hidden_dim', 'num_classes']
    }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> ValidationReport:
        """Validate configuration dictionary."""
        report = ValidationReport()
        
        # Check required sections
        for section in ConfigValidator.REQUIRED_SECTIONS:
            if section not in config:
                report.add_error(f"Missing required section: {section}")
        
        # Validate model section
        if 'model' in config:
            ConfigValidator._validate_model_config(config['model'], report)
        
        # Validate data section
        if 'data' in config:
            ConfigValidator._validate_data_config(config['data'], report)
        
        # Validate training section
        if 'training' in config:
            ConfigValidator._validate_training_config(config['training'], report)
        
        return report
    
    @staticmethod
    def _validate_model_config(model_config: Dict[str, Any], report: ValidationReport):
        """Validate model configuration."""
        model_name = model_config.get('name')
        
        if not model_name:
            report.add_error("Model name not specified", "model.name")
            return
        
        if model_name not in ConfigValidator.MODEL_REQUIRED_FIELDS:
            report.add_error(f"Unknown model type: {model_name}", "model.name", model_name)
            return
        
        # Check required fields for model type
        required_fields = ConfigValidator.MODEL_REQUIRED_FIELDS[model_name]
        for field in required_fields:
            if field not in model_config:
                report.add_error(f"Missing required field for {model_name}: {field}", 
                               f"model.{field}")
        
        # Validate specific fields
        if 'input_dim' in model_config:
            input_dim = model_config['input_dim']
            if not isinstance(input_dim, int) or input_dim <= 0:
                report.add_error("input_dim must be a positive integer", 
                               "model.input_dim", input_dim)
        
        if 'num_classes' in model_config:
            num_classes = model_config['num_classes']
            if not isinstance(num_classes, int) or num_classes < 2:
                report.add_error("num_classes must be an integer >= 2",
                               "model.num_classes", num_classes)
        
        if 'dropout' in model_config:
            dropout = model_config['dropout']
            if not isinstance(dropout, (int, float)) or not (0 <= dropout <= 1):
                report.add_error("dropout must be a float between 0 and 1",
                               "model.dropout", dropout)
    
    @staticmethod
    def _validate_data_config(data_config: Dict[str, Any], report: ValidationReport):
        """Validate data configuration."""
        if 'batch_size' in data_config:
            batch_size = data_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                report.add_error("batch_size must be a positive integer",
                               "data.batch_size", batch_size)
            elif batch_size > 1024:
                report.add_warning("Very large batch size may cause memory issues",
                                 "data.batch_size", batch_size)
        
        if 'num_workers' in data_config:
            num_workers = data_config['num_workers']
            if not isinstance(num_workers, int) or num_workers < 0:
                report.add_error("num_workers must be a non-negative integer",
                               "data.num_workers", num_workers)
        
        if 'sequence_length' in data_config:
            seq_len = data_config['sequence_length']
            if not isinstance(seq_len, int) or seq_len <= 0:
                report.add_error("sequence_length must be a positive integer",
                               "data.sequence_length", seq_len)
    
    @staticmethod
    def _validate_training_config(training_config: Dict[str, Any], report: ValidationReport):
        """Validate training configuration."""
        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                report.add_error("learning_rate must be a positive number",
                               "training.learning_rate", lr)
            elif lr > 0.1:
                report.add_warning("Very high learning rate may cause instability",
                                 "training.learning_rate", lr)
        
        if 'max_epochs' in training_config:
            max_epochs = training_config['max_epochs']
            if not isinstance(max_epochs, int) or max_epochs <= 0:
                report.add_error("max_epochs must be a positive integer",
                               "training.max_epochs", max_epochs)
        
        if 'precision' in training_config:
            precision = training_config['precision']
            valid_precisions = ['16-mixed', '32-true', 'bf16-mixed']
            if precision not in valid_precisions:
                report.add_error(f"precision must be one of {valid_precisions}",
                               "training.precision", precision)

class DataValidator:
    """Validator for data quality and integrity."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          target_column: str = 'failure') -> ValidationReport:
        """Validate a pandas DataFrame."""
        report = ValidationReport()
        
        # Check if DataFrame is empty
        if df.empty:
            report.add_error("DataFrame is empty")
            return report
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                report.add_error(f"Missing required columns: {missing_cols}")
        
        # Check for excessive missing values
        missing_pct = df.isnull().sum() / len(df)
        high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing_cols:
            report.add_warning(f"Columns with >50% missing values: {high_missing_cols}")
        
        # Check target column if specified
        if target_column in df.columns:
            DataValidator._validate_target_column(df[target_column], report, target_column)
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            report.add_warning(f"Found {duplicate_count} duplicate rows")
        
        # Check data types
        DataValidator._validate_data_types(df, report)
        
        # Check for outliers
        DataValidator._check_outliers(df, report)
        
        return report
    
    @staticmethod
    def _validate_target_column(target_series: pd.Series, 
                               report: ValidationReport, 
                               column_name: str):
        """Validate target column."""
        # Check for missing values in target
        missing_count = target_series.isnull().sum()
        if missing_count > 0:
            report.add_error(f"Target column '{column_name}' has {missing_count} missing values")
        
        # Check class distribution for binary classification
        if target_series.dtype in ['int64', 'int32', 'bool']:
            unique_values = target_series.dropna().unique()
            if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                # Binary classification
                class_counts = target_series.value_counts()
                minority_ratio = min(class_counts) / sum(class_counts)
                
                if minority_ratio < 0.01:
                    report.add_warning(f"Severe class imbalance: minority class is {minority_ratio:.2%}")
                elif minority_ratio < 0.1:
                    report.add_info(f"Class imbalance detected: minority class is {minority_ratio:.2%}")
    
    @staticmethod
    def _validate_data_types(df: pd.DataFrame, report: ValidationReport):
        """Validate data types in DataFrame."""
        # Check for object columns that should be numeric
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['timestamp', 'Type']:  # Skip known categorical columns
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    report.add_warning(f"Column '{col}' is object type but appears numeric")
                except (ValueError, TypeError):
                    pass  # Legitimately non-numeric
        
        # Check for extremely large numbers that might cause overflow
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].abs().max() > 1e10:
                report.add_warning(f"Column '{col}' has very large values that might cause numerical issues")
    
    @staticmethod
    def _check_outliers(df: pd.DataFrame, report: ValidationReport):
        """Check for outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'failure':  # Skip target column
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_pct = len(outliers) / len(df) * 100
            
            if outlier_pct > 10:
                report.add_warning(f"Column '{col}' has {outlier_pct:.1f}% outliers")

class ModelInputValidator:
    """Validator for model inputs and outputs."""
    
    @staticmethod
    def validate_model_input(input_tensor: np.ndarray, 
                           expected_shape: Tuple[int, ...],
                           expected_dtype: Optional[np.dtype] = None) -> ValidationReport:
        """Validate model input tensor."""
        report = ValidationReport()
        
        # Check shape
        if input_tensor.shape != expected_shape:
            # Allow flexible batch dimension
            if len(input_tensor.shape) == len(expected_shape):
                if input_tensor.shape[1:] != expected_shape[1:]:
                    report.add_error(f"Input shape mismatch. Expected: {expected_shape}, Got: {input_tensor.shape}")
            else:
                report.add_error(f"Input shape mismatch. Expected: {expected_shape}, Got: {input_tensor.shape}")
        
        # Check data type
        if expected_dtype and input_tensor.dtype != expected_dtype:
            report.add_warning(f"Input dtype mismatch. Expected: {expected_dtype}, Got: {input_tensor.dtype}")
        
        # Check for NaN or infinite values
        if np.isnan(input_tensor).any():
            report.add_error("Input contains NaN values")
        
        if np.isinf(input_tensor).any():
            report.add_error("Input contains infinite values")
        
        # Check value ranges
        if np.abs(input_tensor).max() > 1e6:
            report.add_warning("Input contains very large values that might cause numerical instability")
        
        return report
    
    @staticmethod
    def validate_model_output(output_tensor: np.ndarray,
                            task_type: str = 'classification') -> ValidationReport:
        """Validate model output tensor."""
        report = ValidationReport()
        
        if task_type == 'classification':
            # Check for valid probabilities
            if output_tensor.min() < 0 or output_tensor.max() > 1:
                report.add_warning("Classification output contains values outside [0, 1] range")
            
            # Check if probabilities sum to 1 (for multi-class)
            if output_tensor.shape[-1] > 1:
                prob_sums = output_tensor.sum(axis=-1)
                if not np.allclose(prob_sums, 1.0, rtol=1e-3):
                    report.add_warning("Classification probabilities don't sum to 1")
        
        # Check for NaN or infinite values
        if np.isnan(output_tensor).any():
            report.add_error("Output contains NaN values")
        
        if np.isinf(output_tensor).any():
            report.add_error("Output contains infinite values")
        
        return report

class APIValidator:
    """Validator for API requests and responses."""
    
    PREDICTION_REQUEST_SCHEMA = {
        'type': 'object',
        'required': ['features'],
        'properties': {
            'features': {
                'type': 'array',
                'items': {'type': 'number'}
            },
            'model_version': {'type': 'string'},
            'return_probabilities': {'type': 'boolean'}
        }
    }
    
    @staticmethod
    def validate_prediction_request(request_data: Dict[str, Any]) -> ValidationReport:
        """Validate prediction API request."""
        report = ValidationReport()
        
        # Check required fields
        if 'features' not in request_data:
            report.add_error("Missing required field: features")
            return report
        
        features = request_data['features']
        
        # Validate features
        if not isinstance(features, (list, np.ndarray)):
            report.add_error("Features must be a list or array")
        else:
            # Convert to numpy array for validation
            features_array = np.array(features)
            
            # Check for valid numbers
            if not np.isfinite(features_array).all():
                report.add_error("Features contain invalid values (NaN or infinite)")
            
            # Check feature count
            if len(features_array.shape) == 1:
                if len(features_array) == 0:
                    report.add_error("Features array is empty")
            elif len(features_array.shape) == 2:
                if features_array.shape[1] == 0:
                    report.add_error("Features array is empty")
            else:
                report.add_error("Features must be 1D or 2D array")
        
        return report
    
    @staticmethod
    def validate_prediction_response(response_data: Dict[str, Any]) -> ValidationReport:
        """Validate prediction API response."""
        report = ValidationReport()
        
        # Check required fields
        required_fields = ['prediction', 'confidence', 'model_version']
        for field in required_fields:
            if field not in response_data:
                report.add_error(f"Missing required field: {field}")
        
        # Validate prediction
        if 'prediction' in response_data:
            prediction = response_data['prediction']
            if not isinstance(prediction, (int, float, bool)):
                report.add_error("Prediction must be a number or boolean")
        
        # Validate confidence
        if 'confidence' in response_data:
            confidence = response_data['confidence']
            if not isinstance(confidence, (int, float)):
                report.add_error("Confidence must be a number")
            elif not (0 <= confidence <= 1):
                report.add_error("Confidence must be between 0 and 1")
        
        return report

# Convenience functions
def validate_config_file(config_path: Union[str, Path]) -> ValidationReport:
    """Validate configuration file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        report = ValidationReport()
        report.add_error(f"Configuration file not found: {config_path}")
        return report
    
    try:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            report = ValidationReport()
            report.add_error(f"Unsupported configuration file format: {config_path.suffix}")
            return report
        
        return ConfigValidator.validate_config(config)
        
    except Exception as e:
        report = ValidationReport()
        report.add_error(f"Failed to load configuration file: {e}")
        return report

def validate_training_data(df: pd.DataFrame, config: Dict[str, Any]) -> ValidationReport:
    """Validate training data against configuration."""
    report = ValidationReport()
    
    # Get expected columns from config
    required_columns = []
    if 'data' in config and 'required_columns' in config['data']:
        required_columns = config['data']['required_columns']
    
    # Validate DataFrame
    df_report = DataValidator.validate_dataframe(df, required_columns)
    report.results.extend(df_report.results)
    
    # Additional validation based on config
    if 'model' in config:
        expected_input_dim = config['model'].get('input_dim')
        if expected_input_dim:
            # Count numeric columns (excluding target)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != 'failure']
            
            if len(feature_cols) != expected_input_dim:
                report.add_warning(f"Feature count mismatch. Config expects {expected_input_dim}, "
                                 f"data has {len(feature_cols)} features")
    
    return report