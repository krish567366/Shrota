"""
Comprehensive data preprocessing pipeline for predictive maintenance datasets.
Supports multiple datasets: AI4I, MetroPT2, CMAPSS, SKF Bearing, Elevator IoT.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import warnings
import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

import scipy.signal
from scipy import stats
from scipy.fft import fft, fftfreq
import pywt
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

class DataPreprocessor:
    """Advanced data preprocessor for predictive maintenance datasets."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load and initial preprocessing of dataset."""
        dataset_config = self.config['datasets'][dataset_name]
        
        if dataset_name == 'ai4i':
            return self._load_ai4i(dataset_config)
        elif dataset_name == 'metropt2':
            return self._load_metropt2(dataset_config)
        elif dataset_name == 'cmapss':
            return self._load_cmapss(dataset_config)
        elif dataset_name == 'skf_bearing':
            return self._load_skf_bearing(dataset_config)
        elif dataset_name == 'elevator_iot':
            return self._load_elevator_iot(dataset_config)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_ai4i(self, config: Dict) -> pd.DataFrame:
        """Load AI4I 2020 Predictive Maintenance Dataset."""
        df = pd.read_csv(config['path'])
        
        # Add synthetic timestamp
        df['timestamp'] = pd.date_range(
            start='2020-01-01', periods=len(df), freq='10min'
        )
        
        # Add domain identifier
        df['domain_id'] = config['domain_id']
        
        # Rename columns for consistency
        column_mapping = {
            'Air temperature [K]': 'air_temperature',
            'Process temperature [K]': 'process_temperature',
            'Rotational speed [rpm]': 'rotation_speed',
            'Torque [Nm]': 'torque',
            'Tool wear [min]': 'tool_wear',
            'Machine failure': 'failure'
        }
        df = df.rename(columns=column_mapping)
        
        return df
    
    def _load_metropt2(self, config: Dict) -> pd.DataFrame:
        """Load MetroPT-2 Dataset."""
        # Assuming CSV format, adjust based on actual format
        data_path = Path(config['path'])
        files = list(data_path.glob('*.csv'))
        
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            df['domain_id'] = config['domain_id']
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _load_cmapss(self, config: Dict) -> pd.DataFrame:
        """Load CMAPSS Turbofan Engine Dataset."""
        data_path = Path(config['path'])
        
        # Load training data
        train_file = data_path / 'train_FD001.txt'
        
        # Column names for CMAPSS dataset
        columns = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + \
                 [f'sensor_{i}' for i in range(1, 22)]
        
        df = pd.read_csv(train_file, sep=' ', header=None, names=columns)
        df = df.dropna(axis=1)  # Remove empty columns
        
        # Calculate RUL (Remaining Useful Life)
        df['RUL'] = df.groupby('unit')['cycle'].transform('max') - df['cycle']
        
        # Create failure indicator (RUL < threshold)
        df['failure'] = (df['RUL'] <= 30).astype(int)
        
        # Add timestamp based on cycle
        df['timestamp'] = pd.to_datetime('2000-01-01') + \
                         pd.to_timedelta(df['cycle'], unit='D')
        
        df['domain_id'] = config['domain_id']
        
        return df
    
    def _load_skf_bearing(self, config: Dict) -> pd.DataFrame:
        """Load SKF Bearing Dataset."""
        data_path = Path(config.get('path', './data/skf_bearing'))
        
        try:
            if data_path.exists():
                # Try to load real SKF bearing data
                if (data_path / 'bearing_data.csv').exists():
                    df = pd.read_csv(data_path / 'bearing_data.csv')
                elif any(data_path.glob('*.csv')):
                    # Load first CSV file found
                    csv_files = list(data_path.glob('*.csv'))
                    df = pd.read_csv(csv_files[0])
                else:
                    raise FileNotFoundError("No CSV files found in SKF bearing data directory")
            else:
                # Generate synthetic bearing vibration data
                logger.warning("SKF bearing data not found, generating synthetic data")
                df = self._generate_synthetic_bearing_data()
                
        except Exception as e:
            logger.warning(f"Failed to load SKF bearing data: {e}. Generating synthetic data.")
            df = self._generate_synthetic_bearing_data()
        
        df['domain_id'] = config.get('domain_id', 'skf_bearing')
        return df
    
    def _generate_synthetic_bearing_data(self) -> pd.DataFrame:
        """Generate synthetic bearing vibration data."""
        n_samples = 10000
        
        # Generate time series
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
        
        # Synthetic bearing vibration features
        data = {
            'timestamp': timestamps,
            'vibration_x': np.random.normal(0, 1, n_samples),
            'vibration_y': np.random.normal(0, 1, n_samples), 
            'vibration_z': np.random.normal(0, 1, n_samples),
            'temperature': np.random.normal(40, 5, n_samples),
            'speed': np.random.normal(1800, 100, n_samples),
            'load': np.random.normal(50, 10, n_samples)
        }
        
        # Add degradation patterns
        degradation = np.linspace(0, 1, n_samples)
        data['vibration_x'] += degradation * np.random.normal(0, 2, n_samples)
        data['vibration_y'] += degradation * np.random.normal(0, 2, n_samples)
        data['temperature'] += degradation * 10
        
        # Create failure events (more frequent towards end)
        failure_prob = 0.01 + degradation * 0.2
        data['failure'] = np.random.binomial(1, failure_prob, n_samples)
        
        return pd.DataFrame(data)
    
    def _load_elevator_iot(self, config: Dict) -> pd.DataFrame:
        """Load Elevator IoT Dataset."""
        data_path = Path(config.get('path', './data/elevator_iot'))
        
        try:
            if data_path.exists():
                # Try to load real Elevator IoT data
                if (data_path / 'elevator_data.csv').exists():
                    df = pd.read_csv(data_path / 'elevator_data.csv')
                elif any(data_path.glob('*.csv')):
                    # Load first CSV file found
                    csv_files = list(data_path.glob('*.csv'))
                    df = pd.read_csv(csv_files[0])
                else:
                    raise FileNotFoundError("No CSV files found in Elevator IoT data directory")
            else:
                # Generate synthetic elevator IoT data
                logger.warning("Elevator IoT data not found, generating synthetic data")
                df = self._generate_synthetic_elevator_data()
                
        except Exception as e:
            logger.warning(f"Failed to load Elevator IoT data: {e}. Generating synthetic data.")
            df = self._generate_synthetic_elevator_data()
        
        df['domain_id'] = config.get('domain_id', 'elevator_iot')
        return df
    
    def _generate_synthetic_elevator_data(self) -> pd.DataFrame:
        """Generate synthetic elevator IoT sensor data."""
        n_samples = 15000
        
        # Generate time series
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='30s')
        
        # Synthetic elevator sensor data
        data = {
            'timestamp': timestamps,
            'motor_current': np.random.normal(12, 2, n_samples),
            'motor_voltage': np.random.normal(380, 10, n_samples),
            'motor_temperature': np.random.normal(65, 8, n_samples),
            'door_cycles': np.random.poisson(50, n_samples),
            'load_weight': np.random.gamma(2, 150, n_samples),
            'cabin_position': np.random.randint(1, 21, n_samples),  # Floor number
            'brake_pressure': np.random.normal(8, 1, n_samples),
            'vibration_level': np.random.exponential(2, n_samples),
            'usage_frequency': np.random.poisson(20, n_samples)
        }
        
        # Add wear patterns based on usage
        cumulative_usage = np.cumsum(data['door_cycles'] + data['usage_frequency'])
        wear_factor = cumulative_usage / np.max(cumulative_usage)
        
        # Increase motor temperature and vibration with wear
        data['motor_temperature'] += wear_factor * 15
        data['vibration_level'] += wear_factor * 3
        data['motor_current'] += wear_factor * np.random.normal(0, 1, n_samples)
        
        # Create failure events (higher probability with more wear)
        failure_prob = 0.005 + wear_factor * 0.05
        data['failure'] = np.random.binomial(1, failure_prob, n_samples)
        
        return pd.DataFrame(data)
    
    def create_sequences(self, df: pd.DataFrame, 
                        sequence_length: int,
                        prediction_horizon: int = 1,
                        stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling."""
        feature_columns = [col for col in df.columns 
                          if col not in ['timestamp', 'failure', 'domain_id']]
        
        X, y = [], []
        
        # Group by unit/machine if available
        group_col = 'unit' if 'unit' in df.columns else None
        
        if group_col:
            for unit in df[group_col].unique():
                unit_data = df[df[group_col] == unit].sort_values('timestamp')
                unit_X, unit_y = self._create_sequences_for_unit(
                    unit_data, feature_columns, sequence_length, 
                    prediction_horizon, stride
                )
                X.extend(unit_X)
                y.extend(unit_y)
        else:
            X, y = self._create_sequences_for_unit(
                df.sort_values('timestamp'), feature_columns, 
                sequence_length, prediction_horizon, stride
            )
        
        return np.array(X), np.array(y)
    
    def _create_sequences_for_unit(self, data: pd.DataFrame,
                                  feature_columns: List[str],
                                  sequence_length: int,
                                  prediction_horizon: int,
                                  stride: int) -> Tuple[List, List]:
        """Create sequences for a single unit/machine."""
        X, y = [], []
        
        for i in range(0, len(data) - sequence_length - prediction_horizon + 1, stride):
            # Input sequence
            sequence = data.iloc[i:i+sequence_length][feature_columns].values
            
            # Target (failure probability at prediction horizon)
            target_idx = i + sequence_length + prediction_horizon - 1
            target = data.iloc[target_idx]['failure']
            
            X.append(sequence)
            y.append(target)
        
        return X, y
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for predictive maintenance."""
        df = df.copy()
        
        # Rolling window statistics
        for window in self.config['preprocessing']['rolling_windows']:
            for stat in self.config['preprocessing']['rolling_stats']:
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col not in ['timestamp', 'domain_id']:
                        df[f'{col}_rolling_{window}_{stat}'] = \
                            df[col].rolling(window=window, min_periods=1).agg(stat)
        
        # Lag features
        for lag in self.config['preprocessing']['lag_features']:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in ['timestamp', 'domain_id']:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Difference features
        for diff in self.config['preprocessing']['diff_features']:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in ['timestamp', 'domain_id']:
                    df[f'{col}_diff_{diff}'] = df[col].diff(diff)
        
        # FFT features for vibration/signal analysis
        if self.config['preprocessing']['fft_features']:
            df = self._add_fft_features(df)
        
        # Wavelet features
        if self.config['preprocessing']['wavelet_features']:
            df = self._add_wavelet_features(df)
        
        return df
    
    def _add_fft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add FFT features for frequency domain analysis."""
        vibration_cols = [col for col in df.columns if 'vibration' in col.lower()]
        
        for col in vibration_cols:
            # Compute FFT
            signal = df[col].fillna(0).values
            fft_values = fft(signal)
            freqs = fftfreq(len(signal))
            
            # Extract dominant frequencies
            n_components = self.config['preprocessing']['fft_components']
            dominant_freqs = np.argsort(np.abs(fft_values))[-n_components:]
            
            for i, freq_idx in enumerate(dominant_freqs):
                df[f'{col}_fft_freq_{i}'] = freqs[freq_idx]
                df[f'{col}_fft_magnitude_{i}'] = np.abs(fft_values[freq_idx])
        
        return df
    
    def _add_wavelet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add wavelet features for time-frequency analysis."""
        wavelet = self.config['preprocessing']['wavelet_name']
        levels = self.config['preprocessing']['wavelet_levels']
        
        signal_cols = [col for col in df.columns 
                      if any(x in col.lower() for x in ['vibration', 'current', 'temperature'])]
        
        for col in signal_cols:
            signal = df[col].fillna(0).values
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=levels)
            
            for i, coeff in enumerate(coeffs):
                # Statistical features of wavelet coefficients
                df[f'{col}_wavelet_L{i}_mean'] = np.mean(coeff)
                df[f'{col}_wavelet_L{i}_std'] = np.std(coeff)
                df[f'{col}_wavelet_L{i}_energy'] = np.sum(coeff**2)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using Isolation Forest."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols 
                       if col not in ['timestamp', 'domain_id', 'failure']]
        
        # Isolation Forest for outlier detection
        contamination = self.config['preprocessing']['outlier_contamination']
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        
        outliers = iso_forest.fit_predict(df[numeric_cols])
        df['is_outlier'] = outliers == -1
        
        # Option to remove or cap outliers
        if self.config['preprocessing'].get('remove_outliers', False):
            df = df[df['is_outlier'] == False]
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features using RobustScaler."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols 
                       if col not in ['timestamp', 'domain_id', 'failure', 'is_outlier']]
        
        scaler_type = self.config['preprocessing']['scaler_type']
        
        if scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        
        if fit:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['features'] = scaler
        else:
            df[numeric_cols] = self.scalers['features'].transform(df[numeric_cols])
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with advanced imputation."""
        strategy = self.config['preprocessing']['missing_strategy']
        
        if strategy == 'interpolate':
            method = self.config['preprocessing']['interpolation_method']
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col not in ['domain_id']:
                    df[col] = df[col].interpolate(method=method)
        
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Comprehensive data quality validation."""
        quality_report = {}
        
        # Missing values
        missing_ratio = df.isnull().sum() / len(df)
        quality_report['missing_values'] = missing_ratio[missing_ratio > 0].to_dict()
        
        # Duplicates
        duplicates = df.duplicated().sum()
        quality_report['duplicates'] = duplicates
        
        # Class balance (if failure column exists)
        if 'failure' in df.columns:
            class_dist = df['failure'].value_counts(normalize=True)
            quality_report['class_balance'] = class_dist.to_dict()
        
        # Feature correlation
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        high_corr = np.where(
            (corr_matrix > self.config['quality_checks']['max_correlation_threshold']) & 
            (corr_matrix < 1.0)
        )
        quality_report['high_correlations'] = list(zip(
            corr_matrix.index[high_corr[0]], 
            corr_matrix.columns[high_corr[1]]
        ))
        
        return quality_report
    
    def process_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """Complete preprocessing pipeline for a dataset."""
        print(f"Processing {dataset_name} dataset...")
        
        # Load data
        df = self.load_dataset(dataset_name)
        
        # Data quality validation
        quality_report = self.validate_data_quality(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Outlier detection
        df = self.detect_outliers(df)
        
        # Normalization
        df = self.normalize_features(df)
        
        print(f"Processed {dataset_name}: {df.shape[0]} samples, {df.shape[1]} features")
        
        return df, quality_report
    
    def create_multi_dataset(self, dataset_names: List[str]) -> pd.DataFrame:
        """Combine multiple datasets for transfer learning."""
        combined_dfs = []
        
        for dataset_name in dataset_names:
            df, _ = self.process_dataset(dataset_name)
            combined_dfs.append(df)
        
        # Combine datasets
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        
        # Ensure consistent feature set across domains
        combined_df = self._harmonize_features(combined_df)
        
        return combined_df
    
    def _harmonize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize features across different domains."""
        # This would implement feature harmonization logic
        # For now, just forward fill missing features
        return df.fillna(method='ffill').fillna(method='bfill')
    
    def save_processed_data(self, df: pd.DataFrame, 
                           output_path: str, 
                           metadata: Dict = None):
        """Save processed data with metadata."""
        output_path = Path(output_path)
        
        # Save main data
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, compression='snappy')
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        
        # Save metadata
        if metadata and self.config['export']['save_metadata']:
            metadata_path = output_path.parent / 'metadata.json'
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        print(f"Saved processed data to {output_path}")


def main():
    """Main preprocessing function."""
    import yaml
    
    # Load configuration
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Process individual datasets
    datasets = ['ai4i']  # Add other datasets as available
    
    for dataset_name in datasets:
        try:
            df, quality_report = preprocessor.process_dataset(dataset_name)
            
            # Save processed data
            output_path = f"data/processed/{dataset_name}_processed.parquet"
            preprocessor.save_processed_data(df, output_path, quality_report)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    # Create combined dataset for transfer learning
    try:
        combined_df = preprocessor.create_multi_dataset(datasets)
        preprocessor.save_processed_data(
            combined_df, 
            "data/processed/combined_dataset.parquet"
        )
    except Exception as e:
        print(f"Error creating combined dataset: {e}")


if __name__ == "__main__":
    main()