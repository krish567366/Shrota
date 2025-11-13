"""
Advanced data loaders for predictive maintenance datasets.
Optimized for A100 GPU training with efficient data streaming.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import dask.dataframe as dd
from sklearn.model_selection import train_test_split

class PredictiveMaintenanceDataset(Dataset):
    """PyTorch Dataset for predictive maintenance data."""
    
    def __init__(self, 
                 sequences: np.ndarray,
                 targets: np.ndarray,
                 static_features: Optional[np.ndarray] = None,
                 transform: Optional[callable] = None):
        """
        Args:
            sequences: Time series sequences (N, seq_len, n_features)
            targets: Target values (N,) or (N, n_targets)
            static_features: Static features (N, n_static_features)
            transform: Optional data augmentation transform
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.static_features = torch.FloatTensor(static_features) if static_features is not None else None
        self.transform = transform
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # Apply augmentation if specified
        if self.transform:
            sequence = self.transform(sequence)
        
        sample = {
            'sequence': sequence,
            'target': target
        }
        
        if self.static_features is not None:
            sample['static'] = self.static_features[idx]
        
        return sample

class TimeSeriesAugmentation:
    """Data augmentation for time series data."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to sequence."""
        if np.random.random() < 0.5:  # 50% chance to apply augmentation
            augmentation_type = np.random.choice([
                'gaussian_noise', 'time_warping', 'magnitude_warping', 'window_slicing'
            ])
            
            if augmentation_type == 'gaussian_noise':
                return self._add_gaussian_noise(sequence)
            elif augmentation_type == 'time_warping':
                return self._time_warping(sequence)
            elif augmentation_type == 'magnitude_warping':
                return self._magnitude_warping(sequence)
            elif augmentation_type == 'window_slicing':
                return self._window_slicing(sequence)
        
        return sequence
    
    def _add_gaussian_noise(self, sequence: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to sequence."""
        noise_config = self.config['augmentation']['gaussian_noise']
        if not noise_config['enabled']:
            return sequence
            
        std = noise_config['std']
        noise = torch.randn_like(sequence) * std
        return sequence + noise
    
    def _time_warping(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply time warping augmentation."""
        # Simplified time warping implementation
        # In practice, you'd use more sophisticated warping
        return sequence
    
    def _magnitude_warping(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply magnitude warping augmentation."""
        # Simplified magnitude warping implementation
        scale_factor = 1.0 + 0.1 * torch.randn(1).item()
        return sequence * scale_factor
    
    def _window_slicing(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply window slicing augmentation."""
        seq_len = sequence.shape[0]
        slice_len = int(seq_len * 0.9)  # Keep 90% of sequence
        start_idx = np.random.randint(0, seq_len - slice_len)
        
        # Interpolate to original length
        sliced = sequence[start_idx:start_idx + slice_len]
        return torch.nn.functional.interpolate(
            sliced.unsqueeze(0).transpose(1, 2), 
            size=seq_len, 
            mode='linear'
        ).transpose(1, 2).squeeze(0)

class PredictiveMaintenanceDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for predictive maintenance."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.data_config = config['data']
        self.augmentation = TimeSeriesAugmentation(config)
        
        # Data storage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Feature dimensions
        self.n_features = None
        self.sequence_length = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        if stage == 'fit' or stage is None:
            # Load processed data
            train_data = self._load_data(self.data_config['train_path'])
            val_data = self._load_data(self.data_config['val_path'])
            
            # Create datasets
            self.train_dataset = self._create_dataset(train_data, augment=True)
            self.val_dataset = self._create_dataset(val_data, augment=False)
            
        if stage == 'test' or stage is None:
            test_data = self._load_data(self.data_config['test_path'])
            self.test_dataset = self._create_dataset(test_data, augment=False)
    
    def _load_data(self, path: str) -> Dict:
        """Load preprocessed data."""
        if path.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        
        # Extract sequences and targets
        # This assumes data is already in sequence format
        # Adjust based on your data structure
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'failure', 'domain_id', 'unit']]
        
        return {
            'features': df[feature_cols].values,
            'targets': df['failure'].values if 'failure' in df.columns else None,
            'metadata': df[['timestamp', 'domain_id']].values if 'domain_id' in df.columns else None
        }
    
    def _create_dataset(self, data: Dict, augment: bool = False) -> PredictiveMaintenanceDataset:
        """Create PyTorch dataset from loaded data."""
        sequences = data['features']
        targets = data['targets']
        
        # Reshape if needed (assuming data is flattened)
        if len(sequences.shape) == 2:
            # Reshape to (n_samples, seq_len, n_features)
            n_samples = len(sequences)
            seq_len = self.config['preprocessing']['sequence_length']
            n_features = sequences.shape[1] // seq_len
            
            sequences = sequences.reshape(n_samples, seq_len, n_features)
        
        # Store dimensions
        self.sequence_length = sequences.shape[1]
        self.n_features = sequences.shape[2]
        
        transform = self.augmentation if augment else None
        
        return PredictiveMaintenanceDataset(
            sequences=sequences,
            targets=targets,
            transform=transform
        )
    
    def train_dataloader(self) -> DataLoader:
        """Training data loader optimized for A100."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
            pin_memory=self.data_config['pin_memory'],
            persistent_workers=self.data_config['persistent_workers'],
            prefetch_factor=self.data_config['prefetch_factor'],
            drop_last=True  # For consistent batch sizes
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers'],
            pin_memory=self.data_config['pin_memory'],
            persistent_workers=self.data_config['persistent_workers'],
            prefetch_factor=self.data_config['prefetch_factor']
        )
    
    def test_dataloader(self) -> DataLoader:
        """Test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers'],
            pin_memory=self.data_config['pin_memory']
        )

class StreamingDataLoader:
    """Streaming data loader for very large datasets using Dask."""
    
    def __init__(self, data_path: str, batch_size: int, chunk_size: int = 10000):
        self.data_path = data_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # Load data with Dask for out-of-core processing
        self.df = dd.read_parquet(data_path)
        self.n_chunks = len(self.df) // chunk_size
        
    def __iter__(self):
        """Iterate through data chunks."""
        for i in range(self.n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(self.df))
            
            # Load chunk into memory
            chunk = self.df.iloc[start_idx:end_idx].compute()
            
            # Create batches from chunk
            for batch_start in range(0, len(chunk), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(chunk))
                batch_data = chunk.iloc[batch_start:batch_end]
                
                yield self._prepare_batch(batch_data)
    
    def _prepare_batch(self, batch_data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        feature_cols = [col for col in batch_data.columns 
                       if col not in ['timestamp', 'failure', 'domain_id']]
        
        sequences = torch.FloatTensor(batch_data[feature_cols].values)
        targets = torch.FloatTensor(batch_data['failure'].values) if 'failure' in batch_data.columns else None
        
        return {
            'sequences': sequences,
            'targets': targets
        }

class MultiDomainDataModule(pl.LightningDataModule):
    """DataModule for multi-domain training (transfer learning across datasets)."""
    
    def __init__(self, config: Dict, domain_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.config = config
        self.domain_weights = domain_weights or {}
        
        # Domain-specific data loaders
        self.domain_loaders = {}
        
    def setup(self, stage: Optional[str] = None):
        """Setup multi-domain datasets."""
        datasets = self.config['datasets']
        
        for domain_name, domain_config in datasets.items():
            # Create domain-specific data module
            domain_data_module = PredictiveMaintenanceDataModule(self.config)
            domain_data_module.setup(stage)
            
            self.domain_loaders[domain_name] = domain_data_module
    
    def train_dataloader(self) -> Dict[str, DataLoader]:
        """Return dictionary of domain-specific training loaders."""
        return {
            domain: loader.train_dataloader() 
            for domain, loader in self.domain_loaders.items()
        }
    
    def val_dataloader(self) -> Dict[str, DataLoader]:
        """Return dictionary of domain-specific validation loaders."""
        return {
            domain: loader.val_dataloader() 
            for domain, loader in self.domain_loaders.items()
        }

def create_data_splits(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create temporal train/validation/test splits."""
    split_config = config['data_splits']
    
    if split_config['temporal_split']:
        # Sort by timestamp for temporal splitting
        df = df.sort_values('timestamp')
        
        n_samples = len(df)
        train_size = int(n_samples * split_config['train_ratio'])
        val_size = int(n_samples * split_config['val_ratio'])
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
    else:
        # Random splitting with stratification
        stratify = df[split_config['stratify_column']] if split_config['stratify'] else None
        
        train_df, temp_df = train_test_split(
            df, 
            test_size=1 - split_config['train_ratio'],
            stratify=stratify,
            random_state=42
        )
        
        val_test_ratio = split_config['val_ratio'] / (split_config['val_ratio'] + split_config['test_ratio'])
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_test_ratio,
            stratify=temp_df[split_config['stratify_column']] if split_config['stratify'] else None,
            random_state=42
        )
    
    return train_df, val_df, test_df

def main():
    """Test data loading functionality."""
    import yaml
    
    # Load configuration
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data module
    data_module = PredictiveMaintenanceDataModule(config)
    data_module.setup()
    
    # Test data loading
    train_loader = data_module.train_dataloader()
    
    print("Data module created successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Batch size: {config['data']['batch_size']}")
    
    # Test one batch
    for batch in train_loader:
        print(f"Sequence shape: {batch['sequence'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        break

if __name__ == "__main__":
    main()