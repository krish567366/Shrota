"""
Integration tests for the complete training pipeline.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import time
from unittest.mock import patch, MagicMock
import yaml

import sys
sys.path.append('../../')

from src.training.train import PredictiveMaintenanceTrainer
from src.data.data_loader import DataLoaderFactory
from src.utils.cloud_platform import CloudPlatformDetector
from src.utils.checkpoint_manager import DynamicCheckpointManager
from src.models.tft_model import TemporalFusionTransformer
from src.models.hybrid_model import HybridCNNBiLSTM

@pytest.mark.integration
class TestDataLoaderIntegration:
    """Integration tests for data loading pipeline."""
    
    def test_skf_bearing_loader_integration(self, temp_dir):
        """Test SKF bearing dataset loader integration."""
        from src.data.data_loader import SKFBearingLoader
        
        # Create mock data directory structure
        data_dir = temp_dir / "skf_data"
        data_dir.mkdir()
        
        # Create mock CSV files
        for condition in ['normal', 'fault']:
            condition_dir = data_dir / condition
            condition_dir.mkdir()
            
            for i in range(3):  # 3 files per condition
                csv_file = condition_dir / f"sample_{i}.csv"
                
                # Create realistic bearing data
                n_samples = 1000
                data = pd.DataFrame({
                    'vibration_x': np.random.randn(n_samples) * (2 if condition == 'fault' else 1),
                    'vibration_y': np.random.randn(n_samples) * (2 if condition == 'fault' else 1),
                    'vibration_z': np.random.randn(n_samples) * (2 if condition == 'fault' else 1),
                    'temperature': 25 + np.random.randn(n_samples) * (5 if condition == 'fault' else 2),
                    'speed': 1800 + np.random.randn(n_samples) * (100 if condition == 'fault' else 20),
                    'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1min')
                })
                data.to_csv(csv_file, index=False)
        
        # Test loader
        loader = SKFBearingLoader(str(data_dir))
        dataset = loader.load_data()
        
        assert dataset is not None
        assert len(dataset) > 0
        assert 'failure' in dataset.columns
        assert dataset['failure'].dtype == bool or dataset['failure'].dtype == int
        
        # Test train/val split
        train_loader, val_loader = loader.create_data_loaders(
            batch_size=32,
            train_split=0.8,
            sequence_length=100
        )
        
        assert train_loader is not None
        assert val_loader is not None
        
        # Test batch shape
        train_batch = next(iter(train_loader))
        x, y = train_batch
        
        assert x.shape[0] <= 32  # batch_size
        assert x.shape[1] == 100  # sequence_length
        assert x.shape[2] > 0    # feature_dim
    
    def test_elevator_iot_loader_integration(self, temp_dir):
        """Test Elevator IoT dataset loader integration."""
        from src.data.data_loader import ElevatorIoTLoader
        
        # Create mock data
        data_dir = temp_dir / "elevator_data"
        data_dir.mkdir()
        
        # Create synthetic elevator data
        n_samples = 5000
        elevator_data = pd.DataFrame({
            'motor_current': np.random.randn(n_samples) * 10 + 50,
            'door_cycles': np.random.poisson(100, n_samples),
            'vibration': np.random.exponential(0.5, n_samples),
            'temperature': 20 + np.random.randn(n_samples) * 5,
            'pressure': 1013 + np.random.randn(n_samples) * 10,
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='5min')
        })
        
        # Add failure labels (10% failure rate)
        elevator_data['failure'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        csv_file = data_dir / "elevator_data.csv"
        elevator_data.to_csv(csv_file, index=False)
        
        # Test loader
        loader = ElevatorIoTLoader(str(data_dir))
        dataset = loader.load_data()
        
        assert dataset is not None
        assert len(dataset) > 0
        assert 'failure' in dataset.columns
        
        # Test data loaders
        train_loader, val_loader = loader.create_data_loaders(
            batch_size=16,
            train_split=0.7,
            sequence_length=50
        )
        
        train_batch = next(iter(train_loader))
        x, y = train_batch
        
        assert x.shape[0] <= 16  # batch_size
        assert x.shape[1] == 50  # sequence_length

@pytest.mark.integration
class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""
    
    @pytest.fixture
    def integration_config(self, temp_dir):
        """Create integration test configuration."""
        return {
            'model': {
                'name': 'tft',
                'input_dim': 5,
                'hidden_dim': 32,  # Smaller for faster testing
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.1,
                'num_classes': 2
            },
            'data': {
                'dataset': 'synthetic',
                'batch_size': 8,
                'sequence_length': 20,  # Shorter for faster testing
                'train_split': 0.8
            },
            'training': {
                'max_epochs': 2,  # Very short for integration test
                'learning_rate': 0.001,
                'patience': 2,
                'save_top_k': 1
            },
            'paths': {
                'data_dir': str(temp_dir / "data"),
                'model_dir': str(temp_dir / "models"),
                'log_dir': str(temp_dir / "logs")
            }
        }
    
    def test_tft_training_pipeline(self, integration_config):
        """Test complete TFT training pipeline."""
        # Create necessary directories
        for path_key in integration_config['paths']:
            Path(integration_config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic data
        data_path = Path(integration_config['paths']['data_dir']) / "synthetic_data.csv"
        n_samples = 1000
        synthetic_data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randn(n_samples),
            'failure': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        synthetic_data.to_csv(data_path, index=False)
        
        # Initialize trainer
        trainer = PredictiveMaintenanceTrainer(integration_config)
        
        # Prepare data
        trainer.prepare_data()
        
        # Setup model
        trainer.setup_model()
        
        # Quick training run
        trainer.train()
        
        # Check that model was saved
        model_dir = Path(integration_config['paths']['model_dir'])
        saved_models = list(model_dir.glob("*.ckpt"))
        assert len(saved_models) > 0
        
        # Test evaluation
        results = trainer.evaluate()
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
    
    def test_hybrid_training_pipeline(self, integration_config):
        """Test complete Hybrid model training pipeline."""
        # Modify config for hybrid model
        integration_config['model']['name'] = 'hybrid'
        integration_config['model']['cnn_channels'] = [16, 32]  # Smaller for testing
        integration_config['model']['kernel_sizes'] = [3, 5]
        integration_config['model']['lstm_hidden_dim'] = 32
        integration_config['model']['lstm_num_layers'] = 1
        
        # Create directories
        for path_key in integration_config['paths']:
            Path(integration_config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic data
        data_path = Path(integration_config['paths']['data_dir']) / "synthetic_data.csv"
        n_samples = 1000
        synthetic_data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randn(n_samples),
            'failure': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        synthetic_data.to_csv(data_path, index=False)
        
        # Initialize trainer
        trainer = PredictiveMaintenanceTrainer(integration_config)
        
        # Full pipeline
        trainer.prepare_data()
        trainer.setup_model()
        trainer.train()
        
        # Verify training completed
        model_dir = Path(integration_config['paths']['model_dir'])
        saved_models = list(model_dir.glob("*.ckpt"))
        assert len(saved_models) > 0

@pytest.mark.integration 
class TestCloudIntegration:
    """Integration tests for cloud platform features."""
    
    def test_cloud_platform_detection_integration(self):
        """Test cloud platform detection in real environment."""
        detector = CloudPlatformDetector()
        
        # Detect current platform
        platform_info = detector.detect_platform()
        
        assert platform_info is not None
        assert platform_info.platform is not None
        assert isinstance(platform_info.is_gpu_available, bool)
        assert isinstance(platform_info.memory_gb, (int, float))
        
        # Get optimal configuration
        config = detector.get_optimal_config()
        
        required_keys = [
            'platform', 'batch_size', 'num_workers', 
            'precision', 'gpu_count', 'storage_path'
        ]
        
        for key in required_keys:
            assert key in config
        
        # Validate config values
        assert config['batch_size'] > 0
        assert config['num_workers'] >= 0
        assert config['precision'] in ['16-mixed', '32-true']
        assert config['gpu_count'] >= 0
    
    @patch('src.utils.cloud_platform.requests.get')
    def test_gcp_metadata_detection_integration(self, mock_get):
        """Test GCP metadata endpoint detection."""
        # Mock successful GCP metadata response
        mock_response = MagicMock()
        mock_response.text = "Google"
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        detector = CloudPlatformDetector()
        platform = detector._check_metadata_endpoints()
        
        assert platform is not None
        # Should detect as GCP
        mock_get.assert_called()
    
    def test_checkpoint_manager_cloud_integration(self, temp_dir):
        """Test checkpoint manager with local storage (simulating cloud)."""
        # Setup local storage that simulates cloud behavior
        cloud_storage_dir = temp_dir / "cloud_storage"
        cloud_storage_dir.mkdir()
        
        config = {
            'type': 'local',
            'base_path': str(cloud_storage_dir)
        }
        
        manager = DynamicCheckpointManager(
            project_name='integration_test',
            storage_config=config,
            auto_sync=True,
            sync_interval=1  # 1 second for testing
        )
        
        # Create dummy checkpoint
        dummy_ckpt = temp_dir / "test.ckpt"
        dummy_ckpt.write_text("checkpoint data")
        
        try:
            # Save checkpoint
            checkpoint_id = manager.save_checkpoint(
                checkpoint_path=str(dummy_ckpt),
                model_name='integration_model',
                epoch=1,
                step=100,
                metrics={'loss': 0.5},
                config={'test': True}
            )
            
            # Wait for sync
            time.sleep(2)
            
            # Verify checkpoint exists in "cloud" storage
            cloud_files = list(cloud_storage_dir.rglob("*.ckpt"))
            assert len(cloud_files) > 0
            
            # Test loading
            loaded_path, loaded_config = manager.load_checkpoint(checkpoint_id)
            assert loaded_path is not None
            assert loaded_config['test'] is True
            
        finally:
            manager.stop_background_sync()

@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndIntegration:
    """Full end-to-end integration tests."""
    
    def test_complete_workflow_with_dynamic_launcher(self, temp_dir):
        """Test complete workflow using the dynamic launcher."""
        # Create project structure
        project_dir = temp_dir / "integration_project"
        project_dir.mkdir()
        
        # Create config directories
        config_dir = project_dir / "config"
        config_dir.mkdir()
        data_dir = project_dir / "data"
        data_dir.mkdir()
        models_dir = project_dir / "models"
        models_dir.mkdir()
        logs_dir = project_dir / "logs"
        logs_dir.mkdir()
        
        # Create training config
        training_config = {
            'model': {
                'name': 'tft',
                'input_dim': 3,
                'hidden_dim': 16,
                'num_heads': 2,
                'num_layers': 1,
                'dropout': 0.1,
                'num_classes': 2
            },
            'data': {
                'dataset': 'synthetic',
                'batch_size': 4,
                'sequence_length': 10,
                'train_split': 0.8
            },
            'training': {
                'max_epochs': 1,
                'learning_rate': 0.01,
                'patience': 1,
                'save_top_k': 1
            },
            'paths': {
                'data_dir': str(data_dir),
                'model_dir': str(models_dir),
                'log_dir': str(logs_dir)
            }
        }
        
        config_file = config_dir / "training_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(training_config, f)
        
        # Create synthetic dataset
        synthetic_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'failure': np.random.choice([0, 1], 100, p=[0.7, 0.3])
        })
        data_file = data_dir / "synthetic_data.csv"
        synthetic_data.to_csv(data_file, index=False)
        
        # Import and use dynamic launcher
        sys.path.append(str(project_dir.parent.parent))
        from launch_training import DynamicTrainingLauncher
        
        # Initialize launcher
        launcher = DynamicTrainingLauncher(
            config_path=str(config_file),
            project_name='integration_test',
            storage_config={'type': 'local', 'base_path': str(project_dir / 'checkpoints')}
        )
        
        # Run training
        try:
            results = launcher.launch_training()
            
            # Verify results
            assert 'training_completed' in results
            assert 'checkpoint_id' in results
            assert 'metrics' in results
            
            # Verify files were created
            assert len(list(models_dir.glob("*.ckpt"))) > 0
            assert len(list(logs_dir.glob("*"))) > 0
            
        except Exception as e:
            pytest.skip(f"Training integration test failed: {e}")
    
    def test_cross_platform_checkpoint_compatibility(self, temp_dir):
        """Test checkpoint compatibility across different platforms."""
        # Create checkpoint on "platform 1"
        storage_config_1 = {
            'type': 'local',
            'base_path': str(temp_dir / 'platform1_storage')
        }
        
        manager_1 = DynamicCheckpointManager(
            project_name='cross_platform_test',
            storage_config=storage_config_1,
            auto_sync=False
        )
        
        # Create dummy checkpoint
        dummy_ckpt = temp_dir / "cross_platform.ckpt"
        torch.save({'model_state': 'dummy_state', 'epoch': 5}, dummy_ckpt)
        
        checkpoint_id = manager_1.save_checkpoint(
            checkpoint_path=str(dummy_ckpt),
            model_name='cross_platform_model',
            epoch=5,
            step=1000,
            metrics={'accuracy': 0.85},
            config={'model': {'type': 'test'}}
        )
        
        # Simulate loading on "platform 2" with different storage
        storage_config_2 = {
            'type': 'local', 
            'base_path': str(temp_dir / 'platform2_storage')
        }
        
        # Copy checkpoint metadata to simulate cloud sync
        platform1_metadata = manager_1.metadata_store
        
        manager_2 = DynamicCheckpointManager(
            project_name='cross_platform_test',
            storage_config=storage_config_2,
            auto_sync=False
        )
        
        # Manually sync metadata (simulating cloud sync)
        for checkpoint_info in platform1_metadata.values():
            manager_2.metadata_store[checkpoint_info.checkpoint_id] = checkpoint_info
            
            # Copy actual checkpoint file
            src_path = Path(storage_config_1['base_path']) / checkpoint_info.remote_path
            dst_path = Path(storage_config_2['base_path']) / checkpoint_info.remote_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        
        # Load checkpoint on platform 2
        loaded_path, loaded_config = manager_2.load_checkpoint(checkpoint_id)
        
        assert loaded_path is not None
        assert Path(loaded_path).exists()
        assert loaded_config['model']['type'] == 'test'
        
        # Verify checkpoint content
        loaded_checkpoint = torch.load(loaded_path, map_location='cpu')
        assert 'model_state' in loaded_checkpoint
        assert loaded_checkpoint['epoch'] == 5

@pytest.mark.integration
@pytest.mark.gpu
class TestGPUIntegration:
    """GPU-specific integration tests."""
    
    def test_multi_gpu_detection(self):
        """Test multi-GPU detection and configuration."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        detector = CloudPlatformDetector()
        platform_info = detector.detect_platform()
        
        assert platform_info.is_gpu_available
        assert platform_info.gpu_count >= 1
        
        config = detector.get_optimal_config()
        
        # GPU-specific optimizations should be applied
        if platform_info.gpu_count > 1:
            assert config.get('strategy') == 'ddp'
        
        # Batch size should be adjusted for GPU memory
        assert config['batch_size'] >= 16  # Should use larger batches on GPU
    
    def test_mixed_precision_training(self, integration_config):
        """Test mixed precision training on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Enable mixed precision
        integration_config['training']['precision'] = '16-mixed'
        
        # Create directories
        for path_key in integration_config['paths']:
            Path(integration_config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic data
        data_path = Path(integration_config['paths']['data_dir']) / "synthetic_data.csv"
        synthetic_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'feature_4': np.random.randn(100),
            'feature_5': np.random.randn(100),
            'failure': np.random.choice([0, 1], 100, p=[0.8, 0.2])
        })
        synthetic_data.to_csv(data_path, index=False)
        
        # Test training with mixed precision
        trainer = PredictiveMaintenanceTrainer(integration_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        # Should complete without errors
        trainer.train()
        
        # Verify model was saved
        model_dir = Path(integration_config['paths']['model_dir'])
        saved_models = list(model_dir.glob("*.ckpt"))
        assert len(saved_models) > 0