"""
Unit tests for the predictive maintenance models.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path

import sys
sys.path.append('../../')

from src.models.tft_model import TemporalFusionTransformer
from src.models.hybrid_model import HybridCNNBiLSTM
from src.utils.cloud_platform import CloudPlatformDetector, CloudPlatform
from src.utils.checkpoint_manager import DynamicCheckpointManager
from src.utils.validators import ConfigValidator, DataValidator

class TestTemporalFusionTransformer:
    """Test cases for the TFT model."""
    
    @pytest.fixture
    def tft_config(self):
        """Sample TFT configuration."""
        return {
            'model': {
                'input_dim': 10,
                'hidden_dim': 64,
                'num_heads': 8,
                'num_layers': 3,
                'dropout': 0.1,
                'num_classes': 2
            },
            'training': {
                'learning_rate': 0.001
            }
        }
    
    def test_tft_initialization(self, tft_config):
        """Test TFT model initialization."""
        model = TemporalFusionTransformer(tft_config)
        
        assert model.input_dim == 10
        assert model.hidden_dim == 64
        assert model.num_heads == 8
        assert model.num_layers == 3
        assert model.num_classes == 2
    
    def test_tft_forward_pass(self, tft_config):
        """Test TFT forward pass."""
        model = TemporalFusionTransformer(tft_config)
        
        # Create sample input
        batch_size = 4
        sequence_length = 100
        input_tensor = torch.randn(batch_size, sequence_length, 10)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, 2)  # num_classes
        assert output.shape == expected_shape
        
        # Check output is valid (no NaN/inf)
        assert torch.isfinite(output).all()
    
    @pytest.mark.gpu
    def test_tft_gpu_compatibility(self, tft_config, mock_gpu_environment):
        """Test TFT model on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        model = TemporalFusionTransformer(tft_config)
        model = model.cuda()
        
        input_tensor = torch.randn(2, 50, 10).cuda()
        output = model(input_tensor)
        
        assert output.is_cuda
        assert output.shape == (2, 2)

class TestHybridCNNBiLSTM:
    """Test cases for the Hybrid CNN-BiLSTM model."""
    
    @pytest.fixture
    def hybrid_config(self):
        """Sample hybrid model configuration."""
        return {
            'model': {
                'input_dim': 10,
                'cnn_channels': [32, 64, 128],
                'kernel_sizes': [3, 5, 7],
                'lstm_hidden_dim': 128,
                'lstm_num_layers': 2,
                'dropout': 0.1,
                'num_classes': 2
            },
            'training': {
                'learning_rate': 0.001
            }
        }
    
    def test_hybrid_initialization(self, hybrid_config):
        """Test hybrid model initialization."""
        model = HybridCNNBiLSTM(hybrid_config)
        
        assert model.input_dim == 10
        assert len(model.cnn_channels) == 3
        assert model.lstm_hidden_dim == 128
        assert model.lstm_num_layers == 2
        assert model.num_classes == 2
    
    def test_hybrid_forward_pass(self, hybrid_config):
        """Test hybrid model forward pass."""
        model = HybridCNNBiLSTM(hybrid_config)
        
        # Create sample input
        batch_size = 4
        sequence_length = 100
        input_tensor = torch.randn(batch_size, sequence_length, 10)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, 2)  # num_classes
        assert output.shape == expected_shape
        
        # Check output is valid
        assert torch.isfinite(output).all()
    
    def test_hybrid_parameter_count(self, hybrid_config):
        """Test parameter count is reasonable."""
        model = HybridCNNBiLSTM(hybrid_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable by default
        assert total_params < 10_000_000  # Reasonable upper bound

class TestCloudPlatformDetector:
    """Test cases for cloud platform detection."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = CloudPlatformDetector()
        assert detector.platform_info is None
        assert detector._detection_cache is None
    
    def test_detect_platform_caching(self):
        """Test platform detection caching."""
        detector = CloudPlatformDetector()
        
        # First call
        info1 = detector.detect_platform()
        assert info1 is not None
        
        # Second call should return cached result
        info2 = detector.detect_platform()
        assert info1 is info2  # Same object reference
    
    @patch.dict('os.environ', {'GOOGLE_CLOUD_PROJECT': 'test-project'})
    def test_detect_gcp_environment(self):
        """Test GCP environment detection."""
        detector = CloudPlatformDetector()
        platform = detector._check_environment_variables()
        assert platform == CloudPlatform.GCP
    
    @patch.dict('os.environ', {'COLAB_GPU': '1'})
    def test_detect_colab_environment(self):
        """Test Google Colab detection."""
        detector = CloudPlatformDetector()
        platform = detector._check_environment_variables()
        assert platform == CloudPlatform.COLAB
    
    @patch.dict('os.environ', {'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'})
    def test_detect_kaggle_environment(self):
        """Test Kaggle environment detection."""
        detector = CloudPlatformDetector()
        platform = detector._check_environment_variables()
        assert platform == CloudPlatform.KAGGLE
    
    def test_optimal_config_generation(self):
        """Test optimal configuration generation."""
        detector = CloudPlatformDetector()
        detector.detect_platform()
        
        config = detector.get_optimal_config()
        
        # Check required keys
        required_keys = ['platform', 'batch_size', 'num_workers', 'precision']
        for key in required_keys:
            assert key in config
        
        # Check reasonable values
        assert isinstance(config['batch_size'], int)
        assert config['batch_size'] > 0
        assert isinstance(config['num_workers'], int)
        assert config['num_workers'] >= 0
        assert config['precision'] in ['16-mixed', '32-true']

class TestDynamicCheckpointManager:
    """Test cases for checkpoint management."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self, temp_dir):
        """Create temporary checkpoint directory."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir
    
    def test_checkpoint_manager_initialization(self, temp_checkpoint_dir):
        """Test checkpoint manager initialization."""
        manager = DynamicCheckpointManager(
            project_name='test_project',
            storage_config={'type': 'local', 'base_path': str(temp_checkpoint_dir)},
            auto_sync=False
        )
        
        assert manager.project_name == 'test_project'
        assert manager.local_checkpoint_dir.exists()
    
    def test_save_and_load_checkpoint(self, temp_checkpoint_dir):
        """Test saving and loading checkpoints."""
        manager = DynamicCheckpointManager(
            project_name='test_project',
            storage_config={'type': 'local', 'base_path': str(temp_checkpoint_dir)},
            auto_sync=False
        )
        
        # Create dummy checkpoint file
        dummy_checkpoint = temp_checkpoint_dir / "dummy.ckpt"
        dummy_checkpoint.write_text("dummy checkpoint data")
        
        # Save checkpoint
        checkpoint_id = manager.save_checkpoint(
            checkpoint_path=str(dummy_checkpoint),
            model_name='test_model',
            epoch=5,
            step=100,
            metrics={'accuracy': 0.85, 'loss': 0.3},
            config={'model': {'hidden_dim': 64}}
        )
        
        assert checkpoint_id is not None
        assert 'test_model' in checkpoint_id
        
        # Load checkpoint
        loaded_path, loaded_config = manager.load_checkpoint(checkpoint_id)
        
        assert Path(loaded_path).exists()
        assert loaded_config['model']['hidden_dim'] == 64
    
    def test_list_checkpoints(self, temp_checkpoint_dir):
        """Test listing checkpoints."""
        manager = DynamicCheckpointManager(
            project_name='test_project',
            storage_config={'type': 'local', 'base_path': str(temp_checkpoint_dir)},
            auto_sync=False
        )
        
        # Initially empty
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 0
        
        # Add a checkpoint
        dummy_checkpoint = temp_checkpoint_dir / "dummy.ckpt"
        dummy_checkpoint.write_text("dummy data")
        
        manager.save_checkpoint(
            checkpoint_path=str(dummy_checkpoint),
            model_name='test_model',
            epoch=1,
            step=10,
            metrics={'accuracy': 0.7},
            config={}
        )
        
        # Should have one checkpoint
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].model_name == 'test_model'
    
    def test_get_latest_checkpoint(self, temp_checkpoint_dir):
        """Test getting latest checkpoint."""
        manager = DynamicCheckpointManager(
            project_name='test_project',
            storage_config={'type': 'local', 'base_path': str(temp_checkpoint_dir)},
            auto_sync=False
        )
        
        # No checkpoints initially
        latest = manager.get_latest_checkpoint('test_model')
        assert latest is None
        
        # Add checkpoints
        dummy_checkpoint = temp_checkpoint_dir / "dummy.ckpt"
        dummy_checkpoint.write_text("dummy data")
        
        for epoch in [1, 3, 2]:  # Not in order
            manager.save_checkpoint(
                checkpoint_path=str(dummy_checkpoint),
                model_name='test_model',
                epoch=epoch,
                step=epoch * 10,
                metrics={'epoch': epoch},
                config={}
            )
        
        # Should get the latest by timestamp
        latest = manager.get_latest_checkpoint('test_model')
        assert latest is not None
        assert latest.model_name == 'test_model'

class TestValidators:
    """Test cases for validation system."""
    
    def test_config_validator_valid_config(self, sample_config):
        """Test config validation with valid config."""
        report = ConfigValidator.validate_config(sample_config)
        
        # Should have no errors
        assert not report.has_errors
    
    def test_config_validator_missing_sections(self):
        """Test config validation with missing sections."""
        invalid_config = {'model': {'name': 'tft'}}  # Missing data and training sections
        
        report = ConfigValidator.validate_config(invalid_config)
        
        assert report.has_errors
        errors = report.get_errors()
        error_messages = [e.message for e in errors]
        
        assert any('Missing required section: data' in msg for msg in error_messages)
        assert any('Missing required section: training' in msg for msg in error_messages)
    
    def test_config_validator_invalid_model_config(self):
        """Test config validation with invalid model config."""
        invalid_config = {
            'model': {
                'name': 'tft',
                'input_dim': -5,  # Invalid: negative
                'dropout': 1.5    # Invalid: > 1
            },
            'data': {},
            'training': {}
        }
        
        report = ConfigValidator.validate_config(invalid_config)
        
        assert report.has_errors
        errors = report.get_errors()
        error_messages = [e.message for e in errors]
        
        assert any('input_dim must be a positive integer' in msg for msg in error_messages)
        assert any('dropout must be a float between 0 and 1' in msg for msg in error_messages)
    
    def test_data_validator_valid_dataframe(self, sample_ai4i_data):
        """Test data validation with valid DataFrame."""
        report = DataValidator.validate_dataframe(sample_ai4i_data)
        
        # Should have no errors (warnings are okay)
        assert not report.has_errors
    
    def test_data_validator_empty_dataframe(self):
        """Test data validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        report = DataValidator.validate_dataframe(empty_df)
        
        assert report.has_errors
        errors = report.get_errors()
        assert any('DataFrame is empty' in e.message for e in errors)
    
    def test_data_validator_missing_columns(self, sample_ai4i_data):
        """Test data validation with missing required columns."""
        required_columns = ['missing_column', 'another_missing']
        
        report = DataValidator.validate_dataframe(sample_ai4i_data, required_columns)
        
        assert report.has_errors
        errors = report.get_errors()
        assert any('Missing required columns' in e.message for e in errors)
    
    def test_data_validator_class_imbalance_detection(self):
        """Test detection of severe class imbalance."""
        # Create DataFrame with severe imbalance (99% class 0, 1% class 1)
        n_samples = 1000
        imbalanced_data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'failure': np.concatenate([np.zeros(990), np.ones(10)])  # 1% positive class
        })
        
        report = DataValidator.validate_dataframe(imbalanced_data)
        
        warnings = report.get_warnings()
        assert any('Severe class imbalance' in w.message for w in warnings)

# Parametrized tests for different cloud platforms
@pytest.mark.parametrize("platform", [
    CloudPlatform.GCP,
    CloudPlatform.AZURE, 
    CloudPlatform.AWS,
    CloudPlatform.COLAB,
    CloudPlatform.KAGGLE,
    CloudPlatform.LOCAL
])
def test_platform_specific_config(platform, mock_environment_detector):
    """Test platform-specific configuration generation."""
    detector = CloudPlatformDetector()
    
    # Mock platform detection
    with patch.object(detector, '_detect_cloud_platform', return_value=platform):
        platform_info = detector.detect_platform()
        config = detector.get_optimal_config()
        
        # Check platform-specific configurations
        if platform == CloudPlatform.COLAB:
            assert 'colab' in config.get('storage_path', '').lower() or '/content' in config.get('storage_path', '')
        elif platform == CloudPlatform.KAGGLE:
            assert '/kaggle' in config.get('storage_path', '')
        
        # All platforms should have valid batch sizes
        assert isinstance(config['batch_size'], int)
        assert config['batch_size'] > 0

# Integration tests
@pytest.mark.integration
class TestTrainingIntegration:
    """Integration tests for the training pipeline."""
    
    def test_end_to_end_training_setup(self, sample_config, sample_ai4i_data, temp_dir):
        """Test end-to-end training setup without actual training."""
        # This would test the full pipeline setup
        # (Actual training would be too slow for unit tests)
        
        # Validate config
        config_report = ConfigValidator.validate_config(sample_config)
        assert not config_report.has_errors
        
        # Validate data
        data_report = DataValidator.validate_dataframe(sample_ai4i_data)
        assert not data_report.has_errors
        
        # Test checkpoint manager setup
        checkpoint_manager = DynamicCheckpointManager(
            project_name='integration_test',
            storage_config={'type': 'local', 'base_path': str(temp_dir)},
            auto_sync=False
        )
        
        assert checkpoint_manager.local_checkpoint_dir.exists()
        
        # Cleanup
        checkpoint_manager.stop_background_sync()

# Performance tests
@pytest.mark.slow
class TestPerformance:
    """Performance tests for model inference."""
    
    def test_tft_inference_speed(self, tft_config):
        """Test TFT inference speed."""
        model = TemporalFusionTransformer(tft_config)
        model.eval()
        
        # Create larger batch for performance testing
        batch_size = 64
        sequence_length = 100
        input_tensor = torch.randn(batch_size, sequence_length, 10)
        
        import time
        
        # Warm up
        for _ in range(5):
            _ = model(input_tensor)
        
        # Time inference
        start_time = time.time()
        for _ in range(10):
            _ = model(input_tensor)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        samples_per_second = batch_size / avg_time
        
        # Should be reasonably fast (adjust threshold as needed)
        assert samples_per_second > 100  # At least 100 samples/second
    
    def test_hybrid_inference_speed(self, hybrid_config):
        """Test Hybrid model inference speed."""
        model = HybridCNNBiLSTM(hybrid_config)
        model.eval()
        
        batch_size = 64
        sequence_length = 100
        input_tensor = torch.randn(batch_size, sequence_length, 10)
        
        import time
        
        # Warm up
        for _ in range(5):
            _ = model(input_tensor)
        
        # Time inference
        start_time = time.time()
        for _ in range(10):
            _ = model(input_tensor)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        samples_per_second = batch_size / avg_time
        
        # Should be reasonably fast
        assert samples_per_second > 50  # At least 50 samples/second