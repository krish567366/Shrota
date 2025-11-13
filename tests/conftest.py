"""
Pytest configuration and fixtures for the predictive maintenance ML system.

This module provides shared fixtures and configuration for testing across
different cloud platforms and deployment scenarios.
"""

import os
import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import yaml
from typing import Dict, Any

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from unittest.mock import Mock, MagicMock
import tempfile
import shutil

# Set random seeds for reproducible tests
np.random.seed(42)
torch.manual_seed(42)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
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
            'batch_size': 32,
            'learning_rate': 0.001,
            'max_epochs': 10,
            'patience': 5
        },
        'data': {
            'sequence_length': 100,
            'test_size': 0.2,
            'val_size': 0.2
        },
        'deployment': {
            'platform': 'auto',
            'checkpoint_sync': True,
            'model_registry': 'local'
        }
    }

@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing."""
    n_samples = 1000
    n_features = 10
    sequence_length = 100
    
    # Generate synthetic sensor data
    data = []
    for i in range(n_samples):
        # Create realistic sensor patterns
        time_series = np.random.randn(sequence_length, n_features)
        
        # Add some realistic patterns
        time_series[:, 0] = np.sin(np.linspace(0, 4*np.pi, sequence_length)) + np.random.randn(sequence_length) * 0.1
        time_series[:, 1] = np.cos(np.linspace(0, 4*np.pi, sequence_length)) + np.random.randn(sequence_length) * 0.1
        
        # Add failure indicator (last 20% more likely to be failure)
        failure_prob = 0.05 if i < n_samples * 0.8 else 0.3
        failure = np.random.binomial(1, failure_prob)
        
        data.append({
            'features': time_series,
            'target': failure,
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(hours=i)
        })
    
    return data

@pytest.fixture
def sample_ai4i_data():
    """Generate sample AI4I dataset for testing."""
    n_samples = 1000
    
    data = {
        'air_temperature': np.random.normal(298, 2, n_samples),
        'process_temperature': np.random.normal(308, 1.5, n_samples),
        'rotation_speed': np.random.normal(1500, 100, n_samples),
        'torque': np.random.normal(40, 10, n_samples),
        'tool_wear': np.random.exponential(50, n_samples),
        'Type': np.random.choice(['L', 'M', 'H'], n_samples, p=[0.6, 0.3, 0.1]),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='10min')
    }
    
    # Create realistic failure conditions
    failure_prob = (
        (data['tool_wear'] > 100) * 0.3 +
        (data['torque'] > 60) * 0.2 +
        (data['rotation_speed'] < 1300) * 0.1 +
        (data['process_temperature'] > 310) * 0.15
    )
    
    data['failure'] = np.random.binomial(1, failure_prob, n_samples)
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_gpu_environment():
    """Mock GPU environment for testing."""
    with pytest.MonkeyPatch.context() as m:
        # Mock CUDA availability
        m.setattr(torch.cuda, 'is_available', lambda: True)
        m.setattr(torch.cuda, 'device_count', lambda: 1)
        m.setattr(torch.cuda, 'get_device_name', lambda x: 'NVIDIA A100-SXM4-80GB')
        m.setattr(torch.cuda, 'get_device_properties', 
                 lambda x: Mock(total_memory=85899345920, major=8, minor=0))
        yield

@pytest.fixture
def mock_cpu_environment():
    """Mock CPU-only environment for testing."""
    with pytest.MonkeyPatch.context() as m:
        # Mock no CUDA availability
        m.setattr(torch.cuda, 'is_available', lambda: False)
        m.setattr(torch.cuda, 'device_count', lambda: 0)
        yield

@pytest.fixture
def mock_cloud_storage():
    """Mock cloud storage for testing checkpoint synchronization."""
    storage = {}
    
    class MockCloudStorage:
        def upload(self, local_path: str, remote_path: str):
            with open(local_path, 'rb') as f:
                storage[remote_path] = f.read()
            return True
        
        def download(self, remote_path: str, local_path: str):
            if remote_path in storage:
                with open(local_path, 'wb') as f:
                    f.write(storage[remote_path])
                return True
            return False
        
        def exists(self, remote_path: str):
            return remote_path in storage
        
        def list_files(self, prefix: str):
            return [k for k in storage.keys() if k.startswith(prefix)]
    
    return MockCloudStorage()

@pytest.fixture
def mock_model_registry():
    """Mock model registry for testing."""
    registry = {}
    
    class MockModelRegistry:
        def register_model(self, name: str, version: str, metadata: Dict[str, Any]):
            registry[f"{name}:{version}"] = metadata
            return True
        
        def get_model(self, name: str, version: str = "latest"):
            if version == "latest":
                # Find latest version
                versions = [k for k in registry.keys() if k.startswith(f"{name}:")]
                if not versions:
                    return None
                version = max(versions, key=lambda x: x.split(':')[1])
                return registry[version]
            return registry.get(f"{name}:{version}")
        
        def list_models(self, name_prefix: str = ""):
            return [k for k in registry.keys() if k.startswith(name_prefix)]
    
    return MockModelRegistry()

@pytest.fixture(params=['gcp', 'azure', 'aws', 'local'])
def cloud_platform(request):
    """Parametrized fixture for different cloud platforms."""
    return request.param

@pytest.fixture
def mock_environment_detector():
    """Mock environment detector for cloud platform testing."""
    def _detect_platform(platform_hint=None):
        if platform_hint:
            return platform_hint
        
        # Mock detection logic
        env_vars = os.environ
        if 'GOOGLE_CLOUD_PROJECT' in env_vars:
            return 'gcp'
        elif 'AZURE_SUBSCRIPTION_ID' in env_vars:
            return 'azure'
        elif 'AWS_DEFAULT_REGION' in env_vars:
            return 'aws'
        else:
            return 'local'
    
    return _detect_platform

# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "cloud: Tests for cloud platforms")
    config.addinivalue_line("markers", "slow: Slow running tests")

# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available hardware."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

# Test environment setup
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # Cleanup after test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']