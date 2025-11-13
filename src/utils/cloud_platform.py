"""
Dynamic Cloud Platform Detection and Management

This module provides automatic detection of cloud platforms and configures
the system accordingly for seamless deployment across different environments.
"""

import os
import sys
import json
import socket
import requests
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CloudPlatform(Enum):
    """Supported cloud platforms."""
    GCP = "gcp"
    AZURE = "azure"
    AWS = "aws"
    COLAB = "colab"
    KAGGLE = "kaggle"
    PAPERSPACE = "paperspace"
    LOCAL = "local"
    UNKNOWN = "unknown"

@dataclass
class PlatformInfo:
    """Information about the current platform."""
    platform: CloudPlatform
    instance_type: Optional[str] = None
    gpu_info: Optional[Dict[str, Any]] = None
    storage_info: Optional[Dict[str, Any]] = None
    compute_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class CloudPlatformDetector:
    """Detects the current cloud platform and configures the system accordingly."""
    
    def __init__(self):
        self.platform_info: Optional[PlatformInfo] = None
        self._detection_cache: Optional[CloudPlatform] = None
    
    def detect_platform(self, force_refresh: bool = False) -> PlatformInfo:
        """
        Detect the current cloud platform.
        
        Args:
            force_refresh: Force re-detection even if cached
            
        Returns:
            PlatformInfo object with platform details
        """
        if self.platform_info and not force_refresh:
            return self.platform_info
        
        platform = self._detect_cloud_platform()
        gpu_info = self._detect_gpu_info()
        storage_info = self._detect_storage_info()
        compute_info = self._detect_compute_info()
        instance_type = self._detect_instance_type(platform)
        metadata = self._collect_platform_metadata(platform)
        
        self.platform_info = PlatformInfo(
            platform=platform,
            instance_type=instance_type,
            gpu_info=gpu_info,
            storage_info=storage_info,
            compute_info=compute_info,
            metadata=metadata
        )
        
        logger.info(f"Detected platform: {platform.value}")
        logger.info(f"Instance type: {instance_type}")
        logger.info(f"GPU info: {gpu_info}")
        
        return self.platform_info
    
    def _detect_cloud_platform(self) -> CloudPlatform:
        """Detect the specific cloud platform."""
        if self._detection_cache:
            return self._detection_cache
        
        # Check environment variables first
        env_platform = self._check_environment_variables()
        if env_platform != CloudPlatform.UNKNOWN:
            self._detection_cache = env_platform
            return env_platform
        
        # Check metadata endpoints
        metadata_platform = self._check_metadata_endpoints()
        if metadata_platform != CloudPlatform.UNKNOWN:
            self._detection_cache = metadata_platform
            return metadata_platform
        
        # Check file system indicators
        filesystem_platform = self._check_filesystem_indicators()
        if filesystem_platform != CloudPlatform.UNKNOWN:
            self._detection_cache = filesystem_platform
            return filesystem_platform
        
        # Check process indicators
        process_platform = self._check_process_indicators()
        if process_platform != CloudPlatform.UNKNOWN:
            self._detection_cache = process_platform
            return process_platform
        
        self._detection_cache = CloudPlatform.LOCAL
        return CloudPlatform.LOCAL
    
    def _check_environment_variables(self) -> CloudPlatform:
        """Check environment variables for platform indicators."""
        env_vars = os.environ
        
        # Google Cloud Platform
        if any(key in env_vars for key in [
            'GOOGLE_CLOUD_PROJECT', 'GCP_PROJECT', 'GOOGLE_APPLICATION_CREDENTIALS'
        ]):
            return CloudPlatform.GCP
        
        # Google Colab
        if 'COLAB_GPU' in env_vars or 'COLAB_TPU_ADDR' in env_vars:
            return CloudPlatform.COLAB
        
        # Azure
        if any(key in env_vars for key in [
            'AZURE_SUBSCRIPTION_ID', 'AZURE_CLIENT_ID', 'AZURE_TENANT_ID'
        ]):
            return CloudPlatform.AZURE
        
        # AWS
        if any(key in env_vars for key in [
            'AWS_DEFAULT_REGION', 'AWS_REGION', 'AWS_ACCESS_KEY_ID'
        ]):
            return CloudPlatform.AWS
        
        # Kaggle
        if 'KAGGLE_KERNEL_RUN_TYPE' in env_vars:
            return CloudPlatform.KAGGLE
        
        # Paperspace
        if 'PS_API_KEY' in env_vars or 'PAPERSPACE_API_KEY' in env_vars:
            return CloudPlatform.PAPERSPACE
        
        return CloudPlatform.UNKNOWN
    
    def _check_metadata_endpoints(self) -> CloudPlatform:
        """Check cloud metadata endpoints."""
        try:
            # GCP metadata endpoint
            if self._check_endpoint('http://metadata.google.internal/computeMetadata/v1/'):
                return CloudPlatform.GCP
            
            # Azure metadata endpoint
            if self._check_endpoint('http://169.254.169.254/metadata/instance'):
                return CloudPlatform.AZURE
            
            # AWS metadata endpoint
            if self._check_endpoint('http://169.254.169.254/latest/meta-data/'):
                return CloudPlatform.AWS
                
        except Exception as e:
            logger.debug(f"Error checking metadata endpoints: {e}")
        
        return CloudPlatform.UNKNOWN
    
    def _check_endpoint(self, url: str, timeout: float = 2.0) -> bool:
        """Check if a metadata endpoint is accessible."""
        try:
            headers = {'Metadata-Flavor': 'Google'} if 'google.internal' in url else {}
            response = requests.get(url, headers=headers, timeout=timeout)
            return response.status_code == 200
        except:
            return False
    
    def _check_filesystem_indicators(self) -> CloudPlatform:
        """Check filesystem for platform indicators."""
        # Google Colab
        if Path('/content').exists() and Path('/usr/local/cuda').exists():
            return CloudPlatform.COLAB
        
        # Kaggle
        if Path('/kaggle').exists():
            return CloudPlatform.KAGGLE
        
        # Check for cloud-specific mount points
        if Path('/mnt/batch').exists():  # Azure Batch
            return CloudPlatform.AZURE
        
        return CloudPlatform.UNKNOWN
    
    def _check_process_indicators(self) -> CloudPlatform:
        """Check running processes for platform indicators."""
        try:
            # Check for Google Cloud SDK
            result = subprocess.run(['which', 'gcloud'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return CloudPlatform.GCP
            
            # Check for Azure CLI
            result = subprocess.run(['which', 'az'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return CloudPlatform.AZURE
            
            # Check for AWS CLI
            result = subprocess.run(['which', 'aws'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return CloudPlatform.AWS
                
        except Exception as e:
            logger.debug(f"Error checking processes: {e}")
        
        return CloudPlatform.UNKNOWN
    
    def _detect_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Detect available GPU information."""
        gpu_info = {}
        
        try:
            import torch
            gpu_info['torch_cuda_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                gpu_info['device_count'] = torch.cuda.device_count()
                gpu_info['devices'] = []
                
                for i in range(torch.cuda.device_count()):
                    device_info = {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'properties': {
                            'total_memory': torch.cuda.get_device_properties(i).total_memory,
                            'major': torch.cuda.get_device_properties(i).major,
                            'minor': torch.cuda.get_device_properties(i).minor,
                        }
                    }
                    gpu_info['devices'].append(device_info)
                
                gpu_info['current_device'] = torch.cuda.current_device()
                gpu_info['memory_info'] = {
                    'allocated': torch.cuda.memory_allocated(),
                    'cached': torch.cuda.memory_reserved()
                }
        except ImportError:
            gpu_info['torch_available'] = False
        except Exception as e:
            logger.warning(f"Error detecting GPU info: {e}")
        
        # Try nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info['nvidia_smi'] = result.stdout.strip().split('\n')
        except Exception:
            pass
        
        return gpu_info if gpu_info else None
    
    def _detect_storage_info(self) -> Dict[str, Any]:
        """Detect storage information."""
        storage_info = {}
        
        try:
            import shutil
            # Get disk usage for common mount points
            for path in ['/', '/tmp', '/home', '/content', '/kaggle']:
                if Path(path).exists():
                    usage = shutil.disk_usage(path)
                    storage_info[path] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free
                    }
        except Exception as e:
            logger.warning(f"Error detecting storage info: {e}")
        
        return storage_info
    
    def _detect_compute_info(self) -> Dict[str, Any]:
        """Detect compute information."""
        compute_info = {
            'cpu_count': os.cpu_count(),
            'hostname': socket.gethostname(),
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        try:
            # Get memory info
            import psutil
            memory = psutil.virtual_memory()
            compute_info['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percentage': memory.percent
            }
        except ImportError:
            pass
        
        return compute_info
    
    def _detect_instance_type(self, platform: CloudPlatform) -> Optional[str]:
        """Detect the specific instance type for the platform."""
        try:
            if platform == CloudPlatform.GCP:
                return self._get_gcp_instance_type()
            elif platform == CloudPlatform.AZURE:
                return self._get_azure_instance_type()
            elif platform == CloudPlatform.AWS:
                return self._get_aws_instance_type()
        except Exception as e:
            logger.debug(f"Error detecting instance type: {e}")
        
        return None
    
    def _get_gcp_instance_type(self) -> Optional[str]:
        """Get GCP instance type from metadata."""
        try:
            response = requests.get(
                'http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                headers={'Metadata-Flavor': 'Google'},
                timeout=5
            )
            if response.status_code == 200:
                # Extract instance type from full path
                return response.text.split('/')[-1]
        except:
            pass
        return None
    
    def _get_azure_instance_type(self) -> Optional[str]:
        """Get Azure instance type from metadata."""
        try:
            response = requests.get(
                'http://169.254.169.254/metadata/instance/compute/vmSize',
                headers={'Metadata': 'true'},
                timeout=5
            )
            if response.status_code == 200:
                return response.text
        except:
            pass
        return None
    
    def _get_aws_instance_type(self) -> Optional[str]:
        """Get AWS instance type from metadata."""
        try:
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-type',
                timeout=5
            )
            if response.status_code == 200:
                return response.text
        except:
            pass
        return None
    
    def _collect_platform_metadata(self, platform: CloudPlatform) -> Dict[str, Any]:
        """Collect additional platform-specific metadata."""
        metadata = {'platform': platform.value}
        
        try:
            if platform == CloudPlatform.GCP:
                metadata.update(self._get_gcp_metadata())
            elif platform == CloudPlatform.AZURE:
                metadata.update(self._get_azure_metadata())
            elif platform == CloudPlatform.AWS:
                metadata.update(self._get_aws_metadata())
            elif platform == CloudPlatform.COLAB:
                metadata.update(self._get_colab_metadata())
            elif platform == CloudPlatform.KAGGLE:
                metadata.update(self._get_kaggle_metadata())
        except Exception as e:
            logger.debug(f"Error collecting metadata: {e}")
        
        return metadata
    
    def _get_gcp_metadata(self) -> Dict[str, Any]:
        """Get GCP-specific metadata."""
        metadata = {}
        try:
            # Get project ID
            response = requests.get(
                'http://metadata.google.internal/computeMetadata/v1/project/project-id',
                headers={'Metadata-Flavor': 'Google'},
                timeout=5
            )
            if response.status_code == 200:
                metadata['project_id'] = response.text
            
            # Get zone
            response = requests.get(
                'http://metadata.google.internal/computeMetadata/v1/instance/zone',
                headers={'Metadata-Flavor': 'Google'},
                timeout=5
            )
            if response.status_code == 200:
                metadata['zone'] = response.text.split('/')[-1]
        except:
            pass
        
        return metadata
    
    def _get_azure_metadata(self) -> Dict[str, Any]:
        """Get Azure-specific metadata."""
        metadata = {}
        try:
            response = requests.get(
                'http://169.254.169.254/metadata/instance?api-version=2021-02-01',
                headers={'Metadata': 'true'},
                timeout=5
            )
            if response.status_code == 200:
                azure_metadata = response.json()
                metadata['subscription_id'] = azure_metadata.get('compute', {}).get('subscriptionId')
                metadata['resource_group'] = azure_metadata.get('compute', {}).get('resourceGroupName')
                metadata['location'] = azure_metadata.get('compute', {}).get('location')
        except:
            pass
        
        return metadata
    
    def _get_aws_metadata(self) -> Dict[str, Any]:
        """Get AWS-specific metadata."""
        metadata = {}
        try:
            # Get region
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/placement/region',
                timeout=5
            )
            if response.status_code == 200:
                metadata['region'] = response.text
            
            # Get availability zone
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/placement/availability-zone',
                timeout=5
            )
            if response.status_code == 200:
                metadata['availability_zone'] = response.text
        except:
            pass
        
        return metadata
    
    def _get_colab_metadata(self) -> Dict[str, Any]:
        """Get Google Colab metadata."""
        metadata = {'environment': 'colab'}
        
        # Check for TPU
        if 'COLAB_TPU_ADDR' in os.environ:
            metadata['tpu_address'] = os.environ['COLAB_TPU_ADDR']
        
        return metadata
    
    def _get_kaggle_metadata(self) -> Dict[str, Any]:
        """Get Kaggle metadata."""
        metadata = {'environment': 'kaggle'}
        
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            metadata['kernel_type'] = os.environ['KAGGLE_KERNEL_RUN_TYPE']
        
        return metadata
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal configuration based on detected platform."""
        if not self.platform_info:
            self.detect_platform()
        
        config = {
            'platform': self.platform_info.platform.value,
            'batch_size': self._get_optimal_batch_size(),
            'num_workers': self._get_optimal_num_workers(),
            'precision': self._get_optimal_precision(),
            'storage_path': self._get_storage_path(),
            'checkpoint_sync': self._should_sync_checkpoints(),
            'wandb_mode': self._get_wandb_mode()
        }
        
        return config
    
    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on available GPU memory."""
        if not self.platform_info or not self.platform_info.gpu_info:
            return 32  # Default for CPU
        
        gpu_info = self.platform_info.gpu_info
        if not gpu_info.get('torch_cuda_available'):
            return 32
        
        # Estimate based on GPU memory
        devices = gpu_info.get('devices', [])
        if not devices:
            return 64
        
        total_memory = devices[0].get('properties', {}).get('total_memory', 0)
        
        # Memory-based batch size estimation
        if total_memory > 40 * 1024**3:  # > 40GB (A100)
            return 512
        elif total_memory > 20 * 1024**3:  # > 20GB (V100)
            return 256
        elif total_memory > 10 * 1024**3:  # > 10GB (RTX 3080)
            return 128
        else:
            return 64
    
    def _get_optimal_num_workers(self) -> int:
        """Get optimal number of workers for data loading."""
        cpu_count = self.platform_info.compute_info.get('cpu_count', 4) if self.platform_info else 4
        return min(cpu_count, 8)  # Cap at 8 workers
    
    def _get_optimal_precision(self) -> str:
        """Get optimal precision based on GPU capabilities."""
        if not self.platform_info or not self.platform_info.gpu_info:
            return '32-true'
        
        gpu_info = self.platform_info.gpu_info
        devices = gpu_info.get('devices', [])
        
        if devices:
            # Check for Tensor Core support (compute capability >= 7.0)
            major = devices[0].get('properties', {}).get('major', 0)
            if major >= 7:
                return '16-mixed'  # Use mixed precision for modern GPUs
        
        return '32-true'
    
    def _get_storage_path(self) -> str:
        """Get appropriate storage path for the platform."""
        platform = self.platform_info.platform if self.platform_info else CloudPlatform.LOCAL
        
        if platform == CloudPlatform.COLAB:
            return '/content/drive/MyDrive/predictive_maintenance'
        elif platform == CloudPlatform.KAGGLE:
            return '/kaggle/working'
        elif platform in [CloudPlatform.GCP, CloudPlatform.AZURE, CloudPlatform.AWS]:
            return '/tmp/predictive_maintenance'
        else:
            return './models'
    
    def _should_sync_checkpoints(self) -> bool:
        """Determine if checkpoints should be synced to cloud storage."""
        platform = self.platform_info.platform if self.platform_info else CloudPlatform.LOCAL
        return platform in [CloudPlatform.GCP, CloudPlatform.AZURE, CloudPlatform.AWS, 
                           CloudPlatform.COLAB, CloudPlatform.KAGGLE]
    
    def _get_wandb_mode(self) -> str:
        """Get appropriate Weights & Biases mode."""
        platform = self.platform_info.platform if self.platform_info else CloudPlatform.LOCAL
        
        if platform in [CloudPlatform.COLAB, CloudPlatform.KAGGLE]:
            return 'online'  # These platforms usually have good internet
        else:
            return 'offline'  # Play it safe for other platforms

# Global detector instance
_detector = CloudPlatformDetector()

def get_platform_info(force_refresh: bool = False) -> PlatformInfo:
    """Get current platform information."""
    return _detector.detect_platform(force_refresh)

def get_optimal_config() -> Dict[str, Any]:
    """Get optimal configuration for current platform."""
    return _detector.get_optimal_config()

def is_cloud_platform() -> bool:
    """Check if running on a cloud platform."""
    platform_info = get_platform_info()
    return platform_info.platform != CloudPlatform.LOCAL