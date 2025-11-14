#!/usr/bin/env python3
"""
Advanced Checkpoint Manager for Multi-Platform Training

Features:
- Complete training state preservation
- Dataset progress tracking
- Platform-agnostic checkpoints
- Automatic resume from any interruption
- Incremental dataset training
- Cloud storage integration

Usage:
    checkpoint_manager = AdvancedCheckpointManager(config)
    checkpoint_manager.save_complete_state(model, optimizer, dataset_state)
    state = checkpoint_manager.load_complete_state()
"""

import os
import json
import pickle
import torch
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import hashlib
import psutil
import GPUtil

logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    """Complete training state for seamless resume."""
    
    # Model and training state
    epoch: int
    global_step: int
    phase: str
    current_dataset: str
    dataset_progress: Dict[str, Any]
    
    # Optimizer and scheduler state
    optimizer_state: Dict
    scheduler_state: Dict
    
    # Performance metrics
    best_val_loss: float
    best_val_wer: float
    training_history: List[Dict]
    
    # Dataset and curriculum state
    curriculum_stage: int
    completed_datasets: List[str]
    current_dataset_epoch: int
    dataset_samples_processed: int
    
    # Platform and resource info
    platform_info: Dict
    resource_usage: Dict
    training_time: float
    estimated_remaining_time: float
    
    # Resume information
    checkpoint_path: str
    timestamp: str
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None

class AdvancedCheckpointManager:
    """Advanced checkpoint manager for multi-platform training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.cloud_storage_config = config.get('cloud_storage', {})
        self.auto_save_interval = config.get('auto_save_interval', 300)  # 5 minutes
        self.max_checkpoints = config.get('max_checkpoints', 10)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.local_backup_dir = self.checkpoint_dir / 'local_backup'
        self.local_backup_dir.mkdir(exist_ok=True)
        
        # Initialize cloud storage
        self.cloud_storage = self._init_cloud_storage()
        
        # Training state
        self.training_state = None
        self.last_save_time = 0
        
    def _init_cloud_storage(self):
        """Initialize cloud storage based on configuration."""
        storage_type = self.cloud_storage_config.get('type', 'none')
        
        if storage_type == 'aws_s3':
            try:
                import boto3
                return S3CloudStorage(self.cloud_storage_config)
            except ImportError:
                logger.warning("boto3 not installed, S3 storage disabled")
                
        elif storage_type == 'gcp_gcs':
            try:
                from google.cloud import storage
                return GCSCloudStorage(self.cloud_storage_config)
            except ImportError:
                logger.warning("google-cloud-storage not installed, GCS storage disabled")
                
        elif storage_type == 'azure_blob':
            try:
                from azure.storage.blob import BlobServiceClient
                return AzureBlobStorage(self.cloud_storage_config)
            except ImportError:
                logger.warning("azure-storage-blob not installed, Azure storage disabled")
        
        return None
    
    def save_complete_state(self, 
                           model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer,
                           scheduler: Any,
                           dataset_state: Dict,
                           metrics: Dict,
                           phase: str = "A",
                           force_save: bool = False) -> str:
        """Save complete training state for perfect resume."""
        
        current_time = time.time()
        if not force_save and (current_time - self.last_save_time) < self.auto_save_interval:
            return None  # Skip if too frequent
        
        try:
            # Create training state
            training_state = TrainingState(
                epoch=dataset_state.get('epoch', 0),
                global_step=dataset_state.get('global_step', 0),
                phase=phase,
                current_dataset=dataset_state.get('current_dataset', ''),
                dataset_progress=dataset_state,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler else {},
                best_val_loss=metrics.get('best_val_loss', float('inf')),
                best_val_wer=metrics.get('best_val_wer', float('inf')),
                training_history=metrics.get('history', []),
                curriculum_stage=dataset_state.get('curriculum_stage', 1),
                completed_datasets=dataset_state.get('completed_datasets', []),
                current_dataset_epoch=dataset_state.get('current_dataset_epoch', 0),
                dataset_samples_processed=dataset_state.get('samples_processed', 0),
                platform_info=self._get_platform_info(),
                resource_usage=self._get_resource_usage(),
                training_time=dataset_state.get('training_time', 0),
                estimated_remaining_time=self._estimate_remaining_time(dataset_state),
                checkpoint_path='',
                timestamp=time.strftime('%Y%m%d_%H%M%S'),
                config_hash=self._hash_config()
            )
            
            # Create checkpoint filename
            checkpoint_name = f"phase_{phase}_epoch_{training_state.epoch}_step_{training_state.global_step}_{training_state.timestamp}.ckpt"
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            # Save complete checkpoint
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'training_state': asdict(training_state),
                'config': self.config,
                'pytorch_version': torch.__version__,
                'platform_info': training_state.platform_info
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update training state
            training_state.checkpoint_path = str(checkpoint_path)
            self.training_state = training_state
            
            # Save training state separately (lightweight)
            state_path = self.checkpoint_dir / f"training_state_{training_state.timestamp}.json"
            with open(state_path, 'w') as f:
                json.dump(asdict(training_state), f, indent=2)
            
            # Create symlink to latest
            latest_path = self.checkpoint_dir / f"latest_phase_{phase}.ckpt"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(checkpoint_path.name)
            
            # Upload to cloud storage
            if self.cloud_storage:
                try:
                    cloud_path = self.cloud_storage.upload_checkpoint(checkpoint_path)
                    logger.info(f"✅ Checkpoint uploaded to cloud: {cloud_path}")
                except Exception as e:
                    logger.warning(f"⚠️  Cloud upload failed: {str(e)}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            self.last_save_time = current_time
            
            logger.info(f"💾 Complete checkpoint saved: {checkpoint_name}")
            logger.info(f"   Phase: {phase}, Epoch: {training_state.epoch}, Step: {training_state.global_step}")
            logger.info(f"   Dataset: {training_state.current_dataset}")
            logger.info(f"   Progress: {training_state.dataset_samples_processed:,} samples")
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {str(e)}")
            raise
    
    def load_complete_state(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """Load complete training state for perfect resume."""
        
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self._find_latest_checkpoint()
            
        if checkpoint_path is None:
            logger.info("No checkpoint found, starting from scratch")
            return None
        
        try:
            logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract training state
            training_state_dict = checkpoint_data['training_state']
            self.training_state = TrainingState(**training_state_dict)
            
            logger.info(f"✅ Checkpoint loaded successfully")
            logger.info(f"   Phase: {self.training_state.phase}")
            logger.info(f"   Epoch: {self.training_state.epoch}")
            logger.info(f"   Global Step: {self.training_state.global_step}")
            logger.info(f"   Dataset: {self.training_state.current_dataset}")
            logger.info(f"   Best Val Loss: {self.training_state.best_val_loss:.4f}")
            logger.info(f"   Completed Datasets: {self.training_state.completed_datasets}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {str(e)}")
            return None
    
    def get_next_dataset(self, phase: str) -> Optional[str]:
        """Get next dataset to train on for incremental training."""
        
        phase_datasets = self.config.get('phase_datasets', {}).get(phase, [])
        
        if not self.training_state:
            return phase_datasets[0] if phase_datasets else None
        
        completed = set(self.training_state.completed_datasets)
        
        for dataset in phase_datasets:
            if dataset not in completed:
                return dataset
        
        # All datasets completed for this phase
        return None
    
    def mark_dataset_completed(self, dataset_name: str):
        """Mark a dataset as completed."""
        if self.training_state:
            if dataset_name not in self.training_state.completed_datasets:
                self.training_state.completed_datasets.append(dataset_name)
                logger.info(f"✅ Dataset '{dataset_name}' marked as completed")
    
    def should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved based on time interval."""
        current_time = time.time()
        return (current_time - self.last_save_time) >= self.auto_save_interval
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file."""
        
        # First try to find latest symlink
        for phase in ['A', 'B', 'C', 'D', 'E']:
            latest_path = self.checkpoint_dir / f"latest_phase_{phase}.ckpt"
            if latest_path.exists():
                return str(latest_path.resolve())
        
        # Fallback: find most recent checkpoint
        checkpoint_files = list(self.checkpoint_dir.glob("phase_*.ckpt"))
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoint_files[0])
    
    def _get_platform_info(self) -> Dict:
        """Get current platform information."""
        
        import platform
        platform_info = {
            'hostname': os.uname().nodename,
            'platform': platform.system(),
            'python_version': os.sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add GPU information
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                platform_info['gpu_info'] = [{
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'temperature': gpu.temperature,
                    'load': gpu.load
                } for gpu in gpus]
            except:
                platform_info['gpu_info'] = []
        
        # Detect cloud platform
        platform_info['cloud_platform'] = self._detect_cloud_platform()
        
        return platform_info
    
    def _detect_cloud_platform(self) -> str:
        """Detect which cloud platform we're running on."""
        
        # Check for common cloud platform indicators
        if os.path.exists('/opt/deeplearning/'):
            return 'gcp'
        elif os.path.exists('/opt/ml/'):
            return 'aws_sagemaker'
        elif 'COLAB_GPU' in os.environ:
            return 'colab'
        elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return 'kaggle'
        elif 'RUNPOD_POD_ID' in os.environ:
            return 'runpod'
        elif 'PAPERSPACE_NOTEBOOK_REPO_ID' in os.environ:
            return 'paperspace'
        elif os.path.exists('/mnt/batch/'):
            return 'azure_batch'
        else:
            return 'unknown'
    
    def _get_resource_usage(self) -> Dict:
        """Get current resource usage."""
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'gpu_memory_used': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            'gpu_memory_cached': torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
            'timestamp': time.time()
        }
    
    def _estimate_remaining_time(self, dataset_state: Dict) -> float:
        """Estimate remaining training time."""
        
        if not dataset_state.get('training_time') or not dataset_state.get('global_step'):
            return 0.0
        
        time_per_step = dataset_state['training_time'] / dataset_state['global_step']
        estimated_total_steps = dataset_state.get('estimated_total_steps', 10000)
        remaining_steps = max(0, estimated_total_steps - dataset_state['global_step'])
        
        return remaining_steps * time_per_step
    
    def _hash_config(self) -> str:
        """Create hash of configuration for change detection."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to save space."""
        
        checkpoint_files = list(self.checkpoint_dir.glob("phase_*.ckpt"))
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest checkpoints
        to_remove = checkpoint_files[:-self.max_checkpoints]
        for checkpoint_file in to_remove:
            try:
                checkpoint_file.unlink()
                logger.info(f"🗑️  Removed old checkpoint: {checkpoint_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {checkpoint_file}: {str(e)}")

class CloudStorageBase:
    """Base class for cloud storage implementations."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def upload_checkpoint(self, local_path: Path) -> str:
        raise NotImplementedError
    
    def download_checkpoint(self, cloud_path: str, local_path: Path) -> bool:
        raise NotImplementedError
    
    def list_checkpoints(self) -> List[str]:
        raise NotImplementedError

class S3CloudStorage(CloudStorageBase):
    """AWS S3 cloud storage implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        import boto3
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.get('aws_access_key_id'),
            aws_secret_access_key=config.get('aws_secret_access_key'),
            region_name=config.get('region', 'us-east-1')
        )
        self.bucket_name = config.get('bucket_name')
        self.prefix = config.get('prefix', 'checkpoints/')
    
    def upload_checkpoint(self, local_path: Path) -> str:
        cloud_key = f"{self.prefix}{local_path.name}"
        self.s3_client.upload_file(str(local_path), self.bucket_name, cloud_key)
        return f"s3://{self.bucket_name}/{cloud_key}"

class GCSCloudStorage(CloudStorageBase):
    """Google Cloud Storage implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        from google.cloud import storage
        self.client = storage.Client()
        self.bucket_name = config.get('bucket_name')
        self.prefix = config.get('prefix', 'checkpoints/')
    
    def upload_checkpoint(self, local_path: Path) -> str:
        bucket = self.client.bucket(self.bucket_name)
        blob_name = f"{self.prefix}{local_path.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        return f"gs://{self.bucket_name}/{blob_name}"

class AzureBlobStorage(CloudStorageBase):
    """Azure Blob Storage implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        from azure.storage.blob import BlobServiceClient
        self.blob_service_client = BlobServiceClient(
            account_url=config.get('account_url'),
            credential=config.get('account_key')
        )
        self.container_name = config.get('container_name')
        self.prefix = config.get('prefix', 'checkpoints/')
    
    def upload_checkpoint(self, local_path: Path) -> str:
        blob_name = f"{self.prefix}{local_path.name}"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, 
            blob=blob_name
        )
        with open(local_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        return f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"

if __name__ == "__main__":
    # Example usage
    config = {
        'checkpoint_dir': './checkpoints',
        'auto_save_interval': 300,
        'max_checkpoints': 10,
        'cloud_storage': {
            'type': 'aws_s3',  # 'aws_s3', 'gcp_gcs', 'azure_blob', or 'none'
            'bucket_name': 'my-training-checkpoints',
            'prefix': 'indian-asr/',
            'aws_access_key_id': 'your_key',
            'aws_secret_access_key': 'your_secret'
        },
        'phase_datasets': {
            'A': ['indicvoices', 'spring_inx', 'india_multilingual'],
            'B': ['whisper_dataset', 'fleurs', 'common_voice'],
            'C': ['multichannel_data'],
            'D': ['speaker_data'],
            'E': ['optimization_data']
        }
    }
    
    checkpoint_manager = AdvancedCheckpointManager(config)
    
    # Example of saving state
    # checkpoint_manager.save_complete_state(model, optimizer, scheduler, dataset_state, metrics, phase="A")
    
    # Example of loading state
    # state = checkpoint_manager.load_complete_state()
    
    print("✅ Advanced Checkpoint Manager ready!")