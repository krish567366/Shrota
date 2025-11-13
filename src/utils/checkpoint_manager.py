"""
Dynamic Checkpoint Synchronization System

This module provides cloud-agnostic checkpoint management that automatically
syncs training state across different platforms and allows seamless resumption.
"""

import os
import json
import shutil
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import threading
import time

from .cloud_platform import get_platform_info, CloudPlatform

logger = logging.getLogger(__name__)

@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    model_name: str
    epoch: int
    step: int
    timestamp: str
    platform: str
    metrics: Dict[str, float]
    config_hash: str
    file_size: int
    file_hash: str
    resume_info: Dict[str, Any]

class CloudStorageAdapter:
    """Base class for cloud storage adapters."""
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to cloud storage."""
        raise NotImplementedError
    
    def download(self, remote_path: str, local_path: str) -> bool:
        """Download a file from cloud storage."""
        raise NotImplementedError
    
    def exists(self, remote_path: str) -> bool:
        """Check if a file exists in cloud storage."""
        raise NotImplementedError
    
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix."""
        raise NotImplementedError
    
    def delete(self, remote_path: str) -> bool:
        """Delete a file from cloud storage."""
        raise NotImplementedError

class GCPStorageAdapter(CloudStorageAdapter):
    """Google Cloud Storage adapter."""
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client()
            except ImportError:
                logger.error("google-cloud-storage not installed")
                raise
        return self._client
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            
            # Create directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded gs://{self.bucket_name}/{remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            return False
    
    def exists(self, remote_path: str) -> bool:
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Failed to check existence in GCS: {e}")
            return False
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            bucket = self.client.bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list files in GCS: {e}")
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            blob.delete()
            logger.info(f"Deleted gs://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from GCS: {e}")
            return False

class AzureStorageAdapter(CloudStorageAdapter):
    """Azure Blob Storage adapter."""
    
    def __init__(self, account_name: str, container_name: str, account_key: Optional[str] = None):
        self.account_name = account_name
        self.container_name = container_name
        self.account_key = account_key
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from azure.storage.blob import BlobServiceClient
                if self.account_key:
                    self._client = BlobServiceClient(
                        account_url=f"https://{self.account_name}.blob.core.windows.net",
                        credential=self.account_key
                    )
                else:
                    # Use default credentials
                    self._client = BlobServiceClient(
                        account_url=f"https://{self.account_name}.blob.core.windows.net"
                    )
            except ImportError:
                logger.error("azure-storage-blob not installed")
                raise
        return self._client
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            blob_client = self.client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            logger.info(f"Uploaded {local_path} to Azure blob {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to Azure: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            blob_client = self.client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            
            # Create directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_path, 'wb') as download_file:
                download_file.write(blob_client.download_blob().readall())
            logger.info(f"Downloaded Azure blob {remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from Azure: {e}")
            return False
    
    def exists(self, remote_path: str) -> bool:
        try:
            blob_client = self.client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            return blob_client.exists()
        except Exception as e:
            logger.error(f"Failed to check existence in Azure: {e}")
            return False
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            container_client = self.client.get_container_client(self.container_name)
            blobs = container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list files in Azure: {e}")
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            blob_client = self.client.get_blob_client(
                container=self.container_name, 
                blob=remote_path
            )
            blob_client.delete_blob()
            logger.info(f"Deleted Azure blob {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Azure: {e}")
            return False

class AWSStorageAdapter(CloudStorageAdapter):
    """AWS S3 Storage adapter."""
    
    def __init__(self, bucket_name: str, region: Optional[str] = None):
        self.bucket_name = bucket_name
        self.region = region
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                if self.region:
                    self._client = boto3.client('s3', region_name=self.region)
                else:
                    self._client = boto3.client('s3')
            except ImportError:
                logger.error("boto3 not installed")
                raise
        return self._client
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            self.client.upload_file(local_path, self.bucket_name, remote_path)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            # Create directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.client.download_file(self.bucket_name, remote_path, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False
    
    def exists(self, remote_path: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=remote_path)
            return True
        except Exception:
            return False
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Failed to list files in S3: {e}")
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=remote_path)
            logger.info(f"Deleted s3://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            return False

class LocalStorageAdapter(CloudStorageAdapter):
    """Local filesystem adapter for testing."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            target_path = self.base_path / remote_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, target_path)
            logger.info(f"Copied {local_path} to {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            source_path = self.base_path / remote_path
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, local_path)
            logger.info(f"Copied {source_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False
    
    def exists(self, remote_path: str) -> bool:
        return (self.base_path / remote_path).exists()
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            prefix_path = self.base_path / prefix
            if prefix_path.is_file():
                return [prefix]
            elif prefix_path.is_dir():
                return [str(p.relative_to(self.base_path)) for p in prefix_path.rglob('*') if p.is_file()]
            else:
                # Pattern matching
                pattern_parts = prefix.split('/')
                result = []
                for p in self.base_path.rglob('*'):
                    if p.is_file():
                        rel_path = str(p.relative_to(self.base_path))
                        if rel_path.startswith(prefix):
                            result.append(rel_path)
                return result
        except Exception as e:
            logger.error(f"Failed to list local files: {e}")
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            target_path = self.base_path / remote_path
            if target_path.exists():
                target_path.unlink()
                logger.info(f"Deleted {target_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

class DynamicCheckpointManager:
    """
    Dynamic checkpoint manager that automatically syncs checkpoints
    across different cloud platforms and allows seamless resumption.
    """
    
    def __init__(self, 
                 project_name: str,
                 storage_config: Optional[Dict[str, Any]] = None,
                 auto_sync: bool = True,
                 sync_interval: int = 300):  # 5 minutes
        self.project_name = project_name
        self.auto_sync = auto_sync
        self.sync_interval = sync_interval
        
        # Detect platform and configure storage
        self.platform_info = get_platform_info()
        self.storage_adapter = self._create_storage_adapter(storage_config)
        
        # Local paths
        self.local_checkpoint_dir = Path(f"./checkpoints/{project_name}")
        self.local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata
        self.metadata_file = self.local_checkpoint_dir / "checkpoint_metadata.json"
        self.metadata: Dict[str, CheckpointMetadata] = self._load_metadata()
        
        # Background sync thread
        self._sync_thread = None
        self._stop_sync = threading.Event()
        
        if self.auto_sync and self.storage_adapter:
            self._start_background_sync()
    
    def _create_storage_adapter(self, storage_config: Optional[Dict[str, Any]]) -> Optional[CloudStorageAdapter]:
        """Create appropriate storage adapter based on platform and config."""
        if storage_config:
            storage_type = storage_config.get('type', 'auto')
        else:
            storage_type = 'auto'
        
        if storage_type == 'auto':
            # Auto-detect based on platform
            if self.platform_info.platform == CloudPlatform.GCP:
                bucket_name = storage_config.get('bucket_name', f"{self.project_name}-checkpoints")
                return GCPStorageAdapter(bucket_name)
            
            elif self.platform_info.platform == CloudPlatform.AZURE:
                account_name = storage_config.get('account_name', 'predictivemaintenance')
                container_name = storage_config.get('container_name', f"{self.project_name}-checkpoints")
                account_key = storage_config.get('account_key')
                return AzureStorageAdapter(account_name, container_name, account_key)
            
            elif self.platform_info.platform == CloudPlatform.AWS:
                bucket_name = storage_config.get('bucket_name', f"{self.project_name}-checkpoints")
                region = storage_config.get('region')
                return AWSStorageAdapter(bucket_name, region)
            
            elif self.platform_info.platform in [CloudPlatform.COLAB, CloudPlatform.KAGGLE]:
                # Use Google Drive for Colab, local sync for Kaggle
                if self.platform_info.platform == CloudPlatform.COLAB:
                    drive_path = f"/content/drive/MyDrive/{self.project_name}_checkpoints"
                    return LocalStorageAdapter(drive_path)
                else:
                    return LocalStorageAdapter("./checkpoint_sync")
            
            else:
                # Local development
                return LocalStorageAdapter("./checkpoint_sync")
        
        elif storage_type == 'gcp':
            bucket_name = storage_config.get('bucket_name', f"{self.project_name}-checkpoints")
            return GCPStorageAdapter(bucket_name)
        
        elif storage_type == 'azure':
            account_name = storage_config['account_name']
            container_name = storage_config['container_name']
            account_key = storage_config.get('account_key')
            return AzureStorageAdapter(account_name, container_name, account_key)
        
        elif storage_type == 'aws':
            bucket_name = storage_config['bucket_name']
            region = storage_config.get('region')
            return AWSStorageAdapter(bucket_name, region)
        
        elif storage_type == 'local':
            base_path = storage_config.get('base_path', './checkpoint_sync')
            return LocalStorageAdapter(base_path)
        
        else:
            logger.warning(f"Unknown storage type: {storage_type}")
            return None
    
    def _load_metadata(self) -> Dict[str, CheckpointMetadata]:
        """Load checkpoint metadata from local file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return {
                        k: CheckpointMetadata(**v) for k, v in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save checkpoint metadata to local file."""
        try:
            data = {k: asdict(v) for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for consistency checking."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def save_checkpoint(self, 
                       checkpoint_path: str,
                       model_name: str,
                       epoch: int,
                       step: int,
                       metrics: Dict[str, float],
                       config: Dict[str, Any],
                       resume_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a checkpoint with metadata and optionally sync to cloud storage.
        
        Returns:
            checkpoint_id: Unique identifier for the checkpoint
        """
        # Generate checkpoint ID
        timestamp = datetime.now().isoformat()
        checkpoint_id = f"{model_name}_epoch{epoch}_step{step}_{timestamp.replace(':', '-')}"
        
        # Calculate file info
        file_size = os.path.getsize(checkpoint_path)
        file_hash = self._calculate_file_hash(checkpoint_path)
        config_hash = self._calculate_config_hash(config)
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            model_name=model_name,
            epoch=epoch,
            step=step,
            timestamp=timestamp,
            platform=self.platform_info.platform.value,
            metrics=metrics,
            config_hash=config_hash,
            file_size=file_size,
            file_hash=file_hash,
            resume_info=resume_info or {}
        )
        
        # Save to local checkpoint directory
        local_checkpoint_path = self.local_checkpoint_dir / f"{checkpoint_id}.ckpt"
        shutil.copy2(checkpoint_path, local_checkpoint_path)
        
        # Save config alongside checkpoint
        config_path = self.local_checkpoint_dir / f"{checkpoint_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Update metadata
        self.metadata[checkpoint_id] = metadata
        self._save_metadata()
        
        logger.info(f"Saved checkpoint: {checkpoint_id}")
        
        # Sync to cloud storage if available
        if self.storage_adapter:
            self._sync_checkpoint_to_cloud(checkpoint_id)
        
        return checkpoint_id
    
    def _sync_checkpoint_to_cloud(self, checkpoint_id: str):
        """Sync a specific checkpoint to cloud storage."""
        if not self.storage_adapter:
            return
        
        try:
            # Upload checkpoint file
            local_checkpoint = self.local_checkpoint_dir / f"{checkpoint_id}.ckpt"
            remote_checkpoint = f"checkpoints/{checkpoint_id}.ckpt"
            
            if local_checkpoint.exists():
                self.storage_adapter.upload(str(local_checkpoint), remote_checkpoint)
            
            # Upload config file
            local_config = self.local_checkpoint_dir / f"{checkpoint_id}_config.json"
            remote_config = f"checkpoints/{checkpoint_id}_config.json"
            
            if local_config.exists():
                self.storage_adapter.upload(str(local_config), remote_config)
            
            # Upload metadata
            remote_metadata = "checkpoints/checkpoint_metadata.json"
            self.storage_adapter.upload(str(self.metadata_file), remote_metadata)
            
            logger.info(f"Synced checkpoint {checkpoint_id} to cloud storage")
            
        except Exception as e:
            logger.error(f"Failed to sync checkpoint {checkpoint_id}: {e}")
    
    def list_checkpoints(self, model_name: Optional[str] = None) -> List[CheckpointMetadata]:
        """List available checkpoints, optionally filtered by model name."""
        checkpoints = list(self.metadata.values())
        
        if model_name:
            checkpoints = [c for c in checkpoints if c.model_name == model_name]
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        
        return checkpoints
    
    def get_latest_checkpoint(self, model_name: str) -> Optional[CheckpointMetadata]:
        """Get the latest checkpoint for a specific model."""
        checkpoints = self.list_checkpoints(model_name)
        return checkpoints[0] if checkpoints else None
    
    def load_checkpoint(self, checkpoint_id: str, target_path: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Load a checkpoint from local storage or cloud storage.
        
        Returns:
            (checkpoint_path, config): Path to checkpoint file and configuration
        """
        if checkpoint_id not in self.metadata:
            # Try to sync from cloud storage first
            if self.storage_adapter:
                self._sync_from_cloud()
        
        if checkpoint_id not in self.metadata:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # Check if checkpoint exists locally
        local_checkpoint = self.local_checkpoint_dir / f"{checkpoint_id}.ckpt"
        local_config = self.local_checkpoint_dir / f"{checkpoint_id}_config.json"
        
        if not local_checkpoint.exists() and self.storage_adapter:
            # Download from cloud storage
            remote_checkpoint = f"checkpoints/{checkpoint_id}.ckpt"
            remote_config = f"checkpoints/{checkpoint_id}_config.json"
            
            self.storage_adapter.download(remote_checkpoint, str(local_checkpoint))
            self.storage_adapter.download(remote_config, str(local_config))
        
        if not local_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_id}")
        
        # Load config
        config = {}
        if local_config.exists():
            with open(local_config, 'r') as f:
                config = json.load(f)
        
        # Copy to target path if specified
        if target_path:
            shutil.copy2(local_checkpoint, target_path)
            return target_path, config
        
        return str(local_checkpoint), config
    
    def _sync_from_cloud(self):
        """Sync metadata and checkpoint list from cloud storage."""
        if not self.storage_adapter:
            return
        
        try:
            # Download latest metadata
            remote_metadata = "checkpoints/checkpoint_metadata.json"
            temp_metadata = self.local_checkpoint_dir / "temp_metadata.json"
            
            if self.storage_adapter.download(remote_metadata, str(temp_metadata)):
                # Merge with local metadata
                with open(temp_metadata, 'r') as f:
                    cloud_metadata = json.load(f)
                
                for checkpoint_id, metadata_dict in cloud_metadata.items():
                    if checkpoint_id not in self.metadata:
                        self.metadata[checkpoint_id] = CheckpointMetadata(**metadata_dict)
                
                self._save_metadata()
                temp_metadata.unlink()  # Clean up temp file
                
                logger.info("Synced metadata from cloud storage")
        
        except Exception as e:
            logger.error(f"Failed to sync from cloud: {e}")
    
    def _start_background_sync(self):
        """Start background thread for periodic sync."""
        if self._sync_thread is not None:
            return  # Already running
        
        def sync_worker():
            while not self._stop_sync.wait(self.sync_interval):
                try:
                    self._sync_from_cloud()
                except Exception as e:
                    logger.error(f"Background sync error: {e}")
        
        self._sync_thread = threading.Thread(target=sync_worker, daemon=True)
        self._sync_thread.start()
        logger.info("Started background checkpoint sync")
    
    def stop_background_sync(self):
        """Stop background sync thread."""
        if self._sync_thread is not None:
            self._stop_sync.set()
            self._sync_thread.join(timeout=5)
            self._sync_thread = None
            logger.info("Stopped background checkpoint sync")
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5, model_name: Optional[str] = None):
        """Clean up old checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints(model_name)
        
        if len(checkpoints) <= keep_last_n:
            return  # Nothing to clean up
        
        to_delete = checkpoints[keep_last_n:]
        
        for checkpoint in to_delete:
            try:
                # Delete local files
                local_checkpoint = self.local_checkpoint_dir / f"{checkpoint.checkpoint_id}.ckpt"
                local_config = self.local_checkpoint_dir / f"{checkpoint.checkpoint_id}_config.json"
                
                if local_checkpoint.exists():
                    local_checkpoint.unlink()
                if local_config.exists():
                    local_config.unlink()
                
                # Delete from cloud storage
                if self.storage_adapter:
                    self.storage_adapter.delete(f"checkpoints/{checkpoint.checkpoint_id}.ckpt")
                    self.storage_adapter.delete(f"checkpoints/{checkpoint.checkpoint_id}_config.json")
                
                # Remove from metadata
                del self.metadata[checkpoint.checkpoint_id]
                
                logger.info(f"Cleaned up checkpoint: {checkpoint.checkpoint_id}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup checkpoint {checkpoint.checkpoint_id}: {e}")
        
        self._save_metadata()
        
        # Update cloud metadata
        if self.storage_adapter:
            remote_metadata = "checkpoints/checkpoint_metadata.json"
            self.storage_adapter.upload(str(self.metadata_file), remote_metadata)
    
    def get_resume_info(self, checkpoint_id: str) -> Dict[str, Any]:
        """Get resume information for a checkpoint."""
        if checkpoint_id not in self.metadata:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        return self.metadata[checkpoint_id].resume_info
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.stop_background_sync()