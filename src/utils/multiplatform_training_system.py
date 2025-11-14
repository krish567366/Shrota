#!/usr/bin/env python3
"""
Multi-Platform Training System

Features:
- Automatic platform detection
- Resource optimization per platform
- Cost tracking and management
- Interruption handling
- Platform-specific configurations

Usage:
    platform_manager = MultiPlatformTrainingSystem(config)
    platform_manager.setup_platform()
    platform_manager.start_training()
"""

import os
import json
import time
import logging
import subprocess
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import requests
import socket

logger = logging.getLogger(__name__)

@dataclass
class PlatformConfig:
    """Platform-specific configuration."""
    
    name: str
    max_batch_size: int
    recommended_workers: int
    memory_limit_gb: float
    gpu_memory_gb: float
    max_training_hours: float
    cost_per_hour: float
    storage_path: str
    checkpoint_interval_minutes: int
    mixed_precision: str
    deepspeed_config: Dict
    
class MultiPlatformTrainingSystem:
    """Multi-platform training system with automatic adaptation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.platform_configs = self._load_platform_configs()
        self.current_platform = None
        self.platform_config = None
        self.session_start_time = time.time()
        self.cost_tracker = CostTracker(config.get('cost_limits', {}))
        
        # Detect and setup platform
        self.setup_platform()
    
    def setup_platform(self):
        """Detect and setup current platform."""
        
        platform_name = self._detect_platform()
        self.current_platform = platform_name
        self.platform_config = self.platform_configs.get(platform_name)
        
        if not self.platform_config:
            logger.warning(f"Unknown platform: {platform_name}, using default config")
            self.platform_config = self.platform_configs['default']
        
        logger.info(f"🚀 Platform detected: {platform_name}")
        logger.info(f"   Max batch size: {self.platform_config.max_batch_size}")
        logger.info(f"   Recommended workers: {self.platform_config.recommended_workers}")
        logger.info(f"   Memory limit: {self.platform_config.memory_limit_gb}GB")
        logger.info(f"   GPU memory: {self.platform_config.gpu_memory_gb}GB")
        logger.info(f"   Cost per hour: ${self.platform_config.cost_per_hour}")
        
        # Setup platform-specific environment
        self._setup_platform_environment()
        
        # Initialize cost tracking
        self.cost_tracker.start_session(self.platform_config.cost_per_hour)
    
    def get_optimized_config(self) -> Dict:
        """Get platform-optimized training configuration."""
        
        return {
            'batch_size': self.platform_config.max_batch_size,
            'num_workers': self.platform_config.recommended_workers,
            'mixed_precision': self.platform_config.mixed_precision,
            'deepspeed_config': self.platform_config.deepspeed_config,
            'checkpoint_interval': self.platform_config.checkpoint_interval_minutes * 60,
            'max_training_time': self.platform_config.max_training_hours * 3600,
            'storage_path': self.platform_config.storage_path,
            'memory_limit': self.platform_config.memory_limit_gb * 1024**3,
            'platform_name': self.platform_config.name
        }
    
    def check_platform_limits(self) -> Dict[str, Any]:
        """Check if we're approaching platform limits."""
        
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        # Time limits - handle case where max_training_hours might not be set or is very low
        max_hours = getattr(self.platform_config, 'max_training_hours', 12)  # Default 12 hours
        if max_hours <= 0:  # Handle disabled time limits
            max_time = float('inf')
            time_remaining = float('inf')
            time_warning = False
        else:
            max_time = max_hours * 3600
            time_remaining = max_time - session_duration
            time_warning = time_remaining < 1800  # 30 minutes warning
            
            # Debug logging for time limits
            logger.debug(f"Time limit debug: max_hours={max_hours}, session_duration={session_duration:.1f}s, time_remaining={time_remaining:.1f}s")
        
        # Cost limits
        cost_status = self.cost_tracker.check_limits()
        
        # Memory usage
        memory_percent = psutil.virtual_memory().percent
        memory_warning = memory_percent > 85
        
        # GPU memory usage
        gpu_memory_warning = False
        if hasattr(self, 'torch') and self.torch.cuda.is_available():
            gpu_memory_used = self.torch.cuda.memory_allocated() / 1e9
            gpu_memory_warning = gpu_memory_used > (self.platform_config.gpu_memory_gb * 0.9)
        
        return {
            'time_remaining_minutes': time_remaining / 60,
            'time_warning': time_warning,
            'cost_status': cost_status,
            'memory_percent': memory_percent,
            'memory_warning': memory_warning,
            'gpu_memory_warning': gpu_memory_warning,
            'should_checkpoint': time_warning or cost_status['near_limit'] or memory_warning,
            'should_stop': ((time_remaining < 300 and time_remaining != float('inf')) or cost_status['over_limit']) and session_duration > 60  # 5 minutes buffer, but not if time limits disabled, and allow at least 1 minute of training
        }
    
    def handle_interruption(self, checkpoint_manager, model, optimizer, scheduler, dataset_state, metrics, phase):
        """Handle training interruption gracefully."""
        
        logger.warning("🚨 Training interruption detected!")
        
        # Save emergency checkpoint
        checkpoint_path = checkpoint_manager.save_complete_state(
            model, optimizer, scheduler, dataset_state, metrics, phase, force_save=True
        )
        
        # Save interruption info
        interruption_info = {
            'platform': self.current_platform,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_duration': time.time() - self.session_start_time,
            'cost_spent': self.cost_tracker.get_total_cost(),
            'checkpoint_path': checkpoint_path,
            'reason': 'platform_limit_reached',
            'next_recommended_platform': self._recommend_next_platform(),
            'resume_instructions': self._generate_resume_instructions(phase, dataset_state)
        }
        
        interruption_file = Path('./interruption_info.json')
        with open(interruption_file, 'w') as f:
            json.dump(interruption_info, f, indent=2)
        
        logger.info(f"💾 Emergency checkpoint saved: {checkpoint_path}")
        logger.info(f"📋 Interruption info saved: {interruption_file}")
        logger.info(f"💰 Total cost this session: ${self.cost_tracker.get_total_cost():.2f}")
        logger.info(f"🔄 Recommended next platform: {interruption_info['next_recommended_platform']}")
        
        return interruption_info
    
    def _detect_platform(self) -> str:
        """Detect current cloud/computing platform."""
        
        # Check environment variables first
        if 'COLAB_GPU' in os.environ:
            return 'colab_pro' if self._check_colab_pro() else 'colab'
        elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return 'kaggle'
        elif 'RUNPOD_POD_ID' in os.environ:
            return 'runpod'
        elif 'PAPERSPACE_NOTEBOOK_REPO_ID' in os.environ:
            return 'paperspace'
        
        # Check file system indicators
        if os.path.exists('/opt/deeplearning/'):
            return 'gcp_vertex'
        elif os.path.exists('/opt/ml/'):
            return 'aws_sagemaker'
        elif os.path.exists('/mnt/batch/'):
            return 'azure_batch'
        
        # Check for AWS EC2
        try:
            response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
            if response.status_code == 200:
                return 'aws_ec2'
        except:
            pass
        
        # Check for GCP Compute Engine
        try:
            response = requests.get(
                'http://metadata.google.internal/computeMetadata/v1/instance/name',
                headers={'Metadata-Flavor': 'Google'},
                timeout=2
            )
            if response.status_code == 200:
                return 'gcp_compute'
        except:
            pass
        
        # Check for Azure VM
        try:
            response = requests.get(
                'http://169.254.169.254/metadata/instance?api-version=2021-02-01',
                headers={'Metadata': 'true'},
                timeout=2
            )
            if response.status_code == 200:
                return 'azure_vm'
        except:
            pass
        
        # Check hostname patterns
        hostname = socket.gethostname().lower()
        if 'aws' in hostname or 'ec2' in hostname:
            return 'aws_ec2'
        elif 'gcp' in hostname or 'google' in hostname:
            return 'gcp_compute'
        elif 'azure' in hostname:
            return 'azure_vm'
        elif 'vast' in hostname:
            return 'vast_ai'
        elif 'lambda' in hostname:
            return 'lambda_labs'
        
        return 'local'
    
    def _check_colab_pro(self) -> bool:
        """Check if running on Colab Pro (higher limits)."""
        try:
            # Colab Pro usually has more RAM and better GPUs
            total_memory = psutil.virtual_memory().total / 1e9
            return total_memory > 12  # Pro usually has >12GB RAM
        except:
            return False
    
    def _load_platform_configs(self) -> Dict[str, PlatformConfig]:
        """Load platform-specific configurations."""
        
        return {
            'colab': PlatformConfig(
                name='Google Colab',
                max_batch_size=16,
                recommended_workers=2,
                memory_limit_gb=12,
                gpu_memory_gb=15,
                max_training_hours=12,
                cost_per_hour=0.0,
                storage_path='/content/drive/MyDrive/checkpoints',
                checkpoint_interval_minutes=15,
                mixed_precision='fp16',
                deepspeed_config={'stage': 2}
            ),
            'colab_pro': PlatformConfig(
                name='Google Colab Pro',
                max_batch_size=24,
                recommended_workers=4,
                memory_limit_gb=25,
                gpu_memory_gb=24,
                max_training_hours=24,
                cost_per_hour=0.42,  # $10/month
                storage_path='/content/drive/MyDrive/checkpoints',
                checkpoint_interval_minutes=10,
                mixed_precision='bf16',
                deepspeed_config={'stage': 3}
            ),
            'kaggle': PlatformConfig(
                name='Kaggle',
                max_batch_size=20,
                recommended_workers=4,
                memory_limit_gb=30,
                gpu_memory_gb=16,
                max_training_hours=30,
                cost_per_hour=0.0,
                storage_path='/kaggle/working/checkpoints',
                checkpoint_interval_minutes=20,
                mixed_precision='bf16',
                deepspeed_config={'stage': 2}
            ),
            'runpod': PlatformConfig(
                name='RunPod',
                max_batch_size=32,
                recommended_workers=8,
                memory_limit_gb=64,
                gpu_memory_gb=24,
                max_training_hours=999,
                cost_per_hour=0.89,  # A6000
                storage_path='/workspace/checkpoints',
                checkpoint_interval_minutes=5,
                mixed_precision='bf16',
                deepspeed_config={'stage': 3, 'offload_optimizer': True}
            ),
            'vast_ai': PlatformConfig(
                name='Vast.ai',
                max_batch_size=32,
                recommended_workers=8,
                memory_limit_gb=64,
                gpu_memory_gb=24,
                max_training_hours=999,
                cost_per_hour=0.45,  # Variable
                storage_path='/workspace/checkpoints',
                checkpoint_interval_minutes=5,
                mixed_precision='bf16',
                deepspeed_config={'stage': 3, 'offload_optimizer': True}
            ),
            'lambda_labs': PlatformConfig(
                name='Lambda Labs',
                max_batch_size=40,
                recommended_workers=12,
                memory_limit_gb=128,
                gpu_memory_gb=80,
                max_training_hours=999,
                cost_per_hour=1.50,  # A100
                storage_path='/home/ubuntu/checkpoints',
                checkpoint_interval_minutes=5,
                mixed_precision='bf16',
                deepspeed_config={'stage': 3, 'offload_optimizer': True, 'offload_param': True}
            ),
            'aws_sagemaker': PlatformConfig(
                name='AWS SageMaker',
                max_batch_size=48,
                recommended_workers=16,
                memory_limit_gb=256,
                gpu_memory_gb=80,
                max_training_hours=999,
                cost_per_hour=4.10,  # p4d.xlarge
                storage_path='/opt/ml/checkpoints',
                checkpoint_interval_minutes=3,
                mixed_precision='bf16',
                deepspeed_config={'stage': 3, 'offload_optimizer': True, 'offload_param': True}
            ),
            'gcp_vertex': PlatformConfig(
                name='GCP Vertex AI',
                max_batch_size=48,
                recommended_workers=16,
                memory_limit_gb=256,
                gpu_memory_gb=80,
                max_training_hours=999,
                cost_per_hour=3.67,  # a2-highgpu-1g
                storage_path='/gcs/checkpoints',
                checkpoint_interval_minutes=3,
                mixed_precision='bf16',
                deepspeed_config={'stage': 3, 'offload_optimizer': True, 'offload_param': True}
            ),
            'azure_vm': PlatformConfig(
                name='Azure VM',
                max_batch_size=48,
                recommended_workers=16,
                memory_limit_gb=256,
                gpu_memory_gb=80,
                max_training_hours=999,
                cost_per_hour=3.80,  # Standard_NC24ads_A100_v4
                storage_path='/mnt/checkpoints',
                checkpoint_interval_minutes=3,
                mixed_precision='bf16',
                deepspeed_config={'stage': 3, 'offload_optimizer': True, 'offload_param': True}
            ),
            'paperspace': PlatformConfig(
                name='Paperspace',
                max_batch_size=32,
                recommended_workers=8,
                memory_limit_gb=64,
                gpu_memory_gb=48,
                max_training_hours=999,
                cost_per_hour=2.30,  # A6000
                storage_path='/storage/checkpoints',
                checkpoint_interval_minutes=5,
                mixed_precision='bf16',
                deepspeed_config={'stage': 3, 'offload_optimizer': True}
            ),
            'local': PlatformConfig(
                name='Local Machine',
                max_batch_size=8,
                recommended_workers=4,
                memory_limit_gb=16,
                gpu_memory_gb=12,
                max_training_hours=999,
                cost_per_hour=0.0,
                storage_path='./checkpoints',
                checkpoint_interval_minutes=30,
                mixed_precision='fp16',
                deepspeed_config={'stage': 1}
            ),
            'default': PlatformConfig(
                name='Unknown Platform',
                max_batch_size=16,
                recommended_workers=4,
                memory_limit_gb=16,
                gpu_memory_gb=12,
                max_training_hours=12,
                cost_per_hour=1.0,
                storage_path='./checkpoints',
                checkpoint_interval_minutes=15,
                mixed_precision='fp16',
                deepspeed_config={'stage': 2}
            )
        }
    
    def _setup_platform_environment(self):
        """Setup platform-specific environment."""
        
        # Create checkpoint directory
        os.makedirs(self.platform_config.storage_path, exist_ok=True)
        
        # Set environment variables for optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Platform-specific optimizations
        if self.current_platform in ['colab', 'colab_pro']:
            # Mount Google Drive if available
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                logger.info("✅ Google Drive mounted")
            except:
                logger.warning("Failed to mount Google Drive")
        
        elif self.current_platform == 'kaggle':
            # Setup Kaggle datasets
            os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME', '')
            os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY', '')
        
        elif self.current_platform.startswith('aws'):
            # AWS-specific setup
            os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    def _recommend_next_platform(self) -> str:
        """Recommend next platform based on cost and availability."""
        
        # Platform recommendation based on cost-effectiveness
        cost_ranking = [
            'colab',        # Free
            'kaggle',       # Free
            'vast_ai',      # Cheapest paid
            'runpod',       # Good value
            'colab_pro',    # Monthly subscription
            'paperspace',   # Mid-range
            'gcp_vertex',   # Enterprise
            'azure_vm',     # Enterprise
            'aws_sagemaker' # Most expensive
        ]
        
        # Find next available platform
        current_index = cost_ranking.index(self.current_platform) if self.current_platform in cost_ranking else 0
        
        for i in range(current_index + 1, len(cost_ranking)):
            return cost_ranking[i]
        
        # Loop back to cheapest
        return cost_ranking[0]
    
    def _generate_resume_instructions(self, phase: str, dataset_state: Dict) -> Dict:
        """Generate instructions for resuming training."""
        
        return {
            'current_phase': phase,
            'current_dataset': dataset_state.get('current_dataset', ''),
            'epoch': dataset_state.get('epoch', 0),
            'global_step': dataset_state.get('global_step', 0),
            'completed_datasets': dataset_state.get('completed_datasets', []),
            'resume_command': f"python multiplatform_trainer.py --resume --phase {phase}",
            'estimated_progress': f"{(dataset_state.get('global_step', 0) / dataset_state.get('estimated_total_steps', 1)) * 100:.1f}%"
        }

class CostTracker:
    """Track training costs across platforms."""
    
    def __init__(self, cost_limits: Dict):
        self.cost_limits = cost_limits
        self.session_start_time = None
        self.hourly_rate = 0.0
        self.total_spent = 0.0
        
    def start_session(self, hourly_rate: float):
        """Start cost tracking session."""
        self.session_start_time = time.time()
        self.hourly_rate = hourly_rate
        
    def get_current_cost(self) -> float:
        """Get current session cost."""
        if not self.session_start_time:
            return 0.0
        
        hours = (time.time() - self.session_start_time) / 3600
        return hours * self.hourly_rate
    
    def get_total_cost(self) -> float:
        """Get total cost including previous sessions."""
        return self.total_spent + self.get_current_cost()
    
    def check_limits(self) -> Dict[str, Any]:
        """Check if approaching cost limits."""
        
        current_total = self.get_total_cost()
        daily_limit = self.cost_limits.get('daily_limit', 100.0)
        session_limit = self.cost_limits.get('session_limit', 50.0)
        
        # Handle disabled limits (negative values)
        limits_disabled = daily_limit < 0 or session_limit < 0
        
        if limits_disabled:
            return {
                'current_session_cost': self.get_current_cost(),
                'total_cost': current_total,
                'daily_limit': daily_limit,
                'session_limit': session_limit,
                'near_limit': False,  # Never near limit when disabled
                'over_limit': False   # Never over limit when disabled
            }
        
        return {
            'current_session_cost': self.get_current_cost(),
            'total_cost': current_total,
            'daily_limit': daily_limit,
            'session_limit': session_limit,
            'near_limit': current_total > (daily_limit * 0.8) or self.get_current_cost() > (session_limit * 0.8),
            'over_limit': current_total > daily_limit or self.get_current_cost() > session_limit
        }

if __name__ == "__main__":
    # Example usage
    config = {
        'cost_limits': {
            'daily_limit': 50.0,
            'session_limit': 25.0
        }
    }
    
    platform_system = MultiPlatformTrainingSystem(config)
    optimized_config = platform_system.get_optimized_config()
    
    print(f"Platform: {platform_system.current_platform}")
    print(f"Optimized config: {optimized_config}")
    
    # Check limits
    limits = platform_system.check_platform_limits()
    print(f"Platform limits: {limits}")