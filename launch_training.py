#!/usr/bin/env python3
"""
Dynamic Training Launcher

This script provides a dynamic, cloud-agnostic training launcher that can:
1. Automatically detect the platform (GCP, Azure, AWS, Colab, Kaggle, local)
2. Resume training from the latest checkpoint across platforms
3. Optimize configuration based on available hardware
4. Sync checkpoints to cloud storage for seamless continuation

Usage:
    python launch_training.py --model tft --dataset ai4i
    python launch_training.py --model hybrid --dataset ai4i --resume
    python launch_training.py --config custom_config.yaml
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.cloud_platform import get_platform_info, get_optimal_config, CloudPlatform
from src.utils.checkpoint_manager import DynamicCheckpointManager
from src.training.train import A100OptimizedTrainer
from src.utils.helpers import ConfigManager, setup_logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DynamicTrainingLauncher:
    """Dynamic training launcher with cloud platform support."""
    
    def __init__(self):
        self.platform_info = get_platform_info()
        self.optimal_config = get_optimal_config()
        self.checkpoint_manager = None
        
        logger.info(f"🌐 Platform: {self.platform_info.platform.value}")
        logger.info(f"🖥️  Instance: {self.platform_info.instance_type or 'Unknown'}")
        
        if self.platform_info.gpu_info and self.platform_info.gpu_info.get('torch_cuda_available'):
            gpu_devices = self.platform_info.gpu_info.get('devices', [])
            if gpu_devices:
                logger.info(f"🚀 GPU: {gpu_devices[0]['name']}")
                logger.info(f"💾 VRAM: {gpu_devices[0]['properties']['total_memory'] / 1024**3:.1f} GB")
    
    def create_config(self, args) -> Dict[str, Any]:
        """Create dynamic configuration based on arguments and platform."""
        
        if args.config:
            # Load user-provided config
            config = ConfigManager.load_config(args.config)
            logger.info(f"📄 Config loaded: {args.config}")
        else:
            # Create default config based on model and dataset
            config = self._create_default_config(args.model, args.dataset)
            logger.info(f"🏗️  Default config created for {args.model} on {args.dataset}")
        
        # Apply platform optimizations
        self._apply_platform_optimizations(config)
        
        # Apply command line overrides
        self._apply_cli_overrides(config, args)
        
        return config
    
    def _create_default_config(self, model_name: str, dataset_name: str) -> Dict[str, Any]:
        """Create default configuration for model and dataset."""
        
        # Base configuration
        config = {
            'project_name': 'predictive_maintenance',
            'experiment': {
                'name': f'{model_name}_{dataset_name}_{self.platform_info.platform.value}',
                'tags': [model_name, dataset_name, self.platform_info.platform.value]
            },
            'model': {
                'name': model_name,
            },
            'data': {
                'dataset_name': dataset_name,
                'batch_size': self.optimal_config['batch_size'],
                'num_workers': self.optimal_config['num_workers']
            },
            'training': {
                'max_epochs': 100,
                'patience': 10,
                'precision': self.optimal_config['precision'],
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'checkpoint_storage': {
                'type': 'auto',
                'auto_sync': True,
                'sync_interval': 300  # 5 minutes
            },
            'logging': {
                'wandb': {
                    'enabled': True,
                    'project': 'predictive-maintenance-dynamic',
                    'mode': self.optimal_config['wandb_mode'],
                    'tags': [model_name, dataset_name, self.platform_info.platform.value]
                }
            }
        }
        
        # Model-specific configuration
        if model_name == 'tft':
            config['model'].update({
                'input_dim': 10,  # Will be updated based on data
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'num_classes': 2
            })
        elif model_name == 'hybrid':
            config['model'].update({
                'input_dim': 10,  # Will be updated based on data
                'cnn_channels': [64, 128, 256],
                'kernel_sizes': [3, 5, 7],
                'lstm_hidden_dim': 256,
                'lstm_num_layers': 2,
                'dropout': 0.1,
                'num_classes': 2
            })
        
        return config
    
    def _apply_platform_optimizations(self, config: Dict[str, Any]):
        """Apply platform-specific optimizations."""
        
        # GPU optimizations
        if self.platform_info.gpu_info and self.platform_info.gpu_info.get('torch_cuda_available'):
            gpu_devices = self.platform_info.gpu_info.get('devices', [])
            if gpu_devices:
                gpu_memory = gpu_devices[0]['properties']['total_memory']
                
                # Adjust batch size based on GPU memory
                if gpu_memory > 40 * 1024**3:  # A100 80GB
                    config['data']['batch_size'] = min(config['data']['batch_size'], 512)
                    config['training']['accumulate_grad_batches'] = 1
                elif gpu_memory > 20 * 1024**3:  # V100 32GB
                    config['data']['batch_size'] = min(config['data']['batch_size'], 256)
                    config['training']['accumulate_grad_batches'] = 2
                else:  # Smaller GPUs
                    config['data']['batch_size'] = min(config['data']['batch_size'], 128)
                    config['training']['accumulate_grad_batches'] = 4
        
        # Platform-specific storage configuration
        if self.platform_info.platform == CloudPlatform.COLAB:
            # Mount Google Drive for Colab
            config['checkpoint_storage'].update({
                'type': 'local',
                'base_path': '/content/drive/MyDrive/predictive_maintenance_checkpoints'
            })
            
            # Enable automatic mounting
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                logger.info("📁 Google Drive mounted")
            except ImportError:
                logger.warning("Not running in Colab, skipping Drive mount")
        
        elif self.platform_info.platform == CloudPlatform.KAGGLE:
            config['checkpoint_storage'].update({
                'type': 'local',
                'base_path': '/kaggle/working/checkpoints'
            })
            
            # Kaggle has limited internet, use offline mode
            if 'logging' in config and 'wandb' in config['logging']:
                config['logging']['wandb']['mode'] = 'offline'
        
        elif self.platform_info.platform == CloudPlatform.GCP:
            # Auto-configure GCS bucket
            project_id = self.platform_info.metadata.get('project_id')
            if project_id:
                config['checkpoint_storage'].update({
                    'type': 'gcp',
                    'bucket_name': f'{project_id}-predictive-maintenance-checkpoints'
                })
        
        elif self.platform_info.platform == CloudPlatform.AZURE:
            # Auto-configure Azure blob storage
            resource_group = self.platform_info.metadata.get('resource_group')
            if resource_group:
                config['checkpoint_storage'].update({
                    'type': 'azure',
                    'account_name': 'predictivemaintenance',
                    'container_name': f'{resource_group}-checkpoints'
                })
        
        elif self.platform_info.platform == CloudPlatform.AWS:
            # Auto-configure S3 bucket
            region = self.platform_info.metadata.get('region', 'us-east-1')
            config['checkpoint_storage'].update({
                'type': 'aws',
                'bucket_name': 'predictive-maintenance-checkpoints',
                'region': region
            })
        
        logger.info(f"🔧 Platform optimizations applied")
    
    def _apply_cli_overrides(self, config: Dict[str, Any], args):
        """Apply command line argument overrides."""
        
        if args.batch_size:
            config['data']['batch_size'] = args.batch_size
            logger.info(f"🔄 Batch size override: {args.batch_size}")
        
        if args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
            logger.info(f"📈 Learning rate override: {args.learning_rate}")
        
        if args.max_epochs:
            config['training']['max_epochs'] = args.max_epochs
            logger.info(f"🔄 Max epochs override: {args.max_epochs}")
        
        if args.precision:
            config['training']['precision'] = args.precision
            logger.info(f"🎯 Precision override: {args.precision}")
        
        if args.no_wandb:
            config['logging']['wandb']['enabled'] = False
            logger.info("📈 W&B disabled")
    
    def find_resume_checkpoint(self, config: Dict[str, Any]) -> Optional[str]:
        """Find the latest checkpoint to resume from."""
        
        project_name = config.get('project_name', 'predictive_maintenance')
        model_name = config['model']['name']
        
        # Initialize checkpoint manager
        self.checkpoint_manager = DynamicCheckpointManager(
            project_name=project_name,
            storage_config=config.get('checkpoint_storage', {}),
            auto_sync=config.get('checkpoint_storage', {}).get('auto_sync', True)
        )
        
        # Get latest checkpoint
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(model_name)
        
        if latest_checkpoint:
            logger.info(f"🔄 Found checkpoint: {latest_checkpoint.checkpoint_id}")
            logger.info(f"   Epoch: {latest_checkpoint.epoch}")
            logger.info(f"   Platform: {latest_checkpoint.platform}")
            logger.info(f"   Metrics: {latest_checkpoint.metrics}")
            return latest_checkpoint.checkpoint_id
        
        logger.info("🆕 No checkpoint found, starting fresh training")
        return None
    
    def launch_training(self, config: Dict[str, Any], resume_checkpoint: Optional[str] = None):
        """Launch training with the given configuration."""
        
        logger.info("🚀 Launching training...")
        
        # Create trainer
        trainer = A100OptimizedTrainer(config)
        
        # Setup all components
        trainer.setup_data()
        trainer.setup_model()
        trainer.setup_logger()
        
        # Resume from checkpoint if available
        if resume_checkpoint:
            try:
                checkpoint_path, checkpoint_config = self.checkpoint_manager.load_checkpoint(resume_checkpoint)
                logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
                
                # Update config with resume info
                resume_info = self.checkpoint_manager.get_resume_info(resume_checkpoint)
                if resume_info:
                    config.update(resume_info)
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.info("Starting fresh training instead")
                resume_checkpoint = None
        
        # Start training
        try:
            trainer.train()
            logger.info("✅ Training completed successfully")
            
        except KeyboardInterrupt:
            logger.info("⏹️  Training interrupted by user")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.checkpoint_manager:
                self.checkpoint_manager.stop_background_sync()

def main():
    parser = argparse.ArgumentParser(description="Dynamic Training Launcher")
    
    # Model and dataset
    parser.add_argument('--model', choices=['tft', 'hybrid'], default='tft',
                       help='Model to train')
    parser.add_argument('--dataset', choices=['ai4i', 'metro_pt2', 'cmapss'], default='ai4i',
                       help='Dataset to use')
    
    # Configuration
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int,
                       help='Batch size override')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate override')
    parser.add_argument('--max-epochs', type=int,
                       help='Maximum epochs override')
    parser.add_argument('--precision', choices=['16-mixed', '32-true'],
                       help='Training precision')
    
    # Resume and logging
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create launcher
    launcher = DynamicTrainingLauncher()
    
    # Create configuration
    config = launcher.create_config(args)
    
    # Find resume checkpoint if requested
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = launcher.find_resume_checkpoint(config)
    
    # Save final config for reference
    config_path = Path("./current_training_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    logger.info(f"📄 Config saved: {config_path}")
    
    # Launch training
    launcher.launch_training(config, resume_checkpoint)

if __name__ == "__main__":
    main()