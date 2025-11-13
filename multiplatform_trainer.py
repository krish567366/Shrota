#!/usr/bin/env python3
"""
Multi-Platform Resilient Training System

This is the main training script that combines all components:
- Advanced checkpoint management
- Platform detection and optimization
- Incremental dataset training
- Cost tracking and interruption handling
- Automatic resume capabilities

Usage:
    python multiplatform_trainer.py --phase A
    python multiplatform_trainer.py --resume --checkpoint path/to/checkpoint.ckpt
    python multiplatform_trainer.py --continue-dataset indicvoices
"""

import os
import sys
import argparse
import json
import time
import signal
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.advanced_checkpoint_manager import AdvancedCheckpointManager
from utils.multiplatform_training_system import MultiPlatformTrainingSystem
from utils.incremental_dataset_trainer import IncrementalDatasetTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multiplatform_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiPlatformTrainer:
    """Main training orchestrator for multi-platform, resumable training."""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.checkpoint_manager = AdvancedCheckpointManager(self.config['checkpoint'])
        self.platform_system = MultiPlatformTrainingSystem(self.config['platform'])
        self.dataset_trainer = IncrementalDatasetTrainer(self.config['datasets'])
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.current_phase = None
        self.training_interrupted = False
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._handle_interruption)
        signal.signal(signal.SIGTERM, self._handle_interruption)
    
    def train_phase(self, phase: str, resume: bool = False, checkpoint_path: str = None) -> Dict[str, Any]:
        """Train a complete phase with all datasets."""
        
        logger.info(f"🚀 Starting Phase {phase} training")
        logger.info(f"   Platform: {self.platform_system.current_platform}")
        logger.info(f"   Resume: {resume}")
        
        # Get platform-optimized configuration
        platform_config = self.platform_system.get_optimized_config()
        logger.info(f"   Optimized for: {platform_config['platform_name']}")
        logger.info(f"   Max batch size: {platform_config['batch_size']}")
        logger.info(f"   Workers: {platform_config['num_workers']}")
        
        try:
            # Initialize or resume training
            if resume and checkpoint_path:
                training_state = self._resume_from_checkpoint(checkpoint_path)
                if training_state:
                    self.current_phase = training_state['training_state']['phase']
                    logger.info(f"✅ Resumed from checkpoint: {checkpoint_path}")
                else:
                    logger.error("❌ Failed to resume from checkpoint")
                    return {'status': 'error', 'message': 'Failed to resume'}
            else:
                self._initialize_training(phase, platform_config)
                self.current_phase = phase
            
            # Start incremental dataset training
            phase_result = self._train_phase_with_monitoring(phase)
            
            return phase_result
            
        except KeyboardInterrupt:
            logger.warning("⚠️  Training interrupted by user")
            return self._handle_interruption()
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def continue_dataset(self, dataset_name: str, phase: str = None) -> Dict[str, Any]:
        """Continue training from a specific dataset."""
        
        if phase is None:
            # Try to determine phase from dataset
            phase = self._determine_phase_for_dataset(dataset_name)
        
        if not phase:
            return {'status': 'error', 'message': f'Cannot determine phase for dataset {dataset_name}'}
        
        logger.info(f"🔄 Continuing training from dataset: {dataset_name} (Phase {phase})")
        
        # Resume dataset training
        return self.dataset_trainer.resume_dataset_training(
            dataset_name, phase, self.checkpoint_manager, self.model, self
        )
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status."""
        
        status = {
            'platform': {
                'name': self.platform_system.current_platform,
                'config': self.platform_system.get_optimized_config(),
                'limits': self.platform_system.check_platform_limits()
            },
            'phases': {},
            'overall_progress': {},
            'cost_info': self.platform_system.cost_tracker.check_limits()
        }
        
        # Get progress for each phase
        for phase in ['A', 'B', 'C', 'D', 'E']:
            phase_progress = self.dataset_trainer.get_phase_progress(phase)
            if phase_progress['total_datasets'] > 0:
                status['phases'][phase] = phase_progress
        
        # Calculate overall progress
        total_datasets = sum(p['total_datasets'] for p in status['phases'].values())
        completed_datasets = sum(len(p['completed']) for p in status['phases'].values())
        
        status['overall_progress'] = {
            'total_datasets': total_datasets,
            'completed_datasets': completed_datasets,
            'completion_percentage': (completed_datasets / total_datasets * 100) if total_datasets > 0 else 0,
            'current_phase': self.current_phase
        }
        
        return status
    
    def _train_phase_with_monitoring(self, phase: str) -> Dict[str, Any]:
        """Train phase with continuous monitoring and checkpointing."""
        
        start_time = time.time()
        
        while True:
            # Check platform limits
            limits = self.platform_system.check_platform_limits()
            
            if limits['should_stop']:
                logger.warning("⚠️  Platform limits reached, stopping training")
                interruption_info = self.platform_system.handle_interruption(
                    self.checkpoint_manager, self.model, self.optimizer, 
                    self.scheduler, self._get_dataset_state(), self._get_metrics(), phase
                )
                return {
                    'status': 'interrupted',
                    'reason': 'platform_limits',
                    'interruption_info': interruption_info
                }
            
            if limits['should_checkpoint']:
                logger.info("💾 Saving checkpoint due to platform limits")
                self.checkpoint_manager.save_complete_state(
                    self.model, self.optimizer, self.scheduler,
                    self._get_dataset_state(), self._get_metrics(), phase
                )
            
            # Get next dataset to train
            next_dataset = self.dataset_trainer.get_next_dataset(phase)
            
            if not next_dataset:
                logger.info(f"✅ All datasets completed for Phase {phase}")
                break
            
            # Train on next dataset
            logger.info(f"📚 Training on dataset: {next_dataset}")
            dataset_result = self.dataset_trainer._train_single_dataset(
                next_dataset, phase, self.checkpoint_manager, self.model, self
            )
            
            if dataset_result['status'] == 'interrupted':
                return {
                    'status': 'interrupted',
                    'reason': 'dataset_interrupted',
                    'current_dataset': next_dataset,
                    'dataset_result': dataset_result
                }
            elif dataset_result['status'] == 'failed':
                return {
                    'status': 'failed',
                    'reason': 'dataset_failed',
                    'current_dataset': next_dataset,
                    'dataset_result': dataset_result
                }
            
            # Mark dataset as completed
            self.dataset_trainer.mark_dataset_completed(next_dataset)
            
            # Save progress checkpoint
            self.checkpoint_manager.save_complete_state(
                self.model, self.optimizer, self.scheduler,
                self._get_dataset_state(), self._get_metrics(), phase
            )
        
        # Phase completed successfully
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"✅ Phase {phase} completed successfully")
        logger.info(f"   Training time: {training_time:.1f} seconds")
        
        return {
            'status': 'completed',
            'phase': phase,
            'training_time': training_time,
            'metrics': self.dataset_trainer._calculate_phase_metrics(phase)
        }
    
    def _initialize_training(self, phase: str, platform_config: Dict):
        """Initialize training components."""
        
        logger.info("🔧 Initializing training components...")
        
        # This is where you would initialize your actual model, optimizer, etc.
        # For now, we'll use placeholders
        
        # Initialize model (placeholder)
        self.model = self._create_model(platform_config)
        
        # Initialize optimizer (placeholder)
        self.optimizer = self._create_optimizer(self.model, platform_config)
        
        # Initialize scheduler (placeholder)
        self.scheduler = self._create_scheduler(self.optimizer, platform_config)
        
        logger.info("✅ Training components initialized")
    
    def _resume_from_checkpoint(self, checkpoint_path: str) -> Optional[Dict]:
        """Resume training from checkpoint."""
        
        logger.info(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = self.checkpoint_manager.load_complete_state(checkpoint_path)
        
        if not checkpoint_data:
            return None
        
        # Restore model state (placeholder)
        # self.model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Restore optimizer state (placeholder)
        # self.optimizer.load_state_dict(checkpoint_data['training_state']['optimizer_state'])
        
        # Restore scheduler state (placeholder)
        # if checkpoint_data['training_state']['scheduler_state']:
        #     self.scheduler.load_state_dict(checkpoint_data['training_state']['scheduler_state'])
        
        return checkpoint_data
    
    def _handle_interruption(self, signum=None, frame=None) -> Dict[str, Any]:
        """Handle training interruption gracefully."""
        
        if signum:
            logger.warning(f"⚠️  Received signal {signum}, handling interruption...")
        
        self.training_interrupted = True
        
        # Save emergency checkpoint if components are initialized
        if self.model and self.optimizer and self.current_phase:
            interruption_info = self.platform_system.handle_interruption(
                self.checkpoint_manager, self.model, self.optimizer,
                self.scheduler, self._get_dataset_state(), self._get_metrics(),
                self.current_phase
            )
            
            return {
                'status': 'interrupted',
                'reason': 'user_interruption',
                'interruption_info': interruption_info
            }
        
        return {'status': 'interrupted', 'reason': 'user_interruption'}
    
    def _create_model(self, platform_config: Dict):
        """Create model based on platform configuration."""
        # Placeholder - would create actual model
        logger.info("🧠 Creating model (placeholder)")
        return "model_placeholder"
    
    def _create_optimizer(self, model, platform_config: Dict):
        """Create optimizer based on platform configuration."""
        # Placeholder - would create actual optimizer
        logger.info("⚡ Creating optimizer (placeholder)")
        return "optimizer_placeholder"
    
    def _create_scheduler(self, optimizer, platform_config: Dict):
        """Create scheduler based on platform configuration."""
        # Placeholder - would create actual scheduler
        logger.info("📈 Creating scheduler (placeholder)")
        return "scheduler_placeholder"
    
    def _get_dataset_state(self) -> Dict[str, Any]:
        """Get current dataset training state."""
        # Placeholder - would return actual dataset state
        return {
            'epoch': 1,
            'global_step': 100,
            'current_dataset': 'indicvoices',
            'samples_processed': 1000,
            'estimated_total_steps': 10000,
            'training_time': 3600.0,
            'completed_datasets': [],
            'curriculum_stage': 1,
            'current_dataset_epoch': 1
        }
    
    def _get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        # Placeholder - would return actual metrics
        return {
            'best_val_loss': 2.5,
            'best_val_wer': 0.18,
            'history': []
        }
    
    def _determine_phase_for_dataset(self, dataset_name: str) -> Optional[str]:
        """Determine which phase a dataset belongs to."""
        
        phase_datasets = self.config['datasets']['phase_datasets']
        
        for phase, datasets in phase_datasets.items():
            if dataset_name in datasets:
                return phase
        
        return None
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load training configuration."""
        
        if config_path is None:
            config_path = './config/multiplatform_config.yaml'
        
        # Default configuration
        default_config = {
            'checkpoint': {
                'checkpoint_dir': './checkpoints',
                'auto_save_interval': 300,
                'max_checkpoints': 10,
                'cloud_storage': {
                    'type': 'none'  # 'aws_s3', 'gcp_gcs', 'azure_blob', or 'none'
                }
            },
            'platform': {
                'cost_limits': {
                    'daily_limit': 50.0,
                    'session_limit': 25.0
                }
            },
            'datasets': {
                'phase_datasets': {
                    'A': ['indicvoices', 'spring_inx', 'india_multilingual'],
                    'B': ['whisper_dataset', 'fleurs', 'common_voice'],
                    'C': ['multichannel_data'],
                    'D': ['speaker_data'],
                    'E': ['optimization_data']
                },
                'base_training_config': {
                    'epochs': 5,
                    'learning_rate': 1e-4,
                    'batch_size': 16
                },
                'phase_adjustments': {
                    'A': {'learning_rate': 2e-4},
                    'B': {'learning_rate': 1e-4},
                    'C': {'learning_rate': 5e-5},
                    'D': {'learning_rate': 2e-5},
                    'E': {'learning_rate': 1e-5}
                }
            }
        }
        
        # Try to load from file
        if config_path and os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                # Merge with default config
                for key, value in file_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
                logger.info("Using default configuration")
        
        return default_config

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Multi-Platform Resilient Training System')
    parser.add_argument('--phase', type=str, choices=['A', 'B', 'C', 'D', 'E'], 
                       help='Training phase to run')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                       help='Specific checkpoint to resume from')
    parser.add_argument('--continue-dataset', type=str, 
                       help='Continue training from specific dataset')
    parser.add_argument('--status', action='store_true', 
                       help='Show training status and exit')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MultiPlatformTrainer(args.config)
    
    if args.status:
        # Show status and exit
        status = trainer.get_training_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.continue_dataset:
        # Continue from specific dataset
        result = trainer.continue_dataset(args.continue_dataset)
        print(f"Dataset training result: {result}")
        return
    
    if args.phase:
        # Train specific phase
        result = trainer.train_phase(args.phase, args.resume, args.checkpoint)
        print(f"Phase {args.phase} result: {result}")
        return
    
    # Interactive mode
    print("🚀 Multi-Platform Resilient Training System")
    print("=" * 50)
    
    status = trainer.get_training_status()
    print(f"Platform: {status['platform']['name']}")
    print(f"Overall Progress: {status['overall_progress']['completion_percentage']:.1f}%")
    print(f"Completed Datasets: {status['overall_progress']['completed_datasets']}/{status['overall_progress']['total_datasets']}")
    
    if status['overall_progress']['current_phase']:
        print(f"Current Phase: {status['overall_progress']['current_phase']}")
    
    print("\nAvailable commands:")
    print("  --phase A/B/C/D/E : Train specific phase")
    print("  --resume : Resume from latest checkpoint")
    print("  --status : Show detailed status")
    print("  --continue-dataset <name> : Continue from specific dataset")

if __name__ == "__main__":
    main()