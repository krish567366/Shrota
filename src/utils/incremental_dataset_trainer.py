#!/usr/bin/env python3
"""
Incremental Dataset Training System

Features:
- Dataset-by-dataset training progression
- State accumulation across datasets
- Automatic dataset switching
- Progress tracking per dataset
- Curriculum learning integration

Usage:
    dataset_trainer = IncrementalDatasetTrainer(config)
    dataset_trainer.train_phase("A")
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about a training dataset."""
    
    name: str
    path: str
    size_gb: float
    num_samples: int
    languages: List[str]
    quality_score: float
    difficulty_level: int
    estimated_epochs: int
    preprocessing_required: bool
    
@dataclass
class DatasetProgress:
    """Progress tracking for a dataset."""
    
    dataset_name: str
    status: str  # 'not_started', 'in_progress', 'completed', 'failed'
    epochs_completed: int
    samples_processed: int
    total_samples: int
    best_loss: float
    best_wer: float
    training_time: float
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    checkpoint_path: Optional[str] = None

class IncrementalDatasetTrainer:
    """Incremental dataset training system for robust multi-dataset training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_configs = self._load_dataset_configs()
        self.phase_datasets = config.get('phase_datasets', {})
        self.progress_file = Path(config.get('progress_file', './dataset_progress.json'))
        self.accumulated_state = {}
        
        # Load existing progress
        self.dataset_progress = self._load_progress()
        
        # Initialize dataset information
        self.dataset_info = self._initialize_dataset_info()
    
    def train_phase(self, phase: str, checkpoint_manager=None, model=None, trainer=None) -> Dict[str, Any]:
        """Train all datasets for a specific phase incrementally."""
        
        phase_datasets = self.phase_datasets.get(phase, [])
        if not phase_datasets:
            logger.error(f"No datasets defined for phase {phase}")
            return {'status': 'error', 'message': f'No datasets for phase {phase}'}
        
        logger.info(f"🚀 Starting incremental training for Phase {phase}")
        logger.info(f"   Datasets: {phase_datasets}")
        
        phase_results = {
            'phase': phase,
            'datasets': {},
            'overall_metrics': {},
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'in_progress'
        }
        
        for dataset_name in phase_datasets:
            # Check if dataset already completed
            if self._is_dataset_completed(dataset_name):
                logger.info(f"✅ Dataset '{dataset_name}' already completed, skipping")
                continue
            
            # Train on this dataset
            dataset_result = self._train_single_dataset(
                dataset_name, phase, checkpoint_manager, model, trainer
            )
            
            phase_results['datasets'][dataset_name] = dataset_result
            
            # Update accumulated state
            self._update_accumulated_state(dataset_name, dataset_result)
            
            # Save progress
            self._save_progress()
            
            # Check if training should continue
            if dataset_result['status'] == 'failed':
                logger.error(f"❌ Dataset '{dataset_name}' training failed, stopping phase")
                phase_results['status'] = 'failed'
                break
            elif dataset_result['status'] == 'interrupted':
                logger.warning(f"⚠️  Dataset '{dataset_name}' training interrupted")
                phase_results['status'] = 'interrupted'
                break
        
        # Finalize phase results
        phase_results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        if phase_results['status'] == 'in_progress':
            phase_results['status'] = 'completed'
        
        phase_results['overall_metrics'] = self._calculate_phase_metrics(phase)
        
        logger.info(f"📊 Phase {phase} training completed")
        logger.info(f"   Status: {phase_results['status']}")
        logger.info(f"   Overall metrics: {phase_results['overall_metrics']}")
        
        return phase_results
    
    def get_next_dataset(self, phase: str) -> Optional[str]:
        """Get the next dataset to train for a phase."""
        
        phase_datasets = self.phase_datasets.get(phase, [])
        
        for dataset_name in phase_datasets:
            if not self._is_dataset_completed(dataset_name):
                return dataset_name
        
        return None  # All datasets completed
    
    def get_phase_progress(self, phase: str) -> Dict[str, Any]:
        """Get progress overview for a phase."""
        
        phase_datasets = self.phase_datasets.get(phase, [])
        
        completed = []
        in_progress = []
        not_started = []
        failed = []
        
        for dataset_name in phase_datasets:
            progress = self.dataset_progress.get(dataset_name)
            if not progress:
                not_started.append(dataset_name)
            elif progress.status == 'completed':
                completed.append(dataset_name)
            elif progress.status == 'in_progress':
                in_progress.append(dataset_name)
            elif progress.status == 'failed':
                failed.append(dataset_name)
            else:
                not_started.append(dataset_name)
        
        total_datasets = len(phase_datasets)
        completion_percentage = (len(completed) / total_datasets * 100) if total_datasets > 0 else 0
        
        return {
            'phase': phase,
            'total_datasets': total_datasets,
            'completed': completed,
            'in_progress': in_progress,
            'not_started': not_started,
            'failed': failed,
            'completion_percentage': completion_percentage,
            'status': self._determine_phase_status(completed, in_progress, not_started, failed, total_datasets)
        }
    
    def resume_dataset_training(self, dataset_name: str, phase: str, checkpoint_manager=None, model=None, trainer=None) -> Dict[str, Any]:
        """Resume training for a specific dataset."""
        
        logger.info(f"🔄 Resuming training for dataset: {dataset_name}")
        
        # Load existing progress
        progress = self.dataset_progress.get(dataset_name)
        if progress and progress.checkpoint_path:
            logger.info(f"   Found checkpoint: {progress.checkpoint_path}")
        
        # Continue training
        return self._train_single_dataset(dataset_name, phase, checkpoint_manager, model, trainer, resume=True)
    
    def _train_single_dataset(self, dataset_name: str, phase: str, checkpoint_manager=None, model=None, trainer=None, resume: bool = False) -> Dict[str, Any]:
        """Train on a single dataset."""
        
        logger.info(f"📚 Training on dataset: {dataset_name}")
        
        # Initialize or update progress
        if dataset_name not in self.dataset_progress or not resume:
            self.dataset_progress[dataset_name] = DatasetProgress(
                dataset_name=dataset_name,
                status='in_progress',
                epochs_completed=0,
                samples_processed=0,
                total_samples=self.dataset_info[dataset_name].num_samples,
                best_loss=float('inf'),
                best_wer=float('inf'),
                training_time=0.0,
                start_time=time.strftime('%Y-%m-%d %H:%M:%S')
            )
        
        progress = self.dataset_progress[dataset_name]
        progress.status = 'in_progress'
        
        # Get dataset configuration
        dataset_config = self._get_dataset_training_config(dataset_name, phase)
        
        try:
            # This is where you would integrate with your actual training loop
            # For now, we'll simulate the training process
            
            dataset_result = {
                'dataset_name': dataset_name,
                'status': 'completed',
                'epochs_trained': dataset_config['epochs'],
                'final_loss': 2.3,  # Would come from actual training
                'final_wer': 0.15,  # Would come from actual training
                'training_time': 3600.0,  # Would come from actual training
                'samples_processed': progress.total_samples,
                'checkpoint_path': f"./checkpoints/{dataset_name}_final.ckpt"
            }
            
            # Update progress
            progress.status = 'completed'
            progress.epochs_completed = dataset_result['epochs_trained']
            progress.samples_processed = dataset_result['samples_processed']
            progress.best_loss = dataset_result['final_loss']
            progress.best_wer = dataset_result['final_wer']
            progress.training_time = dataset_result['training_time']
            progress.end_time = time.strftime('%Y-%m-%d %H:%M:%S')
            progress.checkpoint_path = dataset_result['checkpoint_path']
            
            logger.info(f"✅ Dataset '{dataset_name}' training completed")
            logger.info(f"   Final Loss: {dataset_result['final_loss']:.4f}")
            logger.info(f"   Final WER: {dataset_result['final_wer']:.4f}")
            logger.info(f"   Training Time: {dataset_result['training_time']:.1f}s")
            
            return dataset_result
            
        except KeyboardInterrupt:
            logger.warning(f"⚠️  Training interrupted for dataset: {dataset_name}")
            progress.status = 'interrupted'
            return {
                'dataset_name': dataset_name,
                'status': 'interrupted',
                'message': 'Training interrupted by user'
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed for dataset '{dataset_name}': {str(e)}")
            progress.status = 'failed'
            return {
                'dataset_name': dataset_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def _get_dataset_training_config(self, dataset_name: str, phase: str) -> Dict[str, Any]:
        """Get training configuration for a specific dataset."""
        
        base_config = self.config.get('base_training_config', {})
        dataset_config = self.dataset_configs.get(dataset_name, {})
        
        # Merge configurations
        training_config = {**base_config, **dataset_config}
        
        # Apply phase-specific adjustments
        phase_adjustments = self.config.get('phase_adjustments', {}).get(phase, {})
        training_config.update(phase_adjustments)
        
        # Apply dataset-specific adjustments based on characteristics
        dataset_info = self.dataset_info.get(dataset_name)
        if dataset_info:
            # Adjust epochs based on dataset size
            if dataset_info.size_gb > 10:
                training_config['epochs'] = min(training_config.get('epochs', 5), 3)
            elif dataset_info.size_gb < 1:
                training_config['epochs'] = max(training_config.get('epochs', 5), 8)
            
            # Adjust learning rate based on difficulty
            base_lr = training_config.get('learning_rate', 1e-4)
            if dataset_info.difficulty_level > 3:
                training_config['learning_rate'] = base_lr * 0.5
            elif dataset_info.difficulty_level < 2:
                training_config['learning_rate'] = base_lr * 1.5
        
        return training_config
    
    def _is_dataset_completed(self, dataset_name: str) -> bool:
        """Check if a dataset training is completed."""
        
        progress = self.dataset_progress.get(dataset_name)
        return progress is not None and progress.status == 'completed'
    
    def _update_accumulated_state(self, dataset_name: str, dataset_result: Dict):
        """Update accumulated training state across datasets."""
        
        if 'metrics' not in self.accumulated_state:
            self.accumulated_state['metrics'] = []
        
        if 'total_samples' not in self.accumulated_state:
            self.accumulated_state['total_samples'] = 0
        
        if 'total_training_time' not in self.accumulated_state:
            self.accumulated_state['total_training_time'] = 0.0
        
        # Add dataset metrics
        self.accumulated_state['metrics'].append({
            'dataset': dataset_name,
            'final_loss': dataset_result.get('final_loss', 0.0),
            'final_wer': dataset_result.get('final_wer', 0.0),
            'samples_processed': dataset_result.get('samples_processed', 0),
            'training_time': dataset_result.get('training_time', 0.0)
        })
        
        # Update totals
        self.accumulated_state['total_samples'] += dataset_result.get('samples_processed', 0)
        self.accumulated_state['total_training_time'] += dataset_result.get('training_time', 0.0)
        
        # Calculate running averages
        metrics = self.accumulated_state['metrics']
        self.accumulated_state['avg_loss'] = sum(m['final_loss'] for m in metrics) / len(metrics)
        self.accumulated_state['avg_wer'] = sum(m['final_wer'] for m in metrics) / len(metrics)
    
    def _calculate_phase_metrics(self, phase: str) -> Dict[str, Any]:
        """Calculate overall metrics for a phase."""
        
        phase_datasets = self.phase_datasets.get(phase, [])
        completed_datasets = [d for d in phase_datasets if self._is_dataset_completed(d)]
        
        if not completed_datasets:
            return {}
        
        metrics = []
        total_samples = 0
        total_time = 0.0
        
        for dataset_name in completed_datasets:
            progress = self.dataset_progress[dataset_name]
            metrics.append({
                'dataset': dataset_name,
                'loss': progress.best_loss,
                'wer': progress.best_wer,
                'samples': progress.samples_processed,
                'time': progress.training_time
            })
            total_samples += progress.samples_processed
            total_time += progress.training_time
        
        # Calculate weighted averages (by sample count)
        weighted_loss = sum(m['loss'] * m['samples'] for m in metrics) / total_samples if total_samples > 0 else 0
        weighted_wer = sum(m['wer'] * m['samples'] for m in metrics) / total_samples if total_samples > 0 else 0
        
        return {
            'completed_datasets': len(completed_datasets),
            'total_datasets': len(phase_datasets),
            'completion_rate': len(completed_datasets) / len(phase_datasets),
            'weighted_avg_loss': weighted_loss,
            'weighted_avg_wer': weighted_wer,
            'total_samples_processed': total_samples,
            'total_training_time': total_time,
            'avg_time_per_dataset': total_time / len(completed_datasets) if completed_datasets else 0
        }
    
    def _determine_phase_status(self, completed: List, in_progress: List, not_started: List, failed: List, total: int) -> str:
        """Determine overall phase status."""
        
        if len(failed) > 0:
            return 'failed'
        elif len(completed) == total:
            return 'completed'
        elif len(in_progress) > 0:
            return 'in_progress'
        else:
            return 'not_started'
    
    def _load_dataset_configs(self) -> Dict[str, Dict]:
        """Load dataset-specific configurations."""
        
        config_file = self.config.get('dataset_config_file', './config/dataset_configs.yaml')
        
        if not os.path.exists(config_file):
            logger.warning(f"Dataset config file not found: {config_file}")
            return {}
        
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load dataset configs: {str(e)}")
            return {}
    
    def _initialize_dataset_info(self) -> Dict[str, DatasetInfo]:
        """Initialize dataset information."""
        
        # This would normally load from metadata files or scan directories
        # For now, we'll use example configurations
        
        return {
            'indicvoices': DatasetInfo(
                name='IndicVoices',
                path='/data/indicvoices',
                size_gb=15.2,
                num_samples=50000,
                languages=['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'or', 'pa'],
                quality_score=0.95,
                difficulty_level=2,
                estimated_epochs=5,
                preprocessing_required=True
            ),
            'spring_inx': DatasetInfo(
                name='SPRING-INX',
                path='/data/spring_inx',
                size_gb=8.7,
                num_samples=30000,
                languages=['hi', 'en'],
                quality_score=0.92,
                difficulty_level=3,
                estimated_epochs=6,
                preprocessing_required=True
            ),
            'india_multilingual': DatasetInfo(
                name='India Multilingual',
                path='/data/india_multilingual',
                size_gb=22.1,
                num_samples=75000,
                languages=['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'en'],
                quality_score=0.88,
                difficulty_level=4,
                estimated_epochs=4,
                preprocessing_required=True
            ),
            'whisper_dataset': DatasetInfo(
                name='Whisper Dataset',
                path='/data/whisper',
                size_gb=45.6,
                num_samples=120000,
                languages=['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'or', 'pa', 'en'],
                quality_score=0.97,
                difficulty_level=2,
                estimated_epochs=3,
                preprocessing_required=False
            ),
            'fleurs': DatasetInfo(
                name='FLEURS',
                path='/data/fleurs',
                size_gb=12.3,
                num_samples=40000,
                languages=['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'or', 'pa'],
                quality_score=0.93,
                difficulty_level=3,
                estimated_epochs=5,
                preprocessing_required=True
            ),
            'common_voice': DatasetInfo(
                name='Common Voice',
                path='/data/common_voice',
                size_gb=18.9,
                num_samples=65000,
                languages=['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'or', 'pa'],
                quality_score=0.85,
                difficulty_level=3,
                estimated_epochs=4,
                preprocessing_required=True
            )
        }
    
    def _load_progress(self) -> Dict[str, DatasetProgress]:
        """Load existing dataset training progress."""
        
        if not self.progress_file.exists():
            return {}
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Convert dict to DatasetProgress objects
            progress = {}
            for dataset_name, data in progress_data.items():
                progress[dataset_name] = DatasetProgress(**data)
            
            return progress
        except Exception as e:
            logger.error(f"Failed to load progress: {str(e)}")
            return {}
    
    def _save_progress(self):
        """Save current dataset training progress."""
        
        try:
            # Convert DatasetProgress objects to dict
            progress_data = {}
            for dataset_name, progress in self.dataset_progress.items():
                progress_data[dataset_name] = {
                    'dataset_name': progress.dataset_name,
                    'status': progress.status,
                    'epochs_completed': progress.epochs_completed,
                    'samples_processed': progress.samples_processed,
                    'total_samples': progress.total_samples,
                    'best_loss': progress.best_loss,
                    'best_wer': progress.best_wer,
                    'training_time': progress.training_time,
                    'start_time': progress.start_time,
                    'end_time': progress.end_time,
                    'checkpoint_path': progress.checkpoint_path
                }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}")

if __name__ == "__main__":
    # Example usage
    config = {
        'phase_datasets': {
            'A': ['indicvoices', 'spring_inx', 'india_multilingual'],
            'B': ['whisper_dataset', 'fleurs', 'common_voice'],
        },
        'base_training_config': {
            'epochs': 5,
            'learning_rate': 1e-4,
            'batch_size': 16
        },
        'phase_adjustments': {
            'A': {'learning_rate': 2e-4},  # Higher LR for initial phase
            'B': {'learning_rate': 5e-5}   # Lower LR for fine-tuning
        }
    }
    
    trainer = IncrementalDatasetTrainer(config)
    
    # Get phase progress
    progress = trainer.get_phase_progress('A')
    print(f"Phase A progress: {progress}")
    
    # Get next dataset
    next_dataset = trainer.get_next_dataset('A')
    print(f"Next dataset for Phase A: {next_dataset}")
    
    print("✅ Incremental Dataset Trainer ready!")