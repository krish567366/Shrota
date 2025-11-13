"""
Advanced training pipeline for multi-channel, multi-lingual speech recognition models.
Optimized for dynamic cross-platform deployment with A100 GPU optimizations, mixed precision, 
gradient checkpointing, and efficient audio data loading.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, 
    GPUStatsMonitor, RichProgressBar, DeviceStatsMonitor
)
@hydra.main(version_base=None, config_path="../config", config_name="training_config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    print("🚀 Starting Multi-Channel, Multi-Lingual Speech Recognition Training")
    print("=" * 70)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    
    # Initialize speech trainer
    trainer = SpeechRecognitionTrainer(cfg)
    lightning.loggers 
    import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from ..data.speech_data_loader import SpeechDataModule
from ..data.audio_processing import MultiChannelAudioProcessor
from ..models.custom_multilingual_transformer import CustomMultiLingualTransformer
from ..models.custom_conformer import CustomConformerModel
from ..models.custom_cnn_rnn_hybrid import CustomLightweightCNNRNN
from ..models import create_speech_model
from ..utils.multilingual import (
    LanguageDetector, MultiLingualTokenizer, LanguageSpecificProcessor,
    CrossLingualTransferManager
)
from ..utils.helpers import (
    ExperimentTracker, GPUMonitor, ConfigManager, 
    ensure_reproducibility, setup_logging
)
from ..utils.cloud_platform import get_platform_info, get_optimal_config
from ..utils.checkpoint_manager import DynamicCheckpointManager

class SpeechRecognitionTrainer:
    """Training pipeline for multi-channel, multi-lingual speech recognition models."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.training_config = config.training
        self.hardware_config = config.get('hardware', {})
        self.speech_config = config.get('speech', {})
        
        # Setup logging
        setup_logging(
            log_level=config.get('log_level', 'INFO'),
            log_file=config.get('log_file')
        )
        
        # Ensure reproducibility
        ensure_reproducibility(config.get('seed', 42))
        
        # Initialize speech-specific components
        self.tokenizer = None
        self.language_detector = None
        self.language_processor = LanguageSpecificProcessor()
        self.transfer_manager = CrossLingualTransferManager()
        self.audio_processor = None
        
        # Detect platform and merge optimal config
        self.platform_info = get_platform_info()
        optimal_config = get_optimal_config()
        self._merge_optimal_config(optimal_config)
        
        # Initialize components
        self.data_module = None
        self.model = None
        self.trainer = None
        self.logger = None
        
        # Dynamic checkpoint management
        project_name = config.get('project_name', 'predictive_maintenance')
        storage_config = config.get('checkpoint_storage', {})
        self.checkpoint_manager = DynamicCheckpointManager(
            project_name=project_name,
            storage_config=storage_config,
            auto_sync=storage_config.get('auto_sync', True)
        )
        
        # Experiment tracking
        self.experiment_tracker = ExperimentTracker()
        self.gpu_monitor = GPUMonitor()
        
        print(f"🌐 Platform detected: {self.platform_info.platform.value}")
        print(f"💾 Storage: {storage_config.get('type', 'auto')}")
        print(f"🔄 Auto-sync: {storage_config.get('auto_sync', True)}")
    
    def _merge_optimal_config(self, optimal_config: Dict[str, Any]):
        """Merge platform-optimal configuration with user config."""
        # Update batch size if not explicitly set
        if 'batch_size' not in self.config.training:
            self.config.training.batch_size = optimal_config['batch_size']
            print(f"📊 Optimal batch size: {optimal_config['batch_size']}")
        
        # Update num_workers if not explicitly set
        if 'num_workers' not in self.config.data:
            self.config.data.num_workers = optimal_config['num_workers']
            print(f"👥 Optimal workers: {optimal_config['num_workers']}")
        
        # Update precision if not explicitly set
        if 'precision' not in self.config.training:
            self.config.training.precision = optimal_config['precision']
            print(f"🎯 Optimal precision: {optimal_config['precision']}")
        
        # Update wandb mode based on platform
        if 'wandb_mode' in optimal_config and 'logging' in self.config:
            if 'wandb' in self.config.logging:
                self.config.logging.wandb.mode = optimal_config['wandb_mode']
                print(f"📈 W&B mode: {optimal_config['wandb_mode']}")
        
    def _setup_speech_components(self):
        """Setup speech-specific components."""
        # Initialize multi-lingual tokenizer
        vocab_size = self.speech_config.get('vocab_size', 32000)
        self.tokenizer = MultiLingualTokenizer(vocab_size=vocab_size)
        
        # Initialize language detector
        self.language_detector = LanguageDetector(
            num_languages=self.speech_config.get('num_languages', 100),
            feature_dim=self.speech_config.get('feature_dim', 80)
        )
        
        # Initialize audio processor
        num_channels = self.speech_config.get('num_channels', 2)
        sample_rate = self.speech_config.get('sample_rate', 16000)
        self.audio_processor = MultiChannelAudioProcessor(
            num_channels=num_channels,
            sample_rate=sample_rate
        )
        
        print("✅ Speech components initialized")
        print(f"- Tokenizer vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"- Language detector: {self.language_detector.num_languages} languages")
        print(f"- Audio processor: {num_channels} channels, {sample_rate} Hz")

    def setup_data(self):
        """Setup speech data module."""
        print("Setting up speech data module...")
        self.data_module = SpeechDataModule(self.config)
        self.data_module.setup()
        
        print(f"Speech data module initialized:")
        print(f"- Training samples: {len(self.data_module.train_dataset) if self.data_module.train_dataset else 0}")
        print(f"- Validation samples: {len(self.data_module.val_dataset) if self.data_module.val_dataset else 0}")
        print(f"- Test samples: {len(self.data_module.test_dataset) if self.data_module.test_dataset else 0}")
        
    def setup_model(self):
        """Setup speech recognition model based on configuration."""
        model_name = self.config.model.get('name', 'transformer')
        
        print(f"Initializing speech model: {model_name}")
        
        # Get model parameters
        vocab_size = self.speech_config.get('vocab_size', 32000)
        num_languages = self.speech_config.get('num_languages', 100)
        
        # Create speech model
        self.model = create_speech_model(
            model_type=model_name,
            config=self.config,
            vocab_size=vocab_size,
            num_languages=num_languages
        )
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Speech model initialized:")
        print(f"- Model type: {model_name}")
        print(f"- Total parameters: {total_params:,}")
        print(f"- Trainable parameters: {trainable_params:,}")
        print(f"- Model size: {total_params * 4 / 1024**2:.2f} MB")
        print(f"- Vocabulary size: {vocab_size}")
        print(f"- Supported languages: {num_languages}")
        
    def setup_logger(self):
        """Setup experiment logger."""
        logging_config = self.config.get('logging', {})
        
        # Weights & Biases logger
        if logging_config.get('wandb', {}).get('enabled', False):
            wandb_config = logging_config['wandb']
            self.logger = WandbLogger(
                project=wandb_config.get('project', 'predictive-maintenance'),
                entity=wandb_config.get('entity'),
                name=f"{self.config.experiment.get('name', 'experiment')}_{self.config.model.get('name', 'model')}",
                group=wandb_config.get('group'),
                job_type=wandb_config.get('job_type', 'train'),
                tags=wandb_config.get('tags', []),
                config=OmegaConf.to_container(self.config, resolve=True)
            )
            
            # Log additional info
            if self.logger.experiment:
                self.logger.experiment.config.update({
                    'gpu_type': self.hardware_config.get('gpu_type', 'unknown'),
                    'effective_batch_size': self.hardware_config.get('effective_batch_size', 0),
                })
        
        # TensorBoard logger (alternative)
        elif logging_config.get('tensorboard', {}).get('enabled', False):
            tensorboard_config = logging_config['tensorboard']
            self.logger = TensorBoardLogger(
                save_dir=tensorboard_config.get('log_dir', 'logs/tensorboard'),
                name=self.config.model.get('name', 'model'),
                version=self.config.experiment.get('name', 'experiment')
            )
        
        else:
            self.logger = None
            print("No logger configured")
    
    def setup_callbacks(self) -> List[pl.Callback]:
        """Setup PyTorch Lightning callbacks."""
        callbacks = []
        callback_config = self.config.get('callbacks', {})
        
        # Model checkpoint
        if callback_config.get('model_checkpoint', {}).get('enabled', True):
            checkpoint_config = callback_config['model_checkpoint']
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_config.get('dirpath', 'models/checkpoints'),
                filename=checkpoint_config.get('filename', '{epoch:02d}-{val_loss:.2f}'),
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                mode=checkpoint_config.get('mode', 'min'),
                save_top_k=checkpoint_config.get('save_top_k', 3),
                save_last=checkpoint_config.get('save_last', True),
                every_n_epochs=checkpoint_config.get('every_n_epochs', 1),
                verbose=True
            )
            callbacks.append(checkpoint_callback)
        
        # Early stopping
        if callback_config.get('early_stopping', {}).get('enabled', True):
            early_stop_config = callback_config['early_stopping']
            early_stopping = EarlyStopping(
                monitor=early_stop_config.get('monitor', 'val_loss'),
                patience=early_stop_config.get('patience', 15),
                mode=early_stop_config.get('mode', 'min'),
                min_delta=early_stop_config.get('min_delta', 0.001),
                verbose=early_stop_config.get('verbose', True)
            )
            callbacks.append(early_stopping)
        
        # Learning rate monitor
        if callback_config.get('lr_monitor', {}).get('enabled', True):
            lr_monitor = LearningRateMonitor(
                logging_interval=callback_config.get('lr_monitor', {}).get('logging_interval', 'step'),
                log_momentum=callback_config.get('lr_monitor', {}).get('log_momentum', True)
            )
            callbacks.append(lr_monitor)
        
        # GPU stats monitor (A100 specific)
        if callback_config.get('gpu_stats', {}).get('enabled', True):
            gpu_stats = GPUStatsMonitor(
                memory_utilization=True,
                gpu_utilization=True,
                intra_step_time=True,
                inter_step_time=True,
                fan_speed=True,
                temperature=True
            )
            callbacks.append(gpu_stats)
        
        # Device stats monitor
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)
        
        # Rich progress bar
        if callback_config.get('rich_progress', {}).get('enabled', True):
            progress_bar = RichProgressBar(
                leave=callback_config.get('rich_progress', {}).get('leave', True)
            )
            callbacks.append(progress_bar)
        
        return callbacks
    
    def setup_trainer(self):
        """Setup PyTorch Lightning trainer with A100 optimizations."""
        training_config = self.training_config
        hardware_config = self.hardware_config
        
        # A100 optimized settings
        trainer_kwargs = {
            # Hardware settings
            'accelerator': 'gpu',
            'devices': hardware_config.get('num_gpus', 1),
            
            # Mixed precision for A100
            'precision': training_config.get('precision', '16-mixed'),
            
            # Training parameters
            'max_epochs': training_config.get('max_epochs', 100),
            'min_epochs': training_config.get('min_epochs', 1),
            
            # Validation
            'val_check_interval': training_config.get('val_check_interval', 0.25),
            'check_val_every_n_epoch': training_config.get('check_val_every_n_epoch', 1),
            
            # Logging
            'log_every_n_steps': training_config.get('log_every_n_steps', 50),
            
            # Gradient settings
            'gradient_clip_val': training_config.get('gradient_clip_val', 1.0),
            'gradient_clip_algorithm': training_config.get('gradient_clip_algorithm', 'norm'),
            'accumulate_grad_batches': hardware_config.get('gradient_accumulation', 1),
            
            # Reproducibility
            'deterministic': training_config.get('deterministic', False),
            'benchmark': training_config.get('benchmark', True),
            
            # Callbacks and logging
            'callbacks': self.setup_callbacks(),
            'logger': self.logger,
            
            # Performance optimizations
            'enable_checkpointing': True,
            'enable_progress_bar': True,
            'enable_model_summary': True,
            
            # A100 specific optimizations
            'sync_batchnorm': hardware_config.get('sync_batchnorm', True) if hardware_config.get('num_gpus', 1) > 1 else False,
        }
        
        # Multi-GPU strategy
        if hardware_config.get('num_gpus', 1) > 1:
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )
            trainer_kwargs['strategy'] = strategy
        
        # Profiler (for debugging)
        profiler_config = training_config.get('profiler')
        if profiler_config:
            if profiler_config == 'simple':
                trainer_kwargs['profiler'] = 'simple'
            elif profiler_config == 'advanced':
                trainer_kwargs['profiler'] = 'advanced'
            elif profiler_config == 'pytorch':
                from pytorch_lightning.profilers import PyTorchProfiler
                trainer_kwargs['profiler'] = PyTorchProfiler(
                    filename='profiler_output'
                )
        
        self.trainer = pl.Trainer(**trainer_kwargs)
        
        print("Trainer initialized with A100 optimizations:")
        print(f"- Precision: {trainer_kwargs['precision']}")
        print(f"- Gradient accumulation: {trainer_kwargs['accumulate_grad_batches']}")
        print(f"- Max epochs: {trainer_kwargs['max_epochs']}")
        print(f"- Devices: {trainer_kwargs['devices']}")
    
    def train(self):
        """Execute training."""
        print("Starting training...")
        
        # Start experiment tracking
        experiment_name = f"{self.config.experiment.get('name', 'experiment')}_{self.config.model.get('name')}"
        self.experiment_tracker.start_experiment(experiment_name, OmegaConf.to_container(self.config))
        
        # Log initial GPU stats
        gpu_info = self.gpu_monitor.get_gpu_info()
        if gpu_info:
            print("Initial GPU stats:")
            self.gpu_monitor.log_gpu_stats()
        
        try:
            # Train the model
            self.trainer.fit(self.model, self.data_module)
            
            # Test the model
            if self.config.get('evaluation', {}).get('test_after_training', True):
                print("Running final evaluation...")
                test_results = self.trainer.test(self.model, self.data_module)
                
                # Log test results
                if test_results:
                    self.experiment_tracker.log_metrics(test_results[0])
            
            print("Training completed successfully!")
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
        finally:
            # Final GPU stats
            final_gpu_info = self.gpu_monitor.get_gpu_info()
            if final_gpu_info:
                print("Final GPU stats:")
                self.gpu_monitor.log_gpu_stats()
            
            # Finish experiment tracking
            self.experiment_tracker.finish_experiment()
    
    def validate(self):
        """Run validation only."""
        print("Running validation...")
        validation_results = self.trainer.validate(self.model, self.data_module)
        return validation_results
    
    def test(self):
        """Run testing only."""
        print("Running testing...")
        test_results = self.trainer.test(self.model, self.data_module)
        return test_results
    
    def save_model(self, filepath: str):
        """Save trained model."""
        self.trainer.save_checkpoint(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load pretrained speech model."""
        model_name = self.config.model.get('name', 'transformer')
        
        # Load the appropriate speech model
        vocab_size = self.speech_config.get('vocab_size', 32000)
        num_languages = self.speech_config.get('num_languages', 100)
        
        self.model = create_speech_model(
            model_type=model_name,
            config=self.config,
            vocab_size=vocab_size,
            num_languages=num_languages
        )
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        
        print(f"Speech model loaded from {filepath}")

def create_trainer_from_config(config_path: str) -> SpeechRecognitionTrainer:
    """Create speech trainer from configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config = OmegaConf.create(config)
    return SpeechRecognitionTrainer(config)

@hydra.main(version_base=None, config_path="../config", config_name="training_config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    print("🚀 Starting Predictive Maintenance Training with A100 Optimization")
    print("=" * 70)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    
    # Create trainer
    trainer = A100OptimizedTrainer(cfg)
    
    # Setup components
    trainer.setup_data()
    trainer.setup_model()
    trainer.setup_logger()
    trainer.setup_trainer()
    
    # Execute training
    trainer.train()
    
    print("🎉 Training completed!")

def train_model(config_path: str, model_checkpoint: Optional[str] = None):
    """Standalone training function."""
    trainer = create_trainer_from_config(config_path)
    
    # Setup all components
    trainer.setup_data()
    trainer.setup_model()
    trainer.setup_logger()
    trainer.setup_trainer()
    
    # Load checkpoint if provided
    if model_checkpoint:
        trainer.load_model(model_checkpoint)
    
    # Train
    trainer.train()
    
    return trainer

def validate_model(config_path: str, model_checkpoint: str):
    """Standalone validation function."""
    trainer = create_trainer_from_config(config_path)
    
    # Setup components
    trainer.setup_data()
    trainer.setup_model()
    trainer.setup_trainer()
    
    # Load checkpoint
    trainer.load_model(model_checkpoint)
    
    # Validate
    results = trainer.validate()
    return results

def test_model(config_path: str, model_checkpoint: str):
    """Standalone testing function."""
    trainer = create_trainer_from_config(config_path)
    
    # Setup components
    trainer.setup_data()
    trainer.setup_model()
    trainer.setup_trainer()
    
    # Load checkpoint
    trainer.load_model(model_checkpoint)
    
    # Test
    results = trainer.test()
    return results

if __name__ == "__main__":
    main()