#!/usr/bin/env python3
"""
Launch script for Multi-Channel, Multi-Lingual Speech Recognition System

This script provides a unified entry point for training, inference, and evaluation
of the speech recognition models with dynamic cross-platform support.
"""

import sys
import os
from pathlib import Path
import argparse
import torch
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.training.train import SpeechRecognitionTrainer
from src.inference.predict import SpeechPredictor
from src.utils.cloud_platform import get_platform_info, get_optimal_config
from src.utils.multilingual import get_supported_languages, get_language_families
from src.data.dataset_loaders import DatasetDownloader
from src.models import get_available_models

def print_banner():
    """Print system banner."""
    print("🎤" + "=" * 78 + "🎤")
    print("   Multi-Channel, Multi-Lingual Speech Recognition System")
    print("   Built from scratch with dynamic cross-platform support")
    print("=" * 80)
    print()

def print_system_info():
    """Print system and platform information."""
    print("🖥️  System Information:")
    platform_info = get_platform_info()
    
    for key, value in platform_info.items():
        print(f"   {key}: {value}")
    
    print()
    
    # GPU information
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Running on CPU")
    
    print()

def print_model_info():
    """Print available models information."""
    print("🤖 Available Speech Models:")
    models = get_available_models()
    
    for model_type, model_info in models.items():
        print(f"   {model_type}:")
        print(f"      Description: {model_info['description']}")
        print(f"      Parameters: ~{model_info.get('parameters', 'Variable')}")
        print(f"      Best for: {model_info.get('best_for', 'General speech recognition')}")
        print()

def print_language_info():
    """Print supported languages information."""
    print("🌍 Language Support:")
    languages = get_supported_languages()
    families = get_language_families()
    
    print(f"   Total supported languages: {len(languages)}")
    print(f"   Language families: {len(families)}")
    
    # Show some examples
    print("   Example languages:")
    example_langs = list(languages.items())[:10]
    for code, name in example_langs:
        print(f"      {code}: {name}")
    
    if len(languages) > 10:
        print(f"      ... and {len(languages) - 10} more")
    
    print()

def print_audio_info():
    """Print audio processing capabilities."""
    print("🎧 Audio Processing Capabilities:")
    print("   Supported formats: WAV, FLAC, MP3, OGG, M4A")
    print("   Multi-channel support: Mono, Stereo, 5.1, 7.1, Custom")
    print("   Multi-speaker support: 2-3+ speakers with overlapping speech")
    print("   Enhancement features:")
    print("      • Beamforming (Delay-and-Sum, Adaptive)")
    print("      • Speaker separation and diarization")
    print("      • Overlapping speech detection")
    print("      • Noise reduction (Spectral Subtraction, Wiener)")
    print("      • Echo cancellation")
    print("      • Dynamic range compression")
    print("      • Spectral whitening")
    print()

def create_default_config():
    """Create default configuration."""
    config = {
        'model': {
            'name': 'transformer',  # transformer, conformer, cnn_rnn
            'hidden_size': 512,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        },
        'speech': {
            'vocab_size': 32000,
            'num_languages': 100,
            'num_channels': 2,
            'sample_rate': 16000,
            'feature_dim': 80
        },
        'data': {
            'data_dir': './data/speech',
            'batch_size': 16,
            'num_workers': 4,
            'sample_rate': 16000,
            'n_mels': 80,
            'max_audio_length': 30,
            'max_text_length': 512
        },
        'training': {
            'max_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'precision': '16-mixed',
            'strategy': 'auto'
        },
        'logging': {
            'log_level': 'INFO',
            'use_wandb': True,
            'wandb_project': 'speech-recognition',
            'save_dir': './logs'
        },
        'checkpoint_dir': './checkpoints',
        'seed': 42
    }
    
    return OmegaConf.create(config)

def train_model(args):
    """Train a speech recognition model."""
    print("🚀 Starting Speech Recognition Training")
    print("-" * 50)
    
    # Load or create configuration
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config = create_default_config()
        
        # Override with command line arguments
        if args.model:
            config.model.name = args.model
        if args.batch_size:
            config.data.batch_size = args.batch_size
        if args.epochs:
            config.training.max_epochs = args.epochs
        if args.data_dir:
            config.data.data_dir = args.data_dir
    
    # Apply platform optimizations
    platform_info = get_platform_info()
    optimal_config = get_optimal_config(platform_info)
    
    print(f"🔧 Platform: {platform_info['platform']}")
    print(f"💾 Optimal batch size: {optimal_config['batch_size']}")
    print(f"👥 Optimal workers: {optimal_config['num_workers']}")
    print()
    
    # Create trainer
    trainer = SpeechRecognitionTrainer(config)
    
    # Start training
    trainer.train()
    
    print("✅ Training completed!")

def run_inference(args):
    """Run inference on audio files."""
    print("🎯 Running Speech Recognition Inference")
    print("-" * 50)
    
    if not args.input:
        print("❌ Error: No input file specified. Use --input path/to/audio.wav")
        return
    
    # Load model
    if not args.checkpoint:
        print("❌ Error: No checkpoint specified. Use --checkpoint path/to/model.ckpt")
        return
    
    # Create predictor
    predictor = SpeechPredictor(args.checkpoint)
    
    # Run prediction
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        result = predictor.predict_file(str(input_path))
        print(f"📝 Transcription: {result['text']}")
        print(f"🌍 Detected language: {result['language']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
    
    elif input_path.is_dir():
        # Directory of files
        audio_files = []
        for ext in ['.wav', '.flac', '.mp3', '.ogg', '.m4a']:
            audio_files.extend(input_path.glob(f"**/*{ext}"))
        
        print(f"🔍 Found {len(audio_files)} audio files")
        
        for audio_file in audio_files:
            try:
                result = predictor.predict_file(str(audio_file))
                print(f"\n📁 {audio_file.name}:")
                print(f"   Text: {result['text']}")
                print(f"   Language: {result['language']}")
                print(f"   Confidence: {result['confidence']:.3f}")
            except Exception as e:
                print(f"❌ Error processing {audio_file.name}: {e}")
    
    else:
        print(f"❌ Error: Input path {input_path} not found")

def download_datasets(args):
    """Download common speech datasets."""
    print("📥 Dataset Download Utility")
    print("-" * 50)
    
    downloader = DatasetDownloader()
    
    if args.dataset == 'common_voice':
        if not args.language:
            print("❌ Error: Language required for Common Voice. Use --language en")
            return
        
        target_dir = args.output or './data/common_voice'
        downloader.download_common_voice(args.language, target_dir)
    
    elif args.dataset == 'librispeech':
        subset = args.subset or 'train-clean-100'
        target_dir = args.output or './data/librispeech'
        downloader.download_librispeech(subset, target_dir)
    
    else:
        print("❌ Error: Unsupported dataset. Available: common_voice, librispeech")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Channel, Multi-Lingual Speech Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python launch_speech.py train --model transformer --epochs 50 --data-dir ./data/speech
  
  # Inference on single file
  python launch_speech.py predict --checkpoint model.ckpt --input audio.wav
  
  # Inference on directory
  python launch_speech.py predict --checkpoint model.ckpt --input ./audio_folder/
  
  # Download datasets
  python launch_speech.py download --dataset common_voice --language en --output ./data/
  python launch_speech.py download --dataset librispeech --subset train-clean-100
  
  # Show system information
  python launch_speech.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a speech recognition model')
    train_parser.add_argument('--config', type=str, help='Configuration file path')
    train_parser.add_argument('--model', type=str, choices=['transformer', 'conformer', 'cnn_rnn'],
                             help='Model type')
    train_parser.add_argument('--data-dir', type=str, help='Data directory')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--gpu', type=int, help='GPU device ID')
    
    # Inference command
    predict_parser = subparsers.add_parser('predict', help='Run speech recognition inference')
    predict_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    predict_parser.add_argument('--input', type=str, required=True, help='Input audio file or directory')
    predict_parser.add_argument('--output', type=str, help='Output file for results')
    predict_parser.add_argument('--language', type=str, help='Force specific language')
    predict_parser.add_argument('--channels', type=int, default=2, help='Number of audio channels')
    predict_parser.add_argument('--multispeaker', action='store_true', help='Enable multi-speaker processing')
    predict_parser.add_argument('--max-speakers', type=int, default=3, help='Maximum number of speakers')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download speech datasets')
    download_parser.add_argument('--dataset', type=str, required=True,
                                choices=['common_voice', 'librispeech', 'voxforge'],
                                help='Dataset to download')
    download_parser.add_argument('--language', type=str, help='Language code (for Common Voice)')
    download_parser.add_argument('--subset', type=str, help='Dataset subset (for LibriSpeech)')
    download_parser.add_argument('--output', type=str, help='Output directory')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system and model information')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        run_inference(args)
    elif args.command == 'download':
        download_datasets(args)
    elif args.command == 'info' or args.command is None:
        print_system_info()
        print_model_info()
        print_language_info()
        print_audio_info()
        
        if args.command is None:
            print("💡 Use --help to see available commands")
            print("💡 Use 'python launch_speech.py info' for detailed information")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()