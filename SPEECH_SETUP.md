# 🎤 Speech Recognition System - Quick Setup

This is a clean speech-to-text system with multi-lingual support and advanced architectures.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd multilingual-speech-recognition
pip install -r requirements.txt
```

### 2. Download Sample Audio Data
```bash
# Create data directories
mkdir -p data/audio/{train,test,val}

# Example: Download Common Voice data (optional)
# wget -O data/common_voice.tar.gz "https://commonvoice.mozilla.org/datasets"
```

### 3. Train a Speech Model
```bash
# Train lightweight CNN-RNN model
python launch_training.py --config config/hybrid_config.yaml

# Train transformer model
python launch_training.py --config config/transformer_config.yaml

# Train with specific settings
python launch_training.py --model lightweight --epochs 20 --batch-size 16
```

### 4. Run Inference
```bash
# Ultra-fast inference
python launch_ultra_fast.py --model lightweight --benchmark

# Real-time speech recognition
python launch_ultra_fast.py --real-time --model transformer

# Process audio file
python -m src.inference.predict --audio test_audio.wav --model-path models/best_model.ckpt
```

## 🎯 Available Models

1. **Lightweight CNN-RNN**: Fast, efficient model for edge deployment
2. **Multi-lingual Transformer**: Advanced transformer for high accuracy
3. **Custom Conformer**: Combines CNN and attention for optimal performance

## 🔧 Configuration

Edit the YAML files in `config/` folder:
- `hybrid_config.yaml` - CNN-RNN model settings
- `transformer_config.yaml` - Transformer model settings  
- `training_config.yaml` - General training parameters
- `data_config.yaml` - Dataset configuration

## 📊 Supported Features

- ✅ Multi-lingual speech recognition (100+ languages)
- ✅ Multi-channel audio processing
- ✅ Real-time streaming inference
- ✅ Cloud platform compatibility
- ✅ Custom model architectures
- ✅ GPU optimization (A100, V100, etc.)

## 🎤 Usage Examples

### Python API
```python
from src.inference.predict import create_inference_system

# Create speech recognition system
speech_system = create_inference_system(
    config_path='config/transformer_config.yaml',
    model_type='transformer'
)

# Benchmark performance
results = speech_system.benchmark(duration_seconds=30)
print(f"Latency: {results['avg_latency_ms']:.2f}ms")
```

### Command Line
```bash
# Quick transcription
python launch_ultra_fast.py --audio meeting.wav --model transformer --languages en,es,fr

# Streaming from microphone
python launch_ultra_fast.py --real-time --model lightweight --language en
```

## 📂 Project Structure

```
multilingual-speech-recognition/
├── config/              # Configuration files
├── src/
│   ├── models/         # Speech model architectures
│   ├── data/           # Audio processing & data loaders
│   ├── training/       # Training pipeline
│   ├── inference/      # Speech recognition inference
│   └── utils/          # Helper utilities
├── data/               # Audio datasets
├── models/             # Trained model checkpoints
└── logs/               # Training logs
```

Ready to recognize speech! 🎉