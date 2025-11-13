# Google Colab Set# 2. Clone repository
!git clone https://github.com/krish567366/Shrota.git
%cd Shrota

# 3. Install dependencieside for Multi-Platform Training

## 🚀 Quick Start for Google Colab

### Step 1: Setup in Colab Notebook

```python
# 1. Check GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# 2. Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# 3. Clone the repository
!git clone https://github.com/your-username/multilingual-speech-recognition.git
%cd multilingual-speech-recognition

# 4. Install dependencies
!pip install torch torchaudio transformers datasets accelerate deepspeed
!pip install wandb tensorboard librosa soundfile pyyaml
!pip install flash-attn --no-build-isolation

# 5. Make launch script executable
!chmod +x launch_multiplatform_training.sh

# 6. Setup Google Drive checkpoint directory
!mkdir -p "/content/drive/MyDrive/ASR_Checkpoints"

print("✅ Setup completed! Ready to start training.")
```

### Step 2: Configure for Colab

```python
# Update config for Colab-specific paths
import yaml

# Load and modify config
with open('config/multiplatform_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set Colab-specific paths
config['checkpoint']['checkpoint_dir'] = '/content/drive/MyDrive/ASR_Checkpoints'
config['checkpoint']['cloud_storage']['type'] = 'none'  # Use Google Drive instead
config['platform']['cost_limits']['daily_limit'] = 0.0  # Free tier
config['platform']['cost_limits']['session_limit'] = 0.0

# Save updated config
with open('config/multiplatform_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ Configuration updated for Colab")
```

### Step 3: Start Training

```python
# Option 1: Start Phase A training
!./launch_multiplatform_training.sh --phase A

# Option 2: Check status first
!./launch_multiplatform_training.sh --status

# Option 3: Resume from checkpoint (if interrupted)
!./launch_multiplatform_training.sh --resume
```

## 📱 Complete Colab Notebook Template

Here's a complete notebook you can copy-paste:

```python
# =============================================================================
# Multi-Platform Indian ASR Training - Google Colab Setup
# =============================================================================

print("🚀 Setting up Multi-Platform Indian ASR Training System")
print("=" * 60)

# Check GPU
import torch
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ GPU Memory: {gpu_memory:.1f}GB")
    
    # Determine Colab tier
    if "T4" in gpu_name:
        colab_tier = "Free" if gpu_memory < 16 else "Pro"
    elif "V100" in gpu_name:
        colab_tier = "Pro"
    else:
        colab_tier = "Unknown"
    
    print(f"✅ Detected: Google Colab {colab_tier}")
else:
    print("❌ No GPU detected!")

print("\n" + "=" * 60)

# Mount Google Drive
print("📁 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted")

# Clone repository
print("\n📥 Cloning repository...")
!git clone https://github.com/your-username/multilingual-speech-recognition.git
%cd multilingual-speech-recognition
print("✅ Repository cloned")

# Install dependencies
print("\n📦 Installing dependencies...")
!pip install -q torch torchaudio transformers datasets accelerate deepspeed
!pip install -q wandb tensorboard librosa soundfile pyyaml psutil requests
!pip install -q flash-attn --no-build-isolation
print("✅ Dependencies installed")

# Setup directories
print("\n📂 Setting up directories...")
!mkdir -p "/content/drive/MyDrive/ASR_Checkpoints"
!mkdir -p "/content/drive/MyDrive/ASR_Logs"
!mkdir -p logs data models
print("✅ Directories created")

# Configure for Colab
print("\n⚙️  Configuring for Colab...")
import yaml
import os

# Load config
with open('config/multiplatform_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Colab-specific settings
config['checkpoint']['checkpoint_dir'] = '/content/drive/MyDrive/ASR_Checkpoints'
config['checkpoint']['auto_save_interval'] = 900  # 15 minutes for Colab
config['checkpoint']['cloud_storage']['type'] = 'none'  # Use Google Drive

# Platform settings for Colab
config['platform']['cost_limits']['daily_limit'] = 0.0
config['platform']['cost_limits']['session_limit'] = 0.0

# Training optimizations for Colab
if "T4" in gpu_name:
    # T4 GPU settings
    config['datasets']['base_training_config']['batch_size'] = 12
    config['training']['deepspeed']['config']['zero_optimization']['stage'] = 2
elif "V100" in gpu_name:
    # V100 GPU settings (Colab Pro)
    config['datasets']['base_training_config']['batch_size'] = 16
    config['training']['deepspeed']['config']['zero_optimization']['stage'] = 3

# Save updated config
with open('config/multiplatform_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ Configuration updated for Colab")

# Make script executable
!chmod +x launch_multiplatform_training.sh

print("\n" + "=" * 60)
print("🎉 Setup Complete! Ready to start training.")
print("=" * 60)

# Show training options
print("\n🚀 Training Options:")
print("1. Start Phase A: !./launch_multiplatform_training.sh --phase A")
print("2. Check Status:  !./launch_multiplatform_training.sh --status")
print("3. Resume:        !./launch_multiplatform_training.sh --resume")
print("4. Setup Only:    !./launch_multiplatform_training.sh --setup-only")
```

## 🎯 Colab-Specific Training Commands

### Start Training
```python
# Start Phase A (Foundation datasets)
!./launch_multiplatform_training.sh --phase A
```

### Monitor Progress
```python
# Check training status
!./launch_multiplatform_training.sh --status

# Watch logs in real-time
!tail -f multiplatform_training.log

# Check GPU usage
!nvidia-smi
```

### Handle Interruptions
```python
# If Colab disconnects, resume with:
!./launch_multiplatform_training.sh --resume

# Or continue from specific dataset:
!./launch_multiplatform_training.sh --continue-dataset indicvoices
```

## ⚠️ Colab Limitations & Solutions

### Time Limits
- **Free Colab**: ~12 hours max
- **Colab Pro**: ~24 hours max
- **Solution**: Auto-checkpoint every 15 minutes, resume seamlessly

### Memory Limits
- **Free**: T4 with 15GB GPU memory
- **Pro**: V100 with 16GB or T4 with 15GB
- **Solution**: Optimized batch sizes and DeepSpeed

### Disconnections
- **Problem**: Browser/network disconnections
- **Solution**: All progress saved to Google Drive, auto-resume

## 🔧 Colab Optimization Settings

### For Free Colab (T4)
```python
# Optimized settings for T4
config_updates = {
    'batch_size': 12,
    'gradient_accumulation_steps': 6,
    'mixed_precision': 'fp16',
    'deepspeed_stage': 2,
    'checkpoint_interval': 900  # 15 minutes
}
```

### For Colab Pro (V100)
```python
# Optimized settings for V100
config_updates = {
    'batch_size': 16,
    'gradient_accumulation_steps': 4,
    'mixed_precision': 'bf16',
    'deepspeed_stage': 3,
    'checkpoint_interval': 600  # 10 minutes
}
```

## 📊 Expected Performance on Colab

### Free Colab (T4)
- **Phase A**: ~15-20 minutes per epoch
- **Total Phase A**: ~2-3 hours
- **Cost**: Free ✅
- **Sessions needed**: 2-3 (due to time limits)

### Colab Pro (V100)
- **Phase A**: ~10-15 minutes per epoch  
- **Total Phase A**: ~1.5-2 hours
- **Cost**: $10/month
- **Sessions needed**: 1-2

## 🚨 Troubleshooting Colab Issues

### Common Problems

1. **"Runtime disconnected"**
```python
# Check if training was running
!ps aux | grep python

# Resume from checkpoint
!./launch_multiplatform_training.sh --resume
```

2. **"Out of memory"**
```python
# Reduce batch size in config
!python3 -c "
import yaml
with open('config/multiplatform_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['datasets']['base_training_config']['batch_size'] = 8
with open('config/multiplatform_config.yaml', 'w') as f:
    yaml.dump(config, f)
print('Batch size reduced to 8')
"
```

3. **"No space left"**
```python
# Clean up temporary files
!rm -rf /tmp/*
!pip cache purge

# Check space
!df -h
```

4. **"Package conflicts"**
```python
# Reset environment
!pip install --force-reinstall torch torchaudio
!pip install --upgrade transformers datasets
```

## 🔄 Multi-Session Training Workflow

### Session 1 (Start)
```python
# Setup and start Phase A
!./launch_multiplatform_training.sh --phase A
# Training runs for ~12 hours, auto-saves to Drive
```

### Session 2 (Resume)
```python
# Reconnect to new runtime
%cd multilingual-speech-recognition
!./launch_multiplatform_training.sh --resume
# Continues exactly where left off
```

### Session 3 (Next Phase)
```python
# Start Phase B after A completes
!./launch_multiplatform_training.sh --phase B
```

## 📱 Mobile Monitoring

You can monitor training from your phone:

```python
# Setup Weights & Biases for mobile monitoring
import wandb
wandb.login()  # Enter your W&B key

# Training metrics will be available on wandb.ai mobile app
```

## 💡 Pro Tips for Colab

1. **Keep browser tab active** - Colab may disconnect if inactive too long
2. **Use Colab Pro** for longer sessions and better GPUs
3. **Monitor from mobile** using W&B app
4. **Start training during off-peak hours** for better stability
5. **Always check Google Drive space** before starting

## 🎉 Complete Training on Colab

With this setup, you can train your Indian multilingual ASR model completely on Google Colab! The system handles all the complexity of checkpointing, resuming, and optimization automatically.

**Total Expected Time**: 15-25 hours across all phases
**Total Cost**: Free (Colab) or $10/month (Colab Pro)
**Result**: Production-ready multilingual ASR model! 🚀