#!/bin/bash

# Multi-Platform Training Launcher
# Comprehensive script for launching training across different platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/multiplatform_config.yaml"
LOG_FILE="${SCRIPT_DIR}/multiplatform_training.log"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Function to detect platform
detect_platform() {
    if [[ -n "${COLAB_GPU}" ]]; then
        echo "colab"
    elif [[ -n "${KAGGLE_KERNEL_RUN_TYPE}" ]]; then
        echo "kaggle"
    elif [[ -n "${RUNPOD_POD_ID}" ]]; then
        echo "runpod"
    elif [[ -n "${PAPERSPACE_NOTEBOOK_REPO_ID}" ]]; then
        echo "paperspace"
    elif [[ -d "/opt/deeplearning/" ]]; then
        echo "gcp_vertex"
    elif [[ -d "/opt/ml/" ]]; then
        echo "aws_sagemaker"
    elif [[ -d "/mnt/batch/" ]]; then
        echo "azure_batch"
    else
        echo "local"
    fi
}

# Function to setup environment based on platform
setup_environment() {
    local platform=$1
    print_status "Setting up environment for platform: $platform"
    
    case $platform in
        "colab")
            print_status "Setting up Google Colab environment"
            # Mount Google Drive if not already mounted
            if [[ ! -d "/content/drive" ]]; then
                python3 -c "
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Google Drive mounted successfully')
except Exception as e:
    print(f'Failed to mount Google Drive: {e}')
"
            fi
            
            # Install dependencies
            pip install --quiet torch torchaudio transformers datasets accelerate deepspeed
            pip install --quiet wandb tensorboard librosa soundfile
            
            # Set memory optimizations
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
            ;;
            
        "kaggle")
            print_status "Setting up Kaggle environment"
            # Kaggle usually has most packages pre-installed
            pip install --quiet deepspeed accelerate
            
            # Setup Kaggle datasets access
            mkdir -p ~/.kaggle
            if [[ -n "${KAGGLE_USERNAME}" ]] && [[ -n "${KAGGLE_KEY}" ]]; then
                echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" > ~/.kaggle/kaggle.json
                chmod 600 ~/.kaggle/kaggle.json
            fi
            ;;
            
        "runpod"|"vast_ai"|"lambda_labs")
            print_status "Setting up cloud GPU environment: $platform"
            # Install all required packages
            pip install torch torchaudio transformers datasets accelerate deepspeed
            pip install wandb tensorboard librosa soundfile
            pip install flash-attn --no-build-isolation
            
            # Optimize for high-end GPUs
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
            ;;
            
        "aws_sagemaker")
            print_status "Setting up AWS SageMaker environment"
            pip install transformers datasets accelerate deepspeed
            pip install wandb tensorboard librosa soundfile
            
            # AWS-specific optimizations
            export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
            ;;
            
        "gcp_vertex")
            print_status "Setting up GCP Vertex AI environment"
            pip install transformers datasets accelerate deepspeed
            pip install wandb tensorboard librosa soundfile
            ;;
            
        "local")
            print_status "Setting up local environment"
            # Check if virtual environment exists
            if [[ ! -d "venv" ]]; then
                print_warning "Creating virtual environment"
                python3 -m venv venv
            fi
            
            source venv/bin/activate
            pip install -r requirements.txt
            ;;
            
        *)
            print_warning "Unknown platform: $platform, using default setup"
            pip install -r requirements.txt
            ;;
    esac
}

# Function to check system resources
check_resources() {
    print_status "Checking system resources..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "Python version: $python_version"
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_status "GPU: $gpu_info MB"
    else
        print_warning "No NVIDIA GPU detected"
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        memory_gb=$(free -g | awk '/^Mem:/{print $2}')
        print_status "Available RAM: ${memory_gb}GB"
    fi
    
    # Check disk space
    if command -v df &> /dev/null; then
        disk_space=$(df -h . | awk 'NR==2{print $4}')
        print_status "Available disk space: $disk_space"
    fi
}

# Function to create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p checkpoints
    mkdir -p logs
    mkdir -p data
    mkdir -p models
    
    # Create cloud storage directories based on platform
    local platform=$1
    case $platform in
        "colab")
            mkdir -p /content/drive/MyDrive/checkpoints
            ;;
        "kaggle")
            mkdir -p /kaggle/working/checkpoints
            ;;
        "runpod"|"vast_ai"|"lambda_labs")
            mkdir -p /workspace/checkpoints
            ;;
    esac
}

# Function to validate configuration
validate_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    print_status "Configuration file validated: $CONFIG_FILE"
}

# Function to start training
start_training() {
    local phase=$1
    local resume_flag=$2
    local checkpoint_path=$3
    
    print_header "🚀 Starting Multi-Platform Training"
    print_status "Phase: $phase"
    print_status "Resume: $resume_flag"
    print_status "Checkpoint: $checkpoint_path"
    print_status "Config: $CONFIG_FILE"
    print_status "Log file: $LOG_FILE"
    
    # Build command
    local cmd="python3 multiplatform_trainer.py --config $CONFIG_FILE"
    
    if [[ -n "$phase" ]]; then
        cmd="$cmd --phase $phase"
    fi
    
    if [[ "$resume_flag" == "true" ]]; then
        cmd="$cmd --resume"
    fi
    
    if [[ -n "$checkpoint_path" ]]; then
        cmd="$cmd --checkpoint $checkpoint_path"
    fi
    
    print_status "Executing: $cmd"
    
    # Execute with logging
    $cmd 2>&1 | tee -a "$LOG_FILE"
}

# Function to show status
show_status() {
    print_header "📊 Training Status"
    python3 multiplatform_trainer.py --config "$CONFIG_FILE" --status
}

# Function to show help
show_help() {
    cat << EOF
Multi-Platform Training Launcher

Usage: $0 [OPTIONS]

OPTIONS:
    --phase PHASE          Training phase (A, B, C, D, E)
    --resume               Resume from latest checkpoint
    --checkpoint PATH      Resume from specific checkpoint
    --continue-dataset NAME Continue from specific dataset
    --status               Show training status
    --setup-only           Only setup environment, don't start training
    --help                 Show this help message

EXAMPLES:
    # Start Phase A training
    $0 --phase A
    
    # Resume training from latest checkpoint
    $0 --resume
    
    # Resume from specific checkpoint
    $0 --checkpoint ./checkpoints/phase_A_epoch_5.ckpt
    
    # Continue from specific dataset
    $0 --continue-dataset indicvoices
    
    # Show training status
    $0 --status
    
    # Setup environment only
    $0 --setup-only

PLATFORM SUPPORT:
    - Google Colab / Colab Pro
    - Kaggle Notebooks
    - RunPod
    - Vast.ai
    - Lambda Labs
    - AWS SageMaker
    - GCP Vertex AI
    - Azure ML
    - Local machine

FEATURES:
    - Automatic platform detection
    - Platform-specific optimizations
    - Cost tracking and limits
    - Automatic checkpointing
    - Cloud storage integration
    - Graceful interruption handling
    - Incremental dataset training

EOF
}

# Main execution
main() {
    # Parse command line arguments
    PHASE=""
    RESUME="false"
    CHECKPOINT=""
    CONTINUE_DATASET=""
    STATUS="false"
    SETUP_ONLY="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --phase)
                PHASE="$2"
                shift 2
                ;;
            --resume)
                RESUME="true"
                shift
                ;;
            --checkpoint)
                CHECKPOINT="$2"
                shift 2
                ;;
            --continue-dataset)
                CONTINUE_DATASET="$2"
                shift 2
                ;;
            --status)
                STATUS="true"
                shift
                ;;
            --setup-only)
                SETUP_ONLY="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Print header
    print_header "🤖 Multi-Platform Resilient Training System"
    print_header "=========================================="
    
    # Detect platform
    PLATFORM=$(detect_platform)
    print_status "Detected Platform: $PLATFORM"
    
    # Check resources
    check_resources
    
    # Setup environment
    setup_environment "$PLATFORM"
    
    # Setup directories
    setup_directories "$PLATFORM"
    
    # Validate configuration
    validate_config
    
    if [[ "$SETUP_ONLY" == "true" ]]; then
        print_status "Environment setup completed. Exiting."
        exit 0
    fi
    
    # Execute based on command
    if [[ "$STATUS" == "true" ]]; then
        show_status
    elif [[ -n "$CONTINUE_DATASET" ]]; then
        print_status "Continuing training from dataset: $CONTINUE_DATASET"
        python3 multiplatform_trainer.py --config "$CONFIG_FILE" --continue-dataset "$CONTINUE_DATASET" 2>&1 | tee -a "$LOG_FILE"
    elif [[ -n "$PHASE" ]] || [[ "$RESUME" == "true" ]]; then
        start_training "$PHASE" "$RESUME" "$CHECKPOINT"
    else
        print_warning "No action specified. Showing status..."
        show_status
        echo ""
        print_status "Use --help for usage information"
    fi
}

# Trap interruptions for graceful shutdown
trap 'print_warning "Training interrupted. Checkpoints should be saved."; exit 130' INT TERM

# Execute main function
main "$@"