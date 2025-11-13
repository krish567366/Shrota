#!/bin/bash
# IndicVoices Training Commands
# Complete guide to train Indian multilingual ASR using IndicVoices dataset

echo "🎙️  Indian Multilingual ASR Training with IndicVoices"
echo "=================================================="

# Prerequisites
echo "📋 Prerequisites:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Get HuggingFace token: https://huggingface.co/settings/tokens" 
echo "3. Login to HuggingFace: huggingface-cli login"
echo ""

# Phase A: Base Indian ASR Training with IndicVoices
echo "🚀 Phase A: Base Indian ASR Training"
echo "Using IndicVoices dataset (19,550 hours, 22 languages, 29K speakers)"
echo ""

# Single language training (for testing)
echo "🧪 Test with single language (Hindi):"
echo "python train_indian_asr.py \\"
echo "  --phase A \\"
echo "  --config config/indian_asr_phases.yaml \\"
echo "  --language hindi \\"
echo "  --gpus 2 \\"
echo "  --batch-size 32 \\"
echo "  --max-epochs 10"
echo ""

# Multi-language training (full Phase A)
echo "🎯 Full multilingual training (all 22 languages):"
echo "python train_indian_asr.py \\"
echo "  --phase A \\"
echo "  --config config/indian_asr_phases.yaml \\"
echo "  --all-languages \\"
echo "  --gpus 4 \\"
echo "  --batch-size 32 \\"
echo "  --max-epochs 100 \\"
echo "  --checkpoint-every 1000 \\"
echo "  --log-every 100"
echo ""

# Phase B: Cross-lingual transfer
echo "🌍 Phase B: Cross-lingual Transfer Learning:"
echo "python train_indian_asr.py \\"
echo "  --phase B \\"
echo "  --config config/indian_asr_phases.yaml \\"
echo "  --resume-from checkpoints/phase_a_best.ckpt \\"
echo "  --cross-lingual \\"
echo "  --gpus 8 \\"
echo "  --batch-size 64"
echo ""

# Phase C: Multi-channel processing
echo "🔊 Phase C: Multi-channel & Speaker Diarization:"
echo "python train_indian_asr.py \\"
echo "  --phase C \\"
echo "  --config config/indian_asr_phases.yaml \\"
echo "  --resume-from checkpoints/phase_b_best.ckpt \\"
echo "  --multi-channel \\"
echo "  --speaker-diarization \\"
echo "  --channels 8 \\"
echo "  --gpus 4"
echo ""

# Phase D: Speaker recognition
echo "👤 Phase D: Speaker Recognition:"
echo "python train_indian_asr.py \\"
echo "  --phase D \\"
echo "  --config config/indian_asr_phases.yaml \\"
echo "  --resume-from checkpoints/phase_c_best.ckpt \\"
echo "  --speaker-recognition \\"
echo "  --embedding-dim 512 \\"
echo "  --gpus 2"
echo ""

# Phase E: Optimization
echo "⚡ Phase E: Optimization & Deployment:"
echo "python train_indian_asr.py \\"
echo "  --phase E \\"
echo "  --config config/indian_asr_phases.yaml \\"
echo "  --resume-from checkpoints/phase_d_best.ckpt \\"
echo "  --optimize \\"
echo "  --export-onnx \\"
echo "  --export-tensorrt \\"
echo "  --quantize int8"
echo ""

# Data exploration commands
echo "📊 Data Exploration:"
echo "# Show dataset statistics"
echo "python load_indicvoices_example.py --statistics"
echo ""
echo "# Load specific language"
echo "python load_indicvoices_example.py --language tamil --split train"
echo ""
echo "# Demo data processing"
echo "python load_indicvoices_example.py --demo-processing"
echo ""

# Evaluation commands
echo "📈 Evaluation:"
echo "# Evaluate Phase A model"
echo "python evaluate_model.py \\"
echo "  --model checkpoints/phase_a_best.ckpt \\"
echo "  --dataset indicvoices \\"
echo "  --split test \\"
echo "  --languages hindi,bengali,tamil,telugu \\"
echo "  --metrics wer,cer,language_id"
echo ""

# Inference commands
echo "🎤 Inference:"
echo "# Real-time inference"
echo "python src/inference/predict.py \\"
echo "  --model checkpoints/phase_e_optimized.onnx \\"
echo "  --audio audio_sample.wav \\"
echo "  --language auto-detect \\"
echo "  --speaker-diarization \\"
echo "  --real-time"
echo ""

# Expected outcomes
echo "🎯 Expected Results:"
echo "Phase A: WER < 15% on Indian languages"
echo "Phase B: Cross-lingual WER < 18%"
echo "Phase C: Multi-speaker WER < 20%"
echo "Phase D: Speaker ID accuracy > 95%"
echo "Phase E: Latency < 10ms, Model < 100MB"
echo ""

# Resource requirements
echo "💻 Resource Requirements:"
echo "Phase A: 4x A100 80GB, 2-3 weeks, 5TB storage"
echo "Phase B: 8x A100 80GB, 1-2 weeks, 20TB storage"
echo "Phase C: 4x A100 80GB, 1 week, 2TB storage"
echo "Phase D: 2x A100 80GB, 3-5 days, 1TB storage"
echo "Phase E: 2x A100 80GB, 2-3 days, 500GB storage"
echo ""

echo "🚀 Ready to start training! Begin with Phase A:"
echo "python train_indian_asr.py --phase A --config config/indian_asr_phases.yaml --gpus 4"