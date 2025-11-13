#!/usr/bin/env python3
"""
IndicVoices Dataset Loading Example

This script demonstrates how to load and use the IndicVoices dataset 
for Indian multilingual ASR training.

Usage:
    python load_indicvoices_example.py --language hindi --split train
    python load_indicvoices_example.py --all-languages --sample-size 1000
    python load_indicvoices_example.py --statistics
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_indicvoices_sample():
    """Basic example of loading IndicVoices dataset."""
    try:
        from datasets import load_dataset
        
        logger.info("🎯 Loading IndicVoices sample (Assamese, valid split)...")
        
        # Load a small sample for demonstration
        dataset = load_dataset(
            "ai4bharat/IndicVoices", 
            "assamese", 
            split="valid"
        )
        
        logger.info(f"📊 Dataset info:")
        logger.info(f"   - Split: valid")
        logger.info(f"   - Number of samples: {len(dataset)}")
        logger.info(f"   - Features: {dataset.features}")
        
        # Show first sample
        sample = dataset[0]
        logger.info(f"\n📝 Sample data:")
        logger.info(f"   - Text: {sample['text'][:100]}...")
        logger.info(f"   - Duration: {sample['duration']:.2f}s")
        logger.info(f"   - Language: {sample['lang']}")
        logger.info(f"   - Speaker ID: {sample['speaker_id']}")
        logger.info(f"   - Scenario: {sample['scenario']}")
        logger.info(f"   - Gender: {sample['gender']}")
        logger.info(f"   - District: {sample['district']}")
        logger.info(f"   - State: {sample['state']}")
        
        # Audio information
        audio = sample['audio']
        logger.info(f"   - Audio sampling rate: {audio['sampling_rate']}")
        logger.info(f"   - Audio array shape: {len(audio['array'])}")
        
        return dataset
        
    except ImportError:
        logger.error("❌ datasets library not installed. Run: pip install datasets")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {str(e)}")
        return None

def load_multiple_languages(languages: List[str], split: str = "valid"):
    """Load multiple languages from IndicVoices."""
    try:
        from datasets import load_dataset
        
        datasets = {}
        total_samples = 0
        
        logger.info(f"🌍 Loading {len(languages)} languages from IndicVoices...")
        
        for lang in languages:
            try:
                dataset = load_dataset(
                    "ai4bharat/IndicVoices", 
                    lang, 
                    split=split,
                    streaming=True  # Use streaming for large datasets
                )
                
                # Count samples (limited for streaming)
                sample_count = 0
                for _ in dataset:
                    sample_count += 1
                    if sample_count >= 10:  # Limit for demo
                        break
                
                datasets[lang] = dataset
                total_samples += sample_count
                logger.info(f"✅ {lang}: {sample_count}+ samples")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {lang}: {str(e)}")
                continue
        
        logger.info(f"🎉 Successfully loaded {len(datasets)} languages")
        logger.info(f"📊 Total samples (limited): {total_samples}")
        
        return datasets
        
    except ImportError:
        logger.error("❌ datasets library not installed. Run: pip install datasets")
        return None

def get_dataset_statistics():
    """Get comprehensive statistics about IndicVoices dataset."""
    
    # All 22 Indian languages available in IndicVoices
    languages = [
        "assamese", "bengali", "bodo", "dogri", "gujarati", "hindi", 
        "kannada", "kashmiri", "konkani", "maithili", "malayalam", 
        "manipuri", "marathi", "nepali", "odia", "punjabi", 
        "sanskrit", "santali", "sindhi", "tamil", "telugu", "urdu"
    ]
    
    logger.info("📈 IndicVoices Dataset Statistics:")
    logger.info("=" * 50)
    logger.info(f"🌍 Total Languages: {len(languages)}")
    logger.info(f"🎙️  Total Hours: 19,550 (9,200 transcribed)")
    logger.info(f"👥 Total Speakers: ~29,000")
    logger.info(f"🏛️  Districts Covered: 400+")
    logger.info(f"📋 Data Types: Read (8%), Extempore (76%), Conversational (15%)")
    
    logger.info(f"\n📚 Languages Covered:")
    for i, lang in enumerate(languages, 1):
        logger.info(f"   {i:2d}. {lang.capitalize()}")
    
    logger.info(f"\n🎯 Key Features:")
    logger.info(f"   • Natural spontaneous speech (76% extempore)")
    logger.info(f"   • Geographic diversity (400+ districts)")
    logger.info(f"   • Demographic diversity (29K speakers)")
    logger.info(f"   • Constitutional language coverage (22 languages)")
    logger.info(f"   • Quality annotations and metadata")
    
    logger.info(f"\n📊 Data Distribution by Type:")
    logger.info(f"   • Read Speech: 8% (~1,560 hours)")
    logger.info(f"   • Extempore Speech: 76% (~14,860 hours)")  
    logger.info(f"   • Conversational: 15% (~2,930 hours)")
    
    return {
        'languages': languages,
        'total_hours': 19550,
        'transcribed_hours': 9200,
        'speakers': 29000,
        'districts': 400,
        'data_types': ['read', 'extempore', 'conversational'],
        'license': 'CC BY 4.0'
    }

def demonstrate_data_processing():
    """Demonstrate how to process IndicVoices data for training."""
    try:
        from datasets import load_dataset
        import numpy as np
        
        logger.info("🔧 Demonstrating data processing pipeline...")
        
        # Load a small sample
        dataset = load_dataset(
            "ai4bharat/IndicVoices", 
            "hindi", 
            split="valid"
        )
        
        # Process first few samples
        processed_samples = []
        
        for i, sample in enumerate(dataset[:5]):  # Process first 5 samples
            # Extract audio data
            audio_array = np.array(sample['audio']['array'])
            sample_rate = sample['audio']['sampling_rate']
            
            # Extract text and metadata
            text = sample.get('normalized', sample.get('text', ''))
            
            processed_sample = {
                'audio_features': {
                    'duration': sample['duration'],
                    'sample_rate': sample_rate,
                    'num_samples': len(audio_array),
                    'audio_shape': audio_array.shape
                },
                'text_features': {
                    'text': text,
                    'text_length': len(text),
                    'language': sample['lang']
                },
                'speaker_features': {
                    'speaker_id': sample['speaker_id'],
                    'gender': sample['gender'],
                    'age_group': sample['age_group']
                },
                'location_features': {
                    'district': sample['district'],
                    'state': sample['state']
                },
                'domain_features': {
                    'scenario': sample['scenario'],  # read/extempore/conversational
                    'task_name': sample.get('task_name', 'unknown')
                }
            }
            
            processed_samples.append(processed_sample)
            
            logger.info(f"Sample {i+1}:")
            logger.info(f"   Duration: {processed_sample['audio_features']['duration']:.2f}s")
            logger.info(f"   Text: {processed_sample['text_features']['text'][:50]}...")
            logger.info(f"   Speaker: {processed_sample['speaker_features']['speaker_id']}")
            logger.info(f"   Domain: {processed_sample['domain_features']['scenario']}")
            logger.info(f"   Location: {processed_sample['location_features']['district']}, {processed_sample['location_features']['state']}")
        
        logger.info(f"✅ Processed {len(processed_samples)} samples successfully")
        
        return processed_samples
        
    except ImportError:
        logger.error("❌ Required libraries not installed. Run: pip install datasets numpy")
        return None
    except Exception as e:
        logger.error(f"❌ Error processing data: {str(e)}")
        return None

def main():
    """Main function to demonstrate IndicVoices usage."""
    parser = argparse.ArgumentParser(description="IndicVoices Dataset Loading Examples")
    parser.add_argument("--language", type=str, help="Specific language to load")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"], help="Dataset split")
    parser.add_argument("--all-languages", action="store_true", help="Load all available languages")
    parser.add_argument("--statistics", action="store_true", help="Show dataset statistics")
    parser.add_argument("--demo-processing", action="store_true", help="Demonstrate data processing")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of samples to load per language")
    
    args = parser.parse_args()
    
    logger.info("🎙️  IndicVoices Dataset Loader")
    logger.info("=" * 40)
    
    if args.statistics:
        stats = get_dataset_statistics()
        return
    
    if args.demo_processing:
        processed = demonstrate_data_processing()
        return
    
    if args.language:
        # Load specific language
        logger.info(f"Loading {args.language} dataset...")
        dataset = load_dataset(
            "ai4bharat/IndicVoices", 
            args.language, 
            split=args.split
        )
        logger.info(f"✅ Loaded {len(dataset)} samples")
        
    elif args.all_languages:
        # Load all languages
        languages = [
            "assamese", "bengali", "bodo", "dogri", "gujarati", "hindi", 
            "kannada", "kashmiri", "konkani", "maithili", "malayalam", 
            "manipuri", "marathi", "nepali", "odia", "punjabi", 
            "sanskrit", "santali", "sindhi", "tamil", "telugu", "urdu"
        ]
        datasets = load_multiple_languages(languages, args.split)
        
    else:
        # Default: load sample
        dataset = load_indicvoices_sample()
    
    logger.info("\n🎯 Next Steps:")
    logger.info("   1. Install: pip install datasets huggingface-hub")
    logger.info("   2. Get HF token: https://huggingface.co/settings/tokens")
    logger.info("   3. Login: huggingface-cli login")
    logger.info("   4. Start training with Phase A configuration!")

if __name__ == "__main__":
    main()