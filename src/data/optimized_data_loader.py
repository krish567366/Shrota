#!/usr/bin/env python3
"""
High-Performance Data Loading Pipeline for Indian ASR

Implements advanced optimizations:
- Multi-processing with persistent workers
- Dynamic batching by tokens
- Audio feature caching
- Streaming with prefetching
- Memory-efficient preprocessing

Usage:
    from src.data.optimized_data_loader import OptimizedIndianASRDataLoader
    
    loader = OptimizedIndianASRDataLoader(config)
    dataloader = loader.create_dataloader()
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle
import hashlib
from collections import defaultdict
import random
from datasets import load_dataset
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class OptimizedAudioProcessor:
    """Ultra-fast audio preprocessing with caching."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.frame_length = config.get('frame_length', 25)  # ms
        self.frame_shift = config.get('frame_shift', 10)   # ms
        self.use_cache = config.get('precompute_features', True)
        self.cache_dir = Path(config.get('cache_dir', './cache/audio_features'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-compute mel filter bank for efficiency
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(self.sample_rate * self.frame_length / 1000),
            hop_length=int(self.sample_rate * self.frame_shift / 1000),
            n_mels=self.n_mels,
            power=2.0,
            normalized=True
        )
        
    def process_audio(self, audio_data: np.ndarray, audio_id: str = None) -> torch.Tensor:
        """Process audio with caching for maximum speed."""
        
        # Check cache first
        if self.use_cache and audio_id:
            cache_path = self.cache_dir / f"{self._hash_audio_id(audio_id)}.pt"
            if cache_path.exists():
                try:
                    return torch.load(cache_path, map_location='cpu')
                except:
                    pass  # Cache corrupted, recompute
        
        # Convert to tensor and normalize
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data.float()
            
        # Ensure mono
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)
            
        # Normalize audio (optional, can be skipped for speed)
        if self.config.get('normalize_audio', True):
            audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)
        
        # Extract mel spectrogram features
        mel_spec = self.mel_transform(audio_tensor)
        
        # Log mel spectrogram
        mel_spec = torch.log(mel_spec + 1e-8)
        
        # Transpose to (time, freq) format
        features = mel_spec.transpose(0, 1)  # (freq, time) -> (time, freq)
        
        # Cache for future use
        if self.use_cache and audio_id:
            cache_path = self.cache_dir / f"{self._hash_audio_id(audio_id)}.pt"
            try:
                torch.save(features, cache_path)
            except:
                pass  # Cache write failed, continue
                
        return features
    
    def _hash_audio_id(self, audio_id: str) -> str:
        """Create hash for audio ID for caching."""
        return hashlib.md5(audio_id.encode()).hexdigest()

class StreamingIndianASRDataset(IterableDataset):
    """Memory-efficient streaming dataset for large-scale training."""
    
    def __init__(self, dataset_config: Dict, split: str = "train"):
        self.config = dataset_config
        self.split = split
        self.audio_processor = OptimizedAudioProcessor(dataset_config.get('audio_config', {}))
        self.curriculum_config = dataset_config.get('curriculum_learning', {})
        self.current_stage = dataset_config.get('current_curriculum_stage', 3)
        
        # Language list for IndicVoices
        self.languages = [
            "assamese", "bengali", "bodo", "dogri", "gujarati", "hindi", 
            "kannada", "kashmiri", "konkani", "maithili", "malayalam", 
            "manipuri", "marathi", "nepali", "odia", "punjabi", 
            "sanskrit", "santali", "sindhi", "tamil", "telugu", "urdu"
        ]
        
        # Load datasets for all languages
        self.datasets = {}
        self._load_streaming_datasets()
        
    def _load_streaming_datasets(self):
        """Load streaming datasets for all languages."""
        logger.info(f"Loading streaming datasets for {len(self.languages)} languages...")
        
        for lang in self.languages:
            try:
                dataset = load_dataset(
                    "ai4bharat/IndicVoices", 
                    lang, 
                    split=self.split,
                    streaming=True
                )
                self.datasets[lang] = dataset
                logger.info(f"✅ Loaded streaming dataset for {lang}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {lang}: {str(e)}")
                
        logger.info(f"🎉 Loaded {len(self.datasets)} language datasets")
    
    def _should_include_sample(self, sample: Dict) -> bool:
        """Apply curriculum learning filter."""
        if not self.curriculum_config.get('enabled', False):
            return True
            
        stage_config = self.curriculum_config.get(f'stage_{self.current_stage}', {})
        data_filter = stage_config.get('data_filter', {})
        
        # Duration filter
        duration = sample.get('duration', 0)
        if duration < data_filter.get('min_duration', 0):
            return False
        if duration > data_filter.get('max_duration', float('inf')):
            return False
            
        # Scenario filter (read/extempore/conversational)
        scenario = sample.get('scenario', 'unknown')
        allowed_scenarios = data_filter.get('scenarios', ['read', 'extempore', 'conversational'])
        if scenario not in allowed_scenarios:
            return False
            
        return True
    
    def __iter__(self):
        """Iterate through samples with curriculum learning."""
        
        # Create iterators for all languages
        language_iterators = {}
        for lang, dataset in self.datasets.items():
            language_iterators[lang] = iter(dataset)
        
        # Round-robin sampling across languages
        active_languages = list(language_iterators.keys())
        lang_index = 0
        samples_yielded = 0
        max_samples_per_epoch = self.config.get('max_samples_per_epoch', 100000)
        
        while active_languages and samples_yielded < max_samples_per_epoch:
            current_lang = active_languages[lang_index % len(active_languages)]
            
            try:
                # Get next sample from current language
                sample = next(language_iterators[current_lang])
                
                # Apply curriculum learning filter
                if not self._should_include_sample(sample):
                    lang_index += 1
                    continue
                
                # Process the sample
                processed_sample = self._process_sample(sample, current_lang)
                if processed_sample is not None:
                    yield processed_sample
                    samples_yielded += 1
                
                lang_index += 1
                
            except StopIteration:
                # Remove exhausted language
                active_languages.remove(current_lang)
                del language_iterators[current_lang]
                logger.info(f"Finished {current_lang} dataset")
                
            except Exception as e:
                logger.warning(f"Error processing sample from {current_lang}: {str(e)}")
                lang_index += 1
                continue
                
        logger.info(f"Epoch complete: yielded {samples_yielded} samples")
    
    def _process_sample(self, sample: Dict, language: str) -> Optional[Dict]:
        """Process individual sample efficiently."""
        try:
            # Extract audio
            audio_data = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']
            
            # Resample if needed (fast)
            if sample_rate != self.audio_processor.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, 
                    self.audio_processor.sample_rate
                )
                audio_data = resampler(torch.from_numpy(audio_data)).numpy()
            
            # Create unique audio ID for caching
            audio_id = f"{language}_{sample.get('speaker_id', 'unknown')}_{sample.get('duration', 0):.3f}"
            
            # Process audio features
            features = self.audio_processor.process_audio(audio_data, audio_id)
            
            # Extract and clean text
            text = sample.get('normalized', sample.get('text', ''))
            if not text or len(text.strip()) == 0:
                return None
                
            return {
                'features': features,
                'text': text.strip(),
                'language': language,
                'duration': sample.get('duration', 0),
                'speaker_id': sample.get('speaker_id', 'unknown'),
                'scenario': sample.get('scenario', 'unknown'),
                'metadata': {
                    'gender': sample.get('gender', 'unknown'),
                    'age_group': sample.get('age_group', 'unknown'),
                    'district': sample.get('district', 'unknown'),
                    'state': sample.get('state', 'unknown')
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to process sample: {str(e)}")
            return None

class DynamicBatchSampler:
    """Dynamic batching by tokens for optimal GPU utilization."""
    
    def __init__(self, max_tokens: int = 400000, max_sentences: int = 64):
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        
    def create_batches(self, samples: List[Dict]) -> List[List[Dict]]:
        """Group samples into efficient batches."""
        
        # Sort by sequence length for efficiency
        samples.sort(key=lambda x: x['features'].shape[0])
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for sample in samples:
            sample_tokens = sample['features'].shape[0]  # sequence length
            
            # Check if adding this sample would exceed limits
            if (len(current_batch) >= self.max_sentences or 
                current_tokens + sample_tokens > self.max_tokens):
                
                if current_batch:  # Don't add empty batches
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(sample)
            current_tokens += sample_tokens
        
        # Add remaining batch
        if current_batch:
            batches.append(current_batch)
            
        return batches

def collate_fn_optimized(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Optimized collate function with padding and attention masks."""
    
    # Extract features and texts
    features = [sample['features'] for sample in batch]
    texts = [sample['text'] for sample in batch]
    languages = [sample['language'] for sample in batch]
    durations = [sample['duration'] for sample in batch]
    
    # Pad sequences efficiently
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # Create attention masks (1 for real tokens, 0 for padding)
    feature_lengths = torch.tensor([f.shape[0] for f in features])
    max_length = features_padded.shape[1]
    attention_mask = torch.arange(max_length)[None, :] < feature_lengths[:, None]
    
    # Create language IDs (for multilingual models)
    language_to_id = {
        "assamese": 0, "bengali": 1, "bodo": 2, "dogri": 3, "gujarati": 4, 
        "hindi": 5, "kannada": 6, "kashmiri": 7, "konkani": 8, "maithili": 9,
        "malayalam": 10, "manipuri": 11, "marathi": 12, "nepali": 13, 
        "odia": 14, "punjabi": 15, "sanskrit": 16, "santali": 17, 
        "sindhi": 18, "tamil": 19, "telugu": 20, "urdu": 21
    }
    language_ids = torch.tensor([language_to_id.get(lang, 0) for lang in languages])
    
    return {
        'input_features': features_padded,
        'attention_mask': attention_mask,
        'feature_lengths': feature_lengths,
        'texts': texts,
        'language_ids': language_ids,
        'durations': torch.tensor(durations),
        'languages': languages
    }

class OptimizedIndianASRDataLoader:
    """High-performance data loader with all optimizations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.optimization_config = config.get('optimization', {})
        
    def create_dataloader(self, split: str = "train") -> DataLoader:
        """Create optimized DataLoader with all performance enhancements."""
        
        # Create streaming dataset
        dataset = StreamingIndianASRDataset(self.config, split)
        
        # Optimization settings
        data_opt = self.optimization_config.get('data_optimization', {})
        
        # Create DataLoader with optimizations
        dataloader = DataLoader(
            dataset,
            batch_size=None,  # We'll handle batching manually
            num_workers=data_opt.get('num_workers', min(16, mp.cpu_count())),
            pin_memory=data_opt.get('pin_memory', True),
            persistent_workers=data_opt.get('persistent_workers', True),
            prefetch_factor=data_opt.get('prefetch_factor', 4),
            collate_fn=collate_fn_optimized
        )
        
        return dataloader
    
    def create_dynamic_dataloader(self, split: str = "train") -> DataLoader:
        """Create DataLoader with dynamic batching."""
        
        dataset = StreamingIndianASRDataset(self.config, split)
        
        # Dynamic batching parameters
        dynamic_config = self.optimization_config.get('data_optimization', {}).get('dynamic_batching', {})
        max_tokens = dynamic_config.get('max_tokens', 400000)
        max_sentences = dynamic_config.get('max_sentences', 64)
        
        # Custom DataLoader with dynamic batching
        class DynamicDataLoader:
            def __init__(self, dataset, max_tokens, max_sentences, **kwargs):
                self.dataset = dataset
                self.sampler = DynamicBatchSampler(max_tokens, max_sentences)
                self.kwargs = kwargs
                
            def __iter__(self):
                # Collect samples and create dynamic batches
                samples = []
                sample_buffer_size = 1000  # Collect this many samples before batching
                
                for sample in self.dataset:
                    samples.append(sample)
                    
                    if len(samples) >= sample_buffer_size:
                        # Create batches from collected samples
                        batches = self.sampler.create_batches(samples)
                        for batch in batches:
                            yield collate_fn_optimized(batch)
                        samples = []
                
                # Process remaining samples
                if samples:
                    batches = self.sampler.create_batches(samples)
                    for batch in batches:
                        yield collate_fn_optimized(batch)
        
        return DynamicDataLoader(
            dataset, 
            max_tokens, 
            max_sentences,
            num_workers=self.optimization_config.get('data_optimization', {}).get('num_workers', 8),
            pin_memory=True,
            persistent_workers=True
        )

# Performance monitoring
class DataLoaderProfiler:
    """Profile data loading performance."""
    
    def __init__(self):
        self.stats = defaultdict(list)
        
    def profile_dataloader(self, dataloader, num_batches: int = 100):
        """Profile dataloader performance."""
        logger.info(f"Profiling dataloader for {num_batches} batches...")
        
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            batch_start = time.time()
            
            # Simulate processing time
            if torch.cuda.is_available():
                batch['input_features'] = batch['input_features'].cuda(non_blocking=True)
                torch.cuda.synchronize()
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            if i % 20 == 0:
                logger.info(f"Processed batch {i+1}/{num_batches}")
        
        total_time = time.time() - start_time
        avg_batch_time = np.mean(batch_times)
        throughput = num_batches / total_time
        
        logger.info(f"📊 DataLoader Performance:")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Avg batch time: {avg_batch_time:.4f}s")
        logger.info(f"   Throughput: {throughput:.2f} batches/sec")
        logger.info(f"   Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "   CPU only")
        
        return {
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'throughput': throughput,
            'batch_times': batch_times
        }

if __name__ == "__main__":
    # Example usage and benchmarking
    config = {
        'audio_config': {
            'sample_rate': 16000,
            'n_mels': 80,
            'precompute_features': True,
            'normalize_audio': False  # Skip for speed
        },
        'optimization': {
            'data_optimization': {
                'num_workers': 16,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 4,
                'dynamic_batching': {
                    'enabled': True,
                    'max_tokens': 400000,
                    'max_sentences': 64
                }
            }
        },
        'curriculum_learning': {
            'enabled': True,
            'current_curriculum_stage': 1
        },
        'max_samples_per_epoch': 10000  # Limit for demo
    }
    
    # Create and profile optimized data loader
    loader = OptimizedIndianASRDataLoader(config)
    dataloader = loader.create_dataloader("valid")
    
    # Profile performance
    profiler = DataLoaderProfiler()
    stats = profiler.profile_dataloader(dataloader, num_batches=50)
    
    print(f"🚀 Optimized DataLoader ready with {stats['throughput']:.2f} batches/sec!")