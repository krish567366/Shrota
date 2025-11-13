#!/usr/bin/env python3
"""
Hugging Face Dataset Loader for Indian Speech Recognition

Features:
- Automatic download from Hugging Face Hub
- Streaming for large datasets
- Language filtering for Indian languages
- Preprocessing pipeline integration
- Caching for faster subsequent loads

Supported datasets:
- ai4bharat/IndicVoices (18K hours, 10 languages)
- google/fleurs (22 Indian languages)
- mozilla-foundation/common_voice_13_0 (Multiple Indian languages)
- openslr datasets (Hindi, Bengali, etc.)
"""

import os
import logging
from typing import Dict, List, Optional, Union, Iterator
from pathlib import Path
import yaml

try:
    from datasets import load_dataset, Dataset, DatasetDict, IterableDataset
    from huggingface_hub import login, HfFolder
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logging.warning("Hugging Face datasets library not available. Install with: pip install datasets")

logger = logging.getLogger(__name__)

class HuggingFaceDatasetLoader:
    """Load and manage Indian speech datasets from Hugging Face."""
    
    # Available Indian speech datasets on Hugging Face
    AVAILABLE_DATASETS = {
        'ai4bharat/IndicVoices': {
            'languages': ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'or', 'pa'],
            'hours': 18000,
            'quality': 'high',
            'description': 'Large-scale Indian multilingual speech dataset',
            'splits': ['train', 'validation', 'test']
        },
        'google/fleurs': {
            'languages': ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'or', 'pa', 'as', 'bh', 'mai', 'ne', 'sa', 'sd', 'ur'],
            'hours': 100,
            'quality': 'high',
            'description': 'Few-shot Learning Evaluation of Universal Representations of Speech',
            'splits': ['train', 'validation', 'test']
        },
        'mozilla-foundation/common_voice_13_0': {
            'languages': ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'or', 'pa'],
            'hours': 500,
            'quality': 'medium',
            'description': 'Mozilla Common Voice Indian languages',
            'splits': ['train', 'validation', 'test']
        },
        'openslr/slr64': {
            'languages': ['hi'],
            'hours': 40,
            'quality': 'high',
            'description': 'Hindi male speech corpus',
            'splits': ['train']
        },
        'openslr/slr78': {
            'languages': ['bn'],
            'hours': 196,
            'quality': 'high',
            'description': 'Bengali speech corpus',
            'splits': ['train']
        },
        'facebook/multilingual_librispeech': {
            'languages': ['hi'],
            'hours': 44,
            'quality': 'high',
            'description': 'Multilingual LibriSpeech including Hindi',
            'splits': ['train', 'validation', 'test']
        },
        'ai4bharat/Shrutilipi': {
            'languages': ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml'],
            'hours': 6000,
            'quality': 'medium',
            'description': 'Code-mixed Indian speech dataset',
            'splits': ['train', 'validation']
        }
    }
    
    # Indian language codes
    INDIAN_LANGUAGES = [
        'hi',   # Hindi
        'bn',   # Bengali
        'ta',   # Tamil
        'te',   # Telugu
        'mr',   # Marathi
        'gu',   # Gujarati
        'kn',   # Kannada
        'ml',   # Malayalam
        'or',   # Odia
        'pa',   # Punjabi
        'as',   # Assamese
        'bh',   # Bhojpuri
        'mai',  # Maithili
        'ne',   # Nepali
        'sa',   # Sanskrit
        'sd',   # Sindhi
        'ur'    # Urdu
    ]
    
    def __init__(self, config: Dict):
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("Hugging Face datasets library required. Install with: pip install datasets")
        
        self.config = config
        self.cache_dir = config.get('cache_dir', './hf_cache')
        self.streaming = config.get('streaming', True)
        self.target_languages = config.get('languages', self.INDIAN_LANGUAGES)
        self.max_samples_per_dataset = config.get('max_samples_per_dataset', None)
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Setup authentication if token provided
        hf_token = config.get('hf_token') or os.getenv('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
            logger.info("✅ Authenticated with Hugging Face")
    
    def list_available_datasets(self) -> Dict[str, Dict]:
        """List all available Indian speech datasets."""
        return self.AVAILABLE_DATASETS
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a specific dataset."""
        return self.AVAILABLE_DATASETS.get(dataset_name)
    
    def load_dataset(self, 
                    dataset_name: str, 
                    split: str = 'train',
                    streaming: Optional[bool] = None,
                    language_filter: Optional[List[str]] = None) -> Union[Dataset, IterableDataset]:
        """Load a dataset from Hugging Face."""
        
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not available. Available: {list(self.AVAILABLE_DATASETS.keys())}")
        
        streaming = streaming if streaming is not None else self.streaming
        language_filter = language_filter or self.target_languages
        
        logger.info(f"📥 Loading dataset: {dataset_name}")
        logger.info(f"   Split: {split}")
        logger.info(f"   Streaming: {streaming}")
        logger.info(f"   Languages: {language_filter}")
        
        try:
            # Load dataset with appropriate configuration
            if dataset_name == 'ai4bharat/IndicVoices':
                dataset = self._load_indicvoices(split, streaming, language_filter)
            elif dataset_name == 'google/fleurs':
                dataset = self._load_fleurs(split, streaming, language_filter)
            elif dataset_name == 'mozilla-foundation/common_voice_13_0':
                dataset = self._load_common_voice(split, streaming, language_filter)
            elif dataset_name.startswith('openslr/'):
                dataset = self._load_openslr(dataset_name, split, streaming)
            elif dataset_name == 'facebook/multilingual_librispeech':
                dataset = self._load_multilingual_librispeech(split, streaming, language_filter)
            elif dataset_name == 'ai4bharat/Shrutilipi':
                dataset = self._load_shrutilipi(split, streaming, language_filter)
            else:
                # Generic loading
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=streaming,
                    cache_dir=self.cache_dir
                )
            
            # Apply language filtering if needed
            if language_filter and hasattr(dataset, 'filter'):
                dataset = self._filter_by_language(dataset, language_filter)
            
            # Apply sample limiting if specified
            if self.max_samples_per_dataset and hasattr(dataset, 'select'):
                dataset = dataset.select(range(min(len(dataset), self.max_samples_per_dataset)))
            
            logger.info(f"✅ Dataset loaded: {dataset_name}")
            if hasattr(dataset, '__len__'):
                logger.info(f"   Samples: {len(dataset):,}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset '{dataset_name}': {str(e)}")
            raise
    
    def _load_indicvoices(self, split: str, streaming: bool, language_filter: List[str]) -> Union[Dataset, IterableDataset]:
        """Load AI4Bharat IndicVoices dataset."""
        
        # IndicVoices has separate configs for each language
        datasets = []
        
        for lang in language_filter:
            if lang in self.AVAILABLE_DATASETS['ai4bharat/IndicVoices']['languages']:
                try:
                    lang_dataset = load_dataset(
                        'ai4bharat/IndicVoices',
                        name=lang,
                        split=split,
                        streaming=streaming,
                        cache_dir=self.cache_dir
                    )
                    datasets.append(lang_dataset)
                    logger.info(f"   Loaded {lang} subset")
                except Exception as e:
                    logger.warning(f"   Failed to load {lang} subset: {str(e)}")
        
        if not datasets:
            raise ValueError("No datasets loaded for any requested language")
        
        # Combine datasets
        if len(datasets) == 1:
            return datasets[0]
        else:
            # For streaming datasets, we'll need to interleave
            if streaming:
                return self._interleave_datasets(datasets)
            else:
                from datasets import concatenate_datasets
                return concatenate_datasets(datasets)
    
    def _load_fleurs(self, split: str, streaming: bool, language_filter: List[str]) -> Union[Dataset, IterableDataset]:
        """Load Google FLEURS dataset."""
        
        datasets = []
        
        for lang in language_filter:
            if lang in self.AVAILABLE_DATASETS['google/fleurs']['languages']:
                try:
                    lang_dataset = load_dataset(
                        'google/fleurs',
                        name=f'{lang}_in',  # FLEURS uses country codes
                        split=split,
                        streaming=streaming,
                        cache_dir=self.cache_dir
                    )
                    datasets.append(lang_dataset)
                    logger.info(f"   Loaded {lang} subset")
                except Exception as e:
                    logger.warning(f"   Failed to load {lang} subset: {str(e)}")
        
        if not datasets:
            raise ValueError("No datasets loaded for any requested language")
        
        return datasets[0] if len(datasets) == 1 else self._interleave_datasets(datasets)
    
    def _load_common_voice(self, split: str, streaming: bool, language_filter: List[str]) -> Union[Dataset, IterableDataset]:
        """Load Mozilla Common Voice dataset."""
        
        datasets = []
        
        for lang in language_filter:
            if lang in self.AVAILABLE_DATASETS['mozilla-foundation/common_voice_13_0']['languages']:
                try:
                    lang_dataset = load_dataset(
                        'mozilla-foundation/common_voice_13_0',
                        name=lang,
                        split=split,
                        streaming=streaming,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )
                    datasets.append(lang_dataset)
                    logger.info(f"   Loaded {lang} subset")
                except Exception as e:
                    logger.warning(f"   Failed to load {lang} subset: {str(e)}")
        
        if not datasets:
            raise ValueError("No datasets loaded for any requested language")
        
        return datasets[0] if len(datasets) == 1 else self._interleave_datasets(datasets)
    
    def _load_openslr(self, dataset_name: str, split: str, streaming: bool) -> Union[Dataset, IterableDataset]:
        """Load OpenSLR datasets."""
        
        return load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=self.cache_dir
        )
    
    def _load_multilingual_librispeech(self, split: str, streaming: bool, language_filter: List[str]) -> Union[Dataset, IterableDataset]:
        """Load Multilingual LibriSpeech dataset."""
        
        # Filter for Hindi only (main Indian language in MLS)
        if 'hi' in language_filter:
            return load_dataset(
                'facebook/multilingual_librispeech',
                name='hindi',
                split=split,
                streaming=streaming,
                cache_dir=self.cache_dir
            )
        else:
            raise ValueError("Multilingual LibriSpeech only supports Hindi ('hi') among Indian languages")
    
    def _load_shrutilipi(self, split: str, streaming: bool, language_filter: List[str]) -> Union[Dataset, IterableDataset]:
        """Load AI4Bharat Shrutilipi dataset."""
        
        return load_dataset(
            'ai4bharat/Shrutilipi',
            split=split,
            streaming=streaming,
            cache_dir=self.cache_dir
        )
    
    def _filter_by_language(self, dataset: Union[Dataset, IterableDataset], language_filter: List[str]) -> Union[Dataset, IterableDataset]:
        """Filter dataset by language codes."""
        
        def language_filter_fn(example):
            # Check if example has language field
            if 'language' in example:
                return example['language'] in language_filter
            elif 'locale' in example:
                return example['locale'][:2] in language_filter  # Extract language code
            elif 'lang' in example:
                return example['lang'] in language_filter
            else:
                return True  # If no language field, include all
        
        return dataset.filter(language_filter_fn)
    
    def _interleave_datasets(self, datasets: List[Union[Dataset, IterableDataset]]) -> IterableDataset:
        """Interleave multiple datasets."""
        
        if len(datasets) == 1:
            return datasets[0]
        
        # For iterable datasets, create custom interleaving
        if isinstance(datasets[0], IterableDataset):
            return self._custom_interleave(datasets)
        else:
            # For regular datasets, use datasets library interleaving
            from datasets import interleave_datasets
            return interleave_datasets(datasets)
    
    def _custom_interleave(self, datasets: List[IterableDataset]) -> Iterator:
        """Custom interleaving for streaming datasets."""
        
        iterators = [iter(dataset) for dataset in datasets]
        
        while iterators:
            for i, iterator in enumerate(iterators):
                try:
                    yield next(iterator)
                except StopIteration:
                    iterators.pop(i)
                    break
    
    def get_recommended_datasets_for_phase(self, phase: str) -> List[str]:
        """Get recommended datasets for a training phase."""
        
        recommendations = {
            'A': [  # Foundation Phase
                'ai4bharat/IndicVoices',
                'mozilla-foundation/common_voice_13_0'
            ],
            'B': [  # Enhancement Phase
                'google/fleurs',
                'openslr/slr64',
                'openslr/slr78'
            ],
            'C': [  # Specialization Phase
                'facebook/multilingual_librispeech',
                'ai4bharat/Shrutilipi'
            ],
            'D': [  # Optimization Phase - Speaker/Accent variation
                'ai4bharat/IndicVoices',  # Has good speaker diversity
                'google/fleurs'
            ],
            'E': [  # Finalization Phase - High quality
                'ai4bharat/IndicVoices',
                'facebook/multilingual_librispeech'
            ]
        }
        
        return recommendations.get(phase.upper(), [])
    
    def create_phase_config(self, phase: str, custom_datasets: Optional[List[str]] = None) -> Dict:
        """Create dataset configuration for a training phase."""
        
        datasets = custom_datasets or self.get_recommended_datasets_for_phase(phase)
        
        config = {
            'phase_datasets': {
                phase.upper(): datasets
            },
            'huggingface_config': {
                'cache_dir': self.cache_dir,
                'streaming': self.streaming,
                'languages': self.target_languages,
                'max_samples_per_dataset': self.max_samples_per_dataset
            }
        }
        
        return config

def main():
    """Example usage of HuggingFace dataset loader."""
    
    # Configuration
    config = {
        'cache_dir': './hf_cache',
        'streaming': True,
        'languages': ['hi', 'bn', 'ta', 'te', 'mr'],
        'max_samples_per_dataset': 10000,
        'hf_token': None  # Optional HF token
    }
    
    # Initialize loader
    loader = HuggingFaceDatasetLoader(config)
    
    # List available datasets
    print("Available Indian Speech Datasets:")
    for name, info in loader.list_available_datasets().items():
        print(f"  {name}: {info['description']} ({info['hours']} hours)")
    
    # Load a dataset
    try:
        dataset = loader.load_dataset('ai4bharat/IndicVoices', 'train')
        print(f"\nLoaded dataset with {len(dataset) if hasattr(dataset, '__len__') else 'streaming'} samples")
        
        # Show sample
        sample = next(iter(dataset))
        print(f"Sample keys: {list(sample.keys())}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    # Generate phase configuration
    phase_config = loader.create_phase_config('A')
    print(f"\nPhase A Configuration:")
    print(yaml.dump(phase_config, indent=2))

if __name__ == "__main__":
    main()