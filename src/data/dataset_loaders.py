"""
Dataset loaders for Common Voice, LibriSpeech, VoxForge, and custom multi-channel datasets.

This module provides specialized data loaders for popular speech recognition datasets
with multi-channel and multi-lingual support built from scratch.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import librosa
import soundfile as sf
import numpy as np
from omegaconf import DictConfig
import tarfile
import zipfile
import requests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .speech_data_loader import SpeechDataset, SpeechDataModule
from .audio_processing import MultiChannelAudioProcessor
from ..utils.multilingual import MultiLingualTokenizer

class CommonVoiceDataset(SpeechDataset):
    """
    Common Voice dataset loader with multi-lingual support for 100+ languages.
    """
    
    def __init__(self, data_path: Union[str, Path], tokenizer: MultiLingualTokenizer,
                 audio_processor: MultiChannelAudioProcessor, config: DictConfig,
                 split: str = 'train', language: Optional[str] = None):
        
        self.language_filter = language
        super().__init__(data_path, tokenizer, audio_processor, config, split)
        
    def _load_dataset(self) -> List[Dict]:
        """Load Common Voice dataset from TSV files."""
        # Common Voice uses TSV format
        tsv_file = self.data_path / f"{self.split}.tsv"
        
        if not tsv_file.exists():
            print(f"Warning: {tsv_file} not found. Creating empty dataset.")
            return []
        
        # Load TSV file
        try:
            df = pd.read_csv(tsv_file, sep='\t')
        except Exception as e:
            print(f"Error loading TSV file: {e}")
            return []
        
        samples = []
        clips_dir = self.data_path / 'clips'
        
        for _, row in df.iterrows():
            # Extract sample information
            audio_filename = row.get('path', '')
            text = row.get('sentence', '')
            client_id = row.get('client_id', 'unknown')
            
            # Get language from filename or path
            language = self._extract_language_from_path() or 'en'
            
            # Filter by language if specified
            if self.language_filter and language != self.language_filter:
                continue
            
            # Validate required fields
            if not audio_filename or not text:
                continue
            
            # Build full audio path
            audio_path = clips_dir / audio_filename
            
            sample = {
                'audio_path': str(audio_path),
                'text': text.strip(),
                'language': language,
                'duration': row.get('duration', 0.0) / 1000.0 if 'duration' in row else 0.0,
                'speaker_id': client_id,
                'up_votes': row.get('up_votes', 0),
                'down_votes': row.get('down_votes', 0),
                'age': row.get('age', ''),
                'gender': row.get('gender', ''),
                'accent': row.get('accent', '')
            }
            
            if self._validate_sample(sample):
                samples.append(sample)
        
        print(f"Loaded {len(samples)} Common Voice samples for {self.split} split")
        if self.language_filter:
            print(f"Filtered for language: {self.language_filter}")
        
        return samples
    
    def _extract_language_from_path(self) -> Optional[str]:
        """Extract language code from dataset path."""
        # Common Voice datasets are typically organized by language
        # e.g., /path/to/cv-corpus-13.0-2023-03-09/en/
        path_parts = self.data_path.parts
        
        # Look for language codes in path
        for part in reversed(path_parts):
            if len(part) == 2 and part.islower():  # Likely a language code
                return part
        
        return None

class LibriSpeechDataset(SpeechDataset):
    """
    LibriSpeech dataset loader for English speech recognition.
    """
    
    def __init__(self, data_path: Union[str, Path], tokenizer: MultiLingualTokenizer,
                 audio_processor: MultiChannelAudioProcessor, config: DictConfig,
                 split: str = 'train'):
        
        # LibriSpeech split mapping
        self.librispeech_splits = {
            'train': ['train-clean-100', 'train-clean-360', 'train-other-500'],
            'validation': ['dev-clean', 'dev-other'], 
            'test': ['test-clean', 'test-other']
        }
        
        super().__init__(data_path, tokenizer, audio_processor, config, split)
    
    def _load_dataset(self) -> List[Dict]:
        """Load LibriSpeech dataset from FLAC files and transcripts."""
        samples = []
        
        # Get appropriate LibriSpeech subsets for the split
        subsets = self.librispeech_splits.get(self.split, [self.split])
        
        for subset in subsets:
            subset_path = self.data_path / subset
            if not subset_path.exists():
                print(f"Warning: LibriSpeech subset {subset} not found at {subset_path}")
                continue
            
            # Scan for transcript files
            for transcript_file in subset_path.rglob("*.trans.txt"):
                speaker_chapter_dir = transcript_file.parent
                
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) < 2:
                            continue
                        
                        utterance_id = parts[0]
                        text = parts[1]
                        
                        # Build audio file path
                        audio_file = speaker_chapter_dir / f"{utterance_id}.flac"
                        
                        if audio_file.exists():
                            # Extract speaker and chapter info from path
                            speaker_id = speaker_chapter_dir.parent.name
                            chapter_id = speaker_chapter_dir.name
                            
                            sample = {
                                'audio_path': str(audio_file),
                                'text': text.strip(),
                                'language': 'en',  # LibriSpeech is English only
                                'duration': 0.0,  # Will be computed during loading
                                'speaker_id': speaker_id,
                                'chapter_id': chapter_id,
                                'utterance_id': utterance_id,
                                'subset': subset
                            }
                            
                            if self._validate_sample(sample):
                                samples.append(sample)
        
        print(f"Loaded {len(samples)} LibriSpeech samples for {self.split} split")
        return samples

class VoxForgeDataset(SpeechDataset):
    """
    VoxForge dataset loader with multi-lingual support.
    """
    
    def __init__(self, data_path: Union[str, Path], tokenizer: MultiLingualTokenizer,
                 audio_processor: MultiChannelAudioProcessor, config: DictConfig,
                 split: str = 'train', language: Optional[str] = None):
        
        self.language_filter = language
        super().__init__(data_path, tokenizer, audio_processor, config, split)
    
    def _load_dataset(self) -> List[Dict]:
        """Load VoxForge dataset from various audio formats."""
        samples = []
        
        # VoxForge can have different structures, try to handle common ones
        audio_extensions = ['.wav', '.flac']
        
        for speaker_dir in self.data_path.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            
            # Look for transcription files
            for transcript_file in speaker_dir.glob("*.txt"):
                if transcript_file.name.lower() in ['readme.txt', 'license.txt']:
                    continue
                
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # VoxForge transcripts can have different formats
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Try to parse as "filename.wav transcription"
                        parts = line.split(' ', 1)
                        if len(parts) >= 2:
                            audio_filename = parts[0]
                            text = parts[1]
                            
                            # Find corresponding audio file
                            for ext in audio_extensions:
                                audio_file = speaker_dir / audio_filename.replace('.wav', ext)
                                if audio_file.exists():
                                    
                                    # Detect language (VoxForge may have language markers)
                                    language = self._detect_language_from_text_or_path(
                                        text, speaker_dir
                                    )
                                    
                                    # Filter by language if specified
                                    if self.language_filter and language != self.language_filter:
                                        continue
                                    
                                    sample = {
                                        'audio_path': str(audio_file),
                                        'text': text.strip(),
                                        'language': language,
                                        'duration': 0.0,
                                        'speaker_id': speaker_id
                                    }
                                    
                                    if self._validate_sample(sample):
                                        samples.append(sample)
                                    break
                
                except Exception as e:
                    print(f"Error processing transcript {transcript_file}: {e}")
                    continue
        
        print(f"Loaded {len(samples)} VoxForge samples for {self.split} split")
        return samples
    
    def _detect_language_from_text_or_path(self, text: str, path: Path) -> str:
        """Detect language from text content or directory structure."""
        # Simple heuristic-based language detection
        # In practice, you might want to use a more sophisticated language detector
        
        # Check path for language indicators
        path_str = str(path).lower()
        if 'english' in path_str or 'en' in path_str:
            return 'en'
        elif 'german' in path_str or 'de' in path_str:
            return 'de'
        elif 'french' in path_str or 'fr' in path_str:
            return 'fr'
        elif 'spanish' in path_str or 'es' in path_str:
            return 'es'
        elif 'italian' in path_str or 'it' in path_str:
            return 'it'
        
        # Simple text-based detection (very basic)
        text_lower = text.lower()
        if any(word in text_lower for word in ['the', 'and', 'is', 'of', 'to']):
            return 'en'
        elif any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist']):
            return 'de'
        elif any(word in text_lower for word in ['le', 'la', 'et', 'est', 'de']):
            return 'fr'
        elif any(word in text_lower for word in ['el', 'la', 'y', 'es', 'de']):
            return 'es'
        
        return 'en'  # Default to English

class CustomMultiChannelDataset(SpeechDataset):
    """
    Custom dataset loader for multi-channel audio recordings.
    Supports various multi-channel configurations (stereo, 5.1, 7.1, custom).
    """
    
    def __init__(self, data_path: Union[str, Path], tokenizer: MultiLingualTokenizer,
                 audio_processor: MultiChannelAudioProcessor, config: DictConfig,
                 split: str = 'train', channel_config: str = 'stereo'):
        
        self.channel_config = channel_config
        self.expected_channels = self._get_expected_channels(channel_config)
        
        super().__init__(data_path, tokenizer, audio_processor, config, split)
    
    def _get_expected_channels(self, config: str) -> int:
        """Get expected number of channels for the configuration."""
        channel_map = {
            'mono': 1,
            'stereo': 2,
            'quad': 4,
            '5.1': 6,
            '7.1': 8
        }
        return channel_map.get(config, 2)  # Default to stereo
    
    def _load_dataset(self) -> List[Dict]:
        """Load custom multi-channel dataset."""
        samples = []
        
        # Look for metadata file
        metadata_file = self.data_path / f"{self.split}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            for item in metadata:
                sample = {
                    'audio_path': item.get('audio_path', ''),
                    'text': item.get('text', ''),
                    'language': item.get('language', 'en'),
                    'duration': item.get('duration', 0.0),
                    'speaker_id': item.get('speaker_id', 'unknown'),
                    'channel_config': item.get('channel_config', self.channel_config),
                    'recording_environment': item.get('environment', 'unknown'),
                    'microphone_setup': item.get('mic_setup', 'unknown')
                }
                
                # Validate multi-channel requirements
                if self._validate_multichannel_sample(sample):
                    samples.append(sample)
        
        else:
            # Scan directory structure
            audio_extensions = ['.wav', '.flac', '.aiff']
            
            for audio_file in self.data_path.rglob("*"):
                if audio_file.suffix.lower() in audio_extensions:
                    # Check if it's multi-channel
                    if self._is_multichannel_audio(audio_file):
                        
                        # Look for corresponding transcript
                        transcript_file = audio_file.with_suffix('.txt')
                        if transcript_file.exists():
                            with open(transcript_file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            
                            sample = {
                                'audio_path': str(audio_file),
                                'text': text,
                                'language': 'en',  # Default
                                'duration': 0.0,
                                'speaker_id': 'unknown',
                                'channel_config': self.channel_config
                            }
                            
                            if self._validate_multichannel_sample(sample):
                                samples.append(sample)
        
        print(f"Loaded {len(samples)} multi-channel samples for {self.split} split")
        print(f"Channel configuration: {self.channel_config} ({self.expected_channels} channels)")
        
        return samples
    
    def _is_multichannel_audio(self, audio_file: Path) -> bool:
        """Check if audio file has the expected number of channels."""
        try:
            info = sf.info(str(audio_file))
            return info.channels >= self.expected_channels
        except:
            return False
    
    def _validate_multichannel_sample(self, sample: Dict) -> bool:
        """Validate multi-channel specific requirements."""
        # First do basic validation
        if not self._validate_sample(sample):
            return False
        
        # Check channel count
        audio_path = Path(sample['audio_path'])
        if not audio_path.is_absolute():
            audio_path = self.data_path / audio_path
        
        try:
            info = sf.info(str(audio_path))
            if info.channels < self.expected_channels:
                return False
        except:
            return False
        
        return True

# Factory functions for dataset creation
def create_common_voice_dataset(data_path: str, tokenizer: MultiLingualTokenizer,
                               audio_processor: MultiChannelAudioProcessor,
                               config: DictConfig, split: str = 'train',
                               language: Optional[str] = None) -> CommonVoiceDataset:
    """Create Common Voice dataset."""
    return CommonVoiceDataset(data_path, tokenizer, audio_processor, config, split, language)

def create_librispeech_dataset(data_path: str, tokenizer: MultiLingualTokenizer,
                              audio_processor: MultiChannelAudioProcessor,
                              config: DictConfig, split: str = 'train') -> LibriSpeechDataset:
    """Create LibriSpeech dataset."""
    return LibriSpeechDataset(data_path, tokenizer, audio_processor, config, split)

def create_voxforge_dataset(data_path: str, tokenizer: MultiLingualTokenizer,
                           audio_processor: MultiChannelAudioProcessor,
                           config: DictConfig, split: str = 'train',
                           language: Optional[str] = None) -> VoxForgeDataset:
    """Create VoxForge dataset."""
    return VoxForgeDataset(data_path, tokenizer, audio_processor, config, split, language)

def create_multichannel_dataset(data_path: str, tokenizer: MultiLingualTokenizer,
                               audio_processor: MultiChannelAudioProcessor,
                               config: DictConfig, split: str = 'train',
                               channel_config: str = 'stereo') -> CustomMultiChannelDataset:
    """Create custom multi-channel dataset."""
    return CustomMultiChannelDataset(data_path, tokenizer, audio_processor, config, split, channel_config)

# Dataset downloader utilities
class DatasetDownloader:
    """Utility class for downloading common speech datasets."""
    
    @staticmethod
    def download_common_voice(language: str, target_dir: str, version: str = 'cv-corpus-13.0-2023-03-09'):
        """Download Common Voice dataset for a specific language."""
        print(f"Downloading Common Voice {language} dataset...")
        
        # Common Voice download URLs (these would need to be updated with actual URLs)
        base_url = f"https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/{version}/{language}.tar.gz"
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        archive_path = target_path / f"{language}.tar.gz"
        extract_path = target_path / language
        
        # Download if not exists
        if not archive_path.exists():
            print(f"Note: Actual download would require proper Mozilla Common Voice access.")
            print(f"Please download manually from: https://commonvoice.mozilla.org/datasets")
            print(f"Expected archive location: {archive_path}")
            return
        
        # Extract
        if not extract_path.exists():
            print(f"Extracting {archive_path}...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(target_path)
        
        print(f"Common Voice {language} dataset ready at {extract_path}")
    
    @staticmethod
    def download_librispeech(subset: str, target_dir: str):
        """Download LibriSpeech dataset subset."""
        print(f"Downloading LibriSpeech {subset}...")
        
        # LibriSpeech download URLs
        base_url = "http://www.openslr.org/resources/12"
        filename = f"{subset}.tar.gz"
        url = f"{base_url}/{filename}"
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        archive_path = target_path / filename
        
        # Download if not exists
        if not archive_path.exists():
            print(f"Note: Download from {url}")
            print(f"Please download manually and place at: {archive_path}")
            return
        
        # Extract
        extract_path = target_path / subset
        if not extract_path.exists():
            print(f"Extracting {archive_path}...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(target_path)
        
        print(f"LibriSpeech {subset} ready at {extract_path}")

if __name__ == "__main__":
    print("🎤 Speech Dataset Loaders")
    print("Supports: Common Voice, LibriSpeech, VoxForge, Custom Multi-Channel")
    
    # Example dataset info
    datasets_info = {
        'Common Voice': {
            'languages': '100+',
            'format': 'MP3/WAV',
            'transcripts': 'TSV',
            'license': 'CC-0'
        },
        'LibriSpeech': {
            'languages': 'English',
            'format': 'FLAC',
            'transcripts': 'TXT',
            'hours': '1000+'
        },
        'VoxForge': {
            'languages': 'Multiple',
            'format': 'WAV/FLAC',
            'transcripts': 'TXT',
            'license': 'GPL'
        },
        'Custom Multi-Channel': {
            'channels': 'Stereo, 5.1, 7.1, Custom',
            'format': 'WAV/FLAC/AIFF',
            'transcripts': 'JSON/TXT',
            'features': 'Beamforming, Source Separation'
        }
    }
    
    for dataset, info in datasets_info.items():
        print(f"\n{dataset}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\n✅ Dataset loaders ready for multi-channel, multi-lingual speech recognition!")