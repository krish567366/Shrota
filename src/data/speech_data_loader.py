"""
Speech Data Module for Multi-Channel, Multi-Lingual Speech Recognition

Handles loading and preprocessing of various speech datasets including:
- Common Voice dataset (multi-lingual)
- LibriSpeech (English)
- VoxForge (multi-lingual)
- Custom multi-channel datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import librosa
import soundfile as sf
import numpy as np
from omegaconf import DictConfig
import warnings
warnings.filterwarnings('ignore')

from .audio_processing import MultiChannelAudioProcessor
from .multispeaker_processing import MultiSpeakerProcessor
from ..utils.multilingual import MultiLingualTokenizer, get_supported_languages

class SpeechDataset(Dataset):
    """
    Dataset for multi-channel, multi-lingual speech recognition.
    """
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 tokenizer: MultiLingualTokenizer,
                 audio_processor: MultiChannelAudioProcessor,
                 config: DictConfig,
                 split: str = 'train'):
        
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.config = config
        self.split = split
        
        # Audio processing parameters
        self.sample_rate = config.data.get('sample_rate', 16000)
        self.max_audio_length = config.data.get('max_audio_length', 30)  # seconds
        self.max_text_length = config.data.get('max_text_length', 512)  # tokens
        
        # Load dataset metadata
        self.data_samples = self._load_dataset()
        
        print(f"Loaded {len(self.data_samples)} samples for {split} split")
        
    def _load_dataset(self) -> List[Dict]:
        """Load dataset samples from various sources."""
        samples = []
        
        # Try to load from different dataset formats
        manifest_file = self.data_path / f"{self.split}_manifest.json"
        csv_file = self.data_path / f"{self.split}.csv"
        tsv_file = self.data_path / f"{self.split}.tsv"
        
        if manifest_file.exists():
            # JSON manifest format (Common Voice style)
            samples = self._load_from_manifest(manifest_file)
        elif csv_file.exists():
            # CSV format
            samples = self._load_from_csv(csv_file)
        elif tsv_file.exists():
            # TSV format (Common Voice)
            samples = self._load_from_tsv(tsv_file)
        else:
            # Try to scan directory for audio files
            samples = self._scan_audio_directory()
        
        return samples
    
    def _load_from_manifest(self, manifest_file: Path) -> List[Dict]:
        """Load from JSON manifest file."""
        samples = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                if self._validate_sample(sample):
                    samples.append(sample)
        return samples
    
    def _load_from_csv(self, csv_file: Path) -> List[Dict]:
        """Load from CSV file."""
        df = pd.read_csv(csv_file)
        samples = []
        
        for _, row in df.iterrows():
            sample = {
                'audio_path': row.get('audio_path', row.get('path', '')),
                'text': row.get('text', row.get('transcription', '')),
                'language': row.get('language', row.get('lang', 'en')),
                'duration': row.get('duration', 0.0),
                'speaker_id': row.get('speaker_id', 'unknown')
            }
            
            if self._validate_sample(sample):
                samples.append(sample)
        
        return samples
    
    def _load_from_tsv(self, tsv_file: Path) -> List[Dict]:
        """Load from TSV file (Common Voice format)."""
        df = pd.read_csv(tsv_file, sep='\t')
        samples = []
        
        for _, row in df.iterrows():
            sample = {
                'audio_path': row.get('path', ''),
                'text': row.get('sentence', ''),
                'language': row.get('locale', 'en'),
                'duration': row.get('duration', 0.0) / 1000.0,  # Convert ms to seconds
                'speaker_id': row.get('client_id', 'unknown')
            }
            
            if self._validate_sample(sample):
                samples.append(sample)
        
        return samples
    
    def _scan_audio_directory(self) -> List[Dict]:
        """Scan directory for audio files."""
        samples = []
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        for ext in audio_extensions:
            for audio_file in self.data_path.rglob(f"*{ext}"):
                # Look for corresponding text file
                text_file = audio_file.with_suffix('.txt')
                
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    sample = {
                        'audio_path': str(audio_file),
                        'text': text,
                        'language': 'en',  # Default to English
                        'duration': 0.0,  # Will be computed during loading
                        'speaker_id': 'unknown'
                    }
                    
                    samples.append(sample)
        
        return samples
    
    def _validate_sample(self, sample: Dict) -> bool:
        """Validate a data sample."""
        required_keys = ['audio_path', 'text']
        
        # Check required keys
        for key in required_keys:
            if key not in sample or not sample[key]:
                return False
        
        # Check if audio file exists
        audio_path = Path(sample['audio_path'])
        if not audio_path.is_absolute():
            audio_path = self.data_path / audio_path
        
        if not audio_path.exists():
            return False
        
        # Check text length
        if len(sample['text']) > 1000:  # Too long
            return False
        
        # Check duration if available
        if sample.get('duration', 0) > self.max_audio_length:
            return False
        
        return True
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data_samples[idx]
        
        # Load and process audio
        audio_path = Path(sample['audio_path'])
        if not audio_path.is_absolute():
            audio_path = self.data_path / audio_path
        
        try:
            # Load multi-channel audio
            audio_tensor, _ = self.audio_processor.loader.load_audio(str(audio_path))
            
            # Process audio through enhancement pipeline
            processed_audio = self.audio_processor.process_audio(audio_tensor)
            
            # Extract audio features (mel spectrogram)
            features = self._extract_features(processed_audio)
            
            # Tokenize text
            language = sample.get('language', 'en')
            text_tokens = self.tokenizer.encode(
                sample['text'], 
                language=language,
                max_length=self.max_text_length
            )
            
            # Create input and target tensors
            input_tokens = text_tokens[:-1] if len(text_tokens) > 1 else [self.tokenizer.vocab['<sos>']]
            target_tokens = text_tokens[1:] if len(text_tokens) > 1 else [self.tokenizer.vocab['<eos>']]
            
            return {
                'audio_features': features,
                'input_tokens': torch.tensor(input_tokens, dtype=torch.long),
                'target_tokens': torch.tensor(target_tokens, dtype=torch.long),
                'language': language,
                'text': sample['text'],
                'audio_length': torch.tensor(features.shape[0], dtype=torch.long),
                'text_length': torch.tensor(len(target_tokens), dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a dummy sample
            return self._get_dummy_sample()
    
    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract audio features (mel spectrogram)."""
        # Convert to numpy for librosa
        audio_np = audio.numpy()
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np,
            sr=self.sample_rate,
            n_mels=self.config.data.get('n_mels', 80),
            n_fft=self.config.data.get('n_fft', 512),
            hop_length=self.config.data.get('hop_length', 256),
            win_length=self.config.data.get('win_length', 512)
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        # Convert to tensor and transpose to (time, features)
        features = torch.from_numpy(log_mel).float().T
        
        return features
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Get a dummy sample for error cases."""
        dummy_features = torch.zeros(100, self.config.data.get('n_mels', 80))
        dummy_tokens = [self.tokenizer.vocab['<sos>'], self.tokenizer.vocab['<eos>']]
        
        return {
            'audio_features': dummy_features,
            'input_tokens': torch.tensor([self.tokenizer.vocab['<sos>']], dtype=torch.long),
            'target_tokens': torch.tensor([self.tokenizer.vocab['<eos>']], dtype=torch.long),
            'language': 'en',
            'text': '',
            'audio_length': torch.tensor(100, dtype=torch.long),
            'text_length': torch.tensor(1, dtype=torch.long)
        }

class SpeechDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for speech recognition.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.data_config = config.data
        
        # Initialize tokenizer and audio processor
        self.tokenizer = MultiLingualTokenizer(
            vocab_size=config.speech.get('vocab_size', 32000)
        )
        
        self.audio_processor = MultiChannelAudioProcessor(
            num_channels=config.speech.get('num_channels', 2),  
            sample_rate=config.data.get('sample_rate', 16000)
        )
        
        # Initialize multi-speaker processor
        self.multispeaker_processor = MultiSpeakerProcessor(
            max_speakers=config.speech.get('max_speakers', 3),
            sample_rate=config.data.get('sample_rate', 16000)
        )
        
        # Dataset paths
        self.data_dir = Path(self.data_config.data_dir)
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            # Training dataset
            train_path = self.data_dir / 'train'
            if train_path.exists():
                self.train_dataset = SpeechDataset(
                    data_path=train_path,
                    tokenizer=self.tokenizer,
                    audio_processor=self.audio_processor,
                    config=self.config,
                    split='train'
                )
            
            # Validation dataset
            val_path = self.data_dir / 'validation'
            if val_path.exists():
                self.val_dataset = SpeechDataset(
                    data_path=val_path,
                    tokenizer=self.tokenizer,
                    audio_processor=self.audio_processor,
                    config=self.config,
                    split='validation'
                )
        
        if stage == 'test' or stage is None:
            # Test dataset
            test_path = self.data_dir / 'test'
            if test_path.exists():
                self.test_dataset = SpeechDataset(
                    data_path=test_path,
                    tokenizer=self.tokenizer,
                    audio_processor=self.audio_processor,
                    config=self.config,
                    split='test'
                )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        # Separate different data types
        audio_features = [item['audio_features'] for item in batch]
        input_tokens = [item['input_tokens'] for item in batch]
        target_tokens = [item['target_tokens'] for item in batch]
        audio_lengths = torch.stack([item['audio_length'] for item in batch])
        text_lengths = torch.stack([item['text_length'] for item in batch])
        languages = [item['language'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # Pad audio features
        max_audio_len = max(feat.shape[0] for feat in audio_features)
        feature_dim = audio_features[0].shape[1]
        
        padded_audio = torch.zeros(len(batch), max_audio_len, feature_dim)
        for i, feat in enumerate(audio_features):
            length = feat.shape[0]
            padded_audio[i, :length, :] = feat
        
        # Pad token sequences
        max_input_len = max(len(tokens) for tokens in input_tokens)
        max_target_len = max(len(tokens) for tokens in target_tokens)
        
        padded_input = torch.full((len(batch), max_input_len), 
                                 self.tokenizer.vocab['<pad>'], dtype=torch.long)
        padded_target = torch.full((len(batch), max_target_len), 
                                  self.tokenizer.vocab['<pad>'], dtype=torch.long)
        
        for i, (inp, tgt) in enumerate(zip(input_tokens, target_tokens)):
            padded_input[i, :len(inp)] = inp
            padded_target[i, :len(tgt)] = tgt
        
        return {
            'audio_features': padded_audio,
            'input_tokens': padded_input,
            'target_tokens': padded_target,
            'audio_lengths': audio_lengths,
            'text_lengths': text_lengths,
            'languages': languages,
            'texts': texts
        }
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            return None
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.get('batch_size', 16),
            shuffle=True,
            num_workers=self.data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.data_config.get('num_workers', 4) > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            return None
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.data_config.get('num_workers', 4) > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.data_config.get('num_workers', 4) > 0 else False
        )
    
    def get_dataset_info(self) -> Dict:
        """Get information about the datasets."""
        info = {
            'tokenizer_vocab_size': self.tokenizer.get_vocab_size(),
            'supported_languages': len(get_supported_languages()),
            'audio_processor_channels': self.audio_processor.num_channels,
            'sample_rate': self.audio_processor.sample_rate
        }
        
        if self.train_dataset:
            info['train_samples'] = len(self.train_dataset)
        if self.val_dataset:
            info['val_samples'] = len(self.val_dataset)
        if self.test_dataset:
            info['test_samples'] = len(self.test_dataset)
            
        return info

# Factory functions for common datasets
def create_common_voice_datamodule(data_dir: str, config: DictConfig) -> SpeechDataModule:
    """Create data module for Common Voice dataset."""
    # Update config for Common Voice specific settings
    config.data.data_dir = data_dir
    return SpeechDataModule(config)

def create_librispeech_datamodule(data_dir: str, config: DictConfig) -> SpeechDataModule:
    """Create data module for LibriSpeech dataset.""" 
    # Update config for LibriSpeech specific settings
    config.data.data_dir = data_dir
    config.speech.num_channels = 1  # LibriSpeech is mono
    return SpeechDataModule(config)

if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'data': {
            'data_dir': './data/common_voice',
            'batch_size': 16,
            'num_workers': 4,
            'sample_rate': 16000,
            'n_mels': 80,
            'max_audio_length': 30,
            'max_text_length': 512
        },
        'speech': {
            'vocab_size': 32000,
            'num_channels': 2,
            'num_languages': 100
        }
    })
    
    # Create data module
    data_module = SpeechDataModule(config)
    print("✅ Speech data module created successfully!")
    print(f"Dataset info: {data_module.get_dataset_info()}")