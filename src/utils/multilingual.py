"""
Multi-Lingual Language Detection and Tokenization System

Built from scratch for 100+ language support including:
- Automatic language detection from audio features
- Multi-lingual tokenization and vocabulary management
- Language-specific audio processing adaptations
- Cross-lingual transfer learning utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import re
import unicodedata
from collections import defaultdict, Counter
import math

# Language mappings and configurations
LANGUAGE_CODES = {
    # Major languages (Top 20)
    'en': 'English', 'zh': 'Chinese', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
    'ar': 'Arabic', 'bn': 'Bengali', 'ru': 'Russian', 'pt': 'Portuguese', 'id': 'Indonesian',
    'ur': 'Urdu', 'de': 'German', 'ja': 'Japanese', 'sw': 'Swahili', 'mr': 'Marathi',
    'te': 'Telugu', 'tr': 'Turkish', 'ta': 'Tamil', 'vi': 'Vietnamese', 'ko': 'Korean',
    
    # European languages
    'it': 'Italian', 'pl': 'Polish', 'nl': 'Dutch', 'uk': 'Ukrainian', 'ro': 'Romanian',
    'el': 'Greek', 'cs': 'Czech', 'sv': 'Swedish', 'hu': 'Hungarian', 'no': 'Norwegian',
    'fi': 'Finnish', 'da': 'Danish', 'sk': 'Slovak', 'bg': 'Bulgarian', 'hr': 'Croatian',
    'sl': 'Slovenian', 'et': 'Estonian', 'lv': 'Latvian', 'lt': 'Lithuanian', 'mt': 'Maltese',
    
    # Asian languages
    'th': 'Thai', 'my': 'Myanmar', 'km': 'Khmer', 'lo': 'Lao', 'ka': 'Georgian',
    'am': 'Amharic', 'si': 'Sinhala', 'ne': 'Nepali', 'ml': 'Malayalam', 'kn': 'Kannada',
    'gu': 'Gujarati', 'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'mn': 'Mongolian',
    
    # African languages
    'yo': 'Yoruba', 'ig': 'Igbo', 'ha': 'Hausa', 'zu': 'Zulu', 'xh': 'Xhosa',
    'af': 'Afrikaans', 'so': 'Somali', 'rw': 'Kinyarwanda', 'sn': 'Shona', 'ny': 'Chichewa',
    
    # American indigenous languages
    'qu': 'Quechua', 'gn': 'Guarani', 'ay': 'Aymara',
    
    # Middle Eastern languages
    'fa': 'Persian', 'he': 'Hebrew', 'ku': 'Kurdish', 'az': 'Azerbaijani',
    
    # Additional languages
    'ms': 'Malay', 'tl': 'Filipino', 'haw': 'Hawaiian', 'mi': 'Maori', 'is': 'Icelandic',
    'eu': 'Basque', 'cy': 'Welsh', 'ga': 'Irish', 'br': 'Breton', 'co': 'Corsican',
    'la': 'Latin', 'eo': 'Esperanto', 'jv': 'Javanese', 'su': 'Sundanese', 'mg': 'Malagasy',
    
    # Additional Asian languages
    'uz': 'Uzbek', 'kk': 'Kazakh', 'ky': 'Kyrgyz', 'tg': 'Tajik', 'tk': 'Turkmen',
    'hy': 'Armenian', 'be': 'Belarusian', 'mk': 'Macedonian', 'sq': 'Albanian', 'lij': 'Ligurian',
    
    # More African languages  
    'ts': 'Tsonga', 've': 'Venda', 'ss': 'Swati', 'tn': 'Tswana', 'st': 'Sesotho',
    'nso': 'Northern Sotho', 'lg': 'Luganda', 'ak': 'Akan', 'tw': 'Twi', 'bm': 'Bambara',
    'wo': 'Wolof', 'ff': 'Fulfulde', 'ti': 'Tigrinya', 'om': 'Oromo'
}

# Language families for feature extraction
LANGUAGE_FAMILIES = {
    'indo_european': ['en', 'es', 'fr', 'de', 'ru', 'hi', 'bn', 'ur', 'pt', 'it', 'pl', 'nl', 'uk', 'ro', 'el', 'cs', 'sv', 'hu', 'no', 'fi', 'da', 'sk', 'bg', 'hr', 'sl', 'et', 'lv', 'lt', 'fa', 'ku', 'hy', 'be', 'mk', 'sq', 'cy', 'ga', 'br', 'co', 'la', 'eo'],
    'sino_tibetan': ['zh', 'my', 'ne'],
    'afro_asiatic': ['ar', 'he', 'am', 'ti', 'om', 'so', 'ha'],
    'austronesian': ['id', 'ms', 'tl', 'jv', 'su', 'mg', 'haw', 'mi'],
    'dravidian': ['ta', 'te', 'ml', 'kn'],
    'niger_congo': ['sw', 'yo', 'ig', 'zu', 'xh', 'sn', 'ny', 'lg', 'ak', 'tw', 'bm', 'wo', 'ff'],
    'altaic': ['tr', 'az', 'uz', 'kk', 'ky', 'tk', 'mn', 'ja', 'ko'],
    'tai_kadai': ['th', 'lo'],
    'austro_asiatic': ['vi', 'km'],
    'kartvelian': ['ka'],
    'isolates': ['eu', 'is']
}

class LanguageDetector(nn.Module):
    """
    Neural language detector using audio features.
    """
    
    def __init__(self, num_languages: int = len(LANGUAGE_CODES), 
                 feature_dim: int = 80, hidden_dim: int = 256):
        super().__init__()
        self.num_languages = num_languages
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Language codes for indexing
        self.lang_codes = list(LANGUAGE_CODES.keys())
        self.lang_to_idx = {lang: idx for idx, lang in enumerate(self.lang_codes)}
        self.idx_to_lang = {idx: lang for lang, idx in self.lang_to_idx.items()}
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal modeling
        self.temporal_model = nn.LSTM(hidden_dim, hidden_dim // 2, 
                                    num_layers=2, bidirectional=True, 
                                    batch_first=True, dropout=0.1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, 
                                             dropout=0.1, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_languages)
        )
        
        # Language family classifier (auxiliary task)
        self.family_classifier = nn.Linear(hidden_dim, len(LANGUAGE_FAMILIES))
        
    def forward(self, features: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for language detection.
        
        Args:
            features: Audio features (batch_size, seq_len, feature_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with language probabilities and auxiliary outputs
        """
        batch_size, seq_len, _ = features.shape
        
        # Feature extraction
        extracted_features = self.feature_extractor(features)  # (batch, seq_len, hidden_dim)
        
        # Temporal modeling
        lstm_out, _ = self.temporal_model(extracted_features)  # (batch, seq_len, hidden_dim)
        
        # Self-attention
        attended_features, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling (mean + max)
        mean_pooled = torch.mean(attended_features, dim=1)  # (batch, hidden_dim)
        max_pooled = torch.max(attended_features, dim=1)[0]  # (batch, hidden_dim)
        global_features = mean_pooled + max_pooled  # (batch, hidden_dim)
        
        # Language classification
        language_logits = self.classifier(global_features)  # (batch, num_languages)
        language_probs = F.softmax(language_logits, dim=-1)
        
        # Language family classification (auxiliary)
        family_logits = self.family_classifier(global_features)
        family_probs = F.softmax(family_logits, dim=-1)
        
        results = {
            'language_probs': language_probs,
            'language_logits': language_logits,
            'family_probs': family_probs,
            'family_logits': family_logits,
            'features': global_features
        }
        
        if return_attention:
            results['attention_weights'] = attention_weights
            
        return results
    
    def predict_language(self, features: torch.Tensor, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict language from audio features.
        
        Args:
            features: Audio features (seq_len, feature_dim) or (batch_size, seq_len, feature_dim)
            top_k: Number of top predictions to return
            
        Returns:
            List of (language_code, confidence) tuples
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)  # Add batch dimension
            
        with torch.no_grad():
            results = self.forward(features)
            probs = results['language_probs'][0]  # Take first batch item
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, top_k)
            
            predictions = []
            for i in range(top_k):
                lang_code = self.idx_to_lang[top_indices[i].item()]
                confidence = top_probs[i].item()
                predictions.append((lang_code, confidence))
                
        return predictions
    
    def get_language_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Get language-agnostic feature embedding."""
        if features.dim() == 2:
            features = features.unsqueeze(0)
            
        with torch.no_grad():
            results = self.forward(features)
            return results['features'][0]  # Remove batch dimension

class MultiLingualTokenizer:
    """
    Multi-lingual tokenizer supporting 100+ languages with subword tokenization.
    """
    
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3,
            '<blank>': 4,  # For CTC
            '<mask>': 5,
        }
        
        # Language-specific tokens
        self.lang_tokens = {}
        for i, lang_code in enumerate(LANGUAGE_CODES.keys()):
            self.lang_tokens[f'<{lang_code}>'] = len(self.special_tokens) + i
        
        # Initialize vocabularies
        self.vocab = {}
        self.reverse_vocab = {}
        self.char_vocab = {}
        self.subword_vocab = {}
        
        # BPE (Byte Pair Encoding) for subword tokenization
        self.bpe_merges = {}
        self.bpe_vocab = {}
        
        self._initialize_base_vocab()
        
    def _initialize_base_vocab(self):
        """Initialize base vocabulary with special tokens."""
        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
            
        # Add language tokens
        for token, idx in self.lang_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
        
        # Add basic ASCII characters
        current_idx = len(self.special_tokens) + len(self.lang_tokens)
        for i in range(256):  # Extended ASCII
            char = chr(i)
            if char not in self.vocab:
                self.vocab[char] = current_idx
                self.reverse_vocab[current_idx] = char
                current_idx += 1
        
        # Add common Unicode characters for major languages
        unicode_chars = self._get_common_unicode_chars()
        for char in unicode_chars:
            if char not in self.vocab:
                self.vocab[char] = current_idx
                self.reverse_vocab[current_idx] = char
                current_idx += 1
    
    def _get_common_unicode_chars(self) -> List[str]:
        """Get common Unicode characters for multilingual support."""
        chars = []
        
        # Latin extended
        chars.extend([chr(i) for i in range(256, 384)])
        
        # Cyrillic
        chars.extend([chr(i) for i in range(1024, 1280)])
        
        # Arabic
        chars.extend([chr(i) for i in range(1536, 1792)])
        
        # Devanagari (Hindi, etc.)
        chars.extend([chr(i) for i in range(2304, 2432)])
        
        # Bengali
        chars.extend([chr(i) for i in range(2432, 2560)])
        
        # CJK (Chinese, Japanese, Korean) - subset
        chars.extend([chr(i) for i in range(12288, 12352)])  # CJK symbols
        chars.extend([chr(i) for i in range(19968, 20480)])  # Common CJK - subset
        
        # Thai
        chars.extend([chr(i) for i in range(3584, 3712)])
        
        return chars
    
    def train_bpe(self, texts: List[str], target_vocab_size: Optional[int] = None):
        """
        Train Byte Pair Encoding on the provided texts.
        
        Args:
            texts: List of training texts
            target_vocab_size: Target vocabulary size (uses self.vocab_size if None)
        """
        if target_vocab_size is None:
            target_vocab_size = self.vocab_size
            
        # Count character frequencies
        char_freq = Counter()
        for text in texts:
            # Normalize text
            normalized = self._normalize_text(text)
            char_freq.update(normalized)
        
        # Initialize with character vocabulary
        current_vocab = dict(self.vocab)  # Copy base vocab
        current_idx = max(current_vocab.values()) + 1
        
        # Add frequent characters not in base vocab
        for char, freq in char_freq.most_common():
            if freq >= self.min_frequency and char not in current_vocab:
                current_vocab[char] = current_idx
                current_idx += 1
                if len(current_vocab) >= target_vocab_size:
                    break
        
        # BPE training (simplified version)
        word_freqs = Counter()
        for text in texts:
            words = self._tokenize_words(text)
            word_freqs.update(words)
        
        # Convert words to character sequences
        word_chars = {}
        for word, freq in word_freqs.items():
            if freq >= self.min_frequency:
                word_chars[word] = list(word) + ['</w>']  # End of word marker
        
        # BPE merge operations
        merges = []
        while len(current_vocab) < target_vocab_size:
            # Count bigram frequencies
            bigram_freq = Counter()
            for word, chars in word_chars.items():
                freq = word_freqs[word]
                for i in range(len(chars) - 1):
                    bigram = (chars[i], chars[i + 1])
                    bigram_freq[bigram] += freq
            
            if not bigram_freq:
                break
                
            # Get most frequent bigram
            most_frequent_bigram = bigram_freq.most_common(1)[0][0]
            
            # Create new token
            new_token = most_frequent_bigram[0] + most_frequent_bigram[1]
            if new_token.endswith('</w>'):
                new_token = new_token[:-4]  # Remove end marker for vocab
            
            current_vocab[new_token] = current_idx
            current_idx += 1
            merges.append(most_frequent_bigram)
            
            # Update word representations
            new_word_chars = {}
            for word, chars in word_chars.items():
                new_chars = []
                i = 0
                while i < len(chars):
                    if (i < len(chars) - 1 and 
                        chars[i] == most_frequent_bigram[0] and 
                        chars[i + 1] == most_frequent_bigram[1]):
                        new_chars.append(new_token if not new_token.endswith('</w>') else chars[i] + chars[i + 1])
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                new_word_chars[word] = new_chars
            word_chars = new_word_chars
        
        # Update vocabularies
        self.vocab = current_vocab
        self.reverse_vocab = {idx: token for token, idx in current_vocab.items()}
        self.bpe_merges = merges
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Convert to lowercase (optional - might want to preserve case for some languages)
        # text = text.lower()
        
        return text
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Handle multiple scripts and languages
        normalized = self._normalize_text(text)
        
        # Split on whitespace and punctuation
        words = re.findall(r'\w+|[^\w\s]', normalized, re.UNICODE)
        
        return words
    
    def encode(self, text: str, language: Optional[str] = None, 
               max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            language: Language code (optional)
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        # Add language token if specified
        if language and f'<{language}>' in self.vocab:
            tokens.append(self.vocab[f'<{language}>'])
        
        # Add start token
        tokens.append(self.vocab['<sos>'])
        
        # Tokenize text
        normalized = self._normalize_text(text)
        words = self._tokenize_words(normalized)
        
        for word in words:
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)
        
        # Add end token
        tokens.append(self.vocab['<eos>'])
        
        # Truncate if needed
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.vocab['<eos>']]
        
        return tokens
    
    def _encode_word(self, word: str) -> List[int]:
        """Encode a single word using BPE."""
        if not word:
            return []
            
        # Start with character sequence
        chars = list(word) + ['</w>']
        
        # Apply BPE merges
        for merge in self.bpe_merges:
            new_chars = []
            i = 0
            while i < len(chars):
                if (i < len(chars) - 1 and 
                    chars[i] == merge[0] and 
                    chars[i + 1] == merge[1]):
                    new_token = merge[0] + merge[1]
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        
        # Convert to IDs
        token_ids = []
        for char in chars:
            if char == '</w>':
                continue  # Skip end marker
            if char in self.vocab:
                token_ids.append(self.vocab[char])
            else:
                # Handle unknown characters
                if len(char) == 1:
                    token_ids.append(self.vocab.get(char, self.vocab['<unk>']))
                else:
                    # Break down unknown subwords to characters
                    for c in char:
                        token_ids.append(self.vocab.get(c, self.vocab['<unk>']))
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                
                # Skip special tokens if requested
                if skip_special and (
                    token in self.special_tokens or 
                    token.startswith('<') and token.endswith('>')
                ):
                    continue
                    
                tokens.append(token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')  # Convert end markers to spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_language_vocab_info(self) -> Dict:
        """Get information about language support."""
        return {
            'total_languages': len(LANGUAGE_CODES),
            'supported_languages': list(LANGUAGE_CODES.keys()),
            'language_families': list(LANGUAGE_FAMILIES.keys()),
            'vocab_size': self.get_vocab_size(),
            'special_tokens': list(self.special_tokens.keys()),
            'language_tokens': list(self.lang_tokens.keys())
        }

class LanguageSpecificProcessor:
    """
    Language-specific audio processing adaptations.
    """
    
    def __init__(self):
        # Language-specific configurations
        self.language_configs = self._get_language_configs()
        
    def _get_language_configs(self) -> Dict:
        """Get language-specific processing configurations."""
        return {
            # Tonal languages need different pitch processing
            'tonal': {
                'languages': ['zh', 'th', 'vi', 'my'],
                'pitch_emphasis': 1.5,
                'tone_features': True,
                'f0_range': (80, 400)
            },
            
            # Languages with complex consonant clusters
            'consonant_heavy': {
                'languages': ['pl', 'cs', 'sk', 'ru', 'uk', 'bg'],
                'consonant_emphasis': 1.3,
                'formant_focus': [2, 3],  # F2, F3
                'noise_tolerance': 0.8
            },
            
            # Languages with rich vowel systems
            'vowel_rich': {
                'languages': ['fi', 'hu', 'et'],
                'vowel_emphasis': 1.4,
                'formant_focus': [1, 2],  # F1, F2
                'vowel_length_sensitive': True
            },
            
            # Semitic languages with pharyngeal sounds
            'semitic': {
                'languages': ['ar', 'he'],
                'pharyngeal_emphasis': 1.2,
                'low_freq_boost': 1.1,
                'emphasis_detection': True
            },
            
            # Languages with clicks and ejectives
            'click_languages': {
                'languages': ['zu', 'xh'],
                'transient_sensitivity': 1.8,
                'high_freq_emphasis': 1.3,
                'burst_detection': True
            },
            
            # Agglutinative languages
            'agglutinative': {
                'languages': ['tr', 'fi', 'hu', 'ja', 'ko'],
                'long_context': True,
                'morpheme_boundary_detection': True,
                'extended_memory': 1.5
            }
        }
    
    def get_language_adaptations(self, language: str) -> Dict:
        """Get processing adaptations for a specific language."""
        adaptations = {
            'pitch_emphasis': 1.0,
            'consonant_emphasis': 1.0,
            'vowel_emphasis': 1.0,
            'low_freq_boost': 1.0,
            'high_freq_emphasis': 1.0,
            'transient_sensitivity': 1.0,
            'noise_tolerance': 1.0,
            'tone_features': False,
            'vowel_length_sensitive': False,
            'emphasis_detection': False,
            'burst_detection': False,
            'long_context': False,
            'morpheme_boundary_detection': False,
            'extended_memory': 1.0
        }
        
        # Apply language-specific configurations
        for config_type, config in self.language_configs.items():
            if language in config['languages']:
                for key, value in config.items():
                    if key != 'languages':
                        adaptations[key] = value
        
        return adaptations
    
    def adapt_audio_features(self, features: torch.Tensor, language: str) -> torch.Tensor:
        """
        Apply language-specific adaptations to audio features.
        
        Args:
            features: Audio features (seq_len, feature_dim)
            language: Language code
            
        Returns:
            Adapted features
        """
        adaptations = self.get_language_adaptations(language)
        adapted_features = features.clone()
        
        # Apply frequency emphasis
        if adaptations['low_freq_boost'] != 1.0:
            # Boost low frequencies (first 1/3 of features)
            low_freq_end = features.shape[-1] // 3
            adapted_features[:, :low_freq_end] *= adaptations['low_freq_boost']
        
        if adaptations['high_freq_emphasis'] != 1.0:
            # Emphasize high frequencies (last 1/3 of features)
            high_freq_start = (features.shape[-1] * 2) // 3
            adapted_features[:, high_freq_start:] *= adaptations['high_freq_emphasis']
        
        # Apply pitch emphasis for tonal languages
        if adaptations['tone_features']:
            # Emphasize pitch-related features (assuming they're in middle range)
            pitch_start = features.shape[-1] // 3
            pitch_end = (features.shape[-1] * 2) // 3
            adapted_features[:, pitch_start:pitch_end] *= adaptations['pitch_emphasis']
        
        return adapted_features

class CrossLingualTransferManager:
    """
    Manages cross-lingual transfer learning for speech recognition.
    """
    
    def __init__(self):
        self.transfer_matrix = self._build_transfer_matrix()
        
    def _build_transfer_matrix(self) -> torch.Tensor:
        """Build language similarity matrix for transfer learning."""
        languages = list(LANGUAGE_CODES.keys())
        num_langs = len(languages)
        similarity_matrix = torch.eye(num_langs)  # Start with identity
        
        # Add family similarities
        for family, family_langs in LANGUAGE_FAMILIES.items():
            family_indices = [i for i, lang in enumerate(languages) if lang in family_langs]
            
            # Languages in same family have higher similarity
            for i in family_indices:
                for j in family_indices:
                    if i != j:
                        similarity_matrix[i, j] = 0.7
        
        # Add specific high-similarity pairs
        high_similarity_pairs = [
            ('en', 'de'), ('en', 'nl'), ('en', 'sv'), ('en', 'no'),
            ('es', 'pt'), ('es', 'it'), ('es', 'fr'),
            ('zh', 'ja'), ('zh', 'ko'),
            ('ru', 'uk'), ('ru', 'be'), ('ru', 'bg'),
            ('hi', 'ur'), ('hi', 'ne'),
            ('ar', 'he'), ('ar', 'fa'),
            ('id', 'ms'), ('id', 'tl')
        ]
        
        lang_to_idx = {lang: i for i, lang in enumerate(languages)}
        
        for lang1, lang2 in high_similarity_pairs:
            if lang1 in lang_to_idx and lang2 in lang_to_idx:
                i, j = lang_to_idx[lang1], lang_to_idx[lang2]
                similarity_matrix[i, j] = 0.8
                similarity_matrix[j, i] = 0.8
        
        return similarity_matrix
    
    def get_transfer_languages(self, target_language: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get languages most suitable for transfer learning to target language.
        
        Args:
            target_language: Target language code
            top_k: Number of source languages to return
            
        Returns:
            List of (source_language, similarity_score) tuples
        """
        languages = list(LANGUAGE_CODES.keys())
        
        if target_language not in languages:
            return []
        
        target_idx = languages.index(target_language)
        similarities = self.transfer_matrix[target_idx]
        
        # Get top-k most similar languages (excluding target itself)
        similarities[target_idx] = -1  # Exclude self
        top_indices = torch.topk(similarities, top_k).indices
        
        transfer_langs = []
        for idx in top_indices:
            source_lang = languages[idx]
            similarity = similarities[idx].item()
            transfer_langs.append((source_lang, similarity))
        
        return transfer_langs
    
    def compute_transfer_weights(self, source_language: str, target_language: str) -> float:
        """Compute transfer learning weight between two languages."""
        languages = list(LANGUAGE_CODES.keys())
        
        if source_language not in languages or target_language not in languages:
            return 0.0
        
        source_idx = languages.index(source_language)
        target_idx = languages.index(target_language)
        
        return self.transfer_matrix[source_idx, target_idx].item()

# Factory functions
def create_language_detector(feature_dim: int = 80) -> LanguageDetector:
    """Create a language detector model."""
    return LanguageDetector(
        num_languages=len(LANGUAGE_CODES),
        feature_dim=feature_dim,
        hidden_dim=256
    )

def create_multilingual_tokenizer(vocab_size: int = 32000) -> MultiLingualTokenizer:
    """Create a multi-lingual tokenizer."""
    return MultiLingualTokenizer(vocab_size=vocab_size, min_frequency=2)

def get_supported_languages() -> Dict[str, str]:
    """Get all supported languages."""
    return LANGUAGE_CODES.copy()

def get_language_families() -> Dict[str, List[str]]:
    """Get language family mappings."""
    return LANGUAGE_FAMILIES.copy()

if __name__ == "__main__":
    # Example usage
    print("🌍 Multi-Lingual Language Processing System")
    print(f"Supported languages: {len(LANGUAGE_CODES)}")
    
    # Create language detector
    detector = create_language_detector()
    print(f"Language detector created with {detector.num_languages} languages")
    
    # Create tokenizer
    tokenizer = create_multilingual_tokenizer()
    print(f"Tokenizer created with base vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Example text encoding
    test_texts = [
        ("Hello world", "en"),
        ("Hola mundo", "es"),
        ("Bonjour le monde", "fr"),
        ("مرحبا بالعالم", "ar"),
        ("नमस्ते दुनिया", "hi"),
        ("你好世界", "zh")
    ]
    
    for text, lang in test_texts:
        encoded = tokenizer.encode(text, language=lang)
        decoded = tokenizer.decode(encoded)
        print(f"[{lang}] '{text}' -> {len(encoded)} tokens -> '{decoded}'")
    
    # Language processor
    processor = LanguageSpecificProcessor()
    
    # Transfer learning manager
    transfer_manager = CrossLingualTransferManager()
    
    # Example transfer learning suggestions
    target_lang = "hi"  # Hindi
    transfer_langs = transfer_manager.get_transfer_languages(target_lang)
    print(f"\nBest source languages for {target_lang}:")
    for source_lang, similarity in transfer_langs:
        lang_name = LANGUAGE_CODES[source_lang]
        print(f"  {source_lang} ({lang_name}): {similarity:.3f}")
    
    print("\n✅ Multi-lingual system initialized successfully!")