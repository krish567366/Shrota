"""
Custom Lightweight CNN-RNN Hybrid for Multi-Channel Speech Recognition

Built from scratch for efficient real-time processing with multi-channel
audio support and optimized for edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

class MultiChannelCNNFeatureExtractor(nn.Module):
    """
    Multi-channel CNN feature extractor optimized for speech signals.
    Extracts local patterns from each channel independently.
    """
    
    def __init__(self, num_channels: int, output_dim: int = 256):
        super().__init__()
        self.num_channels = num_channels
        self.output_dim = output_dim
        
        # Per-channel CNN layers
        self.channel_cnns = nn.ModuleList([
            nn.Sequential(
                # First conv block
                nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                
                # Second conv block
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                
                # Third conv block
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                
                # Fourth conv block
                nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            ) for _ in range(num_channels)
        ])
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.Linear(128 * num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_channels),
            nn.Sigmoid()
        )
        
        # Channel fusion
        self.channel_fusion = nn.Sequential(
            nn.Linear(128 * num_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Multi-channel audio (batch_size, num_channels, seq_len)
        
        Returns:
            Features (batch_size, reduced_seq_len, output_dim)
        """
        batch_size, num_channels, seq_len = audio.shape
        
        # Extract features from each channel
        channel_features = []
        for ch in range(num_channels):
            ch_audio = audio[:, ch:ch+1, :]  # (batch_size, 1, seq_len)
            ch_features = self.channel_cnns[ch](ch_audio)  # (batch_size, 128, reduced_seq_len)
            channel_features.append(ch_features)
        
        # Get reduced sequence length
        reduced_seq_len = channel_features[0].shape[-1]
        
        # Stack channel features
        stacked_features = torch.stack(channel_features, dim=1)  # (batch_size, num_channels, 128, reduced_seq_len)
        
        # Compute channel attention weights
        # Global average pooling across time
        pooled_features = torch.mean(stacked_features, dim=-1)  # (batch_size, num_channels, 128)
        pooled_features = pooled_features.view(batch_size, -1)  # (batch_size, num_channels * 128)
        
        attention_weights = self.channel_attention(pooled_features)  # (batch_size, num_channels)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # (batch_size, num_channels, 1, 1)
        
        # Apply attention weights
        weighted_features = stacked_features * attention_weights  # (batch_size, num_channels, 128, reduced_seq_len)
        
        # Concatenate channels and transpose
        concat_features = weighted_features.view(batch_size, num_channels * 128, reduced_seq_len)
        concat_features = concat_features.transpose(1, 2)  # (batch_size, reduced_seq_len, num_channels * 128)
        
        # Channel fusion
        fused_features = self.channel_fusion(concat_features)  # (batch_size, reduced_seq_len, output_dim)
        
        return fused_features

class BiDirectionalGRU(nn.Module):
    """
    Bidirectional GRU with layer normalization and residual connections.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Projection layer (if input_dim != hidden_dim * 2)
        if input_dim != hidden_dim * 2:
            self.projection = nn.Linear(input_dim, hidden_dim * 2)
        else:
            self.projection = None
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            lengths: Sequence lengths (batch_size,)
        
        Returns:
            Output features (batch_size, seq_len, hidden_dim * 2)
        """
        residual = x
        
        # Pack sequences if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # GRU forward pass
        gru_output, _ = self.gru(x)
        
        # Unpack sequences
        if lengths is not None:
            gru_output, _ = nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
        
        # Layer normalization
        output = self.layer_norm(gru_output)
        
        # Residual connection with projection if needed
        if self.projection is not None:
            residual = self.projection(residual)
        
        if residual.shape == output.shape:
            output = output + residual
        
        output = self.dropout(output)
        
        return output

class LanguageEmbedding(nn.Module):
    """
    Language embedding module for multi-lingual support.
    """
    
    def __init__(self, num_languages: int, d_model: int):
        super().__init__()
        self.language_embedding = nn.Embedding(num_languages, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, features: torch.Tensor, language_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features (batch_size, seq_len, d_model)
            language_ids: Language IDs (batch_size,)
        
        Returns:
            Features with language embedding (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = features.shape
        
        # Get language embeddings
        lang_emb = self.language_embedding(language_ids)  # (batch_size, d_model)
        lang_emb = lang_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
        
        # Add to features
        enhanced_features = features + lang_emb
        enhanced_features = self.layer_norm(enhanced_features)
        
        return enhanced_features

class CTCOutputHead(nn.Module):
    """
    CTC output head for sequence-to-sequence alignment-free training.
    """
    
    def __init__(self, input_dim: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Output projection layers
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, vocab_size + 1)  # +1 for CTC blank token
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features (batch_size, seq_len, input_dim)
        
        Returns:
            Log probabilities (batch_size, seq_len, vocab_size + 1)
        """
        logits = self.projection(features)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

class CustomLightweightCNNRNN(nn.Module):
    """
    Custom Lightweight CNN-RNN Hybrid for Multi-Channel Speech Recognition.
    
    Optimized for:
    - Real-time processing
    - Multi-channel audio
    - Edge deployment
    - Low latency inference
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.num_channels = config.get('num_channels', 1)
        self.cnn_output_dim = config.get('cnn_output_dim', 256)
        self.rnn_hidden_dim = config.get('rnn_hidden_dim', 256)
        self.rnn_num_layers = config.get('rnn_num_layers', 2)
        self.vocab_size = config.get('vocab_size', 32000)
        self.num_languages = config.get('num_languages', 100)
        self.dropout = config.get('dropout', 0.1)
        
        # Multi-channel CNN feature extractor
        self.cnn_encoder = MultiChannelCNNFeatureExtractor(
            self.num_channels, self.cnn_output_dim
        )
        
        # Bidirectional GRU layers
        self.rnn_encoder = BiDirectionalGRU(
            self.cnn_output_dim, self.rnn_hidden_dim, 
            self.rnn_num_layers, self.dropout
        )
        
        # Language embedding
        self.language_embedding = LanguageEmbedding(
            self.num_languages, self.rnn_hidden_dim * 2
        )
        
        # CTC output head
        self.output_head = CTCOutputHead(
            self.rnn_hidden_dim * 2, self.vocab_size, self.dropout
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, audio: torch.Tensor, language_ids: Optional[torch.Tensor] = None,
                audio_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for speech recognition.
        
        Args:
            audio: Multi-channel audio (batch_size, num_channels, seq_len)
            language_ids: Language IDs (batch_size,)
            audio_lengths: Audio sequence lengths (batch_size,)
        
        Returns:
            Log probabilities (batch_size, time_steps, vocab_size + 1)
        """
        batch_size = audio.shape[0]
        
        # CNN feature extraction
        cnn_features = self.cnn_encoder(audio)  # (batch_size, reduced_seq_len, cnn_output_dim)
        
        # Calculate reduced sequence lengths for RNN
        if audio_lengths is not None:
            # CNN reduces sequence length by factor of 8 (2*2*2 from three stride-2 convs)
            reduced_lengths = (audio_lengths // 8).clamp(min=1, max=cnn_features.shape[1])
        else:
            reduced_lengths = None
        
        # RNN encoding
        rnn_features = self.rnn_encoder(cnn_features, reduced_lengths)  # (batch_size, reduced_seq_len, rnn_hidden_dim * 2)
        
        # Add language embedding if provided
        if language_ids is not None:
            rnn_features = self.language_embedding(rnn_features, language_ids)
        
        # CTC output projection
        log_probs = self.output_head(rnn_features)  # (batch_size, reduced_seq_len, vocab_size + 1)
        
        return log_probs
    
    def transcribe(self, audio: torch.Tensor, language_id: Optional[int] = None) -> List[str]:
        """
        Transcribe audio to text using greedy CTC decoding.
        
        Args:
            audio: Audio tensor (batch_size, num_channels, seq_len)
            language_id: Target language ID
        
        Returns:
            List of transcriptions
        """
        self.eval()
        with torch.no_grad():
            if language_id is not None:
                language_ids = torch.tensor([language_id] * audio.shape[0], 
                                          dtype=torch.long, device=audio.device)
            else:
                language_ids = None
            
            log_probs = self.forward(audio, language_ids)
            
            # Greedy CTC decoding
            predictions = torch.argmax(log_probs, dim=-1)  # (batch_size, time_steps)
            
            return self._decode_predictions(predictions)
    
    def _decode_predictions(self, predictions: torch.Tensor) -> List[str]:
        """Decode CTC predictions to text."""
        batch_transcriptions = []
        blank_id = self.vocab_size  # CTC blank token
        
        for batch_idx in range(predictions.shape[0]):
            pred_seq = predictions[batch_idx].cpu().numpy()
            
            # CTC decoding: remove blanks and consecutive duplicates
            decoded = []
            prev_token = None
            for token in pred_seq:
                if token != blank_id and token != prev_token:
                    decoded.append(token)
                prev_token = token
            
            # Convert to text (placeholder)
            transcription = self._tokens_to_text(decoded)
            batch_transcriptions.append(transcription)
        
        return batch_transcriptions
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert tokens to text (placeholder)."""
        return " ".join([f"token_{t}" for t in tokens])
    
    def get_model_size(self) -> float:
        """Get model size in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / 1024 / 1024  # 4 bytes per float32 parameter
    
    def get_inference_time(self, audio_duration: float, device: str = 'cpu') -> Dict[str, float]:
        """
        Estimate inference time for given audio duration.
        
        Args:
            audio_duration: Duration in seconds
            device: Device type ('cpu' or 'cuda')
        
        Returns:
            Dictionary with timing estimates
        """
        # These are rough estimates based on model complexity
        if device == 'cpu':
            # CPU inference times (rough estimates)
            cnn_time = audio_duration * 0.05  # 50ms per second of audio
            rnn_time = audio_duration * 0.03  # 30ms per second of audio
            total_time = cnn_time + rnn_time
        else:
            # GPU inference times (much faster)
            cnn_time = audio_duration * 0.01  # 10ms per second of audio
            rnn_time = audio_duration * 0.005  # 5ms per second of audio
            total_time = cnn_time + rnn_time
        
        return {
            'cnn_time': cnn_time,
            'rnn_time': rnn_time,
            'total_time': total_time,
            'real_time_factor': audio_duration / total_time if total_time > 0 else float('inf')
        }

def create_lightweight_cnn_rnn_model(config: Dict) -> CustomLightweightCNNRNN:
    """
    Factory function to create a lightweight CNN-RNN model.
    
    Args:
        config: Model configuration
    
    Returns:
        Initialized model
    """
    model = CustomLightweightCNNRNN(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = model.get_model_size()
    
    print(f"Custom Lightweight CNN-RNN Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size_mb:.1f} MB")
    print(f"  Architecture: {config['num_channels']}-channel CNN + {config['rnn_num_layers']}-layer BiGRU")
    
    # Performance estimates
    cpu_timing = model.get_inference_time(1.0, 'cpu')
    gpu_timing = model.get_inference_time(1.0, 'cuda')
    
    print(f"  Estimated CPU inference: {cpu_timing['total_time']*1000:.1f}ms per 1s audio")
    print(f"  Estimated GPU inference: {gpu_timing['total_time']*1000:.1f}ms per 1s audio")
    print(f"  Real-time factor (GPU): {gpu_timing['real_time_factor']:.1f}x")
    
    return model

# Example usage
if __name__ == "__main__":
    # Lightweight model configuration
    config = {
        'num_channels': 2,          # Stereo input
        'cnn_output_dim': 256,      # CNN feature dimension
        'rnn_hidden_dim': 256,      # RNN hidden dimension
        'rnn_num_layers': 2,        # Number of RNN layers
        'vocab_size': 32000,        # Vocabulary size
        'num_languages': 100,       # Number of supported languages
        'dropout': 0.1              # Dropout rate
    }
    
    # Create model
    model = create_lightweight_cnn_rnn_model(config)
    
    # Example forward pass
    batch_size = 2
    num_channels = 2
    audio_length = 16000 * 3  # 3 seconds at 16kHz
    
    # Dummy input
    audio = torch.randn(batch_size, num_channels, audio_length)
    language_ids = torch.tensor([0, 1])
    audio_lengths = torch.tensor([audio_length, audio_length // 2])
    
    # Forward pass
    log_probs = model(audio, language_ids, audio_lengths)
    print(f"Output shape: {log_probs.shape}")
    
    # Transcription
    transcriptions = model.transcribe(audio[:1], language_id=0)
    print(f"Transcription: {transcriptions[0]}")
    
    # Performance analysis
    timing = model.get_inference_time(3.0, 'cuda')
    print(f"3-second audio inference time: {timing['total_time']*1000:.1f}ms")
    print(f"Real-time factor: {timing['real_time_factor']:.1f}x")