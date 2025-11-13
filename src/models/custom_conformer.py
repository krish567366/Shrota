"""
Custom Conformer-Inspired Architecture for Multi-Channel Speech Recognition

Built from scratch combining CNN and Transformer blocks for optimal
speech recognition performance with multi-channel audio support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
import numpy as np

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for efficient processing.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.bn1(self.depthwise(x))
        x = F.relu(x)
        x = self.bn2(self.pointwise(x))
        return x

class ConvolutionModule(nn.Module):
    """
    Convolution module in Conformer architecture.
    Captures local patterns in audio sequences.
    """
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        # Layernorm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise convolution 1
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        
        # GLU activation
        self.glu = nn.GLU(dim=1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model)
        
        # Swish activation
        self.swish = nn.SiLU()
        
        # Pointwise convolution 2
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Layer normalization
        residual = x
        x = self.layer_norm(x)
        
        # Transpose for convolution (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)
        
        # Pointwise convolution 1 + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        
        # Pointwise convolution 2
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)
        
        # Residual connection
        return residual + x

class FeedForwardModule(nn.Module):
    """
    Feed-forward module with swish activation.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + x

class MultiHeadSelfAttentionModule(nn.Module):
    """
    Multi-head self-attention module with relative positional encoding.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative positional encoding
        self.relative_attention_bias = nn.Parameter(torch.randn(num_heads, 32, 32))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative positional bias
        if seq_len <= 32:
            rel_bias = self.relative_attention_bias[:, :seq_len, :seq_len]
            scores = scores + rel_bias.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        output = self.dropout(output)
        
        # Residual connection
        return residual + output

class ConformerBlock(nn.Module):
    """
    Single Conformer block combining feed-forward, attention, and convolution.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 conv_kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        # First feed-forward module (half-step residual)
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        
        # Convolution module
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Second feed-forward module (half-step residual)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First feed-forward (half-step)
        x = x + 0.5 * self.ff1(x)
        
        # Multi-head self-attention
        x = self.attention(x, mask)
        
        # Convolution module
        x = self.conv(x)
        
        # Second feed-forward (half-step)
        x = x + 0.5 * self.ff2(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x

class MultiChannelAudioSubsampling(nn.Module):
    """
    Multi-channel audio subsampling using strided convolutions.
    Reduces sequence length while extracting features.
    """
    
    def __init__(self, num_channels: int, d_model: int):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        
        # Per-channel subsampling
        self.channel_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ) for _ in range(num_channels)
        ])
        
        # Channel fusion
        if num_channels > 1:
            self.channel_fusion = nn.Linear(128 * num_channels, d_model)
        else:
            self.channel_fusion = nn.Linear(128, d_model)
        
        # Position-wise feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: (batch_size, num_channels, seq_len)
        Returns:
            features: (batch_size, reduced_seq_len, d_model)
            lengths: (batch_size,) - lengths after subsampling
        """
        batch_size, num_channels, seq_len = audio.shape
        
        # Process each channel
        channel_features = []
        for ch in range(num_channels):
            ch_audio = audio[:, ch:ch+1, :]  # (batch_size, 1, seq_len)
            ch_features = self.channel_convs[ch](ch_audio)  # (batch_size, 128, reduced_seq_len)
            channel_features.append(ch_features)
        
        # Get reduced sequence length
        reduced_seq_len = channel_features[0].shape[-1]
        
        # Concatenate and transpose
        if num_channels > 1:
            combined_features = torch.cat(channel_features, dim=1)  # (batch_size, 128*num_channels, reduced_seq_len)
        else:
            combined_features = channel_features[0]
        
        combined_features = combined_features.transpose(1, 2)  # (batch_size, reduced_seq_len, 128*num_channels)
        
        # Channel fusion
        features = self.channel_fusion(combined_features)  # (batch_size, reduced_seq_len, d_model)
        
        # Position-wise feed-forward
        features = self.ff(features)
        
        # Calculate new sequence lengths (reduced by factor of 4 due to 2x stride twice)
        lengths = torch.full((batch_size,), reduced_seq_len, dtype=torch.long, device=audio.device)
        
        return features, lengths

class CustomConformerModel(nn.Module):
    """
    Custom Conformer-inspired model for multi-channel, multi-lingual speech recognition.
    Built from scratch with optimizations for streaming and multi-channel audio.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.num_channels = config.get('num_channels', 1)
        self.d_model = config.get('d_model', 512)
        self.num_heads = config.get('num_heads', 8)
        self.num_blocks = config.get('num_blocks', 16)
        self.d_ff = config.get('d_ff', 2048)
        self.conv_kernel_size = config.get('conv_kernel_size', 31)
        self.vocab_size = config.get('vocab_size', 32000)
        self.dropout = config.get('dropout', 0.1)
        
        # Audio subsampling
        self.subsampling = MultiChannelAudioSubsampling(self.num_channels, self.d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                self.d_model, self.num_heads, self.d_ff, 
                self.conv_kernel_size, self.dropout
            ) for _ in range(self.num_blocks)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size + 1)  # +1 for CTC blank
        
        # Language embedding
        self.language_embedding = nn.Embedding(100, self.d_model)  # 100 languages
        
        # Initialize weights
        self._init_weights()
        
    def _create_positional_encoding(self, max_len: int = 5000):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, audio: torch.Tensor, language_ids: Optional[torch.Tensor] = None,
                audio_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for speech recognition.
        
        Args:
            audio: Raw audio (batch_size, num_channels, seq_len)
            language_ids: Language IDs (batch_size,)
            audio_lengths: Audio lengths (batch_size,)
        
        Returns:
            Log probabilities (batch_size, time_steps, vocab_size + 1)
        """
        batch_size = audio.shape[0]
        
        # Audio subsampling and feature extraction
        features, feature_lengths = self.subsampling(audio)  # (batch_size, reduced_seq_len, d_model)
        seq_len = features.shape[1]
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.shape[1]:
            features = features + self.pos_encoding[:, :seq_len, :]
        
        # Add language embedding
        if language_ids is not None:
            lang_emb = self.language_embedding(language_ids)  # (batch_size, d_model)
            lang_emb = lang_emb.unsqueeze(1).expand(-1, seq_len, -1)
            features = features + lang_emb
        
        # Create attention mask
        mask = None
        if audio_lengths is not None:
            # Adjust lengths for subsampling (reduced by factor of 4)
            adjusted_lengths = (audio_lengths // 4).clamp(max=seq_len)
            mask = torch.arange(seq_len, device=audio.device).expand(
                batch_size, seq_len
            ) < adjusted_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(1)  # For multi-head attention
        
        # Pass through conformer blocks
        x = features
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        # Output projection
        log_probs = F.log_softmax(self.output_projection(x), dim=-1)
        
        return log_probs
    
    def transcribe(self, audio: torch.Tensor, language_id: Optional[int] = None,
                   beam_size: int = 1) -> List[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio tensor (batch_size, num_channels, seq_len)
            language_id: Target language ID
            beam_size: Beam search size (1 for greedy)
        
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
            
            if beam_size == 1:
                # Greedy decoding
                predictions = torch.argmax(log_probs, dim=-1)
                return self._decode_predictions(predictions)
            else:
                # Beam search (placeholder)
                return self._beam_search_decode(log_probs, beam_size)
    
    def _decode_predictions(self, predictions: torch.Tensor) -> List[str]:
        """Decode CTC predictions to text."""
        batch_transcriptions = []
        blank_id = self.vocab_size
        
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
    
    def _beam_search_decode(self, log_probs: torch.Tensor, beam_size: int) -> List[str]:
        """Beam search decoding (placeholder implementation)."""
        # This would implement proper beam search CTC decoding
        # For now, fall back to greedy
        predictions = torch.argmax(log_probs, dim=-1)
        return self._decode_predictions(predictions)
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert tokens to text (placeholder)."""
        return " ".join([f"token_{t}" for t in tokens])

def create_custom_conformer_model(config: Dict) -> CustomConformerModel:
    """
    Factory function to create a custom Conformer model.
    
    Args:
        config: Model configuration
    
    Returns:
        Initialized Conformer model
    """
    model = CustomConformerModel(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Custom Conformer Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model

# Example usage
if __name__ == "__main__":
    # Model configuration
    config = {
        'num_channels': 2,          # Stereo input
        'd_model': 256,             # Smaller for faster training
        'num_heads': 8,             # Attention heads
        'num_blocks': 12,           # Conformer blocks
        'd_ff': 1024,              # Feed-forward dimension
        'conv_kernel_size': 31,     # Convolution kernel size
        'vocab_size': 32000,       # Vocabulary size
        'dropout': 0.1             # Dropout rate
    }
    
    # Create model
    model = create_custom_conformer_model(config)
    
    # Example forward pass
    batch_size = 2
    num_channels = 2
    audio_length = 16000 * 5  # 5 seconds at 16kHz
    
    # Dummy input
    audio = torch.randn(batch_size, num_channels, audio_length)
    language_ids = torch.tensor([0, 1])  # Different languages
    audio_lengths = torch.tensor([audio_length, audio_length // 2])
    
    # Forward pass
    log_probs = model(audio, language_ids, audio_lengths)
    print(f"Output shape: {log_probs.shape}")
    
    # Transcription
    transcriptions = model.transcribe(audio[:1], language_id=0)
    print(f"Transcription: {transcriptions[0]}")