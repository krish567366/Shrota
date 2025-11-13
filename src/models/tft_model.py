"""
Temporal Fusion Transformer (TFT) implementation for predictive maintenance.
Optimized for interpretable time-series forecasting with attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_model import BasePredictor, AttentionModule, PositionalEncoding, get_activation_function

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for controlling information flow."""
    
    def __init__(self, input_size: int, hidden_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GLU."""
        x = self.linear(x)
        x = self.dropout(x)
        
        # Split into two parts for gating
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) component of TFT."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: Optional[int] = None,
                 context_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        
        if output_size is None:
            output_size = input_size
            
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        
        # Linear transformations
        self.linear_hidden = nn.Linear(input_size, hidden_size)
        
        if context_size is not None:
            self.context_linear = nn.Linear(context_size, hidden_size, bias=False)
        
        # GLU for non-linear processing
        self.glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        
        # Output projection
        self.linear_out = nn.Linear(hidden_size, output_size)
        
        # Skip connection (if input and output sizes differ)
        self.skip_connection = None
        if input_size != output_size:
            self.skip_connection = nn.Linear(input_size, output_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GRN."""
        # Linear transformation
        hidden = self.linear_hidden(x)
        
        # Add context if provided
        if context is not None and self.context_size is not None:
            hidden = hidden + self.context_linear(context)
        
        # Apply ELU activation
        hidden = F.elu(hidden)
        
        # Apply GLU
        hidden = self.glu(hidden)
        
        # Output projection
        output = self.linear_out(hidden)
        
        # Skip connection
        if self.skip_connection is not None:
            skip = self.skip_connection(x)
        else:
            skip = x
        
        # Add and normalize
        output = self.layer_norm(output + skip)
        
        return output

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature selection in TFT."""
    
    def __init__(self, input_size: int, num_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # Feature-specific GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout=dropout)
            for _ in range(num_features)
        ])
        
        # Variable selection weights
        self.selection_grn = GatedResidualNetwork(
            input_size * num_features, hidden_size, num_features, dropout=dropout
        )
        
    def forward(self, flattened_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through variable selection network.
        
        Args:
            flattened_embedding: Flattened embeddings of shape (batch_size, seq_len, input_size * num_features)
            
        Returns:
            selected_features: Selected features (batch_size, seq_len, hidden_size)
            selection_weights: Variable selection weights (batch_size, seq_len, num_features)
        """
        batch_size, seq_len, _ = flattened_embedding.shape
        
        # Reshape to separate features
        features = flattened_embedding.view(batch_size, seq_len, self.num_features, self.input_size)
        
        # Apply feature-specific transformations
        transformed_features = []
        for i, grn in enumerate(self.feature_grns):
            feature = features[:, :, i, :]  # (batch_size, seq_len, input_size)
            transformed = grn(feature)  # (batch_size, seq_len, hidden_size)
            transformed_features.append(transformed)
        
        transformed_features = torch.stack(transformed_features, dim=2)  # (batch_size, seq_len, num_features, hidden_size)
        
        # Calculate selection weights
        selection_weights = self.selection_grn(flattened_embedding)  # (batch_size, seq_len, num_features)
        selection_weights = F.softmax(selection_weights, dim=-1)
        
        # Apply selection weights
        selection_weights = selection_weights.unsqueeze(-1)  # (batch_size, seq_len, num_features, 1)
        selected_features = (transformed_features * selection_weights).sum(dim=2)  # (batch_size, seq_len, hidden_size)
        
        return selected_features, selection_weights.squeeze(-1)

class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention with attention weights output."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = AttentionModule(hidden_size, num_heads, dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention weights."""
        output, attention_weights = self.attention(query, key, value, mask)
        return output, attention_weights

class TemporalFusionTransformer(BasePredictor):
    """
    Temporal Fusion Transformer for predictive maintenance.
    
    Features:
    - Variable selection for interpretability
    - Static and temporal feature processing
    - Multi-head attention for temporal dependencies
    - Quantile prediction for uncertainty estimation
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        model_config = self.model_config
        
        # Model dimensions
        self.hidden_size = model_config['hidden_size']
        self.num_heads = model_config.get('attention_head_size', 4)
        self.dropout = model_config.get('dropout', 0.1)
        self.max_encoder_length = model_config['max_encoder_length']
        self.max_prediction_length = model_config['max_prediction_length']
        
        # Feature dimensions
        self.static_reals = model_config.get('static_reals', [])
        self.static_categoricals = model_config.get('static_categoricals', [])
        self.time_varying_known_reals = model_config.get('time_varying_known_reals', [])
        self.time_varying_known_categoricals = model_config.get('time_varying_known_categoricals', [])
        self.time_varying_unknown_reals = model_config.get('time_varying_unknown_reals', [])
        
        # Calculate feature sizes
        self.num_static_reals = len(self.static_reals)
        self.num_static_categoricals = len(self.static_categoricals)
        self.num_time_varying_known = len(self.time_varying_known_reals) + len(self.time_varying_known_categoricals)
        self.num_time_varying_unknown = len(self.time_varying_unknown_reals)
        
        # Embedding layers
        self.static_embedding = nn.Linear(self.num_static_reals + self.num_static_categoricals, self.hidden_size)
        self.time_varying_embedding = nn.Linear(
            self.num_time_varying_known + self.num_time_varying_unknown, self.hidden_size
        )
        
        # Variable selection networks
        self.static_vsn = VariableSelectionNetwork(
            self.hidden_size, 1, self.hidden_size, self.dropout
        )
        
        self.encoder_vsn = VariableSelectionNetwork(
            self.hidden_size, self.num_time_varying_known + self.num_time_varying_unknown,
            self.hidden_size, self.dropout
        )
        
        self.decoder_vsn = VariableSelectionNetwork(
            self.hidden_size, self.num_time_varying_known,
            self.hidden_size, self.dropout
        )
        
        # LSTM layers for temporal processing
        self.encoder_lstm = nn.LSTM(
            self.hidden_size, self.hidden_size, 
            model_config.get('lstm_layers', 2),
            batch_first=True, dropout=self.dropout if model_config.get('lstm_layers', 2) > 1 else 0
        )
        
        self.decoder_lstm = nn.LSTM(
            self.hidden_size, self.hidden_size,
            model_config.get('lstm_layers', 2),
            batch_first=True, dropout=self.dropout if model_config.get('lstm_layers', 2) > 1 else 0
        )
        
        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size,
            context_size=self.hidden_size, dropout=self.dropout
        )
        
        # Multi-head attention
        self.self_attention = InterpretableMultiHeadAttention(
            self.hidden_size, self.num_heads, self.dropout
        )
        
        # Position-wise feed forward
        self.positionwise_grn = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, dropout=self.dropout
        )
        
        # Output layers
        self.output_size = model_config.get('output_size', 1)
        self.quantiles = model_config.get('quantiles', [0.1, 0.5, 0.9])
        
        if self.quantiles:
            # Quantile prediction for uncertainty
            self.output_layer = nn.Linear(self.hidden_size, len(self.quantiles))
        else:
            self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
        self.last_variable_selection_weights = None
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, static_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through TFT.
        
        Args:
            x: Time series data (batch_size, seq_len, num_features)
            static_features: Static features (batch_size, num_static_features)
            
        Returns:
            predictions: Model predictions (batch_size, prediction_length, output_size)
        """
        batch_size, seq_len, num_features = x.shape
        
        # Split encoder and decoder sequences
        encoder_length = self.max_encoder_length
        decoder_length = seq_len - encoder_length
        
        encoder_x = x[:, :encoder_length, :]
        decoder_x = x[:, encoder_length:, :] if decoder_length > 0 else None
        
        # Process static features
        if static_features is not None:
            static_embedding = self.static_embedding(static_features)
            static_embedding = static_embedding.unsqueeze(1)  # Add time dimension
            
            # Static variable selection
            static_selected, static_weights = self.static_vsn(
                static_embedding.expand(-1, encoder_length, -1).reshape(batch_size, encoder_length, -1)
            )
            static_context = static_selected.mean(dim=1)  # Average over time
        else:
            static_context = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Encoder processing
        encoder_embedding = self.time_varying_embedding(encoder_x)
        
        # Encoder variable selection
        encoder_selected, encoder_var_weights = self.encoder_vsn(
            encoder_embedding.reshape(batch_size, encoder_length, -1)
        )
        
        # LSTM encoding
        encoder_output, (h_n, c_n) = self.encoder_lstm(encoder_selected)
        
        # Decoder processing (if available)
        if decoder_x is not None:
            decoder_embedding = self.time_varying_embedding(decoder_x)
            decoder_selected, decoder_var_weights = self.decoder_vsn(
                decoder_embedding.reshape(batch_size, decoder_length, -1)
            )
            
            # LSTM decoding
            decoder_output, _ = self.decoder_lstm(decoder_selected, (h_n, c_n))
            
            # Combine encoder and decoder outputs
            lstm_output = torch.cat([encoder_output, decoder_output], dim=1)
        else:
            lstm_output = encoder_output
        
        # Static enrichment
        enriched_output = self.static_enrichment(
            lstm_output, static_context.unsqueeze(1).expand(-1, lstm_output.size(1), -1)
        )
        
        # Self-attention
        attended_output, attention_weights = self.self_attention(
            enriched_output, enriched_output, enriched_output
        )
        
        # Position-wise processing
        processed_output = self.positionwise_grn(attended_output)
        
        # Final predictions
        predictions = self.output_layer(processed_output)
        
        # Store attention weights for interpretability
        self.last_attention_weights = attention_weights.detach()
        if 'encoder_var_weights' in locals():
            self.last_variable_selection_weights = {
                'encoder': encoder_var_weights.detach(),
                'decoder': decoder_var_weights.detach() if 'decoder_var_weights' in locals() else None
            }
        
        # Return predictions for the prediction horizon
        if self.max_prediction_length > 0:
            predictions = predictions[:, -self.max_prediction_length:, :]
        
        return predictions.squeeze(-1) if predictions.size(-1) == 1 else predictions
    
    def compute_quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss for uncertainty estimation."""
        if not self.quantiles:
            return F.mse_loss(predictions, targets)
        
        targets = targets.unsqueeze(-1).expand_as(predictions)
        
        quantile_losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[:, :, i]
            target_q = targets[:, :, i]
            
            error = target_q - pred_q
            loss = torch.max(q * error, (q - 1) * error)
            quantile_losses.append(loss)
        
        return torch.stack(quantile_losses, dim=-1).sum(dim=-1).mean()
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss with quantile support."""
        if self.quantiles:
            return self.compute_quantile_loss(predictions, targets)
        else:
            return super().compute_loss(predictions, targets, **kwargs)
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights for interpretability."""
        return self.last_attention_weights
    
    def get_variable_selection_weights(self) -> Optional[Dict]:
        """Get last computed variable selection weights for interpretability."""
        return self.last_variable_selection_weights
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                static_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Make predictions with uncertainty estimates."""
        self.eval()
        with torch.no_grad():
            predictions = self(x, static_features)
        
        if self.quantiles:
            results = {}
            for i, q in enumerate(self.quantiles):
                results[f'quantile_{q}'] = predictions[:, :, i]
            
            # Extract median as point prediction
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            results['prediction'] = predictions[:, :, median_idx]
            
            return results
        else:
            return {'prediction': predictions}


if __name__ == "__main__":
    # Test TFT model
    print("Testing Temporal Fusion Transformer...")
    
    # Mock configuration
    config = {
        'model': {
            'hidden_size': 256,
            'attention_head_size': 4,
            'dropout': 0.1,
            'max_encoder_length': 60,
            'max_prediction_length': 10,
            'lstm_layers': 2,
            'output_size': 1,
            'quantiles': [0.1, 0.5, 0.9],
            'static_reals': ['machine_age'],
            'static_categoricals': ['machine_type'],
            'time_varying_known_reals': ['scheduled_maintenance'],
            'time_varying_known_categoricals': ['shift'],
            'time_varying_unknown_reals': ['temperature', 'vibration', 'pressure', 'humidity']
        },
        'training': {
            'optimizer': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 0.01
        }
    }
    
    # Create model
    model = TemporalFusionTransformer(config)
    
    # Test forward pass
    batch_size = 32
    seq_len = 70  # 60 encoder + 10 decoder
    num_features = 4  # time_varying_unknown_reals
    
    x = torch.randn(batch_size, seq_len, num_features)
    static_features = torch.randn(batch_size, 2)  # 1 real + 1 categorical
    
    predictions = model(x, static_features)
    print(f"TFT predictions shape: {predictions.shape}")
    
    # Test uncertainty prediction
    uncertainty_results = model.predict_with_uncertainty(x, static_features)
    print("Uncertainty prediction keys:", list(uncertainty_results.keys()))
    
    # Test attention weights
    attention_weights = model.get_attention_weights()
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")
    
    print("TFT test completed successfully!")