"""
Hybrid CNN-BiLSTM model for predictive maintenance.
Combines convolutional layers for local pattern detection with BiLSTM for temporal dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .base_model import BasePredictor, AttentionModule, get_activation_function

class Conv1DBlock(nn.Module):
    """1D Convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, activation: str = 'relu',
                 batch_norm: bool = True, dropout: float = 0.1):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        self.activation = get_activation_function(activation)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for local pattern detection in time series."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        cnn_config = config['cnn']
        
        self.input_channels = config.get('n_features', 8)
        self.filters = cnn_config['filters']
        self.kernel_sizes = cnn_config['kernel_sizes']
        self.activation = cnn_config.get('activation', 'relu')
        self.batch_norm = cnn_config.get('batch_norm', True)
        self.dropout = cnn_config.get('dropout', 0.2)
        self.pooling = cnn_config.get('pooling', 'max')
        self.pool_size = cnn_config.get('pool_size', 2)
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = self.input_channels
        for i, (out_channels, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            # Convolutional block
            conv_block = Conv1DBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                activation=self.activation,
                batch_norm=self.batch_norm,
                dropout=self.dropout
            )
            self.conv_layers.append(conv_block)
            
            # Pooling layer
            if self.pooling == 'max':
                pool_layer = nn.MaxPool1d(kernel_size=self.pool_size, stride=self.pool_size)
            elif self.pooling == 'avg':
                pool_layer = nn.AvgPool1d(kernel_size=self.pool_size, stride=self.pool_size)
            else:
                pool_layer = nn.Identity()
            
            self.pool_layers.append(pool_layer)
            in_channels = out_channels
        
        self.output_channels = in_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN feature extractor.
        
        Args:
            x: Input tensor (batch_size, seq_len, n_features)
            
        Returns:
            features: Extracted features (batch_size, seq_len_reduced, output_channels)
        """
        # Transpose for Conv1d: (batch_size, n_features, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x = conv_layer(x)
            x = pool_layer(x)
        
        # Transpose back: (batch_size, seq_len_reduced, output_channels)
        x = x.transpose(1, 2)
        
        return x

class BiLSTMModule(nn.Module):
    """Bidirectional LSTM module for temporal dependency modeling."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        bilstm_config = config['bilstm']
        
        self.input_size = config.get('cnn_output_size', 256)  # From CNN output
        self.hidden_size = bilstm_config['hidden_size']
        self.num_layers = bilstm_config['num_layers']
        self.dropout = bilstm_config.get('dropout', 0.3)
        self.bidirectional = bilstm_config.get('bidirectional', True)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Output size
        self.output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through BiLSTM.
        
        Args:
            x: Input features (batch_size, seq_len, input_size)
            
        Returns:
            output: LSTM output (batch_size, seq_len, output_size)
            hidden_state: Final hidden and cell states
        """
        # LSTM forward pass
        output, (h_n, c_n) = self.lstm(x)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output, (h_n, c_n)

class HybridCNNBiLSTM(BasePredictor):
    """
    Hybrid CNN-BiLSTM model for predictive maintenance.
    
    Architecture:
    1. CNN layers for local pattern detection
    2. BiLSTM for temporal dependencies
    3. Multi-head attention for feature importance
    4. Output layers for prediction (with optional dual head)
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        model_config = self.model_config
        
        # Model dimensions
        self.sequence_length = model_config.get('sequence_length', 100)
        self.n_features = model_config.get('n_features', 8)
        self.dual_head = model_config.get('dual_head', True)
        
        # CNN Feature Extractor
        cnn_config = {
            'cnn': model_config['cnn'],
            'n_features': self.n_features
        }
        self.cnn_extractor = CNNFeatureExtractor(cnn_config)
        
        # Calculate CNN output size
        self.cnn_output_size = self.cnn_extractor.output_channels
        
        # BiLSTM Module
        bilstm_config = {
            'bilstm': model_config['bilstm'],
            'cnn_output_size': self.cnn_output_size
        }
        self.bilstm = BiLSTMModule(bilstm_config)
        
        # Attention mechanism
        attention_config = model_config.get('attention', {})
        self.attention_type = attention_config.get('type', 'multi_head')
        self.num_heads = attention_config.get('num_heads', 8)
        self.attention_hidden_size = attention_config.get('hidden_size', self.bilstm.output_size)
        self.attention_dropout = attention_config.get('dropout', 0.1)
        
        if self.attention_type == 'multi_head':
            self.attention = AttentionModule(
                hidden_size=self.attention_hidden_size,
                num_heads=self.num_heads,
                dropout=self.attention_dropout
            )
        
        # Feature projection (if needed)
        if self.bilstm.output_size != self.attention_hidden_size:
            self.feature_projection = nn.Linear(self.bilstm.output_size, self.attention_hidden_size)
        else:
            self.feature_projection = nn.Identity()
        
        # Output layers
        output_config = model_config['output']
        self.hidden_layers = output_config['hidden_layers']
        self.output_activation = output_config.get('activation', 'relu')
        self.output_dropout = output_config.get('dropout', 0.3)
        self.final_activation = output_config.get('final_activation', 'sigmoid')
        
        # Build output network
        self.output_layers = self._build_output_network()
        
        # Dual head for classification + regression
        if self.dual_head:
            self.classification_head = nn.Sequential(
                nn.Linear(self.hidden_layers[-1], 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
    
    def _build_output_network(self) -> nn.ModuleList:
        """Build the output network layers."""
        layers = nn.ModuleList()
        
        input_size = self.attention_hidden_size
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(get_activation_function(self.output_activation))
            layers.append(nn.Dropout(self.output_dropout))
            input_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(input_size, 1))
        if self.final_activation != 'identity':
            layers.append(get_activation_function(self.final_activation))
        
        return layers
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input sequences (batch_size, seq_len, n_features)
            
        Returns:
            predictions: Regression predictions (batch_size, 1)
            classification_output: Classification predictions (batch_size, 1) if dual_head=True
        """
        batch_size, seq_len, n_features = x.shape
        
        # CNN feature extraction for local patterns
        cnn_features = self.cnn_extractor(x)  # (batch_size, reduced_seq_len, cnn_output_size)
        
        # BiLSTM for temporal dependencies
        lstm_output, (h_n, c_n) = self.bilstm(cnn_features)  # (batch_size, reduced_seq_len, bilstm_output_size)
        
        # Project features if needed
        projected_features = self.feature_projection(lstm_output)
        
        # Multi-head attention for feature importance
        if self.attention_type == 'multi_head':
            attended_features, attention_weights = self.attention(
                projected_features, projected_features, projected_features
            )
            # Store attention weights for interpretability
            self.last_attention_weights = attention_weights.detach()
        else:
            attended_features = projected_features
        
        # Global pooling (mean over time dimension)
        global_features = attended_features.mean(dim=1)  # (batch_size, attention_hidden_size)
        
        # Output network
        output = global_features
        for layer in self.output_layers:
            output = layer(output)
        
        # Regression predictions
        regression_output = output  # (batch_size, 1)
        
        # Dual head: classification output
        if self.dual_head:
            classification_output = self.classification_head(global_features)
            return regression_output, classification_output
        else:
            return regression_output
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    classification_outputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute combined loss for dual-head model."""
        if self.dual_head and classification_outputs is not None:
            # Regression loss
            reg_loss = F.mse_loss(predictions, targets)
            
            # Classification loss (binary)
            threshold = self.model_config.get('classification_threshold', 0.5)
            class_targets = (targets > threshold).float()
            
            # Focal loss for classification
            alpha = self.training_config.get('focal_loss_alpha', 0.25)
            gamma = self.training_config.get('focal_loss_gamma', 2.0)
            
            bce_loss = F.binary_cross_entropy(classification_outputs, class_targets, reduction='none')
            p_t = class_targets * classification_outputs + (1 - class_targets) * (1 - classification_outputs)
            focal_loss = alpha * (1 - p_t) ** gamma * bce_loss
            class_loss = focal_loss.mean()
            
            # Combine losses
            reg_weight = self.training_config.get('regression_weight', 0.7)
            class_weight = self.training_config.get('classification_weight', 0.3)
            
            total_loss = reg_weight * reg_loss + class_weight * class_loss
            
            # Log individual losses
            self.log('reg_loss', reg_loss, on_step=True, on_epoch=True)
            self.log('class_loss', class_loss, on_step=True, on_epoch=True)
            
            return total_loss
        else:
            return F.mse_loss(predictions, targets)
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights for interpretability."""
        return self.last_attention_weights
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate features for analysis."""
        features = {}
        
        # CNN features
        cnn_features = self.cnn_extractor(x)
        features['cnn_features'] = cnn_features
        
        # LSTM features
        lstm_output, _ = self.bilstm(cnn_features)
        features['lstm_features'] = lstm_output
        
        # Attention features
        projected_features = self.feature_projection(lstm_output)
        if self.attention_type == 'multi_head':
            attended_features, attention_weights = self.attention(
                projected_features, projected_features, projected_features
            )
            features['attention_weights'] = attention_weights
        else:
            attended_features = projected_features
        
        features['attended_features'] = attended_features
        features['global_features'] = attended_features.mean(dim=1)
        
        return features
    
    def predict_with_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions and return intermediate features."""
        self.eval()
        with torch.no_grad():
            # Get predictions
            outputs = self(x)
            if isinstance(outputs, tuple):
                predictions, classification_outputs = outputs
            else:
                predictions = outputs
                classification_outputs = None
            
            # Get features
            features = self.extract_features(x)
            
            results = {
                'predictions': predictions,
                'features': features
            }
            
            if classification_outputs is not None:
                results['classification'] = classification_outputs
            
            return results


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN for capturing patterns at different time scales."""
    
    def __init__(self, input_channels: int, output_channels: int, 
                 kernel_sizes: List[int] = [3, 5, 7, 11]):
        super().__init__()
        
        self.kernel_sizes = kernel_sizes
        self.conv_branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(input_channels, output_channels // len(kernel_sizes), 
                         kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(output_channels // len(kernel_sizes)),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.conv_branches.append(branch)
        
        # 1x1 conv for dimensionality adjustment
        self.output_conv = nn.Conv1d(output_channels, output_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale CNN."""
        # Apply different scale convolutions
        branch_outputs = []
        for branch in self.conv_branches:
            branch_output = branch(x)
            branch_outputs.append(branch_output)
        
        # Concatenate outputs
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # Final output projection
        output = self.output_conv(concatenated)
        
        return output


if __name__ == "__main__":
    # Test Hybrid CNN-BiLSTM model
    print("Testing Hybrid CNN-BiLSTM model...")
    
    # Mock configuration
    config = {
        'model': {
            'sequence_length': 100,
            'n_features': 8,
            'dual_head': True,
            'classification_threshold': 0.5,
            'cnn': {
                'filters': [64, 128, 256],
                'kernel_sizes': [3, 5, 7],
                'activation': 'relu',
                'batch_norm': True,
                'dropout': 0.2,
                'pooling': 'max',
                'pool_size': 2
            },
            'bilstm': {
                'hidden_size': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': True
            },
            'attention': {
                'type': 'multi_head',
                'num_heads': 8,
                'hidden_size': 512,
                'dropout': 0.1
            },
            'output': {
                'hidden_layers': [512, 256, 128],
                'activation': 'relu',
                'dropout': 0.3,
                'final_activation': 'sigmoid'
            }
        },
        'training': {
            'optimizer': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'regression_weight': 0.7,
            'classification_weight': 0.3,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0
        }
    }
    
    # Create model
    model = HybridCNNBiLSTM(config)
    
    # Test forward pass
    batch_size = 32
    seq_len = 100
    n_features = 8
    
    x = torch.randn(batch_size, seq_len, n_features)
    
    # Forward pass
    outputs = model(x)
    if isinstance(outputs, tuple):
        reg_pred, class_pred = outputs
        print(f"Regression predictions shape: {reg_pred.shape}")
        print(f"Classification predictions shape: {class_pred.shape}")
    else:
        print(f"Predictions shape: {outputs.shape}")
    
    # Test feature extraction
    features = model.extract_features(x)
    print("Feature extraction keys:", list(features.keys()))
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value.shape}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights()
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")
    
    print("Hybrid CNN-BiLSTM model test completed successfully!")