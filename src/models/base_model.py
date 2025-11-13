"""
Base model classes for predictive maintenance ML system.
Provides foundation for different model architectures with A100 GPU optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from abc import ABC, abstractmethod

class BasePredictor(pl.LightningModule, ABC):
    """Base class for predictive maintenance models."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        
        # Save hyperparameters
        self.save_hyperparameters(config)
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # Model architecture will be defined in subclasses
        self.encoder = None
        self.decoder = None
        self.classifier = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        pass
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Optimizer
        optimizer_name = self.training_config.get('optimizer', 'AdamW')
        lr = self.training_config.get('learning_rate', 0.001)
        weight_decay = self.training_config.get('weight_decay', 0.01)
        
        if optimizer_name == 'AdamW':
            optimizer = AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'Ranger':
            # Ranger optimizer (AdamW + Lookahead + RAdam)
            try:
                from ranger_adabelief import RangerAdaBelief
                optimizer = RangerAdaBelief(
                    self.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            except ImportError:
                print("Ranger optimizer not available, falling back to AdamW")
                optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler_name = self.training_config.get('lr_scheduler', 'CosineAnnealingWarmRestarts')
        
        if scheduler_name == 'CosineAnnealingWarmRestarts':
            scheduler_params = self.training_config.get('lr_scheduler_params', {})
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_params.get('T_0', 10),
                T_mult=scheduler_params.get('T_mult', 2),
                eta_min=scheduler_params.get('eta_min', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        elif scheduler_name == 'OneCycleLR':
            scheduler_params = self.training_config.get('lr_scheduler_params', {})
            scheduler = OneCycleLR(
                optimizer,
                max_lr=scheduler_params.get('max_lr', 0.01),
                epochs=self.training_config.get('max_epochs', 100),
                steps_per_epoch=scheduler_params.get('steps_per_epoch', 100),
                pct_start=scheduler_params.get('pct_start', 0.3),
                anneal_strategy=scheduler_params.get('anneal_strategy', 'cos')
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        
        return optimizer
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    classification_outputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss based on model configuration."""
        loss = 0.0
        
        # Regression loss (e.g., for RUL prediction)
        if self.model_config.get('loss', 'RMSE') == 'RMSE':
            reg_loss = F.mse_loss(predictions, targets)
        elif self.model_config['loss'] == 'MAE':
            reg_loss = F.l1_loss(predictions, targets)
        else:
            reg_loss = F.mse_loss(predictions, targets)
        
        # Classification loss (for dual-head models)
        if classification_outputs is not None and self.model_config.get('dual_head', False):
            # Binary classification for failure prediction
            class_targets = (targets > self.model_config.get('classification_threshold', 0.5)).float()
            
            # Focal loss for imbalanced data
            alpha = self.training_config.get('focal_loss_alpha', 0.25)
            gamma = self.training_config.get('focal_loss_gamma', 2.0)
            
            ce_loss = F.binary_cross_entropy_with_logits(classification_outputs, class_targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
            class_loss = focal_loss.mean()
            
            # Combine losses
            reg_weight = self.training_config.get('regression_weight', 0.7)
            class_weight = self.training_config.get('classification_weight', 0.3)
            
            loss = reg_weight * reg_loss + class_weight * class_loss
        else:
            loss = reg_loss
        
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        sequences = batch['sequence']
        targets = batch['target']
        
        # Forward pass
        outputs = self(sequences)
        
        if isinstance(outputs, tuple):
            predictions, classification_outputs = outputs
            loss = self.compute_loss(predictions, targets, classification_outputs)
        else:
            predictions = outputs
            loss = self.compute_loss(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end calculations
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'predictions': predictions.detach(),
            'targets': targets.detach()
        })
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        sequences = batch['sequence']
        targets = batch['target']
        
        # Forward pass
        outputs = self(sequences)
        
        if isinstance(outputs, tuple):
            predictions, classification_outputs = outputs
            loss = self.compute_loss(predictions, targets, classification_outputs)
        else:
            predictions = outputs
            loss = self.compute_loss(predictions, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end calculations
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'predictions': predictions.detach(),
            'targets': targets.detach()
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if not self.training_step_outputs:
            return
        
        # Calculate epoch metrics
        all_predictions = torch.cat([x['predictions'] for x in self.training_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.training_step_outputs])
        
        # RMSE
        rmse = torch.sqrt(F.mse_loss(all_predictions, all_targets))
        self.log('train_rmse', rmse, prog_bar=True)
        
        # MAE
        mae = F.l1_loss(all_predictions, all_targets)
        self.log('train_mae', mae)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if not self.validation_step_outputs:
            return
        
        # Calculate epoch metrics
        all_predictions = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # RMSE
        rmse = torch.sqrt(F.mse_loss(all_predictions, all_targets))
        self.log('val_rmse', rmse, prog_bar=True)
        
        # MAE
        mae = F.l1_loss(all_predictions, all_targets)
        self.log('val_mae', mae)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        sequences = batch['sequence']
        outputs = self(sequences)
        
        if isinstance(outputs, tuple):
            predictions, _ = outputs
        else:
            predictions = outputs
        
        return predictions


class AttentionModule(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through attention."""
        batch_size, seq_len, _ = query.shape
        
        # Linear transformations
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and apply output linear layer
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.output_linear(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models."""
    
    def __init__(self, hidden_size: int, max_length: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-np.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = x
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Residual connection
        x = x + residual
        x = self.layer_norm(x)
        
        return x


class FeatureExtractor(nn.Module):
    """Feature extraction module for time series data."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            current_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        return self.feature_extractor(x)


class OutputHead(nn.Module):
    """Output head for different prediction tasks."""
    
    def __init__(self, hidden_size: int, output_size: int, 
                 task_type: str = 'regression', dropout: float = 0.3):
        super().__init__()
        self.task_type = task_type
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        if task_type == 'classification':
            self.activation = nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=-1)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output head."""
        x = self.dropout(x)
        x = self.output_layer(x)
        x = self.activation(x)
        return x


class ModelCheckpointCallback(pl.Callback):
    """Custom model checkpoint callback with additional functionality."""
    
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Check if current model is the best."""
        current_metric = trainer.callback_metrics.get(self.monitor)
        
        if current_metric is not None:
            is_better = (
                current_metric < self.best_metric if self.mode == 'min' 
                else current_metric > self.best_metric
            )
            
            if is_better:
                self.best_metric = current_metric
                # Save best model artifacts
                self._save_model_artifacts(trainer, pl_module)
    
    def _save_model_artifacts(self, trainer, pl_module):
        """Save additional model artifacts."""
        # This could include attention weights, feature importance, etc.
        pass


def get_activation_function(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'swish': nn.SiLU(),
        'mish': nn.Mish()
    }
    
    return activations.get(name.lower(), nn.ReLU())


def initialize_weights(model: nn.Module):
    """Initialize model weights using Xavier/He initialization."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


if __name__ == "__main__":
    # Test base model components
    print("Testing base model components...")
    
    # Test attention module
    attention = AttentionModule(hidden_size=256, num_heads=8)
    x = torch.randn(32, 100, 256)  # (batch, seq_len, hidden_size)
    output, weights = attention(x, x, x)
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Test positional encoding
    pos_enc = PositionalEncoding(hidden_size=256)
    encoded = pos_enc(x)
    print(f"Positional encoding output shape: {encoded.shape}")
    
    # Test feature extractor
    feature_extractor = FeatureExtractor(input_size=8, hidden_size=256)
    input_features = torch.randn(32, 100, 8)
    extracted = feature_extractor(input_features)
    print(f"Feature extraction output shape: {extracted.shape}")
    
    print("Base model components test completed!")