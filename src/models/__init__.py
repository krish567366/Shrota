"""
Models package for multi-lingual speech-to-text system.

Contains custom-built speech recognition models:
- Multi-lingual Transformer for 100+ languages
- Conformer-inspired architecture with CNN+Transformer
- Lightweight CNN-RNN hybrid for edge deployment
"""

# Speech Recognition Models (Built from Scratch)
from .custom_multilingual_transformer import CustomMultiLingualTransformer, create_custom_multilingual_model
from .custom_conformer import CustomConformerModel, create_custom_conformer_model
from .custom_cnn_rnn_hybrid import CustomLightweightCNNRNN, create_lightweight_cnn_rnn_model

# Legacy base components (keeping for reference)
from .base_model import (
    BasePredictor,
    AttentionModule,
    PositionalEncoding,
    ResidualBlock,
    FeatureExtractor,
    OutputHead,
    ModelCheckpointCallback,
    get_activation_function,
    initialize_weights
)

__all__ = [
    # Custom Speech Recognition Models
    'CustomMultiLingualTransformer',
    'CustomConformerModel', 
    'CustomLightweightCNNRNN',
    'create_custom_multilingual_model',
    'create_custom_conformer_model',
    'create_lightweight_cnn_rnn_model',
    
    # Legacy Base Components
    'BasePredictor',
    'AttentionModule',
    'PositionalEncoding',
    'ResidualBlock',
    'FeatureExtractor',
    'OutputHead',
    'ModelCheckpointCallback',
    'get_activation_function',
    'initialize_weights'
]

# Model type mapping for easy access
SPEECH_MODELS = {
    'multilingual_transformer': CustomMultiLingualTransformer,
    'conformer': CustomConformerModel,
    'lightweight_cnn_rnn': CustomLightweightCNNRNN,
}

MODEL_FACTORIES = {
    'multilingual_transformer': create_custom_multilingual_model,
    'conformer': create_custom_conformer_model,
    'lightweight_cnn_rnn': create_lightweight_cnn_rnn_model,
}

def create_speech_model(model_type: str, config: dict):
    """
    Factory function to create speech recognition models.
    
    Args:
        model_type: Type of model ('multilingual_transformer', 'conformer', 'lightweight_cnn_rnn')
        config: Model configuration dictionary
    
    Returns:
        Initialized speech recognition model
    """
    if model_type not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_FACTORIES.keys())}")
    
    return MODEL_FACTORIES[model_type](config)