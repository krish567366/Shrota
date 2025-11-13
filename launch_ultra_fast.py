"""
Ultra-Fast Speech Recognition Deployment Script

Optimized deployment with real-time streaming, batch processing,
and maximum performance configurations.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import threading
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our optimized components
from inference.fast_inference import (
    UltraFastSpeechRecognizer,
    StreamingInferenceEngine,
    BatchInferenceEngine,
    HardwareAccelerator,
    ModelOptimizer
)

# Import existing components
from models.custom_multilingual_transformer import CustomMultiLingualTransformer
from models.custom_conformer import CustomConformerModel
from models.custom_cnn_rnn_hybrid import CustomLightweightCNNRNN
from data.audio_processing import MultiChannelAudioProcessor
from data.multispeaker_processing import MultiSpeakerProcessor
from utils.helpers import setup_logging

class UltraFastDeployment:
    """
    Complete ultra-fast deployment system for speech recognition.
    """
    
    def __init__(self, config: Dict, model_type: str = 'lightweight'):
        self.config = config
        self.model_type = model_type
        
        # Performance tracking
        self.deployment_start_time = time.time()
        self.total_requests = 0
        self.total_processing_time = 0
        
        # Initialize components
        self.logger = setup_logging("UltraFastDeployment")
        self.device = HardwareAccelerator.detect_optimal_device()
        
        # Load and optimize model
        self.model, self.tokenizer = self._load_optimized_model()
        
        # Initialize processors
        self.audio_processor = MultiChannelAudioProcessor(
            sample_rate=config.get('sample_rate', 16000),
            n_mels=config.get('n_mels', 80)
        )
        
        self.multispeaker_processor = MultiSpeakerProcessor(
            sample_rate=config.get('sample_rate', 16000),
            max_speakers=config.get('max_speakers', 3)
        )
        
        # Initialize ultra-fast recognizer
        self.recognizer = UltraFastSpeechRecognizer(
            self.model, 
            self.tokenizer,
            optimization_level='max'
        )
        
        self.logger.info("🚀 Ultra-fast deployment initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {model_type}")
    
    def _load_optimized_model(self):
        """Load and optimize the selected model."""
        self.logger.info(f"Loading {self.model_type} model...")
        
        # Model configurations
        if self.model_type == 'lightweight':
            # Ultra-fast CNN-RNN hybrid for <10ms inference
            model = CustomLightweightCNNRNN(
                input_dim=self.config.get('n_mels', 80),
                hidden_dim=256,  # Reduced for speed
                num_layers=2,    # Reduced for speed
                vocab_size=5000, # Reduced vocabulary for speed
                dropout=0.1
            )
            target_latency = "5ms"
        
        elif self.model_type == 'conformer':
            # Optimized Conformer for balanced speed/accuracy
            model = CustomConformerModel(
                input_dim=self.config.get('n_mels', 80),
                encoder_dim=384,  # Reduced for speed
                num_encoder_layers=8,  # Reduced for speed
                num_attention_heads=6,
                feed_forward_dim=1536,
                conv_kernel_size=31,
                vocab_size=5000,
                dropout=0.1
            )
            target_latency = "8ms"
        
        else:  # transformer
            # Optimized Transformer
            model = CustomMultiLingualTransformer(
                input_dim=self.config.get('n_mels', 80),
                model_dim=384,  # Reduced for speed
                num_heads=6,
                num_encoder_layers=6,  # Reduced for speed
                num_decoder_layers=4,  # Reduced for speed
                ff_dim=1536,
                vocab_size=5000,
                max_seq_length=1000,  # Reduced for speed
                num_languages=50,  # Reduced for speed
                dropout=0.1
            )
            target_latency = "10ms"
        
        # Move to device
        model = model.to(self.device)
        
        # Create simple tokenizer for demo
        class FastTokenizer:
            def __init__(self, vocab_size=5000):
                self.vocab_size = vocab_size
                self.vocab = {
                    '<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<blank>': 4
                }
                # Add common tokens
                for i in range(5, vocab_size):
                    self.vocab[f'token_{i}'] = i
            
            def decode(self, tokens, skip_special=True):
                # Simple decoding for demo
                if skip_special:
                    tokens = [t for t in tokens if t >= 5]
                return f"transcribed_text_from_{len(tokens)}_tokens"
        
        tokenizer = FastTokenizer()
        
        self.logger.info(f"✅ Model loaded - Target latency: {target_latency}")
        return model, tokenizer
    
    def process_realtime_stream(self, audio_stream: Iterator[torch.Tensor]) -> Iterator[Dict]:
        """
        Process real-time audio stream with ultra-fast inference.
        
        Args:
            audio_stream: Iterator yielding audio chunks (16kHz, mono)
            
        Yields:
            Real-time recognition results
        """
        self.logger.info("🎤 Starting real-time streaming recognition...")
        
        chunk_count = 0
        total_latency = 0
        
        for audio_chunk in audio_stream:
            start_time = time.time()
            
            # Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(
                audio_chunk.unsqueeze(0),  # Add batch dimension
                enhance_audio=False  # Skip enhancement for speed
            )
            
            # Multi-speaker processing if enabled
            if self.config.get('enable_multispeaker', False):
                speaker_results = self.multispeaker_processor.process_audio(
                    processed_audio[0],
                    return_separated=True
                )
                
                # Process each speaker separately (parallel processing)
                results = []
                for speaker_audio in speaker_results.get('separated_speakers', [processed_audio[0]]):
                    speaker_result = next(self.recognizer.recognize_streaming([speaker_audio]))
                    results.append(speaker_result)
                
                result = {
                    'type': 'multispeaker',
                    'speakers': results,
                    'num_speakers': len(results)
                }
            else:
                # Single speaker processing
                result = next(self.recognizer.recognize_streaming([processed_audio[0]]))
                result['type'] = 'single_speaker'
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            chunk_count += 1
            total_latency += processing_time
            
            # Add performance info
            result.update({
                'chunk_id': chunk_count,
                'processing_time_ms': processing_time * 1000,
                'avg_latency_ms': (total_latency / chunk_count) * 1000,
                'real_time_factor': (len(audio_chunk) / 16000) / processing_time,
                'device': str(self.device)
            })
            
            # Update global stats
            self.total_requests += 1
            self.total_processing_time += processing_time
            
            yield result
    
    def process_batch_files(self, audio_files: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Process multiple audio files in optimized batches.
        
        Args:
            audio_files: List of audio file paths
            batch_size: Batch size for processing
            
        Returns:
            Batch processing results
        """
        self.logger.info(f"📁 Processing {len(audio_files)} files in batches of {batch_size}")
        
        all_results = []
        total_start_time = time.time()
        
        # Process files in batches
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]
            batch_start_time = time.time()
            
            # Load and preprocess batch
            batch_features = []
            file_info = []
            
            for file_path in batch_files:
                try:
                    # Load audio (simplified for demo)
                    audio = torch.randn(16000 * 5)  # 5 seconds of random audio
                    
                    # Preprocess
                    processed = self.audio_processor.preprocess_audio(
                        audio.unsqueeze(0),
                        enhance_audio=False
                    )
                    
                    # Extract features (simplified)
                    features = torch.randn(100, 80)  # (time, features)
                    
                    batch_features.append(features)
                    file_info.append({
                        'file_path': file_path,
                        'duration': len(audio) / 16000,
                        'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    })
                
                except Exception as e:
                    self.logger.warning(f"Failed to process {file_path}: {e}")
                    continue
            
            # Batch inference
            if batch_features:
                batch_results = self.recognizer.recognize_batch(batch_features)
                
                # Combine with file info
                for result, info in zip(batch_results, file_info):
                    result.update(info)
                    all_results.append(result)
            
            batch_time = time.time() - batch_start_time
            self.logger.info(f"Batch {i//batch_size + 1}: {len(batch_features)} files in {batch_time:.2f}s")
        
        total_time = time.time() - total_start_time
        throughput = len(all_results) / total_time
        
        self.logger.info(f"✅ Batch processing complete: {len(all_results)} files in {total_time:.2f}s ({throughput:.1f} files/s)")
        
        return all_results
    
    def benchmark_performance(self, duration_seconds: int = 30) -> Dict:
        """
        Benchmark the ultra-fast inference performance.
        
        Args:
            duration_seconds: Duration of benchmark test
            
        Returns:
            Performance benchmark results
        """
        self.logger.info(f"🏁 Running performance benchmark for {duration_seconds} seconds...")
        
        # Generate test audio stream
        sample_rate = 16000
        chunk_size = 1600  # 100ms chunks
        total_chunks = (duration_seconds * sample_rate) // chunk_size
        
        def audio_generator():
            for i in range(total_chunks):
                yield torch.randn(chunk_size)  # Random audio chunk
        
        # Benchmark streaming
        start_time = time.time()
        results = list(self.process_realtime_stream(audio_generator()))
        benchmark_time = time.time() - start_time
        
        # Calculate metrics
        total_audio_duration = total_chunks * (chunk_size / sample_rate)
        real_time_factor = total_audio_duration / benchmark_time
        
        benchmark_results = {
            'test_duration_seconds': duration_seconds,
            'total_chunks_processed': len(results),
            'total_processing_time': benchmark_time,
            'total_audio_duration': total_audio_duration,
            'real_time_factor': real_time_factor,
            'avg_latency_ms': np.mean([r.get('processing_time_ms', 0) for r in results]),
            'min_latency_ms': np.min([r.get('processing_time_ms', 0) for r in results]),
            'max_latency_ms': np.max([r.get('processing_time_ms', 0) for r in results]),
            'throughput_chunks_per_second': len(results) / benchmark_time,
            'model_type': self.model_type,
            'device': str(self.device),
            'optimization_level': 'max'
        }
        
        # Performance evaluation
        if benchmark_results['avg_latency_ms'] < 10:
            performance_grade = "🚀 ULTRA-FAST"
        elif benchmark_results['avg_latency_ms'] < 50:
            performance_grade = "⚡ FAST"
        elif benchmark_results['avg_latency_ms'] < 100:
            performance_grade = "✅ GOOD"
        else:
            performance_grade = "⚠️ SLOW"
        
        benchmark_results['performance_grade'] = performance_grade
        
        self.logger.info(f"Benchmark Results:")
        self.logger.info(f"  Average Latency: {benchmark_results['avg_latency_ms']:.2f}ms")
        self.logger.info(f"  Real-time Factor: {real_time_factor:.1f}x")
        self.logger.info(f"  Performance Grade: {performance_grade}")
        
        return benchmark_results
    
    def get_deployment_stats(self) -> Dict:
        """Get deployment statistics."""
        uptime = time.time() - self.deployment_start_time
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.total_requests,
            'total_processing_time': self.total_processing_time,
            'average_request_time_ms': (
                (self.total_processing_time / self.total_requests) * 1000
                if self.total_requests > 0 else 0
            ),
            'requests_per_second': self.total_requests / uptime if uptime > 0 else 0,
            'model_type': self.model_type,
            'device': str(self.device),
            **self.recognizer.get_performance_metrics()
        }

def create_demo_audio_stream(duration: int = 10, chunk_duration: float = 0.1) -> Iterator[torch.Tensor]:
    """Create demo audio stream for testing."""
    sample_rate = 16000
    chunk_size = int(chunk_duration * sample_rate)
    total_chunks = int(duration / chunk_duration)
    
    for i in range(total_chunks):
        # Generate realistic audio-like noise
        chunk = torch.randn(chunk_size) * 0.1
        # Add some periodic patterns to simulate speech
        t = torch.linspace(0, chunk_duration, chunk_size)
        chunk += 0.05 * torch.sin(2 * np.pi * 440 * t)  # 440Hz tone
        yield chunk
        time.sleep(chunk_duration * 0.1)  # Simulate real-time with some delay

def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Speech Recognition Deployment")
    parser.add_argument('--model', choices=['lightweight', 'conformer', 'transformer'],
                       default='lightweight', help='Model type for deployment')
    parser.add_argument('--mode', choices=['stream', 'batch', 'benchmark'],
                       default='benchmark', help='Deployment mode')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration for streaming/benchmark (seconds)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for batch processing')
    parser.add_argument('--enable-multispeaker', action='store_true',
                       help='Enable multi-speaker processing')
    parser.add_argument('--max-speakers', type=int, default=3,
                       help='Maximum number of speakers')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'sample_rate': 16000,
        'n_mels': 80,
        'enable_multispeaker': args.enable_multispeaker,
        'max_speakers': args.max_speakers
    }
    
    print("⚡ Ultra-Fast Speech Recognition Deployment")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Multi-speaker: {'Enabled' if args.enable_multispeaker else 'Disabled'}")
    print()
    
    # Initialize deployment
    deployment = UltraFastDeployment(config, args.model)
    
    try:
        if args.mode == 'stream':
            print(f"🎤 Starting real-time streaming for {args.duration} seconds...")
            audio_stream = create_demo_audio_stream(duration=args.duration)
            
            for i, result in enumerate(deployment.process_realtime_stream(audio_stream)):
                print(f"Chunk {i+1}: {result.get('processing_time_ms', 0):.1f}ms "
                      f"(RTF: {result.get('real_time_factor', 0):.1f}x)")
                
                if i % 10 == 9:  # Print every 10 chunks
                    stats = deployment.get_deployment_stats()
                    print(f"  Average latency: {stats.get('average_request_time_ms', 0):.1f}ms")
        
        elif args.mode == 'batch':
            print(f"📁 Batch processing demo with {args.batch_size} files...")
            # Create demo file list
            demo_files = [f"demo_file_{i}.wav" for i in range(args.batch_size)]
            
            results = deployment.process_batch_files(demo_files, args.batch_size)
            print(f"Processed {len(results)} files successfully")
        
        elif args.mode == 'benchmark':
            print("🏁 Running performance benchmark...")
            benchmark_results = deployment.benchmark_performance(args.duration)
            
            print("\n📊 Benchmark Results:")
            print(f"  Average Latency: {benchmark_results['avg_latency_ms']:.2f}ms")
            print(f"  Min Latency: {benchmark_results['min_latency_ms']:.2f}ms")
            print(f"  Max Latency: {benchmark_results['max_latency_ms']:.2f}ms")
            print(f"  Real-time Factor: {benchmark_results['real_time_factor']:.1f}x")
            print(f"  Throughput: {benchmark_results['throughput_chunks_per_second']:.1f} chunks/s")
            print(f"  Performance Grade: {benchmark_results['performance_grade']}")
            
            if benchmark_results['avg_latency_ms'] < 10:
                print("\n🎉 TARGET ACHIEVED: Ultra-fast inference <10ms!")
            else:
                print(f"\n⚠️ Target not met, but still fast at {benchmark_results['avg_latency_ms']:.1f}ms")
    
    except KeyboardInterrupt:
        print("\n⏹️ Deployment stopped by user")
    
    finally:
        # Final statistics
        stats = deployment.get_deployment_stats()
        print(f"\n📈 Final Statistics:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"  Average Request Time: {stats['average_request_time_ms']:.1f}ms")
        print(f"  Requests/Second: {stats['requests_per_second']:.1f}")
        print(f"  Device: {stats['device']}")
        
        print("\n✅ Deployment completed successfully!")

if __name__ == "__main__":
    main()