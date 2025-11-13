"""
Multi-Channel Audio Processing Pipeline

Built from scratch for advanced audio preprocessing including:
- Multi-channel audio loading and processing
- Beamforming and source separation
- Noise reduction and echo cancellation
- Audio enhancement and normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Dict, List, Union
import warnings
from scipy import signal
from scipy.spatial.distance import cdist
import math

class MultiChannelAudioLoader:
    """
    Multi-channel audio loader supporting various formats and configurations.
    """
    
    def __init__(self, sample_rate: int = 16000, target_channels: Optional[int] = None):
        self.sample_rate = sample_rate
        self.target_channels = target_channels
        
    def load_audio(self, file_path: str, normalize: bool = True) -> Tuple[torch.Tensor, int]:
        """
        Load multi-channel audio file.
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize audio
            
        Returns:
            audio: Tensor of shape (channels, samples)
            original_sr: Original sample rate
        """
        try:
            # Load audio with librosa (handles most formats)
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            
            # Ensure audio is 2D (channels, samples)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]  # Add channel dimension
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            # Adjust channels if specified
            if self.target_channels is not None:
                audio = self._adjust_channels(audio, self.target_channels)
            
            # Normalize
            if normalize:
                audio = self._normalize_audio(audio)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            return audio_tensor, sr
            
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            # Return silence if loading fails
            channels = self.target_channels or 1
            samples = self.sample_rate  # 1 second of silence
            return torch.zeros(channels, samples), self.sample_rate
    
    def _adjust_channels(self, audio: np.ndarray, target_channels: int) -> np.ndarray:
        """Adjust number of audio channels."""
        current_channels = audio.shape[0]
        
        if current_channels == target_channels:
            return audio
        elif current_channels > target_channels:
            # Downmix by taking first N channels
            return audio[:target_channels]
        else:
            # Upmix by repeating channels
            repeat_factor = target_channels // current_channels
            remainder = target_channels % current_channels
            
            repeated = np.tile(audio, (repeat_factor, 1))
            if remainder > 0:
                repeated = np.vstack([repeated, audio[:remainder]])
            
            return repeated
    
    def _normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level."""
        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms > 0:
            # Convert target dB to linear scale
            target_rms = 10 ** (target_db / 20)
            scaling_factor = target_rms / rms
            audio = audio * scaling_factor
        
        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio

class BeamformingProcessor:
    """
    Multi-channel beamforming for source separation and noise reduction.
    Implements delay-and-sum and MVDR (Minimum Variance Distortionless Response) beamforming.
    """
    
    def __init__(self, num_channels: int, sample_rate: int, mic_positions: Optional[np.ndarray] = None):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.mic_positions = mic_positions or self._default_mic_positions()
        
        # Speed of sound (m/s)
        self.c = 343.0
        
    def _default_mic_positions(self) -> np.ndarray:
        """Create default microphone positions for common configurations."""
        if self.num_channels == 2:
            # Stereo: 10cm apart
            return np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        elif self.num_channels == 4:
            # Quad: square array
            return np.array([
                [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]
            ])
        elif self.num_channels == 6:
            # 5.1 surround approximation
            return np.array([
                [0.0, 0.0, 0.0],   # Front left
                [0.1, 0.0, 0.0],   # Front right
                [0.05, -0.1, 0.0], # Center
                [0.05, 0.15, 0.0], # LFE
                [-0.05, 0.05, 0.0],# Surround left
                [0.15, 0.05, 0.0]  # Surround right
            ])
        else:
            # Linear array
            positions = []
            spacing = 0.05  # 5cm spacing
            for i in range(self.num_channels):
                positions.append([i * spacing, 0.0, 0.0])
            return np.array(positions)
    
    def delay_and_sum_beamform(self, audio: torch.Tensor, target_angle: float = 0.0) -> torch.Tensor:
        """
        Delay-and-sum beamforming for a target direction.
        
        Args:
            audio: Multi-channel audio (channels, samples)
            target_angle: Target angle in degrees (0 = front, 90 = right)
            
        Returns:
            Beamformed single-channel audio (samples,)
        """
        channels, samples = audio.shape
        
        # Convert angle to radians
        angle_rad = np.radians(target_angle)
        
        # Calculate delays for each microphone
        delays = []
        for i in range(channels):
            # Calculate delay based on microphone position and target angle
            mic_pos = self.mic_positions[i]
            delay = -mic_pos[0] * np.cos(angle_rad) / self.c  # Simplified 2D case
            delay_samples = int(delay * self.sample_rate)
            delays.append(delay_samples)
        
        # Apply delays and sum
        delayed_signals = []
        max_delay = max(abs(d) for d in delays)
        
        for i in range(channels):
            delay = delays[i]
            if delay > 0:
                # Positive delay: pad at beginning
                padded = F.pad(audio[i], (delay, max_delay - delay))
            else:
                # Negative delay: pad at end
                padded = F.pad(audio[i], (max_delay + delay, -delay))
            
            delayed_signals.append(padded)
        
        # Sum all delayed signals
        beamformed = torch.stack(delayed_signals).mean(dim=0)
        
        # Trim to original length
        beamformed = beamformed[max_delay:max_delay + samples]
        
        return beamformed
    
    def adaptive_beamform(self, audio: torch.Tensor, noise_segment: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adaptive beamforming using MVDR (simplified version).
        
        Args:
            audio: Multi-channel audio (channels, samples)
            noise_segment: Optional noise-only segment for noise estimation
            
        Returns:
            Beamformed audio (samples,)
        """
        channels, samples = audio.shape
        
        # Frame-based processing
        frame_size = 512
        hop_size = 256
        num_frames = (samples - frame_size) // hop_size + 1
        
        beamformed_frames = []
        
        for frame_idx in range(num_frames):
            start = frame_idx * hop_size
            end = start + frame_size
            
            # Extract frame from all channels
            frame = audio[:, start:end]  # (channels, frame_size)
            
            # Compute covariance matrix
            R = torch.mm(frame, frame.t()) / frame_size  # (channels, channels)
            
            # Add regularization for numerical stability
            R = R + 1e-6 * torch.eye(channels)
            
            # MVDR weight calculation (simplified)
            # In practice, you'd need steering vector and noise covariance
            try:
                R_inv = torch.inverse(R)
                # Use first channel as reference (simplified)
                e1 = torch.zeros(channels)
                e1[0] = 1.0
                
                weights = torch.mv(R_inv, e1)
                weights = weights / torch.sum(weights)
                
                # Apply weights
                beamformed_frame = torch.mv(weights, frame).sum(dim=0)
                beamformed_frames.append(beamformed_frame)
                
            except:
                # Fallback to simple averaging if matrix inversion fails
                beamformed_frame = frame.mean(dim=0)
                beamformed_frames.append(beamformed_frame)
        
        # Overlap-add reconstruction
        beamformed = self._overlap_add(beamformed_frames, hop_size, samples)
        
        return beamformed
    
    def _overlap_add(self, frames: List[torch.Tensor], hop_size: int, total_length: int) -> torch.Tensor:
        """Reconstruct signal from overlapping frames."""
        reconstructed = torch.zeros(total_length)
        
        for i, frame in enumerate(frames):
            start = i * hop_size
            end = min(start + len(frame), total_length)
            frame_end = end - start
            
            reconstructed[start:end] += frame[:frame_end]
        
        return reconstructed

class NoiseReductionProcessor:
    """
    Noise reduction using spectral subtraction and Wiener filtering.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def spectral_subtraction(self, audio: torch.Tensor, noise_profile: Optional[torch.Tensor] = None,
                           alpha: float = 2.0, beta: float = 0.01) -> torch.Tensor:
        """
        Spectral subtraction noise reduction.
        
        Args:
            audio: Input audio (samples,)
            noise_profile: Noise spectrum estimate
            alpha: Over-subtraction factor
            beta: Spectral floor factor
            
        Returns:
            Denoised audio (samples,)
        """
        # STFT parameters
        n_fft = 512
        hop_length = 256
        win_length = n_fft
        
        # Compute STFT
        stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                         window=torch.hann_window(win_length), return_complex=True)
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Estimate noise spectrum if not provided
        if noise_profile is None:
            # Use first 10 frames as noise estimate
            noise_frames = magnitude[:, :10]
            noise_profile = torch.mean(noise_frames, dim=1, keepdim=True)
        
        # Spectral subtraction
        noise_power = noise_profile ** 2
        signal_power = magnitude ** 2
        
        # Over-subtraction
        enhanced_power = signal_power - alpha * noise_power
        
        # Apply spectral floor
        enhanced_power = torch.maximum(enhanced_power, beta * signal_power)
        
        # Compute enhanced magnitude
        enhanced_magnitude = torch.sqrt(enhanced_power)
        
        # Reconstruct complex spectrum
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        
        # Inverse STFT
        enhanced_audio = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, window=torch.hann_window(win_length))
        
        return enhanced_audio
    
    def wiener_filter(self, audio: torch.Tensor, noise_power: float = 0.01) -> torch.Tensor:
        """
        Wiener filtering for noise reduction.
        
        Args:
            audio: Input audio (samples,)
            noise_power: Estimated noise power
            
        Returns:
            Filtered audio (samples,)
        """
        # STFT
        n_fft = 512
        hop_length = 256
        win_length = n_fft
        
        stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                         window=torch.hann_window(win_length), return_complex=True)
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Estimate signal power
        signal_power = magnitude ** 2
        
        # Wiener gain
        wiener_gain = signal_power / (signal_power + noise_power)
        
        # Apply gain
        enhanced_magnitude = magnitude * wiener_gain
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_audio = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, window=torch.hann_window(win_length))
        
        return enhanced_audio

class AudioEnhancementProcessor:
    """
    Advanced audio enhancement including echo cancellation and dynamic range compression.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def echo_cancellation(self, audio: torch.Tensor, echo_delay: float = 0.1,
                         echo_strength: float = 0.3) -> torch.Tensor:
        """
        Simple echo cancellation using adaptive filtering.
        
        Args:
            audio: Input audio (samples,)
            echo_delay: Echo delay in seconds
            echo_strength: Echo strength (0-1)
            
        Returns:
            Echo-cancelled audio (samples,)
        """
        delay_samples = int(echo_delay * self.sample_rate)
        
        if delay_samples >= len(audio):
            return audio
        
        # Create delayed version (simulated echo)
        delayed_audio = torch.zeros_like(audio)
        delayed_audio[delay_samples:] = audio[:-delay_samples] * echo_strength
        
        # Simple echo cancellation (in practice, would use adaptive algorithms)
        cancelled_audio = audio - delayed_audio * 0.5
        
        return cancelled_audio
    
    def dynamic_range_compression(self, audio: torch.Tensor, threshold: float = -20.0,
                                ratio: float = 4.0, attack_time: float = 0.003,
                                release_time: float = 0.1) -> torch.Tensor:
        """
        Dynamic range compression to even out volume levels.
        
        Args:
            audio: Input audio (samples,)
            threshold: Compression threshold in dB
            ratio: Compression ratio
            attack_time: Attack time in seconds
            release_time: Release time in seconds
            
        Returns:
            Compressed audio (samples,)
        """
        # Convert to dB
        audio_db = 20 * torch.log10(torch.abs(audio) + 1e-8)
        
        # Calculate gain reduction
        gain_reduction = torch.zeros_like(audio_db)
        mask = audio_db > threshold
        gain_reduction[mask] = (threshold - audio_db[mask]) * (1 - 1/ratio)
        
        # Apply attack and release smoothing
        attack_coeff = torch.exp(-1.0 / (attack_time * self.sample_rate))
        release_coeff = torch.exp(-1.0 / (release_time * self.sample_rate))
        
        smoothed_gain = torch.zeros_like(gain_reduction)
        for i in range(1, len(gain_reduction)):
            if gain_reduction[i] < smoothed_gain[i-1]:
                # Attack
                smoothed_gain[i] = attack_coeff * smoothed_gain[i-1] + (1 - attack_coeff) * gain_reduction[i]
            else:
                # Release
                smoothed_gain[i] = release_coeff * smoothed_gain[i-1] + (1 - release_coeff) * gain_reduction[i]
        
        # Apply gain
        linear_gain = 10 ** (smoothed_gain / 20)
        compressed_audio = audio * linear_gain
        
        return compressed_audio
    
    def spectral_whitening(self, audio: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
        """
        Spectral whitening to balance frequency content.
        
        Args:
            audio: Input audio (samples,)
            alpha: Whitening factor (0-1)
            
        Returns:
            Whitened audio (samples,)
        """
        # STFT
        n_fft = 512
        hop_length = 256
        win_length = n_fft
        
        stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                         window=torch.hann_window(win_length), return_complex=True)
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Compute spectral envelope
        spectral_envelope = torch.mean(magnitude, dim=1, keepdim=True)
        
        # Whitening
        whitened_magnitude = magnitude / (alpha * spectral_envelope + (1 - alpha) * torch.ones_like(spectral_envelope))
        
        # Reconstruct
        whitened_stft = whitened_magnitude * torch.exp(1j * phase)
        whitened_audio = torch.istft(whitened_stft, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, window=torch.hann_window(win_length))
        
        return whitened_audio

class MultiChannelAudioProcessor:
    """
    Complete multi-channel audio processing pipeline combining all enhancement techniques.
    Now includes multi-speaker support for handling 2-3+ speakers with overlapping speech.
    """
    
    def __init__(self, num_channels: int, sample_rate: int = 16000,
                 processing_config: Optional[Dict] = None, enable_multispeaker: bool = True):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.config = processing_config or self._default_config()
        self.enable_multispeaker = enable_multispeaker
        
        # Initialize processors
        self.loader = MultiChannelAudioLoader(sample_rate, num_channels)
        self.beamformer = BeamformingProcessor(num_channels, sample_rate)
        self.noise_reducer = NoiseReductionProcessor(sample_rate)
        self.enhancer = AudioEnhancementProcessor(sample_rate)
        
        # Initialize multi-speaker processor if enabled
        if self.enable_multispeaker:
            from .multispeaker_processing import MultiSpeakerProcessor
            self.multispeaker_processor = MultiSpeakerProcessor(
                max_speakers=self.config.get('max_speakers', 3),
                sample_rate=sample_rate
            )
        
    def _default_config(self) -> Dict:
        """Default processing configuration."""
        return {
            'enable_beamforming': True,
            'beamforming_method': 'delay_and_sum',  # or 'adaptive'
            'target_angle': 0.0,
            'enable_noise_reduction': True,
            'noise_reduction_method': 'spectral_subtraction',  # or 'wiener'
            'enable_echo_cancellation': True,
            'enable_compression': True,
            'enable_whitening': False,
            'normalize_output': True,
            # Multi-speaker processing
            'enable_multispeaker': True,
            'max_speakers': 3,
            'speaker_separation_threshold': 0.3,
            'overlap_detection': True
        }
    
    def process_audio_file(self, file_path: str) -> torch.Tensor:
        """
        Process an audio file through the complete pipeline.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Processed single-channel audio
        """
        # Load multi-channel audio
        audio, _ = self.loader.load_audio(file_path)
        
        return self.process_audio(audio)
    
    def process_audio(self, audio: torch.Tensor, return_multispeaker: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Process multi-channel audio through the enhancement pipeline.
        Now supports multi-speaker processing for handling overlapping speech.
        
        Args:
            audio: Multi-channel audio (channels, samples)
            return_multispeaker: If True, returns multi-speaker analysis
            
        Returns:
            Enhanced single-channel audio (samples,) or multi-speaker analysis dict
        """
        processed = audio
        
        # Step 1: Multi-speaker processing (if enabled and requested)
        if (self.enable_multispeaker and 
            self.config.get('enable_multispeaker', True) and 
            return_multispeaker):
            
            # Use first channel for speaker separation (can be enhanced later)
            mono_audio = processed[0] if processed.ndim > 1 else processed
            
            # Process through multi-speaker pipeline
            multispeaker_results = self.multispeaker_processor.process_multispeaker_audio(
                mono_audio
            )
            
            # Process each separated speaker through enhancement pipeline
            enhanced_speakers = {}
            for speaker_id, speaker_audio in multispeaker_results['separated_speakers'].items():
                enhanced_audio = self._process_single_channel(speaker_audio.squeeze())
                enhanced_speakers[speaker_id] = enhanced_audio
            
            multispeaker_results['enhanced_speakers'] = enhanced_speakers
            return multispeaker_results
        
        # Step 2: Standard processing pipeline
        # Beamforming
        if self.config['enable_beamforming'] and audio.shape[0] > 1:
            if self.config['beamforming_method'] == 'delay_and_sum':
                processed = self.beamformer.delay_and_sum_beamform(
                    processed, self.config['target_angle']
                )
            else:
                processed = self.beamformer.adaptive_beamform(processed)
        else:
            # Just take first channel if no beamforming
            processed = processed[0] if processed.ndim > 1 else processed
        
        # Apply single-channel processing
        processed = self._process_single_channel(processed)
        
        return processed
    
    def _process_single_channel(self, audio: torch.Tensor) -> torch.Tensor:
        """Process single-channel audio through enhancement pipeline."""
        processed = audio
        
        # Noise reduction
        if self.config['enable_noise_reduction']:
            if self.config['noise_reduction_method'] == 'spectral_subtraction':
                processed = self.noise_reducer.spectral_subtraction(processed)
            else:
                processed = self.noise_reducer.wiener_filter(processed)
        
        # Echo cancellation
        if self.config['enable_echo_cancellation']:
            processed = self.enhancer.echo_cancellation(processed)
        
        # Dynamic range compression
        if self.config['enable_compression']:
            processed = self.enhancer.dynamic_range_compression(processed)
        
        # Spectral whitening
        if self.config['enable_whitening']:
            processed = self.enhancer.spectral_whitening(processed)
        
        # Final normalization
        if self.config['normalize_output']:
            processed = self._normalize_final(processed)
        
        return processed
    
    def _normalize_final(self, audio: torch.Tensor) -> torch.Tensor:
        """Final normalization to prevent clipping."""
        max_val = torch.max(torch.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        return audio
    
    def get_processing_info(self) -> Dict:
        """Get information about the processing pipeline."""
        return {
            'num_channels': self.num_channels,
            'sample_rate': self.sample_rate,
            'config': self.config,
            'multispeaker_enabled': self.enable_multispeaker,
            'max_speakers': self.config.get('max_speakers', 3),
            'pipeline_steps': [
                'Multi-channel loading',
                'Multi-speaker separation' if self.enable_multispeaker else None,
                'Speaker diarization' if self.enable_multispeaker else None,
                'Overlapping speech detection' if self.enable_multispeaker else None,
                'Beamforming' if self.config['enable_beamforming'] else 'Channel selection',
                'Noise reduction' if self.config['enable_noise_reduction'] else None,
                'Echo cancellation' if self.config['enable_echo_cancellation'] else None,
                'Dynamic compression' if self.config['enable_compression'] else None,
                'Spectral whitening' if self.config['enable_whitening'] else None,
                'Normalization' if self.config['normalize_output'] else None
            ]
        }

# Example usage and factory functions
def create_stereo_processor(sample_rate: int = 16000, enable_all: bool = True, 
                           enable_multispeaker: bool = True) -> MultiChannelAudioProcessor:
    """Create a stereo audio processor with multi-speaker support."""
    config = {
        'enable_beamforming': enable_all,
        'beamforming_method': 'delay_and_sum',
        'target_angle': 0.0,
        'enable_noise_reduction': enable_all,
        'noise_reduction_method': 'spectral_subtraction',
        'enable_echo_cancellation': enable_all,
        'enable_compression': enable_all,
        'enable_whitening': False,
        'normalize_output': True,
        'enable_multispeaker': enable_multispeaker,
        'max_speakers': 3,
        'speaker_separation_threshold': 0.3,
        'overlap_detection': True
    }
    return MultiChannelAudioProcessor(2, sample_rate, config, enable_multispeaker)

def create_surround_processor(sample_rate: int = 16000, 
                             enable_multispeaker: bool = True) -> MultiChannelAudioProcessor:
    """Create a 5.1 surround sound processor with multi-speaker support."""
    config = {
        'enable_beamforming': True,
        'beamforming_method': 'adaptive',
        'target_angle': 0.0,
        'enable_noise_reduction': True,
        'noise_reduction_method': 'spectral_subtraction',
        'enable_echo_cancellation': True,
        'enable_compression': True,
        'enable_whitening': True,
        'normalize_output': True,
        'enable_multispeaker': enable_multispeaker,
        'max_speakers': 5,  # More speakers possible with surround
        'speaker_separation_threshold': 0.2,
        'overlap_detection': True
    }
    return MultiChannelAudioProcessor(6, sample_rate, config, enable_multispeaker)

def create_multispeaker_processor(sample_rate: int = 16000, max_speakers: int = 3,
                                 num_channels: int = 2) -> MultiChannelAudioProcessor:
    """Create a processor optimized for multi-speaker scenarios."""
    config = {
        'enable_beamforming': True,
        'beamforming_method': 'adaptive',  # Better for multi-speaker
        'target_angle': 0.0,
        'enable_noise_reduction': True,
        'noise_reduction_method': 'spectral_subtraction',
        'enable_echo_cancellation': True,
        'enable_compression': False,  # Might interfere with speaker separation
        'enable_whitening': False,
        'normalize_output': True,
        'enable_multispeaker': True,
        'max_speakers': max_speakers,
        'speaker_separation_threshold': 0.25,
        'overlap_detection': True
    }
    return MultiChannelAudioProcessor(num_channels, sample_rate, config, True)

if __name__ == "__main__":
    # Example usage
    print("🎤 Multi-Channel Audio Processing Pipeline with Multi-Speaker Support")
    
    # Create multi-speaker processor
    processor = create_multispeaker_processor(max_speakers=3)
    print(f"Created processor: {processor.get_processing_info()}")
    
    # Simulate stereo audio (2 channels, 3 seconds) with multiple speakers
    sample_rate = 16000
    duration = 3
    samples = sample_rate * duration
    
    # Create synthetic stereo audio with multiple speakers
    # Speaker 1: Low frequency content
    speaker1 = torch.sin(2 * np.pi * 200 * torch.linspace(0, duration, samples)) * 0.3
    # Speaker 2: High frequency content  
    speaker2 = torch.sin(2 * np.pi * 800 * torch.linspace(0, duration, samples)) * 0.3
    # Mix with some noise and spatial separation
    left_channel = speaker1 + torch.randn(samples) * 0.1
    right_channel = speaker2 + torch.randn(samples) * 0.1
    stereo_audio = torch.stack([left_channel, right_channel])
    
    print(f"Input audio shape: {stereo_audio.shape}")
    
    # Process audio normally
    enhanced_audio = processor.process_audio(stereo_audio)
    print(f"Enhanced audio shape: {enhanced_audio.shape}")
    
    # Process with multi-speaker analysis
    print("\n🔄 Processing with multi-speaker analysis...")
    multispeaker_results = processor.process_audio(stereo_audio, return_multispeaker=True)
    
    if isinstance(multispeaker_results, dict):
        print(f"✅ Multi-speaker analysis completed!")
        print(f"   Separated speakers: {list(multispeaker_results['separated_speakers'].keys())}")
        print(f"   Enhanced speakers: {list(multispeaker_results['enhanced_speakers'].keys())}")
        print(f"   Overlapping regions: {len(multispeaker_results['overlapping_regions'])}")
        
        # Show speaker statistics
        stats = multispeaker_results.get('speaker_statistics', {})
        if 'overall' in stats:
            print(f"   Overall overlap: {stats['overall'].get('overlap_percentage', 0):.1f}%")
    
    print(f"\n🎉 Multi-speaker processing completed! Ready for speech recognition with 2-3+ speakers.")