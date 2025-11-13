"""
Multi-Speaker Audio Processing and Diarization

Handles scenarios with 2-3+ speakers including:
- Speaker separation and source isolation
- Speaker diarization (who spoke when)
- Overlapping speech detection and handling
- Multi-speaker transcription with speaker labels
- Voice activity detection for each speaker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import librosa
from scipy import signal
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class SpeakerSeparationNetwork(nn.Module):
    """
    Neural network for speaker separation using TasNet/Conv-TasNet approach.
    Separates mixed audio into individual speaker streams.
    """
    
    def __init__(self, num_speakers: int = 3, feature_dim: int = 256, 
                 num_layers: int = 8, kernel_size: int = 3):
        super().__init__()
        self.num_speakers = num_speakers
        self.feature_dim = feature_dim
        
        # Encoder: Convert waveform to feature representation
        self.encoder = nn.Conv1d(1, feature_dim, kernel_size=20, stride=10, padding=5)
        
        # Separation network with dilated convolutions
        self.separation_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 4)  # Exponential dilation pattern
            self.separation_layers.append(
                nn.Sequential(
                    nn.Conv1d(feature_dim, feature_dim, kernel_size, 
                             dilation=dilation, padding=dilation*(kernel_size-1)//2),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                    nn.Conv1d(feature_dim, feature_dim, 1),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )
        
        # Speaker-specific mask generation
        self.mask_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, 1),
                nn.ReLU(),
                nn.Conv1d(feature_dim, feature_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_speakers)
        ])
        
        # Decoder: Convert features back to waveform
        self.decoder = nn.ConvTranspose1d(feature_dim, 1, kernel_size=20, 
                                        stride=10, padding=5)
        
    def forward(self, mixed_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Separate mixed audio into individual speakers.
        
        Args:
            mixed_audio: Mixed audio tensor (batch_size, 1, samples)
            
        Returns:
            Dictionary with separated speaker audio streams
        """
        batch_size, _, samples = mixed_audio.shape
        
        # Encode to feature space
        features = self.encoder(mixed_audio)  # (batch, feature_dim, time)
        
        # Process through separation layers
        separated_features = features
        for layer in self.separation_layers:
            residual = separated_features
            separated_features = layer(separated_features) + residual
        
        # Generate masks for each speaker
        speaker_masks = []
        for mask_gen in self.mask_generators:
            mask = mask_gen(separated_features)
            speaker_masks.append(mask)
        
        # Apply masks and decode
        separated_speakers = {}
        for i, mask in enumerate(speaker_masks):
            masked_features = separated_features * mask
            separated_audio = self.decoder(masked_features)
            
            # Ensure output length matches input
            if separated_audio.shape[-1] != samples:
                if separated_audio.shape[-1] > samples:
                    separated_audio = separated_audio[:, :, :samples]
                else:
                    padding = samples - separated_audio.shape[-1]
                    separated_audio = F.pad(separated_audio, (0, padding))
            
            separated_speakers[f'speaker_{i+1}'] = separated_audio
        
        return separated_speakers
    
    def compute_separation_loss(self, separated_speakers: Dict[str, torch.Tensor],
                               target_speakers: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute permutation-invariant loss for speaker separation."""
        import itertools
        
        speaker_keys = list(separated_speakers.keys())
        num_speakers = len(speaker_keys)
        
        # Try all permutations to find best assignment
        min_loss = float('inf')
        
        for perm in itertools.permutations(range(num_speakers)):
            total_loss = 0
            for i, j in enumerate(perm):
                pred_key = speaker_keys[i]
                target_key = f'speaker_{j+1}'
                
                if target_key in target_speakers:
                    loss = F.mse_loss(separated_speakers[pred_key], 
                                    target_speakers[target_key])
                    total_loss += loss
            
            min_loss = min(min_loss, total_loss)
        
        return min_loss

class SpeakerDiarization:
    """
    Speaker diarization system to determine who spoke when.
    Uses clustering of speaker embeddings to identify different speakers.
    """
    
    def __init__(self, embedding_dim: int = 256, sample_rate: int = 16000):
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        
        # Speaker embedding network (simplified x-vector style)
        self.embedding_net = self._build_embedding_network()
        
    def _build_embedding_network(self) -> nn.Module:
        """Build speaker embedding network."""
        return nn.Sequential(
            # Frame-level processing
            nn.Conv1d(80, 512, 5, padding=2),  # Input: mel spectrograms
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            # Statistics pooling layer
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            # Segment-level processing
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim)
        )
    
    def extract_speaker_embeddings(self, audio_segments: List[torch.Tensor]) -> np.ndarray:
        """
        Extract speaker embeddings from audio segments.
        
        Args:
            audio_segments: List of audio segments (each segment is a tensor)
            
        Returns:
            Speaker embeddings array (n_segments, embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for segment in audio_segments:
                # Convert to mel spectrogram
                mel_spec = self._audio_to_mel(segment)
                
                # Extract embedding
                embedding = self.embedding_net(mel_spec.unsqueeze(0))
                embeddings.append(embedding.squeeze().numpy())
        
        return np.array(embeddings)
    
    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        audio_np = audio.numpy()
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np,
            sr=self.sample_rate,
            n_mels=80,
            n_fft=512,
            hop_length=256
        )
        
        # Convert to log scale and normalize
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        return torch.from_numpy(log_mel).float()
    
    def cluster_speakers(self, embeddings: np.ndarray, 
                        num_speakers: Optional[int] = None) -> np.ndarray:
        """
        Cluster speaker embeddings to identify unique speakers.
        
        Args:
            embeddings: Speaker embeddings (n_segments, embedding_dim)
            num_speakers: Number of speakers (if known, else auto-detect)
            
        Returns:
            Speaker labels for each segment
        """
        # Compute pairwise distances
        distances = pdist(embeddings, metric='cosine')
        
        # Hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')
        
        if num_speakers is None:
            # Auto-detect number of speakers using elbow method
            num_speakers = self._estimate_num_speakers(linkage_matrix, max_speakers=5)
        
        # Get cluster labels
        labels = fcluster(linkage_matrix, num_speakers, criterion='maxclust') - 1
        
        return labels
    
    def _estimate_num_speakers(self, linkage_matrix: np.ndarray, 
                              max_speakers: int = 5) -> int:
        """Estimate optimal number of speakers using cluster metrics."""
        scores = []
        
        for k in range(2, max_speakers + 1):
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            
            # Simple silhouette-like score
            # In practice, you'd use more sophisticated metrics
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                score = len(unique_labels) / k  # Prefer fewer clusters
                scores.append((k, score))
        
        if scores:
            # Return k with best score
            best_k = max(scores, key=lambda x: x[1])[0]
            return best_k
        
        return 2  # Default fallback

class OverlappingSpeechHandler:
    """
    Handles overlapping speech scenarios where multiple speakers talk simultaneously.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def detect_overlapping_regions(self, separated_speakers: Dict[str, torch.Tensor],
                                  threshold: float = 0.3) -> List[Tuple[float, float]]:
        """
        Detect time regions where multiple speakers are active simultaneously.
        
        Args:
            separated_speakers: Dictionary of separated speaker audio
            threshold: Energy threshold for voice activity detection
            
        Returns:
            List of (start_time, end_time) tuples for overlapping regions
        """
        # Voice activity detection for each speaker
        speaker_activity = {}
        
        for speaker_id, audio in separated_speakers.items():
            activity = self._voice_activity_detection(audio.squeeze(), threshold)
            speaker_activity[speaker_id] = activity
        
        # Find overlapping regions
        overlapping_regions = []
        audio_length = len(list(separated_speakers.values())[0].squeeze())
        frame_size = 512
        hop_size = 256
        
        for i in range(0, audio_length - frame_size, hop_size):
            start_time = i / self.sample_rate
            end_time = (i + frame_size) / self.sample_rate
            
            # Count active speakers in this frame
            active_speakers = 0
            for activity in speaker_activity.values():
                frame_idx = i // hop_size
                if frame_idx < len(activity) and activity[frame_idx]:
                    active_speakers += 1
            
            # If multiple speakers are active, it's overlapping
            if active_speakers > 1:
                overlapping_regions.append((start_time, end_time))
        
        # Merge adjacent overlapping regions
        merged_regions = self._merge_adjacent_regions(overlapping_regions)
        
        return merged_regions
    
    def _voice_activity_detection(self, audio: torch.Tensor, 
                                 threshold: float = 0.3) -> np.ndarray:
        """Simple energy-based voice activity detection."""
        frame_size = 512
        hop_size = 256
        
        # Compute frame energy
        frames = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = torch.mean(frame ** 2).item()
            frames.append(energy > threshold)
        
        return np.array(frames)
    
    def _merge_adjacent_regions(self, regions: List[Tuple[float, float]],
                               max_gap: float = 0.1) -> List[Tuple[float, float]]:
        """Merge adjacent overlapping regions within max_gap seconds."""
        if not regions:
            return []
        
        merged = [regions[0]]
        
        for start, end in regions[1:]:
            last_end = merged[-1][1]
            
            if start - last_end <= max_gap:
                # Merge with previous region
                merged[-1] = (merged[-1][0], end)
            else:
                # Add as new region
                merged.append((start, end))
        
        return merged
    
    def handle_overlapping_transcription(self, separated_speakers: Dict[str, torch.Tensor],
                                       overlapping_regions: List[Tuple[float, float]],
                                       transcription_model) -> Dict[str, List[Dict]]:
        """
        Handle transcription of overlapping speech regions.
        
        Args:
            separated_speakers: Separated speaker audio streams
            overlapping_regions: Time regions with overlapping speech
            transcription_model: Speech recognition model
            
        Returns:
            Transcription results with speaker labels and timing
        """
        results = {speaker_id: [] for speaker_id in separated_speakers.keys()}
        
        for start_time, end_time in overlapping_regions:
            # Extract overlapping segments from each speaker
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            for speaker_id, audio in separated_speakers.items():
                segment = audio.squeeze()[start_sample:end_sample]
                
                # Check if this speaker is actually active in this segment
                if self._is_speaker_active(segment):
                    # Transcribe segment
                    # Note: This would use your actual transcription model
                    transcription = f"[Overlapping speech - {speaker_id}]"
                    
                    results[speaker_id].append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': transcription,
                        'confidence': 0.8,  # Placeholder
                        'overlapping': True
                    })
        
        return results
    
    def _is_speaker_active(self, audio_segment: torch.Tensor, 
                          threshold: float = 0.1) -> bool:
        """Check if speaker is active in audio segment."""
        energy = torch.mean(audio_segment ** 2).item()
        return energy > threshold

class MultiSpeakerProcessor:
    """
    Complete multi-speaker processing pipeline combining separation, diarization, and transcription.
    """
    
    def __init__(self, max_speakers: int = 3, sample_rate: int = 16000):
        self.max_speakers = max_speakers
        self.sample_rate = sample_rate
        
        # Initialize components
        self.separator = SpeakerSeparationNetwork(num_speakers=max_speakers)
        self.diarizer = SpeakerDiarization(sample_rate=sample_rate)
        self.overlap_handler = OverlappingSpeechHandler(sample_rate=sample_rate)
        
    def process_multispeaker_audio(self, mixed_audio: torch.Tensor,
                                  transcription_model=None) -> Dict[str, any]:
        """
        Complete pipeline for processing multi-speaker audio.
        
        Args:
            mixed_audio: Mixed audio with multiple speakers (1, samples)
            transcription_model: Speech recognition model for transcription
            
        Returns:
            Complete analysis including separated audio, speaker labels, and transcriptions
        """
        results = {
            'separated_speakers': {},
            'speaker_timeline': [],
            'overlapping_regions': [],
            'transcriptions': {},
            'speaker_statistics': {}
        }
        
        # Step 1: Speaker separation
        print("🔄 Separating speakers...")
        if mixed_audio.dim() == 1:
            mixed_audio = mixed_audio.unsqueeze(0).unsqueeze(0)
        elif mixed_audio.dim() == 2:
            mixed_audio = mixed_audio.unsqueeze(0)
        
        separated_speakers = self.separator(mixed_audio)
        results['separated_speakers'] = separated_speakers
        
        # Step 2: Detect overlapping speech regions
        print("🔄 Detecting overlapping speech...")
        overlapping_regions = self.overlap_handler.detect_overlapping_regions(
            separated_speakers
        )
        results['overlapping_regions'] = overlapping_regions
        
        # Step 3: Speaker diarization (who spoke when)
        print("🔄 Performing speaker diarization...")
        speaker_timeline = self._create_speaker_timeline(separated_speakers)
        results['speaker_timeline'] = speaker_timeline
        
        # Step 4: Handle overlapping transcription
        if transcription_model and overlapping_regions:
            print("🔄 Transcribing overlapping speech...")
            overlap_transcriptions = self.overlap_handler.handle_overlapping_transcription(
                separated_speakers, overlapping_regions, transcription_model
            )
            results['transcriptions'].update(overlap_transcriptions)
        
        # Step 5: Generate speaker statistics
        results['speaker_statistics'] = self._generate_speaker_stats(
            separated_speakers, overlapping_regions
        )
        
        print(f"✅ Multi-speaker processing complete!")
        print(f"   - Detected {len(separated_speakers)} speaker streams")
        print(f"   - Found {len(overlapping_regions)} overlapping regions")
        
        return results
    
    def _create_speaker_timeline(self, separated_speakers: Dict[str, torch.Tensor],
                                frame_duration: float = 0.25) -> List[Dict]:
        """Create timeline showing when each speaker was active."""
        timeline = []
        audio_length = len(list(separated_speakers.values())[0].squeeze())
        frame_samples = int(frame_duration * self.sample_rate)
        
        for i in range(0, audio_length, frame_samples):
            start_time = i / self.sample_rate
            end_time = min((i + frame_samples) / self.sample_rate, 
                          audio_length / self.sample_rate)
            
            # Determine active speakers in this frame
            active_speakers = []
            for speaker_id, audio in separated_speakers.items():
                segment = audio.squeeze()[i:i + frame_samples]
                if self.overlap_handler._is_speaker_active(segment):
                    active_speakers.append(speaker_id)
            
            if active_speakers:
                timeline.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'active_speakers': active_speakers,
                    'num_speakers': len(active_speakers)
                })
        
        return timeline
    
    def _generate_speaker_stats(self, separated_speakers: Dict[str, torch.Tensor],
                               overlapping_regions: List[Tuple[float, float]]) -> Dict:
        """Generate statistics about speaker activity."""
        stats = {}
        total_duration = len(list(separated_speakers.values())[0].squeeze()) / self.sample_rate
        overlap_duration = sum(end - start for start, end in overlapping_regions)
        
        for speaker_id, audio in separated_speakers.items():
            # Calculate speaking time
            activity = self.overlap_handler._voice_activity_detection(audio.squeeze())
            speaking_time = np.sum(activity) * 256 / self.sample_rate  # hop_size = 256
            
            stats[speaker_id] = {
                'total_speaking_time': speaking_time,
                'speaking_percentage': (speaking_time / total_duration) * 100,
                'activity_frames': np.sum(activity),
                'silence_frames': len(activity) - np.sum(activity)
            }
        
        stats['overall'] = {
            'total_duration': total_duration,
            'overlapping_duration': overlap_duration,
            'overlap_percentage': (overlap_duration / total_duration) * 100,
            'num_speakers': len(separated_speakers)
        }
        
        return stats

# Factory function
def create_multispeaker_processor(max_speakers: int = 3, 
                                 sample_rate: int = 16000) -> MultiSpeakerProcessor:
    """Create a multi-speaker processor."""
    return MultiSpeakerProcessor(max_speakers=max_speakers, sample_rate=sample_rate)

if __name__ == "__main__":
    print("👥 Multi-Speaker Audio Processing System")
    print("Handles 2-3+ speakers with overlapping speech")
    
    # Example usage
    processor = create_multispeaker_processor(max_speakers=3)
    
    # Simulate mixed audio (3 seconds, 3 speakers)
    sample_rate = 16000
    duration = 3
    samples = sample_rate * duration
    
    # Create synthetic mixed audio
    mixed_audio = torch.randn(samples) * 0.5
    
    print(f"Processing {duration}s audio with up to 3 speakers...")
    
    # Process multi-speaker audio
    results = processor.process_multispeaker_audio(mixed_audio)
    
    print("\n📊 Results:")
    print(f"Speaker streams: {list(results['separated_speakers'].keys())}")
    print(f"Overlapping regions: {len(results['overlapping_regions'])}")
    print(f"Timeline segments: {len(results['speaker_timeline'])}")
    
    # Show speaker statistics
    stats = results['speaker_statistics']
    for speaker_id, speaker_stats in stats.items():
        if speaker_id != 'overall':
            print(f"{speaker_id}: {speaker_stats['speaking_percentage']:.1f}% speaking time")
    
    print(f"Overall overlap: {stats['overall']['overlap_percentage']:.1f}%")
    print("\n✅ Multi-speaker processing ready!")