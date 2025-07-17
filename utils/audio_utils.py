"""
Audio processing utilities for voice-to-action system
Provides common audio manipulation, validation, and conversion functions
"""
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union, List
import io
import tempfile
import os
from pathlib import Path
import mimetypes
from dataclasses import dataclass

from config.logging import get_logger

@dataclass
class AudioInfo:
    """Audio file information"""
    duration: float
    sample_rate: int
    channels: int
    samples: int
    format: str
    size_bytes: int
    bit_depth: Optional[int] = None
    is_valid: bool = True
    error_message: Optional[str] = None

class AudioProcessor:
    """Audio processing utilities"""
    
    def __init__(self):
        self.logger = get_logger("audio_processor")
        
        # Standard audio settings
        self.target_sample_rate = 16000
        self.target_channels = 1  # Mono
        self.max_duration = 300  # 5 minutes
        self.min_duration = 0.1  # 100ms
        
        # Supported formats
        self.supported_formats = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mp3',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac',
            '.webm': 'audio/webm'
        }
    
    def get_audio_info(self, audio_data: bytes, filename: str = None) -> AudioInfo:
        """Get comprehensive audio file information"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=self._get_extension(filename)) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                
                # Load with librosa to get info
                try:
                    y, sr = librosa.load(temp_file.name, sr=None)
                    
                    # Calculate properties
                    duration = len(y) / sr
                    channels = 1 if y.ndim == 1 else y.shape[0]
                    samples = len(y)
                    
                    # Try to get format info
                    format_info = sf.info(temp_file.name)
                    file_format = format_info.format if hasattr(format_info, 'format') else 'unknown'
                    
                    return AudioInfo(
                        duration=duration,
                        sample_rate=sr,
                        channels=channels,
                        samples=samples,
                        format=file_format,
                        size_bytes=len(audio_data),
                        is_valid=self._validate_audio_properties(duration, sr, channels)
                    )
                    
                except Exception as e:
                    return AudioInfo(
                        duration=0.0,
                        sample_rate=0,
                        channels=0,
                        samples=0,
                        format='unknown',
                        size_bytes=len(audio_data),
                        is_valid=False,
                        error_message=str(e)
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to get audio info: {e}")
            return AudioInfo(
                duration=0.0,
                sample_rate=0,
                channels=0,
                samples=0,
                format='unknown',
                size_bytes=len(audio_data),
                is_valid=False,
                error_message=str(e)
            )
    
    def _get_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        if not filename:
            return '.wav'
        return Path(filename).suffix.lower() or '.wav'
    
    def _validate_audio_properties(self, duration: float, sample_rate: int, channels: int) -> bool:
        """Validate audio properties"""
        if duration < self.min_duration or duration > self.max_duration:
            return False
        if sample_rate < 8000 or sample_rate > 48000:
            return False
        if channels < 1 or channels > 2:
            return False
        return True
    
    def normalize_audio(self, audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
        """Normalize audio to target level"""
        try:
            if np.max(np.abs(audio)) > 0:
                return audio / np.max(np.abs(audio)) * target_level
            return audio
        except Exception as e:
            self.logger.error(f"Audio normalization failed: {e}")
            return audio
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono"""
        try:
            if audio.ndim > 1:
                return librosa.to_mono(audio)
            return audio
        except Exception as e:
            self.logger.error(f"Mono conversion failed: {e}")
            return audio
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int = None) -> Tuple[np.ndarray, int]:
        """Resample audio to target sample rate"""
        try:
            target_sr = target_sr or self.target_sample_rate
            
            if orig_sr != target_sr:
                audio_resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
                return audio_resampled, target_sr
            
            return audio, orig_sr
            
        except Exception as e:
            self.logger.error(f"Audio resampling failed: {e}")
            return audio, orig_sr
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Trim silence from beginning and end of audio"""
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return trimmed
        except Exception as e:
            self.logger.error(f"Silence trimming failed: {e}")
            return audio
    
    def apply_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply basic noise reduction"""
        try:
            # Simple high-pass filter to remove low-frequency noise
            audio_filtered = librosa.effects.preemphasis(audio, coef=0.97)
            
            # Simple spectral gating
            stft = librosa.stft(audio_filtered)
            magnitude = np.abs(stft)
            
            # Calculate noise threshold (bottom 10% of magnitude)
            noise_threshold = np.percentile(magnitude, 10)
            
            # Apply spectral gating
            mask = magnitude > noise_threshold * 2
            stft_cleaned = stft * mask
            
            # Convert back to time domain
            audio_cleaned = librosa.istft(stft_cleaned)
            
            return audio_cleaned
            
        except Exception as e:
            self.logger.error(f"Noise reduction failed: {e}")
            return audio
    
    def split_audio_by_silence(
        self, 
        audio: np.ndarray, 
        sr: int, 
        min_silence_len: float = 0.5,
        silence_thresh: float = -40
    ) -> List[Tuple[np.ndarray, float, float]]:
        """Split audio into segments based on silence"""
        try:
            # Convert silence threshold from dB to amplitude
            silence_thresh_amp = librosa.db_to_amplitude(silence_thresh)
            
            # Find segments
            intervals = librosa.effects.split(
                audio, 
                top_db=abs(silence_thresh),
                frame_length=2048,
                hop_length=512
            )
            
            segments = []
            for start_sample, end_sample in intervals:
                start_time = start_sample / sr
                end_time = end_sample / sr
                duration = end_time - start_time
                
                # Only include segments longer than minimum duration
                if duration >= min_silence_len:
                    segment_audio = audio[start_sample:end_sample]
                    segments.append((segment_audio, start_time, end_time))
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Audio splitting failed: {e}")
            return [(audio, 0.0, len(audio) / sr)]
    
    def convert_format(
        self, 
        audio_data: bytes, 
        input_format: str, 
        output_format: str = 'wav'
    ) -> bytes:
        """Convert audio between formats"""
        try:
            # Create temporary input file
            input_suffix = f".{input_format.lower()}"
            output_suffix = f".{output_format.lower()}"
            
            with tempfile.NamedTemporaryFile(suffix=input_suffix) as input_file:
                input_file.write(audio_data)
                input_file.flush()
                
                # Load audio
                audio, sr = librosa.load(input_file.name, sr=None)
                
                # Create temporary output file
                with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as output_file:
                    output_path = output_file.name
                
                try:
                    # Write in target format
                    sf.write(output_path, audio, sr, format=output_format.upper())
                    
                    # Read converted file
                    with open(output_path, 'rb') as f:
                        converted_data = f.read()
                    
                    return converted_data
                    
                finally:
                    # Clean up temporary output file
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                        
        except Exception as e:
            self.logger.error(f"Format conversion failed: {e}")
            return audio_data  # Return original if conversion fails
    
    def extract_features(self, audio: np.ndarray, sr: int) -> dict:
        """Extract audio features for analysis"""
        try:
            features = {}
            
            # Basic features
            features['duration'] = len(audio) / sr
            features['sample_rate'] = sr
            features['channels'] = 1 if audio.ndim == 1 else audio.shape[0]
            
            # Energy features
            features['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # MFCC features (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_means'] = [float(np.mean(mfcc)) for mfcc in mfccs]
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {
                'duration': len(audio) / sr if sr > 0 else 0,
                'sample_rate': sr,
                'error': str(e)
            }
    
    def validate_audio_quality(self, audio: np.ndarray, sr: int) -> dict:
        """Validate audio quality and provide recommendations"""
        try:
            quality_report = {
                'overall_quality': 'good',
                'issues': [],
                'recommendations': [],
                'metrics': {}
            }
            
            # Check duration
            duration = len(audio) / sr
            quality_report['metrics']['duration'] = duration
            
            if duration < self.min_duration:
                quality_report['issues'].append('Audio too short')
                quality_report['recommendations'].append('Record longer audio')
                quality_report['overall_quality'] = 'poor'
            elif duration > self.max_duration:
                quality_report['issues'].append('Audio too long')
                quality_report['recommendations'].append('Trim audio to under 5 minutes')
                quality_report['overall_quality'] = 'fair'
            
            # Check sample rate
            quality_report['metrics']['sample_rate'] = sr
            if sr < 16000:
                quality_report['issues'].append('Low sample rate')
                quality_report['recommendations'].append('Use at least 16kHz sample rate')
                quality_report['overall_quality'] = 'fair'
            
            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
            quality_report['metrics']['clipping_ratio'] = clipping_ratio
            if clipping_ratio > 0.01:  # More than 1% clipped
                quality_report['issues'].append('Audio clipping detected')
                quality_report['recommendations'].append('Reduce recording volume')
                quality_report['overall_quality'] = 'fair'
            
            # Check signal-to-noise ratio (simple estimation)
            rms_energy = np.sqrt(np.mean(audio**2))
            quality_report['metrics']['rms_energy'] = rms_energy
            
            if rms_energy < 0.01:  # Very quiet audio
                quality_report['issues'].append('Very low audio level')
                quality_report['recommendations'].append('Increase recording volume')
                quality_report['overall_quality'] = 'fair'
            elif rms_energy > 0.7:  # Very loud audio
                quality_report['issues'].append('Very high audio level')
                quality_report['recommendations'].append('Reduce recording volume')
            
            # Check for silence
            silence_ratio = np.sum(np.abs(audio) < 0.001) / len(audio)
            quality_report['metrics']['silence_ratio'] = silence_ratio
            if silence_ratio > 0.8:  # More than 80% silence
                quality_report['issues'].append('Too much silence')
                quality_report['recommendations'].append('Remove silence or re-record')
                quality_report['overall_quality'] = 'poor'
            
            # Overall quality assessment
            if not quality_report['issues']:
                quality_report['overall_quality'] = 'excellent'
            elif len(quality_report['issues']) == 1 and quality_report['overall_quality'] == 'good':
                quality_report['overall_quality'] = 'good'
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Audio quality validation failed: {e}")
            return {
                'overall_quality': 'unknown',
                'issues': ['Quality validation failed'],
                'recommendations': ['Check audio file format'],
                'error': str(e)
            }
    
    def process_for_speech_recognition(self, audio_data: bytes, filename: str = None) -> Tuple[bytes, dict]:
        """Process audio for optimal speech recognition"""
        try:
            # Get audio info
            audio_info = self.get_audio_info(audio_data, filename)
            
            if not audio_info.is_valid:
                raise ValueError(f"Invalid audio: {audio_info.error_message}")
            
            # Load audio
            with tempfile.NamedTemporaryFile(suffix=self._get_extension(filename)) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                
                audio, sr = librosa.load(temp_file.name, sr=None)
            
            processing_info = {
                'original_duration': audio_info.duration,
                'original_sample_rate': audio_info.sample_rate,
                'processing_steps': []
            }
            
            # Convert to mono
            if audio_info.channels > 1:
                audio = self.convert_to_mono(audio)
                processing_info['processing_steps'].append('converted_to_mono')
            
            # Resample to target rate
            if sr != self.target_sample_rate:
                audio, sr = self.resample_audio(audio, sr, self.target_sample_rate)
                processing_info['processing_steps'].append(f'resampled_to_{self.target_sample_rate}Hz')
            
            # Normalize audio
            original_max = np.max(np.abs(audio))
            audio = self.normalize_audio(audio, target_level=0.95)
            if original_max < 0.1 or original_max > 0.98:
                processing_info['processing_steps'].append('normalized')
            
            # Trim silence
            original_length = len(audio)
            audio = self.trim_silence(audio, top_db=20)
            if len(audio) < original_length * 0.9:
                processing_info['processing_steps'].append('trimmed_silence')
            
            # Apply noise reduction for poor quality audio
            quality_report = self.validate_audio_quality(audio, sr)
            if quality_report['overall_quality'] in ['poor', 'fair']:
                audio = self.apply_noise_reduction(audio, sr)
                processing_info['processing_steps'].append('noise_reduction')
            
            # Convert back to bytes (WAV format)
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_output:
                sf.write(temp_output.name, audio, sr, format='WAV')
                temp_output.flush()
                
                with open(temp_output.name, 'rb') as f:
                    processed_audio_data = f.read()
            
            # Update processing info
            processing_info.update({
                'final_duration': len(audio) / sr,
                'final_sample_rate': sr,
                'final_channels': 1,
                'size_reduction': len(audio_data) - len(processed_audio_data),
                'quality_assessment': quality_report
            })
            
            return processed_audio_data, processing_info
            
        except Exception as e:
            self.logger.error(f"Audio processing for speech recognition failed: {e}")
            return audio_data, {'error': str(e), 'processing_steps': []}

# Global audio processor instance
audio_processor = AudioProcessor()

# Convenience functions
def get_audio_info(audio_data: bytes, filename: str = None) -> AudioInfo:
    """Get audio information using the global processor"""
    return audio_processor.get_audio_info(audio_data, filename)

def normalize_audio(audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
    """Normalize audio using the global processor"""
    return audio_processor.normalize_audio(audio, target_level)

def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert to mono using the global processor"""
    return audio_processor.convert_to_mono(audio)

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Resample audio using the global processor"""
    return audio_processor.resample_audio(audio, orig_sr, target_sr)

def process_for_speech(audio_data: bytes, filename: str = None) -> Tuple[bytes, dict]:
    """Process audio for speech recognition using the global processor"""
    return audio_processor.process_for_speech_recognition(audio_data, filename)

def validate_audio_quality(audio: np.ndarray, sr: int) -> dict:
    """Validate audio quality using the global processor"""
    return audio_processor.validate_audio_quality(audio, sr)

def extract_audio_features(audio: np.ndarray, sr: int) -> dict:
    """Extract audio features using the global processor"""
    return audio_processor.extract_features(audio, sr)

