"""
Voice Activity Detection (VAD) Service
Detects speech segments in audio to improve processing efficiency
"""
import numpy as np
import librosa
from typing import List, Tuple, Optional
from dataclasses import dataclass
import webrtcvad
import struct
from config.logging import get_logger

@dataclass
class VoiceSegment:
    """Voice activity segment"""
    start_time: float
    end_time: float
    confidence: float
    duration: float
    
    @property
    def start_sample(self) -> int:
        """Start sample index at 16kHz"""
        return int(self.start_time * 16000)
    
    @property
    def end_sample(self) -> int:
        """End sample index at 16kHz"""
        return int(self.end_time * 16000)

class VoiceActivityDetector:
    """Voice Activity Detection using WebRTC VAD and energy-based methods"""
    
    def __init__(self):
        self.logger = get_logger("voice_activity_detector")
        
        # WebRTC VAD settings
        self.webrtc_vad = webrtcvad.Vad()
        self.webrtc_aggressiveness = 2  # 0-3, higher = more aggressive
        self.webrtc_vad.set_mode(self.webrtc_aggressiveness)
        
        # Audio processing settings
        self.sample_rate = 16000  # WebRTC VAD requires 16kHz
        self.frame_duration_ms = 30  # 10, 20, or 30 ms
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # Energy-based VAD settings
        self.energy_threshold_ratio = 0.02  # Ratio of max energy
        self.min_voice_duration = 0.1  # Minimum voice segment duration (seconds)
        self.max_silence_duration = 0.5  # Maximum silence within voice segment
        
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to proper range for VAD"""
        # Ensure audio is in [-1, 1] range
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Convert to 16-bit PCM range for WebRTC VAD
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16
    
    def _audio_to_frames(self, audio: np.ndarray) -> List[bytes]:
        """Convert audio array to frames for WebRTC VAD"""
        frames = []
        
        # Pad audio to ensure complete frames
        total_samples_needed = (len(audio) // self.frame_size + 1) * self.frame_size
        if len(audio) < total_samples_needed:
            audio = np.pad(audio, (0, total_samples_needed - len(audio)))
        
        # Split into frames
        for i in range(0, len(audio), self.frame_size):
            frame = audio[i:i + self.frame_size]
            if len(frame) == self.frame_size:
                # Convert to bytes
                frame_bytes = struct.pack('<' + 'h' * len(frame), *frame)
                frames.append(frame_bytes)
        
        return frames
    
    def _webrtc_vad_detect(self, audio: np.ndarray) -> List[bool]:
        """Detect voice activity using WebRTC VAD"""
        try:
            # Normalize and convert audio
            audio_int16 = self._normalize_audio(audio)
            
            # Convert to frames
            frames = self._audio_to_frames(audio_int16)
            
            # Process each frame
            voice_flags = []
            for frame in frames:
                try:
                    is_speech = self.webrtc_vad.is_speech(frame, self.sample_rate)
                    voice_flags.append(is_speech)
                except Exception as e:
                    # If WebRTC VAD fails, assume no speech
                    voice_flags.append(False)
            
            return voice_flags
            
        except Exception as e:
            self.logger.error(f"WebRTC VAD failed: {e}")
            # Return all False if VAD fails
            return [False] * (len(audio) // self.frame_size)
    
    def _energy_based_vad(self, audio: np.ndarray) -> List[bool]:
        """Detect voice activity using energy-based method"""
        try:
            # Calculate frame-wise energy
            frame_length = self.frame_size
            hop_length = frame_length // 2
            
            # Use librosa for frame-wise energy calculation
            energy = librosa.feature.rms(
                y=audio,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # Calculate dynamic threshold
            max_energy = np.max(energy)
            threshold = max_energy * self.energy_threshold_ratio
            
            # Apply threshold
            voice_flags = energy > threshold
            
            # Convert to frame-aligned flags
            aligned_flags = []
            for i in range(0, len(audio), self.frame_size):
                frame_idx = min(i // hop_length, len(voice_flags) - 1)
                aligned_flags.append(voice_flags[frame_idx])
            
            return aligned_flags
            
        except Exception as e:
            self.logger.error(f"Energy-based VAD failed: {e}")
            return [True] * (len(audio) // self.frame_size)  # Assume all speech if failed
    
    def _combine_vad_results(self, webrtc_flags: List[bool], energy_flags: List[bool]) -> List[bool]:
        """Combine WebRTC and energy-based VAD results"""
        # Use logical OR - if either method detects speech, consider it speech
        min_length = min(len(webrtc_flags), len(energy_flags))
        combined = []
        
        for i in range(min_length):
            # Combine results with WebRTC having higher weight
            combined.append(webrtc_flags[i] or energy_flags[i])
        
        return combined
    
    def _smooth_vad_results(self, voice_flags: List[bool]) -> List[bool]:
        """Smooth VAD results to remove short gaps and segments"""
        if not voice_flags:
            return voice_flags
        
        smoothed = voice_flags.copy()
        
        # Fill short gaps in voice activity
        max_gap_frames = int(self.max_silence_duration * 1000 / self.frame_duration_ms)
        
        i = 0
        while i < len(smoothed) - max_gap_frames:
            if smoothed[i] and not smoothed[i + max_gap_frames]:
                # Check if there's voice activity after the gap
                for j in range(i + 1, min(i + max_gap_frames + 1, len(smoothed))):
                    if smoothed[j]:
                        # Fill the gap
                        for k in range(i, j):
                            smoothed[k] = True
                        break
            i += 1
        
        # Remove short voice segments
        min_voice_frames = int(self.min_voice_duration * 1000 / self.frame_duration_ms)
        
        i = 0
        while i < len(smoothed):
            if smoothed[i]:
                # Find end of voice segment
                segment_start = i
                while i < len(smoothed) and smoothed[i]:
                    i += 1
                segment_end = i
                
                # If segment is too short, remove it
                if segment_end - segment_start < min_voice_frames:
                    for j in range(segment_start, segment_end):
                        smoothed[j] = False
            else:
                i += 1
        
        return smoothed
    
    def _flags_to_segments(self, voice_flags: List[bool]) -> List[VoiceSegment]:
        """Convert voice flags to time segments"""
        segments = []
        
        if not voice_flags:
            return segments
        
        i = 0
        while i < len(voice_flags):
            if voice_flags[i]:
                # Start of voice segment
                segment_start = i
                
                # Find end of voice segment
                while i < len(voice_flags) and voice_flags[i]:
                    i += 1
                segment_end = i
                
                # Convert frame indices to time
                start_time = segment_start * self.frame_duration_ms / 1000
                end_time = segment_end * self.frame_duration_ms / 1000
                duration = end_time - start_time
                
                # Calculate confidence based on density of voice frames
                voice_frame_count = sum(voice_flags[segment_start:segment_end])
                total_frames = segment_end - segment_start
                confidence = voice_frame_count / total_frames if total_frames > 0 else 0.0
                
                segments.append(VoiceSegment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    duration=duration
                ))
            else:
                i += 1
        
        return segments
    
    def detect_voice_activity(
        self, 
        audio: np.ndarray, 
        sample_rate: int = None,
        method: str = "combined"
    ) -> List[VoiceSegment]:
        """
        Detect voice activity in audio
        
        Args:
            audio: Audio signal as numpy array
            sample_rate: Sample rate of audio (will resample to 16kHz if different)
            method: Detection method ("webrtc", "energy", "combined")
        
        Returns:
            List of voice segments with timing information
        """
        try:
            # Resample to 16kHz if necessary
            if sample_rate and sample_rate != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Ensure audio is mono
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            
            # Apply different VAD methods
            if method == "webrtc":
                voice_flags = self._webrtc_vad_detect(audio)
            elif method == "energy":
                voice_flags = self._energy_based_vad(audio)
            elif method == "combined":
                webrtc_flags = self._webrtc_vad_detect(audio)
                energy_flags = self._energy_based_vad(audio)
                voice_flags = self._combine_vad_results(webrtc_flags, energy_flags)
            else:
                raise ValueError(f"Unknown VAD method: {method}")
            
            # Smooth results
            smoothed_flags = self._smooth_vad_results(voice_flags)
            
            # Convert to segments
            segments = self._flags_to_segments(smoothed_flags)
            
            self.logger.info(
                f"VAD detected {len(segments)} voice segments "
                f"in {len(audio)/self.sample_rate:.1f}s audio using {method} method"
            )
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Voice activity detection failed: {e}")
            # Return full audio as single segment if VAD fails
            duration = len(audio) / self.sample_rate
            return [VoiceSegment(
                start_time=0.0,
                end_time=duration,
                confidence=0.5,
                duration=duration
            )]
    
    def extract_voice_segments(
        self, 
        audio: np.ndarray, 
        segments: List[VoiceSegment],
        sample_rate: int = None
    ) -> List[Tuple[np.ndarray, VoiceSegment]]:
        """
        Extract audio data for voice segments
        
        Args:
            audio: Original audio signal
            segments: Voice segments from detect_voice_activity
            sample_rate: Sample rate of audio
        
        Returns:
            List of (audio_segment, segment_info) tuples
        """
        try:
            actual_sample_rate = sample_rate or self.sample_rate
            extracted_segments = []
            
            for segment in segments:
                # Calculate sample indices
                start_sample = int(segment.start_time * actual_sample_rate)
                end_sample = int(segment.end_time * actual_sample_rate)
                
                # Extract audio segment
                if start_sample < len(audio) and end_sample > start_sample:
                    audio_segment = audio[start_sample:min(end_sample, len(audio))]
                    extracted_segments.append((audio_segment, segment))
            
            return extracted_segments
            
        except Exception as e:
            self.logger.error(f"Voice segment extraction failed: {e}")
            return [(audio, VoiceSegment(
                start_time=0.0,
                end_time=len(audio) / (sample_rate or self.sample_rate),
                confidence=0.5,
                duration=len(audio) / (sample_rate or self.sample_rate)
            ))]
    
    def get_voice_statistics(self, segments: List[VoiceSegment]) -> dict:
        """Get statistics about voice activity"""
        if not segments:
            return {
                "total_segments": 0,
                "total_voice_duration": 0.0,
                "average_segment_duration": 0.0,
                "average_confidence": 0.0
            }
        
        total_duration = sum(seg.duration for seg in segments)
        average_duration = total_duration / len(segments)
        average_confidence = sum(seg.confidence for seg in segments) / len(segments)
        
        return {
            "total_segments": len(segments),
            "total_voice_duration": total_duration,
            "average_segment_duration": average_duration,
            "average_confidence": average_confidence,
            "longest_segment": max(seg.duration for seg in segments),
            "shortest_segment": min(seg.duration for seg in segments)
        }

# Global VAD instance
voice_activity_detector = VoiceActivityDetector()

# Convenience functions
def detect_voice(audio: np.ndarray, sample_rate: int = None, **kwargs) -> List[VoiceSegment]:
    """Detect voice activity using the global detector"""
    return voice_activity_detector.detect_voice_activity(audio, sample_rate, **kwargs)

def extract_voice_audio(
    audio: np.ndarray, 
    segments: List[VoiceSegment], 
    sample_rate: int = None
) -> List[Tuple[np.ndarray, VoiceSegment]]:
    """Extract voice segments using the global detector"""
    return voice_activity_detector.extract_voice_segments(audio, segments, sample_rate)


