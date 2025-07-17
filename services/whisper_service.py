"""
Whisper Service - Real Speech-to-Text Implementation
Replaces mock transcription with actual OpenAI Whisper
"""
import whisper
import torch
import numpy as np
import librosa
import tempfile
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time
from dataclasses import dataclass

from config.settings import settings
from config.logging import get_logger

@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription"""
    text: str
    language: str
    confidence: float
    duration: float
    processing_time: float
    segments: list = None
    word_timestamps: list = None

class WhisperService:
    """OpenAI Whisper speech-to-text service"""
    
    def __init__(self, model_size: str = "base"):
        self.logger = get_logger("whisper_service")
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"
        ]
        
        # Audio preprocessing settings
        self.target_sample_rate = 16000
        self.max_duration = 300  # 5 minutes max
        self.min_duration = 0.1  # 100ms minimum
        
    async def initialize(self):
        """Initialize Whisper model"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            start_time = time.time()
            
            # Load the model
            self.model = whisper.load_model(self.model_size, device=self.device)
            
            load_time = time.time() - start_time
            self.logger.info(f"Whisper model loaded in {load_time:.2f}s on {self.device}")
            
            # Test the model with a simple transcription
            await self._test_model()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper: {e}")
            raise
    
    async def _test_model(self):
        """Test the model with a simple audio"""
        try:
            # Create a simple test audio (1 second of silence)
            test_audio = np.zeros(self.target_sample_rate)
            result = self.model.transcribe(test_audio, language="en")
            self.logger.info("Whisper model test successful")
        except Exception as e:
            self.logger.warning(f"Whisper model test failed: {e}")
    
    def _validate_audio_data(self, audio_data: bytes) -> bool:
        """Validate audio data before processing"""
        try:
            # Check file size (max 50MB)
            if len(audio_data) > 50 * 1024 * 1024:
                raise ValueError("Audio file too large (max 50MB)")
            
            # Check minimum size (at least 1KB)
            if len(audio_data) < 1024:
                raise ValueError("Audio file too small (min 1KB)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio validation failed: {e}")
            return False
    
    def _preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocess audio file for Whisper"""
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            
            # Validate duration
            duration = len(audio) / sr
            if duration > self.max_duration:
                self.logger.warning(f"Audio too long ({duration:.1f}s), truncating to {self.max_duration}s")
                audio = audio[:int(self.max_duration * sr)]
            elif duration < self.min_duration:
                raise ValueError(f"Audio too short ({duration:.1f}s), minimum {self.min_duration}s")
            
            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            # Apply basic noise reduction (simple high-pass filter)
            audio = librosa.effects.preemphasis(audio, coef=0.97)
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score from Whisper result"""
        try:
            # Whisper doesn't provide direct confidence scores
            # We'll estimate based on various factors
            
            segments = result.get("segments", [])
            if not segments:
                return 0.5  # Default moderate confidence
            
            # Average the probability scores from segments
            probabilities = []
            for segment in segments:
                # Whisper provides avg_logprob which we convert to probability
                avg_logprob = segment.get("avg_logprob", -1.0)
                prob = np.exp(avg_logprob) if avg_logprob > -10 else 0.1
                probabilities.append(prob)
            
            if probabilities:
                base_confidence = np.mean(probabilities)
            else:
                base_confidence = 0.5
            
            # Adjust based on text characteristics
            text = result.get("text", "").strip()
            
            # Longer texts generally more reliable
            length_factor = min(len(text) / 100, 1.0)
            
            # Penalize very short transcriptions
            if len(text) < 10:
                length_factor *= 0.5
            
            # Penalize texts with many repeated characters
            if len(set(text.lower())) < len(text) * 0.3:
                length_factor *= 0.7
            
            # Final confidence score
            confidence = base_confidence * (0.7 + 0.3 * length_factor)
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        include_segments: bool = False,
        include_word_timestamps: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe audio data to text using Whisper
        
        Args:
            audio_data: Raw audio bytes
            language: Target language code (auto-detect if None)
            task: "transcribe" or "translate" 
            temperature: Sampling temperature (0.0 for deterministic)
            include_segments: Include segment-level timestamps
            include_word_timestamps: Include word-level timestamps
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not self.model:
                raise RuntimeError("Whisper model not initialized")
            
            if not self._validate_audio_data(audio_data):
                raise ValueError("Invalid audio data")
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Preprocess audio
                audio = self._preprocess_audio(temp_path)
                audio_duration = len(audio) / self.target_sample_rate
                
                # Prepare Whisper options
                options = {
                    "task": task,
                    "temperature": temperature,
                    "verbose": False
                }
                
                if language and language in self.supported_languages:
                    options["language"] = language
                
                if include_word_timestamps and self.model_size in ["small", "medium", "large"]:
                    options["word_timestamps"] = True
                
                # Run transcription
                transcription_start = time.time()
                result = self.model.transcribe(audio, **options)
                transcription_time = time.time() - transcription_start
                
                # Extract results
                text = result["text"].strip()
                detected_language = result.get("language", "unknown")
                confidence = self._calculate_confidence(result)
                
                # Extract segments if requested
                segments = None
                if include_segments and "segments" in result:
                    segments = [
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"].strip(),
                            "confidence": np.exp(seg.get("avg_logprob", -1.0))
                        }
                        for seg in result["segments"]
                    ]
                
                # Extract word timestamps if available
                word_timestamps = None
                if include_word_timestamps and segments:
                    word_timestamps = []
                    for seg in result.get("segments", []):
                        for word in seg.get("words", []):
                            word_timestamps.append({
                                "word": word["word"],
                                "start": word["start"], 
                                "end": word["end"],
                                "confidence": word.get("probability", 0.5)
                            })
                
                processing_time = time.time() - start_time
                
                self.logger.info(
                    f"Transcription complete: {len(text)} chars, "
                    f"{audio_duration:.1f}s audio, {processing_time:.2f}s processing, "
                    f"confidence: {confidence:.2f}"
                )
                
                return TranscriptionResult(
                    text=text,
                    language=detected_language,
                    confidence=confidence,
                    duration=audio_duration,
                    processing_time=processing_time,
                    segments=segments,
                    word_timestamps=word_timestamps
                )
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Transcription failed after {processing_time:.2f}s: {e}")
            
            # Return error result
            return TranscriptionResult(
                text="",
                language="unknown",
                confidence=0.0,
                duration=0.0,
                processing_time=processing_time
            )
    
    async def transcribe_file(
        self,
        file_path: str,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio file by path"""
        try:
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            return await self.transcribe_audio(audio_data, **kwargs)
            
        except Exception as e:
            self.logger.error(f"File transcription failed: {e}")
            return TranscriptionResult(
                text="",
                language="unknown", 
                confidence=0.0,
                duration=0.0,
                processing_time=0.0
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "is_loaded": self.model is not None,
            "supported_languages": self.supported_languages,
            "max_duration": self.max_duration,
            "target_sample_rate": self.target_sample_rate
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Whisper service"""
        try:
            if not self.model:
                return {"status": "unhealthy", "reason": "Model not loaded"}
            
            # Quick test with minimal audio
            test_audio = np.zeros(1000)  # Very short audio
            start_time = time.time()
            result = self.model.transcribe(test_audio, language="en", verbose=False)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "model_size": self.model_size,
                "device": self.device,
                "response_time": response_time,
                "test_result": len(result.get("text", "")) >= 0
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e)
            }

# Global Whisper service instance
whisper_service = WhisperService(model_size=settings.whisper_model_size if hasattr(settings, 'whisper_model_size') else "base")

# Convenience functions
async def initialize_whisper():
    """Initialize the global Whisper service"""
    await whisper_service.initialize()

async def transcribe_audio(audio_data: bytes, **kwargs) -> TranscriptionResult:
    """Transcribe audio using the global service"""
    return await whisper_service.transcribe_audio(audio_data, **kwargs)

async def transcribe_file(file_path: str, **kwargs) -> TranscriptionResult:
    """Transcribe file using the global service"""
    return await whisper_service.transcribe_file(file_path, **kwargs)

