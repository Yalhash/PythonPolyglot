"""Local transcription service using faster-whisper"""

import numpy as np
from typing import Optional, Dict, Any
from faster_whisper import WhisperModel
import warnings

# Suppress warnings from faster-whisper
warnings.filterwarnings("ignore", category=UserWarning)


class TranscriptionService:
    """Handles audio transcription using local faster-whisper model"""
    
    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "auto"):
        """
        Initialize transcription service with local Whisper model
        
        Args:
            model_size: Model size - "tiny", "base", "small", "medium", "large-v2", "large-v3"
                       Smaller = faster but less accurate, Larger = slower but more accurate
            device: "cpu", "cuda", or "auto" (auto-detects)
            compute_type: "int8", "int8_float16", "int16", "float16", "float32", or "auto"
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None
        self.current_transcription: Optional[str] = None
        self.detected_language: Optional[str] = None
        
        # Initialize model (will be loaded on first use)
        print(f"Transcription service initialized (model: {model_size}, device: {device})")
        print("Note: Model will be downloaded on first use (~150MB-3GB depending on size)")
    
    def _ensure_model_loaded(self):
        """Load the Whisper model if not already loaded"""
        if self.model is None:
            try:
                print(f"Loading Whisper model '{self.model_size}'...")
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
                raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000, 
                   language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio data using local Whisper model
        
        Args:
            audio_data: Audio samples as numpy array (float32, -1.0 to 1.0)
            sample_rate: Sample rate in Hz
            language: Optional language code (e.g., 'en', 'es', 'fr') for faster processing
            
        Returns:
            Dictionary with 'text' and 'language' keys
        """
        try:
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Ensure audio is in the correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize to prevent clipping
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # faster-whisper expects audio in float32 format with values between -1.0 and 1.0
            # and sample rate of 16000 Hz (which we already have)
            
            # Transcribe using faster-whisper
            segments, info = self.model.transcribe(
                audio_data,
                language=language,
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection - filters out silence
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Get detected language
            detected_lang = info.language if hasattr(info, 'language') else None
            
            # Collect all segments into text
            text_segments = []
            for segment in segments:
                text_segments.append(segment.text.strip())
            
            # Join segments
            text = " ".join(text_segments).strip()
            
            self.current_transcription = text
            self.detected_language = detected_lang
            
            return {
                'text': text,
                'language': detected_lang
            }
        
        except Exception as e:
            error_msg = str(e)
            print(f"Transcription error: {error_msg}")
            
            return {
                'text': '',
                'language': None,
                'error': error_msg
            }
    
    async def transcribe_async(self, audio_data: np.ndarray, sample_rate: int = 16000,
                              language: Optional[str] = None) -> Dict[str, Any]:
        """
        Asynchronously transcribe audio data
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            language: Optional language code
            
        Returns:
            Dictionary with 'text' and 'language' keys
        """
        # For local model, we can just call transcribe directly
        # It's already fast enough, but we keep async interface for compatibility
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio_data, sample_rate, language)
    
    def get_last_transcription(self) -> Optional[str]:
        """Get the last transcription result"""
        return self.current_transcription
    
    def get_detected_language(self) -> Optional[str]:
        """Get the last detected language"""
        return self.detected_language
    
    def change_model(self, model_size: str):
        """Change the Whisper model size (requires reload)"""
        self.model_size = model_size
        self.model = None  # Force reload on next transcription
        print(f"Model changed to '{model_size}'. Will reload on next transcription.")
