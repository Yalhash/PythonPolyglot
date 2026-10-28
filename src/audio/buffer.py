"""Audio buffering and chunking utilities"""

import numpy as np
from collections import deque
from typing import Optional, Callable


class AudioBuffer:
    """Manages audio buffering and chunking for real-time processing"""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration_seconds: float = 5.0):
        """
        Initialize audio buffer
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_seconds: Duration of each chunk in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration_seconds = chunk_duration_seconds
        self.chunk_size = int(sample_rate * chunk_duration_seconds)
        self.buffer = deque(maxlen=self.chunk_size * 2)  # Keep some overlap
        self.chunk_callback: Optional[Callable[[np.ndarray], None]] = None
    
    def add_audio(self, audio_data: np.ndarray):
        """
        Add audio data to buffer and trigger callback when chunk is ready
        
        Args:
            audio_data: Audio samples as numpy array
        """
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Ensure mono channel
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Add to buffer
        self.buffer.extend(audio_data)
        
        # Check if we have enough data for a chunk
        if len(self.buffer) >= self.chunk_size:
            chunk = np.array(list(self.buffer)[:self.chunk_size], dtype=np.float32)
            # Remove processed samples (keep some overlap)
            overlap = self.chunk_size // 4  # 25% overlap
            for _ in range(self.chunk_size - overlap):
                if self.buffer:
                    self.buffer.popleft()
            
            # Trigger callback if set
            if self.chunk_callback:
                self.chunk_callback(chunk)
    
    def set_chunk_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function to be called when a chunk is ready"""
        self.chunk_callback = callback
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def get_current_buffer(self) -> np.ndarray:
        """Get current buffer contents as numpy array"""
        return np.array(list(self.buffer), dtype=np.float32)

