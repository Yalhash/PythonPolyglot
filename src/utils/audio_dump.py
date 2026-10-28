"""Utility for dumping audio chunks to files for debugging"""

import os
import wave
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional


class AudioDumper:
    """Handles dumping audio chunks to WAV files for debugging"""
    
    def __init__(self, dump_folder: str = "dumps", enabled: bool = False):
        """
        Initialize audio dumper
        
        Args:
            dump_folder: Folder to save audio chunks
            enabled: Whether dumping is enabled
        """
        self.dump_folder = Path(dump_folder)
        self.enabled = enabled
        self.chunk_counter = 0
        
        # Create dump folder if it doesn't exist
        if self.enabled:
            self.dump_folder.mkdir(exist_ok=True)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable audio dumping"""
        self.enabled = enabled
        if self.enabled:
            self.dump_folder.mkdir(exist_ok=True)
    
    def dump_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        Dump audio chunk to WAV file
        
        Args:
            audio_data: Audio samples as numpy array (float32, -1.0 to 1.0)
            sample_rate: Sample rate in Hz
            
        Returns:
            Path to saved file if successful, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            # Ensure audio is in the correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize to prevent clipping
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Convert to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.chunk_counter += 1
            filename = f"audio_{timestamp}_{self.chunk_counter:04d}.wav"
            filepath = self.dump_folder / filename
            
            # Write WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            return str(filepath)
        
        except Exception as e:
            print(f"Error dumping audio chunk: {e}")
            return None
    
    def clear_dumps(self):
        """Clear all dumped audio files"""
        if self.dump_folder.exists():
            for file in self.dump_folder.glob("*.wav"):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error deleting {file}: {e}")

