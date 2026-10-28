"""System audio capture module with cross-platform support"""

import sounddevice as sd
import numpy as np
import platform
from typing import Optional, Callable, List
from .buffer import AudioBuffer


class AudioCapture:
    """Captures system audio using platform-specific loopback"""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration_seconds: float = 5.0):
        """
        Initialize audio capture
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_seconds: Duration of each chunk in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration_seconds = chunk_duration_seconds
        self.buffer = AudioBuffer(sample_rate, chunk_duration_seconds)
        self.stream: Optional[sd.InputStream] = None
        self.is_capturing = False
        self.device_id: Optional[int] = None
    
    def list_devices(self) -> List[dict]:
        """List available audio devices"""
        devices = sd.query_devices()
        result = []
        for i, device in enumerate(devices):
            result.append({
                'id': i,
                'name': device.get('name', f'Device {i}'),
                'max_input_channels': device.get('max_input_channels', 0),
                'max_output_channels': device.get('max_output_channels', 0),
                'sample_rate': device.get('default_samplerate', 44100),
                'is_input': device.get('max_input_channels', 0) > 0
            })
        return result
    
    def find_loopback_device(self) -> Optional[int]:
        """
        Find the appropriate loopback device for system audio capture
        
        Returns:
            Device ID if found, None otherwise
        """
        system = platform.system()
        devices = sd.query_devices()
        
        if system == "Windows":
            # On Windows, look for WASAPI loopback devices
            # WASAPI loopback devices are typically named with "(loopback)" suffix
            for i, device in enumerate(devices):
                name = device['name'].lower()
                hostapi = device.get('hostapi', 0)
                
                # Check if it's a WASAPI device with loopback in the name
                if 'loopback' in name:
                    # Verify it has input channels (loopback devices show as input)
                    if device.get('max_input_channels', 0) > 0:
                        return i
                
                # Also check for WASAPI devices that might be loopback-capable
                # WASAPI hostapi is typically 3
                if hostapi == 3 and device.get('max_input_channels', 0) > 0:
                    # Check if it's a loopback by checking the name pattern
                    if '(loopback)' in device['name'] or 'loopback' in name:
                        return i
            
            # Try to find default WASAPI output and see if there's a corresponding loopback
            try:
                default_output = sd.query_devices(kind='output')
                if default_output:
                    # Look for a loopback version of the default output
                    default_name = default_output['name'].lower()
                    for i, device in enumerate(devices):
                        if 'loopback' in device['name'].lower() and default_name.split('(')[0].strip() in device['name'].lower():
                            if device.get('max_input_channels', 0) > 0:
                                return i
            except:
                pass
        
        elif system == "Darwin":  # macOS
            # On macOS, look for CoreAudio devices that support loopback
            for i, device in enumerate(devices):
                name = device['name'].lower()
                # macOS loopback devices might be named differently
                if 'blackhole' in name or 'soundflower' in name or 'loopback' in name:
                    if device.get('max_input_channels', 0) > 0:
                        return i
            # Note: macOS requires special virtual audio devices for loopback
            # Users may need to install BlackHole or similar
        
        # Fallback: return None to use default
        # Note: On Windows, this might not work for system audio capture
        # Users may need to install a virtual audio cable or use OBS Virtual Audio
        return None
    
    def start_capture(self, device_id: Optional[int] = None, 
                     chunk_callback: Optional[Callable[[np.ndarray], None]] = None):
        """
        Start capturing audio
        
        Args:
            device_id: Audio device ID (None for auto-detect)
            chunk_callback: Callback function for audio chunks
        """
        if self.is_capturing:
            self.stop_capture()
        
        # Set up chunk callback
        if chunk_callback:
            self.buffer.set_chunk_callback(chunk_callback)
        
        # Determine device
        if device_id is not None:
            self.device_id = device_id
        else:
            self.device_id = self.find_loopback_device()
        
        # Get device info to determine channel count
        device_info = None
        if self.device_id is not None:
            try:
                device_info = sd.query_devices(self.device_id)
            except:
                pass
        
        # Determine number of channels to use
        # For loopback devices, we typically need to use the output channels as input
        if device_info:
            # Try to use input channels if available, otherwise use output channels
            max_input_channels = device_info.get('max_input_channels', 0)
            max_output_channels = device_info.get('max_output_channels', 0)
            
            # For loopback, we often need to use output channels
            # Start with 2 (stereo) as most devices support it, fallback to 1 (mono)
            channels = 2 if max_output_channels >= 2 or max_input_channels >= 2 else 1
        else:
            # Default to stereo (most common)
            channels = 2
        
        # Create audio stream
        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=channels,
                samplerate=self.sample_rate,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            )
            self.stream.start()
            self.is_capturing = True
            print(f"Audio capture started with {channels} channel(s) on device {self.device_id}")
        except Exception as e:
            # Try with mono if stereo failed
            if channels == 2:
                try:
                    print(f"Failed with {channels} channels, trying mono...")
                    self.stream = sd.InputStream(
                        device=self.device_id,
                        channels=1,
                        samplerate=self.sample_rate,
                        dtype=np.float32,
                        callback=self._audio_callback,
                        blocksize=int(self.sample_rate * 0.1)
                    )
                    self.stream.start()
                    self.is_capturing = True
                    print(f"Audio capture started with 1 channel on device {self.device_id}")
                except Exception as e2:
                    print(f"Error starting audio capture: {e2}")
                    raise e2
            else:
                print(f"Error starting audio capture: {e}")
                raise
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add audio data to buffer
        self.buffer.add_audio(indata)
    
    def stop_capture(self):
        """Stop capturing audio"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_capturing = False
        self.buffer.clear()
    
    def is_active(self) -> bool:
        """Check if audio capture is active"""
        return self.is_capturing and self.stream is not None and self.stream.active
    
    def get_current_audio(self) -> np.ndarray:
        """Get current buffered audio"""
        return self.buffer.get_current_buffer()

