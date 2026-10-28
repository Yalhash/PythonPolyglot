"""Configuration management for Polyglot application"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

CONFIG_DIR = Path.home() / ".polyglot"
CONFIG_FILE = CONFIG_DIR / "config.json"


class Settings:
    """Manages application settings and configuration"""
    
    def __init__(self):
        """Initialize settings from environment and config file"""
        self.config_dir = CONFIG_DIR
        self.config_file = CONFIG_FILE
        self._ensure_config_dir()
        self._load_config()
    
    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist"""
        self.config_dir.mkdir(exist_ok=True)
    
    def _load_config(self):
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._config = {}
        else:
            self._config = {}
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            print(f"Error saving config: {e}")
    
    # API Keys
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment or config"""
        return os.getenv("OPENAI_API_KEY") or self._config.get("openai_api_key")
    
    @openai_api_key.setter
    def openai_api_key(self, value: str):
        """Set OpenAI API key"""
        self._config["openai_api_key"] = value
        self._save_config()
    
    @property
    def google_translate_api_key(self) -> Optional[str]:
        """Get Google Translate API key from environment or config"""
        return os.getenv("GOOGLE_TRANSLATE_API_KEY") or self._config.get("google_translate_api_key")
    
    @google_translate_api_key.setter
    def google_translate_api_key(self, value: str):
        """Set Google Translate API key"""
        self._config["google_translate_api_key"] = value
        self._save_config()
    
    # Language Settings
    @property
    def user_language(self) -> str:
        """Get user's target language"""
        return self._config.get("user_language", "en")
    
    @user_language.setter
    def user_language(self, value: str):
        """Set user's target language"""
        self._config["user_language"] = value
        self._save_config()
    
    @property
    def auto_detect_source_language(self) -> bool:
        """Check if source language should be auto-detected"""
        return self._config.get("auto_detect_source_language", True)
    
    @auto_detect_source_language.setter
    def auto_detect_source_language(self, value: bool):
        """Set auto-detect source language flag"""
        self._config["auto_detect_source_language"] = value
        self._save_config()
    
    # Audio Settings
    @property
    def audio_device_id(self) -> Optional[int]:
        """Get audio device ID (None for default)"""
        return self._config.get("audio_device_id")
    
    @audio_device_id.setter
    def audio_device_id(self, value: Optional[int]):
        """Set audio device ID"""
        self._config["audio_device_id"] = value
        self._save_config()
    
    @property
    def sample_rate(self) -> int:
        """Get audio sample rate"""
        return self._config.get("sample_rate", 16000)
    
    @sample_rate.setter
    def sample_rate(self, value: int):
        """Set audio sample rate"""
        try:
            self._config["sample_rate"] = int(value)
            self._save_config()
        except (ValueError, TypeError):
            pass
    
    @property
    def chunk_duration_seconds(self) -> float:
        """Get audio chunk duration in seconds"""
        return self._config.get("chunk_duration_seconds", 5.0)
    
    @chunk_duration_seconds.setter
    def chunk_duration_seconds(self, value: float):
        """Set audio chunk duration in seconds"""
        try:
            self._config["chunk_duration_seconds"] = float(value)
            self._save_config()
        except (ValueError, TypeError):
            pass
    
    # Window Settings
    @property
    def window_opacity(self) -> float:
        """Get window opacity (0.0 to 1.0)"""
        opacity = self._config.get("window_opacity", 0.9)
        return max(0.0, min(1.0, opacity))
    
    @window_opacity.setter
    def window_opacity(self, value: float):
        """Set window opacity"""
        self._config["window_opacity"] = max(0.0, min(1.0, value))
        self._save_config()
    
    @property
    def window_position(self) -> tuple[int, int]:
        """Get window position (x, y)"""
        x = self._config.get("window_x", 100)
        y = self._config.get("window_y", 100)
        return (x, y)
    
    @window_position.setter
    def window_position(self, value: tuple[int, int]):
        """Set window position"""
        x, y = value
        self._config["window_x"] = x
        self._config["window_y"] = y
        self._save_config()
    
    # Whisper Model Settings
    @property
    def whisper_model_size(self) -> str:
        """Get Whisper model size"""
        return self._config.get("whisper_model_size", "base")
    
    @whisper_model_size.setter
    def whisper_model_size(self, value: str):
        """Set Whisper model size"""
        valid_sizes = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        if value not in valid_sizes:
            value = "base"
        self._config["whisper_model_size"] = value
        self._save_config()
    
    @property
    def whisper_device(self) -> str:
        """Get Whisper device (cpu, cuda, or auto)"""
        return self._config.get("whisper_device", "auto")
    
    @whisper_device.setter
    def whisper_device(self, value: str):
        """Set Whisper device"""
        valid_devices = ["cpu", "cuda", "auto"]
        if value not in valid_devices:
            value = "auto"
        self._config["whisper_device"] = value
        self._save_config()
    
    # Debug Settings
    @property
    def dump_audio_chunks(self) -> bool:
        """Get dump audio chunks setting"""
        return self._config.get("dump_audio_chunks", False)
    
    @dump_audio_chunks.setter
    def dump_audio_chunks(self, value: bool):
        """Set dump audio chunks setting"""
        self._config["dump_audio_chunks"] = bool(value)
        self._save_config()
    
    @property
    def transcript_only_mode(self) -> bool:
        """Get transcript-only mode setting"""
        return self._config.get("transcript_only_mode", False)
    
    @transcript_only_mode.setter
    def transcript_only_mode(self, value: bool):
        """Set transcript-only mode setting"""
        self._config["transcript_only_mode"] = bool(value)
        self._save_config()
    
    @property
    def debug_logging(self) -> bool:
        """Get debug logging setting"""
        return self._config.get("debug_logging", False)
    
    @debug_logging.setter
    def debug_logging(self, value: bool):
        """Set debug logging setting"""
        self._config["debug_logging"] = bool(value)
        self._save_config()
    
    @property
    def mock_mode(self) -> bool:
        """Get mock mode setting"""
        return self._config.get("mock_mode", False)
    
    @mock_mode.setter
    def mock_mode(self, value: bool):
        """Set mock mode setting"""
        self._config["mock_mode"] = bool(value)
        self._save_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value"""
        self._config[key] = value
        self._save_config()

