"""Settings dialog for audio device and language selection"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QGroupBox, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QSlider)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from typing import List, Dict, Optional


class SettingsDialog(QDialog):
    """Settings dialog for configuring audio device and languages"""
    
    settings_changed = pyqtSignal(int, str, bool, int, float, str, str, float, bool, bool, bool, bool)  
    # device_id, target_language, auto_detect, sample_rate, chunk_duration, 
    # whisper_model_size, whisper_device, window_opacity, dump_audio, transcript_only, debug_logging, mock_mode
    
    def __init__(self, audio_devices: List[Dict], current_device_id: Optional[int],
                 current_target_language: str, current_auto_detect: bool = True,
                 current_sample_rate: int = 16000, current_chunk_duration: float = 5.0,
                 current_whisper_model_size: str = "base", current_whisper_device: str = "auto",
                 current_window_opacity: float = 0.9, current_dump_audio: bool = False,
                 current_transcript_only: bool = False, current_debug_logging: bool = False,
                 current_mock_mode: bool = False, parent=None):
        super().__init__(parent)
        self.audio_devices = audio_devices
        self.current_device_id = current_device_id
        self.current_target_language = current_target_language
        self.current_auto_detect = current_auto_detect
        self.current_sample_rate = current_sample_rate
        self.current_chunk_duration = current_chunk_duration
        self.current_whisper_model_size = current_whisper_model_size
        self.current_whisper_device = current_whisper_device
        self.current_window_opacity = current_window_opacity
        self.current_dump_audio = current_dump_audio
        self.current_transcript_only = current_transcript_only
        self.current_debug_logging = current_debug_logging
        self.current_mock_mode = current_mock_mode
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the settings dialog UI"""
        self.setWindowTitle("Polyglot Settings")
        self.setMinimumSize(450, 700)
        self.setModal(True)
        
        # Style
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: white;
                padding: 5px;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QComboBox:hover {
                border: 1px solid #777;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                color: white;
                selection-background-color: #555;
            }
            QPushButton {
                background-color: #4a9eff;
                color: white;
                padding: 8px 20px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5aaeff;
            }
            QPushButton:pressed {
                background-color: #3a8eef;
            }
            QGroupBox {
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QCheckBox {
                color: white;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #555;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #777;
            }
            QCheckBox::indicator:checked {
                background-color: #4a9eff;
                border: 2px solid #4a9eff;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #5aaeff;
                border: 2px solid #5aaeff;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                color: white;
                padding: 5px;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #777;
            }
            QSlider::groove:horizontal {
                background: #3c3c3c;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a9eff;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #5aaeff;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Audio Device Selection
        audio_group = QGroupBox("Audio Device")
        audio_layout = QVBoxLayout()
        audio_layout.setSpacing(10)
        
        audio_label = QLabel("Select audio input device:")
        audio_label.setFont(QFont("Arial", 10))
        audio_layout.addWidget(audio_label)
        
        self.audio_combo = QComboBox()
        self.audio_combo.setFont(QFont("Arial", 9))
        
        # Add "Default" option first
        self.audio_combo.addItem("Default (Auto-detect)", None)
        
        # Add devices to combo box
        # Filter to show only input-capable devices or loopback devices
        for device in self.audio_devices:
            device_name = device['name']
            device_id = device['id']
            is_input = device.get('is_input', False)
            is_loopback = 'loopback' in device_name.lower()
            is_virtual = any(kw in device_name.lower() for kw in ['vb-audio', 'virtual', 'blackhole', 'soundflower', 'obs', 'voicemeeter'])
            
            # Show input devices, loopback devices, or virtual devices
            if is_input or is_loopback or is_virtual:
                display_name = device_name
                if is_loopback or is_virtual:
                    display_name += " (Recommended for system audio)"
                
                self.audio_combo.addItem(display_name, device_id)
        
        # Set current selection
        if self.current_device_id is not None:
            index = self.audio_combo.findData(self.current_device_id)
            if index >= 0:
                self.audio_combo.setCurrentIndex(index)
        
        audio_layout.addWidget(self.audio_combo)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        # Language Selection
        language_group = QGroupBox("Language")
        language_layout = QVBoxLayout()
        language_layout.setSpacing(10)
        
        language_label = QLabel("Target language (your language):")
        language_label.setFont(QFont("Arial", 10))
        language_layout.addWidget(language_label)
        
        self.language_combo = QComboBox()
        self.language_combo.setFont(QFont("Arial", 9))
        
        # Common languages
        languages = [
            ("English", "en"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Italian", "it"),
            ("Portuguese", "pt"),
            ("Chinese (Simplified)", "zh"),
            ("Chinese (Traditional)", "zh-TW"),
            ("Japanese", "ja"),
            ("Korean", "ko"),
            ("Russian", "ru"),
            ("Arabic", "ar"),
            ("Hindi", "hi"),
            ("Dutch", "nl"),
            ("Polish", "pl"),
            ("Turkish", "tr"),
            ("Vietnamese", "vi"),
            ("Thai", "th"),
        ]
        
        for lang_name, lang_code in languages:
            self.language_combo.addItem(lang_name, lang_code)
        
        # Set current selection
        index = self.language_combo.findData(self.current_target_language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        else:
            # Default to English
            self.language_combo.setCurrentIndex(0)
        
        language_layout.addWidget(self.language_combo)
        
        # Auto-detect source language checkbox
        self.auto_detect_checkbox = QCheckBox("Auto-detect source language")
        self.auto_detect_checkbox.setFont(QFont("Arial", 9))
        self.auto_detect_checkbox.setToolTip("Automatically detect the language being spoken")
        self.auto_detect_checkbox.setChecked(self.current_auto_detect)
        language_layout.addWidget(self.auto_detect_checkbox)
        
        language_group.setLayout(language_layout)
        layout.addWidget(language_group)
        
        # Audio Settings
        audio_settings_group = QGroupBox("Audio Settings")
        audio_settings_layout = QVBoxLayout()
        audio_settings_layout.setSpacing(10)
        
        # Sample Rate
        sample_rate_layout = QHBoxLayout()
        sample_rate_label = QLabel("Sample Rate (Hz):")
        sample_rate_label.setFont(QFont("Arial", 10))
        sample_rate_label.setMinimumWidth(150)
        sample_rate_layout.addWidget(sample_rate_label)
        
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.setFont(QFont("Arial", 9))
        self.sample_rate_combo.addItems(["8000", "16000", "22050", "44100", "48000"])
        # Set current value
        current_sr = str(self.current_sample_rate)
        index = self.sample_rate_combo.findText(current_sr)
        if index >= 0:
            self.sample_rate_combo.setCurrentIndex(index)
        else:
            self.sample_rate_combo.setCurrentText(current_sr)
        sample_rate_layout.addWidget(self.sample_rate_combo)
        audio_settings_layout.addLayout(sample_rate_layout)
        
        # Chunk Duration
        chunk_duration_layout = QHBoxLayout()
        chunk_duration_label = QLabel("Chunk Duration (sec):")
        chunk_duration_label.setFont(QFont("Arial", 10))
        chunk_duration_label.setMinimumWidth(150)
        chunk_duration_layout.addWidget(chunk_duration_label)
        
        self.chunk_duration_spin = QDoubleSpinBox()
        self.chunk_duration_spin.setFont(QFont("Arial", 9))
        self.chunk_duration_spin.setRange(1.0, 15.0)
        self.chunk_duration_spin.setSingleStep(0.5)
        self.chunk_duration_spin.setValue(self.current_chunk_duration)
        self.chunk_duration_spin.setDecimals(1)
        chunk_duration_layout.addWidget(self.chunk_duration_spin)
        audio_settings_layout.addLayout(chunk_duration_layout)
        
        audio_settings_group.setLayout(audio_settings_layout)
        layout.addWidget(audio_settings_group)
        
        # Whisper Model Settings
        whisper_group = QGroupBox("Whisper Model")
        whisper_layout = QVBoxLayout()
        whisper_layout.setSpacing(10)
        
        # Model Size
        model_size_layout = QHBoxLayout()
        model_size_label = QLabel("Model Size:")
        model_size_label.setFont(QFont("Arial", 10))
        model_size_label.setMinimumWidth(150)
        model_size_layout.addWidget(model_size_label)
        
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.setFont(QFont("Arial", 9))
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium", "large-v2", "large-v3"])
        index = self.whisper_model_combo.findText(self.current_whisper_model_size)
        if index >= 0:
            self.whisper_model_combo.setCurrentIndex(index)
        model_size_layout.addWidget(self.whisper_model_combo)
        whisper_layout.addLayout(model_size_layout)
        
        # Device
        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        device_label.setFont(QFont("Arial", 10))
        device_label.setMinimumWidth(150)
        device_layout.addWidget(device_label)
        
        self.whisper_device_combo = QComboBox()
        self.whisper_device_combo.setFont(QFont("Arial", 9))
        self.whisper_device_combo.addItems(["auto", "cpu", "cuda"])
        index = self.whisper_device_combo.findText(self.current_whisper_device)
        if index >= 0:
            self.whisper_device_combo.setCurrentIndex(index)
        device_layout.addWidget(self.whisper_device_combo)
        whisper_layout.addLayout(device_layout)
        
        whisper_group.setLayout(whisper_layout)
        layout.addWidget(whisper_group)
        
        # Window Settings
        window_group = QGroupBox("Window Settings")
        window_layout = QVBoxLayout()
        window_layout.setSpacing(10)
        
        # Opacity
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity:")
        opacity_label.setFont(QFont("Arial", 10))
        opacity_label.setMinimumWidth(150)
        opacity_layout.addWidget(opacity_label)
        
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.current_window_opacity * 100))
        self.opacity_slider.setToolTip(f"{int(self.current_window_opacity * 100)}%")
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_slider.setToolTip(f"{v}%"))
        opacity_layout.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel(f"{int(self.current_window_opacity * 100)}%")
        self.opacity_label.setFont(QFont("Arial", 9))
        self.opacity_label.setMinimumWidth(40)
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v}%"))
        opacity_layout.addWidget(self.opacity_label)
        
        window_layout.addLayout(opacity_layout)
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)
        
        # Debug Settings
        debug_group = QGroupBox("Debug Settings")
        debug_layout = QVBoxLayout()
        debug_layout.setSpacing(10)
        
        self.dump_audio_checkbox = QCheckBox("Dump audio chunks to file")
        self.dump_audio_checkbox.setFont(QFont("Arial", 9))
        self.dump_audio_checkbox.setToolTip("Saves audio chunks to 'dumps' folder with timestamps")
        self.dump_audio_checkbox.setChecked(self.current_dump_audio)
        debug_layout.addWidget(self.dump_audio_checkbox)
        
        self.transcript_only_checkbox = QCheckBox("Transcript-only mode")
        self.transcript_only_checkbox.setFont(QFont("Arial", 9))
        self.transcript_only_checkbox.setToolTip("Disable translation and GPT responses - show only transcription")
        self.transcript_only_checkbox.setChecked(self.current_transcript_only)
        debug_layout.addWidget(self.transcript_only_checkbox)
        
        self.debug_logging_checkbox = QCheckBox("Enable debug logging")
        self.debug_logging_checkbox.setFont(QFont("Arial", 9))
        self.debug_logging_checkbox.setToolTip("Print detailed logging for translation and response generation to terminal")
        self.debug_logging_checkbox.setChecked(self.current_debug_logging)
        debug_layout.addWidget(self.debug_logging_checkbox)
        
        self.mock_mode_checkbox = QCheckBox("Mock mode (return dummy text)")
        self.mock_mode_checkbox.setFont(QFont("Arial", 9))
        self.mock_mode_checkbox.setToolTip("Return 'dummy text' instead of making API calls for translation and responses")
        self.mock_mode_checkbox.setChecked(self.current_mock_mode)
        debug_layout.addWidget(self.mock_mode_checkbox)
        
        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_settings)
        button_layout.addWidget(apply_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _apply_settings(self):
        """Apply the selected settings"""
        device_id = self.audio_combo.currentData()  # Can be None for default
        target_language = self.language_combo.currentData()
        auto_detect = self.auto_detect_checkbox.isChecked()
        sample_rate = int(self.sample_rate_combo.currentText())
        chunk_duration = self.chunk_duration_spin.value()
        whisper_model_size = self.whisper_model_combo.currentText()
        whisper_device = self.whisper_device_combo.currentText()
        window_opacity = self.opacity_slider.value() / 100.0
        dump_audio = self.dump_audio_checkbox.isChecked()
        transcript_only = self.transcript_only_checkbox.isChecked()
        debug_logging = self.debug_logging_checkbox.isChecked()
        mock_mode = self.mock_mode_checkbox.isChecked()
        
        self.settings_changed.emit(
            device_id, target_language, auto_detect, sample_rate, chunk_duration,
            whisper_model_size, whisper_device, window_opacity, dump_audio, transcript_only, debug_logging, mock_mode
        )
        self.accept()

