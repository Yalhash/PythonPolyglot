"""Main application entry point for Polyglot"""

import os
import sys
from pathlib import Path

# Load .env into environment before any other project imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import numpy as np
from typing import Optional
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QMessageBox
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QPixmap
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import Settings
from src.audio.capture import AudioCapture
from src.services.transcription import TranscriptionService
from src.services.translation import TranslationService
from src.services.response import ResponseGenerationService
from src.ui.overlay_window import OverlayWindow
from src.ui.settings_dialog import SettingsDialog
from src.utils.audio_dump import AudioDumper
from src.detection.screen_share import ScreenShareDetector


class AudioProcessor(QObject):
    """Handles audio processing in background threads"""
    
    transcription_ready = pyqtSignal(dict)  # Emits {'text': str, 'language': str}
    
    def __init__(self, transcription_service: TranscriptionService):
        super().__init__()
        self.transcription_service = transcription_service
        self.executor = ThreadPoolExecutor(max_workers=2)  # Allow parallel processing to prevent backlog
        self.processing_queue = []
        self.is_processing = False
    
    def process_audio_chunk(self, audio_data: np.ndarray):
        """Process audio chunk in background thread"""
        # Submit for processing - executor will handle queueing
        future = self.executor.submit(self._transcribe_sync, audio_data)
        future.add_done_callback(self._on_transcription_done)
    
    def _transcribe_sync(self, audio_data: np.ndarray) -> dict:
        """Synchronous transcription wrapper"""
        try:
            return self.transcription_service.transcribe(audio_data)
        except Exception as e:
            print(f"Error in transcription: {e}")
            return {'text': '', 'language': None, 'error': str(e)}
    
    def _on_transcription_done(self, future):
        """Handle transcription completion"""
        try:
            result = future.result()
            self.transcription_ready.emit(result)
        except Exception as e:
            print(f"Error getting transcription result: {e}")


class PolyglotApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.app = QApplication(sys.argv)
        self.settings = Settings()
        
        # Initialize transcription service (local, no API key needed)
        self.transcription_service = TranscriptionService(
            model_size=self.settings.whisper_model_size,
            device=self.settings.whisper_device
        )
        
        # Translation and response services need OpenAI API key
        # Check if we have at least one API key for translation/response
        if not self.settings.openai_api_key and not self.settings.google_translate_api_key:
            self._show_api_key_warning()
            # Continue anyway - user can still use transcription
        
        # Initialize translation service (OpenAI or Google)
        self.translation_service = TranslationService(
            openai_api_key=self.settings.openai_api_key,
            google_api_key=self.settings.google_translate_api_key,
            debug_logging=self.settings.debug_logging,
            mock_mode=self.settings.mock_mode
        )
        
        # Response service - allow creation even without API key if mock mode is enabled
        if self.settings.openai_api_key or self.settings.mock_mode:
            self.response_service = ResponseGenerationService(
                self.settings.openai_api_key or "",  # Empty string if mock mode
                debug_logging=self.settings.debug_logging,
                mock_mode=self.settings.mock_mode
            )
        else:
            print("Warning: OpenAI API key not set. Response generation will be disabled.")
            self.response_service = None
        
        # Initialize audio capture
        self.audio_capture = AudioCapture(
            sample_rate=self.settings.sample_rate,
            chunk_duration_seconds=self.settings.chunk_duration_seconds
        )
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(self.transcription_service)
        self.audio_processor.transcription_ready.connect(self._on_transcription_ready)
        
        # Thread pool for translation and response generation
        self.translation_executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize audio dumper
        self.audio_dumper = AudioDumper(
            dump_folder="dumps",
            enabled=self.settings.dump_audio_chunks
        )
        
        # Initialize UI
        self.overlay_window = OverlayWindow(opacity=self.settings.window_opacity)
        window_x, window_y = self.settings.window_position
        self.overlay_window.set_position(window_x, window_y)
        
        # Set initial status
        device_id = self.settings.audio_device_id
        if device_id is not None:
            self.overlay_window.set_capture_status(False, f"Ready - Device: {device_id}")
        else:
            self.overlay_window.set_capture_status(False, "Configure settings and press Start")
        
        self.overlay_window.show()
        
        # Set up system tray
        self._setup_system_tray()
        
        # Connect settings button
        self.overlay_window.settings_btn.clicked.connect(self._show_settings)
        
        # Connect quit button and signal
        self.overlay_window.close_btn.clicked.connect(self.quit)
        self.overlay_window.quit_requested.connect(self.quit)
        
        # Connect start/stop button
        self.overlay_window.start_stop_requested.connect(self._toggle_capture)
        
        # Connect audio capture callback
        self.audio_capture.buffer.set_chunk_callback(self._on_audio_chunk)
        
        # State
        self.current_source_language = None
        self.last_transcription = ""
        self.is_capturing = False
    
    def _show_api_key_warning(self):
        """Show warning message about missing API keys"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("API Keys Not Set")
        msg.setText("Translation and response features require API keys")
        msg.setInformativeText(
            "Transcription will work locally (free).\n\n"
            "For translation and response generation, set one of:\n"
            "- OPENAI_API_KEY (for translation + responses)\n"
            "- GOOGLE_TRANSLATE_API_KEY (for translation only)\n\n"
            "You can still use transcription without API keys."
        )
        msg.exec()
    
    def _setup_system_tray(self):
        """Set up system tray icon"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            print("System tray is not available")
            return
        
        self.tray_icon = QSystemTrayIcon(self.app)
        
        # Create a simple icon (you can replace this with an actual icon file)
        pixmap = QPixmap(16, 16)
        pixmap.fill(self.app.palette().color(self.app.palette().ColorRole.Window))
        self.tray_icon.setIcon(QIcon(pixmap))
        
        # Create menu
        menu = QMenu()
        
        show_action = menu.addAction("Show Window")
        show_action.triggered.connect(self.overlay_window.show)
        
        hide_action = menu.addAction("Hide Window")
        hide_action.triggered.connect(self.overlay_window.hide)
        
        menu.addSeparator()
        
        settings_action = menu.addAction("Settings")
        settings_action.triggered.connect(self._show_settings)
        
        menu.addSeparator()
        
        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(lambda: self.quit(confirm=True))
        
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.setToolTip("Polyglot - Interview Translation Assistant")
        self.tray_icon.show()
    
    def _on_audio_chunk(self, audio_data: np.ndarray):
        """Handle audio chunk from capture"""
        # Dump audio chunk if enabled
        if self.settings.dump_audio_chunks:
            self.audio_dumper.dump_audio_chunk(audio_data, self.settings.sample_rate)
        
        # Process audio in background
        self.audio_processor.process_audio_chunk(audio_data)
    
    def _on_transcription_ready(self, result: dict):
        """Handle transcription result"""
        # Check for errors
        if 'error' in result:
            error_msg = result.get('error', 'Unknown error')
            # Show error message for quota issues
            if 'quota' in error_msg.lower() or '429' in error_msg:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("API Quota Exceeded")
                msg.setText("OpenAI API quota has been exceeded")
                msg.setInformativeText(
                    "Please check your OpenAI billing and plan details.\n\n"
                    "You can:\n"
                    "1. Upgrade your OpenAI plan\n"
                    "2. Wait for your quota to reset\n"
                    "3. Check your usage at https://platform.openai.com/usage"
                )
                msg.exec()
            return
        
        text = result.get('text', '').strip()
        language = result.get('language')
        
        if not text:
            return
        
        # Update source language if detected
        if language:
            self.current_source_language = language
        
        # Improved duplicate detection - allow similar but not identical text
        # This handles continuous speech where chunks may overlap
        if text and text == self.last_transcription:
            return
        # Also check if text is just a substring of last (likely duplicate)
        if text and self.last_transcription and text in self.last_transcription:
            return
        
        self.last_transcription = text
        
        # Update UI with original text (transcription signal is already thread-safe)
        lang_display = self.current_source_language or "Unknown"
        self.overlay_window.update_original_text(text, lang_display)
        
        # Check if transcript-only mode is enabled
        if self.settings.transcript_only_mode:
            # Skip translation and response generation - NO API CALLS
            print("Transcript-only mode: Skipping translation and response generation")
            self.overlay_window.update_translated_text("(Transcript-only mode)")
            self.overlay_window.update_response_original("(Transcript-only mode)")
            self.overlay_window.update_response_translated("(Transcript-only mode)")
            return
        
        # Translate and generate responses in background
        target_lang = self.settings.user_language
        source_lang = self.current_source_language if self.current_source_language else None
        
        future = self.translation_executor.submit(
            self._process_translation_and_response,
            text, target_lang, source_lang
        )
        future.add_done_callback(self._on_translation_done)
    
    def _process_translation_and_response(self, text: str, target_lang: str, source_lang: str) -> dict:
        """Process translation and response generation (runs in background thread)"""
        # Double-check transcript-only mode to prevent any API calls
        if self.settings.transcript_only_mode:
            return {
                'translated': '(Transcript-only mode)',
                'responses': {
                    'original_response': '(Transcript-only mode)',
                    'translated_response': '(Transcript-only mode)'
                }
            }
        
        try:
            translated = self.translation_service.translate(text, target_lang, source_lang)
            
            # Generate responses (if service is available)
            if self.response_service:
                responses = self.response_service.generate_response(
                    text, translated, source_lang or "auto", target_lang
                )
                # Add to conversation history
                self.response_service.add_to_history(text, translated, source_lang or "auto", target_lang)
            else:
                responses = {
                    'original_response': '(Requires OpenAI API key)',
                    'translated_response': '(Requires OpenAI API key)'
                }
            
            return {
                'translated': translated,
                'responses': responses
            }
        except Exception as e:
            print(f"Error in translation/response generation: {e}")
            return {
                'translated': '',
                'responses': {'original_response': '', 'translated_response': ''}
            }
    
    def _on_translation_done(self, future):
        """Handle translation and response completion (thread-safe UI updates)"""
        try:
            result = future.result()
            translated_text = result.get('translated', '')
            responses = result.get('responses', {})
            original_response = responses.get('original_response', '')
            translated_response = responses.get('translated_response', '')
            # Emit signal on OverlayWindow — Qt auto-connection detects the
            # cross-thread call and queues delivery to the main/GUI thread.
            self.overlay_window.translation_result_ready.emit(
                translated_text, original_response, translated_response
            )
        except Exception as e:
            print(f"Error getting translation result: {e}")
    
    
    def _show_settings(self):
        """Show settings dialog"""
        # Get available audio devices
        devices = self.audio_capture.list_devices()
        
        # Get current settings
        current_device_id = self.settings.audio_device_id
        current_target_language = self.settings.user_language
        current_auto_detect = self.settings.auto_detect_source_language
        current_sample_rate = self.settings.sample_rate
        current_chunk_duration = self.settings.chunk_duration_seconds
        current_whisper_model_size = self.settings.whisper_model_size
        current_whisper_device = self.settings.whisper_device
        current_window_opacity = self.settings.window_opacity
        current_dump_audio = self.settings.dump_audio_chunks
        current_transcript_only = self.settings.transcript_only_mode
        current_debug_logging = self.settings.debug_logging
        current_mock_mode = self.settings.mock_mode
        
        # Create and show settings dialog
        dialog = SettingsDialog(
            audio_devices=devices,
            current_device_id=current_device_id,
            current_target_language=current_target_language,
            current_auto_detect=current_auto_detect,
            current_sample_rate=current_sample_rate,
            current_chunk_duration=current_chunk_duration,
            current_whisper_model_size=current_whisper_model_size,
            current_whisper_device=current_whisper_device,
            current_window_opacity=current_window_opacity,
            current_dump_audio=current_dump_audio,
            current_transcript_only=current_transcript_only,
            current_debug_logging=current_debug_logging,
            current_mock_mode=current_mock_mode,
            parent=self.overlay_window
        )
        
        # Connect settings changed signal
        dialog.settings_changed.connect(self._on_settings_changed)
        
        dialog.exec()
    
    def _on_settings_changed(self, device_id: Optional[int], target_language: str,
                            auto_detect: bool, sample_rate: int, chunk_duration: float,
                            whisper_model_size: str, whisper_device: str, window_opacity: float,
                            dump_audio: bool, transcript_only: bool, debug_logging: bool, mock_mode: bool):
        """Handle settings changes"""
        # Update language settings
        self.settings.user_language = target_language
        self.settings.auto_detect_source_language = auto_detect
        
        # Update audio settings
        sample_rate_changed = sample_rate != self.settings.sample_rate
        chunk_duration_changed = chunk_duration != self.settings.chunk_duration_seconds
        self.settings.sample_rate = sample_rate
        self.settings.chunk_duration_seconds = chunk_duration
        
        # Update whisper model settings
        whisper_model_changed = whisper_model_size != self.settings.whisper_model_size
        whisper_device_changed = whisper_device != self.settings.whisper_device
        self.settings.whisper_model_size = whisper_model_size
        self.settings.whisper_device = whisper_device
        
        # Update window settings
        self.settings.window_opacity = window_opacity
        self.overlay_window.set_opacity(window_opacity)
        
        # Update debug settings
        self.settings.dump_audio_chunks = dump_audio
        self.settings.transcript_only_mode = transcript_only
        self.settings.debug_logging = debug_logging
        self.settings.mock_mode = mock_mode
        
        # Update audio dumper
        self.audio_dumper.set_enabled(dump_audio)
        
        # Update debug logging in services
        if hasattr(self.translation_service, 'debug_logging'):
            self.translation_service.debug_logging = debug_logging
        if self.response_service and hasattr(self.response_service, 'debug_logging'):
            self.response_service.debug_logging = debug_logging
        
        # Update mock mode in services
        if hasattr(self.translation_service, 'mock_mode'):
            self.translation_service.mock_mode = mock_mode
        if self.response_service and hasattr(self.response_service, 'mock_mode'):
            self.response_service.mock_mode = mock_mode
        
        # Handle settings that require restart
        needs_restart = False
        restart_reasons = []
        
        if whisper_model_changed or whisper_device_changed:
            needs_restart = True
            restart_reasons.append("Whisper model settings")
            # Reinitialize transcription service (will reload model on next use)
            self.transcription_service = TranscriptionService(
                model_size=whisper_model_size,
                device=whisper_device
            )
            self.audio_processor.transcription_service = self.transcription_service
        
        if sample_rate_changed or chunk_duration_changed:
            needs_restart = True
            restart_reasons.append("audio settings")
            # Reinitialize audio capture with new settings
            if self.is_capturing:
                self.audio_capture.stop_capture()
            self.audio_capture = AudioCapture(
                sample_rate=sample_rate,
                chunk_duration_seconds=chunk_duration
            )
            # Reconnect callback
            self.audio_capture.buffer.set_chunk_callback(self._on_audio_chunk)
        
        if needs_restart and self.is_capturing:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Settings Changed")
            msg.setText("Some settings require restarting audio capture")
            msg.setInformativeText(
                f"The following settings have changed: {', '.join(restart_reasons)}\n\n"
                "Audio capture will be restarted. Please press Start again."
            )
            msg.exec()
            # Stop capture - user will need to press Start again
            if self.is_capturing:
                self._stop_capture()
        
        # Update status message
        if transcript_only:
            status_msg = "Transcript-only mode enabled (NO API calls)"
        else:
            device_name = "Default" if device_id is None else f"Device {device_id}"
            status_msg = f"Ready - Device: {device_name}"
        
        if not self.is_capturing:
            self.overlay_window.set_capture_status(False, status_msg)
        
        # Update audio device if changed
        if device_id != self.settings.audio_device_id:
            self.settings.audio_device_id = device_id
            
            # If currently capturing, restart with new device
            if self.is_capturing:
                try:
                    self.audio_capture.stop_capture()
                    self.audio_capture.start_capture(device_id=device_id)
                    device_name = "Default" if device_id is None else f"Device {device_id}"
                    self.overlay_window.set_capture_status(True, f"Using {device_name}")
                    print(f"Audio capture restarted with {device_name}")
                except Exception as e:
                    print(f"Error restarting audio capture: {e}")
                    self.is_capturing = False
                    self.overlay_window.set_capture_status(False, "Error - Check device")
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Icon.Warning)
                    msg.setWindowTitle("Audio Device Error")
                    msg.setText(f"Could not switch to selected audio device: {e}")
                    msg.setInformativeText("Audio capture has been stopped. Please press Start again after fixing the device.")
                    msg.exec()
            else:
                # Just update the status message
                device_name = "Default" if device_id is None else f"Device {device_id}"
                self.overlay_window.set_capture_status(False, f"Ready - Device: {device_name}")
    
    def _toggle_capture(self):
        """Toggle audio capture on/off"""
        if self.is_capturing:
            self._stop_capture()
        else:
            self._start_capture()
    
    def _start_capture(self):
        """Start audio capture"""
        # Check if device is configured
        device_id = self.settings.audio_device_id
        
        try:
            # Start audio capture
            self.audio_capture.start_capture(device_id=device_id)
            self.is_capturing = True
            
            # Update UI
            device_name = "Default" if device_id is None else f"Device {device_id}"
            self.overlay_window.set_capture_status(True, f"Using {device_name}")
            print("Audio capture started")
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            self.is_capturing = False
            
            # List available devices for debugging
            try:
                devices = self.audio_capture.list_devices()
                device_list = "\n".join([
                    f"  - Device {d['id']}: {d['name']}"
                    for d in devices[:10]  # Show first 10 devices
                ])
                device_info = f"\n\nAvailable audio devices:\n{device_list}"
            except:
                device_info = ""
            
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Audio Capture Error")
            msg.setText(f"Could not start audio capture: {e}")
            msg.setInformativeText(
                "Please configure your audio device in Settings first.\n\n"
                "Make sure your audio device is available and you have the necessary permissions."
                f"{device_info}\n\n"
                "On Windows, you may need to use a loopback device or virtual audio cable."
            )
            msg.exec()
            
            # Update UI to show error
            self.overlay_window.set_capture_status(False, "Error - Check settings")
    
    def _stop_capture(self):
        """Stop audio capture"""
        try:
            self.audio_capture.stop_capture()
            self.is_capturing = False
            self.overlay_window.set_capture_status(False, "Stopped")
            print("Audio capture stopped")
        except Exception as e:
            print(f"Error stopping audio capture: {e}")
            self.overlay_window.set_capture_status(False, "Error stopping")
    
    def start(self):
        """Start the application"""
        # Don't start audio capture automatically - wait for user to press Start
        
        # Save window position on close
        self.app.aboutToQuit.connect(self._save_window_position)
        
        # Run application
        return self.app.exec()
    
    def _save_window_position(self):
        """Save window position to settings"""
        if self.overlay_window.isVisible():
            x, y = self.overlay_window.get_position()
            self.settings.window_position = (x, y)
    
    def quit(self, confirm: bool = True):
        """
        Quit the application
        
        Args:
            confirm: If True, show confirmation dialog
        """
        if confirm:
            reply = QMessageBox.question(
                self.overlay_window,
                'Quit Polyglot',
                'Are you sure you want to quit?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        print("Shutting down Polyglot...")
        # Stop capture if running
        if self.is_capturing:
            self._stop_capture()
        try:
            self.audio_capture.stop_capture()
        except:
            pass
        
        try:
            self.audio_processor.executor.shutdown(wait=False)
        except:
            pass
        
        try:
            self.translation_executor.shutdown(wait=False)
        except:
            pass
        
        # Save window position before quitting
        self._save_window_position()
        
        # Quit the application
        self.app.quit()
        sys.exit(0)


def main():
    """Main entry point"""
    app = PolyglotApp()
    sys.exit(app.start())


if __name__ == "__main__":
    main()

