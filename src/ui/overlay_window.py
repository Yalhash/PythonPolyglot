"""Transparent overlay window for displaying translations and responses"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QMouseEvent, QKeySequence, QShortcut
from .components import OverlayPanel
import platform


class OverlayWindow(QWidget):
    """Transparent overlay window that stays on top"""
    
    # Signal emitted when window should be hidden/shown
    visibility_changed = pyqtSignal(bool)
    # Signal emitted when quit is requested
    quit_requested = pyqtSignal()
    # Signal emitted when start/stop button is clicked
    start_stop_requested = pyqtSignal()
    # Thread-safe signal for delivering translation results to the main thread
    translation_result_ready = pyqtSignal(str, str, str)
    
    def __init__(self, opacity: float = 0.9, parent=None):
        super().__init__(parent)
        self.opacity = opacity
        self.drag_position = QPoint()
        self.is_dragging = False
        self._setup_ui()
        self.translation_result_ready.connect(self._apply_translation_result)
    
    def _setup_ui(self):
        """Set up the overlay window"""
        # Set window flags for transparency and always on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        
        # Enable transparency
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowOpacity(self.opacity)
        
        # Make window invisible to screen capture (Windows only)
        self._set_screen_capture_exclusion()
        
        # Set window size
        self.setMinimumSize(400, 550)
        self.resize(400, 650)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Title bar (for dragging)
        title_bar = self._create_title_bar()
        main_layout.addWidget(title_bar)
        
        # Control panel with Start/Stop button
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Content panel
        self.panel = OverlayPanel(self)
        main_layout.addWidget(self.panel)
        
        self.setLayout(main_layout)
        
        # Style the window
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(20, 20, 20, 230);
                border-radius: 8px;
            }
        """)
        
        # Add keyboard shortcuts
        self._setup_shortcuts()
    
    def _create_title_bar(self) -> QWidget:
        """Create a draggable title bar"""
        title_widget = QWidget()
        title_widget.setFixedHeight(30)
        title_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 40, 40, 255);
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Title label
        title_label = QLabel("Polyglot")
        title_label.setStyleSheet("color: white; font-weight: bold;")
        title_label.setFont(QFont("Arial", 10))
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        # Settings button
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(20, 20)
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 60, 255);
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 80, 255);
            }
        """)
        layout.addWidget(self.settings_btn)
        
        # Minimize button
        self.minimize_btn = QPushButton("−")
        self.minimize_btn.setFixedSize(20, 20)
        self.minimize_btn.setToolTip("Minimize")
        self.minimize_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 60, 255);
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 80, 255);
            }
        """)
        self.minimize_btn.clicked.connect(self.hide)
        layout.addWidget(self.minimize_btn)
        
        # Close/Quit button
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setToolTip("Quit Application")
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(200, 60, 60, 255);
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: rgba(220, 80, 80, 255);
            }
        """)
        layout.addWidget(self.close_btn)
        
        title_widget.setLayout(layout)
        return title_widget
    
    def _create_control_panel(self) -> QWidget:
        """Create control panel with Start/Stop button"""
        control_widget = QWidget()
        control_widget.setFixedHeight(50)
        control_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 255);
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)
        
        # Status label
        self.status_label = QLabel("Ready - Configure settings and press Start")
        self.status_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        self.status_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Start/Stop button
        self.start_stop_btn = QPushButton("Start")
        self.start_stop_btn.setFixedSize(80, 30)
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
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
        """)
        self.start_stop_btn.clicked.connect(self.start_stop_requested.emit)
        layout.addWidget(self.start_stop_btn)
        
        control_widget.setLayout(layout)
        return control_widget
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for window dragging"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.is_dragging = True
            event.accept()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for window dragging"""
        if self.is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        self.is_dragging = False
        event.accept()
    
    def update_original_text(self, text: str, language: str = ""):
        """Update original language text"""
        self.panel.update_original(text, language)
    
    def update_translated_text(self, text: str):
        """Update translated text"""
        self.panel.update_translated(text)
    
    def update_response_original(self, text: str):
        """Update original language response"""
        self.panel.update_response_original(text)
    
    def update_response_translated(self, text: str):
        """Update translated response"""
        self.panel.update_response_translated(text)
    
    def _apply_translation_result(self, translated_text: str, response_original: str, response_translated: str):
        """Apply translation results to UI (invoked on main thread via queued signal connection)"""
        if translated_text:
            self.panel.update_translated(translated_text)
        if response_original:
            self.panel.update_response_original(response_original)
        if response_translated:
            self.panel.update_response_translated(response_translated)

    def clear_all(self):
        """Clear all text displays"""
        self.panel.clear_all()
    
    def set_opacity(self, opacity: float):
        """Set window opacity (0.0 to 1.0)"""
        self.opacity = max(0.0, min(1.0, opacity))
        self.setWindowOpacity(self.opacity)
    
    def set_position(self, x: int, y: int):
        """Set window position"""
        self.move(x, y)
    
    def get_position(self) -> tuple[int, int]:
        """Get current window position"""
        pos = self.pos()
        return (pos.x(), pos.y())
    
    def set_capture_status(self, is_capturing: bool, status_message: str = ""):
        """
        Update the capture status and button
        
        Args:
            is_capturing: True if capturing, False if stopped
            status_message: Optional status message to display
        """
        if is_capturing:
            self.start_stop_btn.setText("Stop")
            self.start_stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """)
            if status_message:
                self.status_label.setText(f"● Recording - {status_message}")
            else:
                self.status_label.setText("● Recording...")
            self.status_label.setStyleSheet("color: #4a9eff; font-size: 9pt; font-weight: bold;")
        else:
            self.start_stop_btn.setText("Start")
            self.start_stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a9eff;
                    color: white;
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
            """)
            if status_message:
                self.status_label.setText(f"Stopped - {status_message}")
            else:
                self.status_label.setText("Ready - Press Start to begin")
            self.status_label.setStyleSheet("color: #aaa; font-size: 9pt;")
    
    def showEvent(self, event):
        """Handle show event"""
        super().showEvent(event)
        # Re-apply screen capture exclusion after showing
        self._set_screen_capture_exclusion()
        self.visibility_changed.emit(True)
    
    def hideEvent(self, event):
        """Handle hide event"""
        super().hideEvent(event)
        self.visibility_changed.emit(False)
    
    def _setup_shortcuts(self):
        """Set up keyboard shortcuts"""
        # Ctrl+Q to quit
        quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        quit_shortcut.activated.connect(self.quit_requested.emit)
        
        # Escape to hide
        hide_shortcut = QShortcut(QKeySequence("Escape"), self)
        hide_shortcut.activated.connect(self.hide)
    
    def _set_screen_capture_exclusion(self):
        """Make window invisible to screen capture while visible to user"""
        if platform.system() == "Windows":
            try:
                import ctypes
                
                # Get window handle (need to get the actual HWND)
                hwnd = int(self.winId())
                
                # SetWindowDisplayAffinity is available on Windows 10 version 2004+
                # WDA_EXCLUDEFROMCAPTURE = 0x00000011
                WDA_EXCLUDEFROMCAPTURE = 0x11
                
                # Try to set display affinity to exclude from capture
                result = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
                if result:
                    print("Window set to be excluded from screen capture")
                else:
                    # Get last error
                    error = ctypes.get_last_error()
                    if error != 0:
                        print(f"Could not set screen capture exclusion (error {error})")
                    else:
                        print("SetWindowDisplayAffinity not available on this Windows version")
                
            except (AttributeError, OSError) as e:
                # Function not available or other error
                print(f"Could not set screen capture exclusion: {e}")
                print("Note: This feature requires Windows 10 version 2004 or later")
        elif platform.system() == "Darwin":  # macOS
            try:
                # On macOS, screen capture exclusion is handled differently
                # We can use NSWindow's sharingType property via PyObjC if available
                # For now, the transparent window should be less visible in captures
                pass
            except Exception as e:
                print(f"Could not set screen capture exclusion: {e}")
    

