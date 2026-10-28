"""UI components for the overlay window"""

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPalette, QTextOption


class TextDisplay(QLabel):
    """Custom label for displaying text with styling"""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the label styling"""
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(30, 30, 30, 200);
                color: white;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 50);
            }
        """)
        
        # Set font
        font = QFont("Arial", 10)
        self.setFont(font)
    
    def set_title(self, title: str):
        """Set the title/label for this display"""
        self.title = title
    
    def set_text(self, text: str):
        """Set the text content"""
        if self.title:
            display_text = f"<b>{self.title}:</b><br>{text}"
        else:
            display_text = text
        self.setText(display_text)
    
    def clear_text(self):
        """Clear the text"""
        if self.title:
            self.setText(f"<b>{self.title}:</b>")
        else:
            self.setText("")


class ScrollingTextDisplay(QWidget):
    """Scrolling text display that appends text instead of replacing it"""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the scrolling text display"""
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        
        # Title label (always create, show if title is set)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: transparent;
                padding: 0px;
            }
        """)
        font = QFont("Arial", 10, QFont.Weight.Bold)
        self.title_label.setFont(font)
        if self.title:
            self.title_label.setText(f"<b>{self.title}:</b>")
            self.title_label.show()
        else:
            self.title_label.hide()
        self.layout.addWidget(self.title_label)
        
        # Scrolling text edit
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 30, 30, 200);
                color: white;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 50);
            }
        """)
        
        # Set font
        font = QFont("Arial", 10)
        self.text_edit.setFont(font)
        
        # Set minimum height for better visibility
        self.text_edit.setMinimumHeight(100)
        self.text_edit.setMaximumHeight(300)
        
        self.layout.addWidget(self.text_edit)
        self.setLayout(self.layout)
    
    def set_title(self, title: str):
        """Set the title/label for this display"""
        self.title = title
        if self.title:
            self.title_label.setText(f"<b>{self.title}:</b>")
            self.title_label.show()
        else:
            self.title_label.hide()
    
    def append_text(self, text: str):
        """Append text to the display (scrolls to bottom automatically)"""
        if text:
            # Get current text to check if we need to append to it
            current_text = self.text_edit.toPlainText()
            
            if current_text:
                # Always add a new paragrpah
                self.text_edit.append(text)
            else:
                # First text - just set it
                self.text_edit.setPlainText(text)
            
            # Scroll to bottom
            scrollbar = self.text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def set_text(self, text: str):
        """Set the text content (replaces existing text)"""
        self.text_edit.setPlainText(text)
        # Scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_text(self):
        """Clear the text"""
        self.text_edit.clear()


class OverlayPanel(QWidget):
    """Panel containing multiple text displays"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel layout"""
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create text displays - use ScrollingTextDisplay for original transcript and translated text
        self.original_display = ScrollingTextDisplay("Original", self)
        self.translated_display = ScrollingTextDisplay("Translated", self)
        self.response_original_display = TextDisplay("Suggested Response (Original)", self)
        self.response_translated_display = TextDisplay("Suggested Response (Your Language)", self)
        
        # Add to layout
        layout.addWidget(self.original_display)
        layout.addWidget(self.translated_display)
        layout.addWidget(self.response_original_display)
        layout.addWidget(self.response_translated_display)
        
        # Add stretch at the end
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_original(self, text: str, language: str = ""):
        """Update original text display - appends new transcriptions"""
        print(f"Updating original text display: {text}")
        if language:
            self.original_display.set_title(f"Original ({language})")
        self.original_display.append_text(text)
    
    def update_translated(self, text: str):
        """Update translated text display - appends new translations"""
        print(f"Updating translated text display: {text}")
        self.translated_display.append_text(text)
    
    def update_response_original(self, text: str):
        print(f"Updating original language response display: {text}")
        """Update original language response display"""
        self.response_original_display.set_text(text)
    
    def update_response_translated(self, text: str):
        """Update translated response display"""
        print(f"Updating translated response display: {text}")
        self.response_translated_display.set_text(text)
    
    def clear_all(self):
        """Clear all displays"""
        self.original_display.clear_text()
        self.translated_display.clear_text()
        self.response_original_display.clear_text()
        self.response_translated_display.clear_text()

