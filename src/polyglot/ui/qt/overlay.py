"""The overlay window: a passive view. It draws what the presenter gives it and emits
intent signals for clicks. No logic, no services, no threads."""

from __future__ import annotations

import bisect
from importlib import resources

from PyQt6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QGuiApplication, QKeySequence, QMouseEvent, QResizeEvent, QShortcut
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from polyglot.config import UiConfig
from polyglot.ui.follow import FollowLock
from polyglot.ui.presenter import CaptureVM, OptionVM, RowVM, StatusVM, SuggestionVM


def load_stylesheet() -> str:
    return resources.files("polyglot.ui.qt").joinpath("theme.qss").read_text(encoding="utf-8")


def _label(text: str = "", name: str = "", *, wrap: bool = True, selectable: bool = True) -> QLabel:
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    if selectable:
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return lbl


#: While following, the view moves at most once per this many milliseconds, however often
#: the live text of the newest sentence changes, and glides there instead of jumping.
_SETTLE_MS = 250
_GLIDE_MS = 220
#: Space kept under the newest bubble, so it ends a little above the window's edge.
_BOTTOM_GAP = 28


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


class RowWidget(QFrame):
    """One utterance: original, translation underneath, and a note."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Row")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)
        self.who = _label("You", "Who", selectable=False)
        self.original = _label(name="Original")
        self.translation = _label(name="Translation")
        self.note = _label(name="Note", selectable=False)
        for w in (self.who, self.original, self.translation, self.note):
            layout.addWidget(w)
        self.who.setVisible(False)

    def apply(self, vm: RowVM) -> None:
        self.original.setText(vm.original or "…")
        self.translation.setText(vm.translation or "")
        self.translation.setVisible(bool(vm.translation))
        self.note.setText(vm.note or "")
        self.note.setVisible(bool(vm.note))
        self.setProperty("partial", vm.partial)
        self.setProperty("failed", vm.failed)
        self.setProperty("mine", vm.mine)
        self.who.setVisible(vm.mine)
        _repolish(self)


class OptionWidget(QFrame):
    """One suggested reply. Click to copy what to say."""

    def __init__(self, vm: OptionVM) -> None:
        super().__init__()
        self.setObjectName("Option")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to copy")
        self._say = vm.say
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)
        layout.addWidget(_label(vm.say, "Say", selectable=False))
        layout.addWidget(_label(vm.means, "Means", selectable=False))

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._say)
        self.setProperty("copied", True)
        _repolish(self)
        QTimer.singleShot(600, self._uncopy)

    def _uncopy(self) -> None:
        self.setProperty("copied", False)
        _repolish(self)


class TitleBar(QWidget):
    """Drag handle for the frameless window."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self._drag_from: QPoint | None = None

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        win = self.window()
        if a0 is not None and win is not None and a0.button() == Qt.MouseButton.LeftButton:
            self._drag_from = a0.globalPosition().toPoint() - win.frameGeometry().topLeft()

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        win = self.window()
        if a0 is not None and win is not None and self._drag_from is not None:
            win.move(a0.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        self._drag_from = None


def _button(text: str, name: str, tip: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(name)
    b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return b


class OverlayWindow(QWidget):
    start_stop_clicked = pyqtSignal()
    suggest_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    quit_clicked = pyqtSignal()
    language_clicked = pyqtSignal()

    def __init__(self, ui: UiConfig) -> None:
        super().__init__()
        self.setObjectName("Overlay")
        self.setWindowTitle("Polyglot")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setStyleSheet(load_stylesheet())
        self._rows: dict[int, RowWidget] = {}
        self._order: list[int] = []
        self._follow = FollowLock()
        self._build()
        self.apply_ui(ui)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.suggest_clicked.emit)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.hide)

    # ------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        title = TitleBar(self)
        tl = QHBoxLayout(title)
        tl.setContentsMargins(10, 4, 6, 4)
        name = _label("Polyglot", "AppName", wrap=False, selectable=False)
        self.language = _button("Language: auto", "LanguageButton", "Click to unlock and re-detect")
        self.language.clicked.connect(self.language_clicked.emit)
        tl.addWidget(name)
        tl.addWidget(self.language)
        tl.addStretch(1)
        for text, obj, tip, signal in (
            ("⟲", "IconButton", "Clear conversation", self.clear_clicked),
            ("⚙", "IconButton", "Settings", self.settings_clicked),
        ):
            b = _button(text, obj, tip)
            b.clicked.connect(signal.emit)
            tl.addWidget(b)
        hide = _button("–", "IconButton", "Hide (tray icon or hotkey brings it back)")
        hide.clicked.connect(self.hide)
        tl.addWidget(hide)
        close = _button("×", "CloseButton", "Quit Polyglot")
        close.clicked.connect(self.quit_clicked.emit)
        tl.addWidget(close)
        root.addWidget(title)

        controls = QWidget()
        controls.setObjectName("Controls")
        cl = QHBoxLayout(controls)
        cl.setContentsMargins(10, 6, 10, 6)
        self.start = _button("Start", "StartButton", "Start or stop listening")
        self.start.clicked.connect(self.start_stop_clicked.emit)
        self.suggest = _button("Suggest", "SuggestButton", "Suggest replies to the last sentence")
        self.suggest.clicked.connect(self.suggest_clicked.emit)
        self.status = _label("Stopped", "Status", wrap=False, selectable=False)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lock_button = _button("", "LockButton", "")
        self.lock_button.clicked.connect(self.toggle_lock)
        cl.addWidget(self.start)
        cl.addWidget(self.suggest)
        cl.addWidget(self.lock_button)
        cl.addWidget(self.status, 1)
        root.addWidget(controls)

        self.transcript = QScrollArea()
        self.transcript.setObjectName("Transcript")
        self.transcript.setWidgetResizable(True)
        self.transcript.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        holder.setObjectName("TranscriptBody")
        self.rows_layout = QVBoxLayout(holder)
        # The extra space at the bottom keeps the newest bubble clear of the window's edge.
        self.rows_layout.setContentsMargins(6, 6, 6, _BOTTOM_GAP)
        self.rows_layout.setSpacing(6)
        self.rows_layout.addStretch(1)
        self.transcript.setWidget(holder)
        self.transcript.setMinimumHeight(90)
        bar = self.transcript.verticalScrollBar()
        if bar is not None:
            self._glide = QPropertyAnimation(bar, b"value", self)
            self._glide.setDuration(_GLIDE_MS)
            self._glide.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._settle = QTimer(self)
            self._settle.setSingleShot(True)
            self._settle.setInterval(_SETTLE_MS)
            self._settle.timeout.connect(self._glide_to_bottom)
            bar.rangeChanged.connect(self._on_range_changed)  # the content grew or shrank
            bar.actionTriggered.connect(self._on_user_scroll)  # dragging, keys, arrows
        viewport = self.transcript.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)  # mouse wheel and touchpad
        self._show_lock()

        # Suggestions get a pane of their own below the transcript. A splitter gives each its
        # own rectangle, so one can never cover the other, and the divider can be dragged.
        self.suggestions = QFrame()  # a plain panel: no scrolling, sized to its replies
        self.suggestions.setObjectName("Suggestions")
        self.suggestions.setMinimumHeight(70)
        sl = QVBoxLayout(self.suggestions)
        sl.setContentsMargins(10, 8, 10, 8)
        sl.setSpacing(6)
        self.reply_to = _label("", "ReplyTo")
        self.suggest_message = _label("", "SuggestMessage", selectable=False)
        self.stale = _label(
            "Newer sentences since. Press Suggest to refresh.", "Stale", selectable=False
        )
        self.options_layout = QVBoxLayout()
        self.options_layout.setSpacing(6)
        sl.addWidget(self.reply_to)
        sl.addLayout(self.options_layout)
        sl.addWidget(self.suggest_message)
        sl.addWidget(self.stale)
        sl.addStretch(1)  # spare height goes below the replies, not between them
        self.suggestions.hide()
        self.panes = QSplitter(Qt.Orientation.Vertical)
        self.panes.setChildrenCollapsible(False)
        self.panes.addWidget(self.transcript)
        self.panes.addWidget(self.suggestions)
        self.panes.setStretchFactor(0, 3)
        self.panes.setStretchFactor(1, 2)
        self._pane_dragged = False  # once the user drags the divider, stop sizing it for them
        self.panes.splitterMoved.connect(self._on_divider_moved)
        root.addWidget(self.panes, 1)

        footer = QWidget()
        footer.setObjectName("Footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 2, 2, 2)
        self.capture = _label("", "Capture", wrap=False, selectable=False)
        fl.addWidget(self.capture, 1)
        fl.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        root.addWidget(footer)

    def apply_ui(self, ui: UiConfig) -> None:
        self.setWindowOpacity(ui.opacity)
        self.setGeometry(ui.x, ui.y, ui.width, ui.height)

    def toggle_visible(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()

    # ------------------------------------------------------------ following the latest
    @property
    def locked(self) -> bool:
        return self._follow.locked

    def toggle_lock(self) -> None:
        """Lock or unlock following the newest sentence (button and hotkey)."""
        if self._follow.toggle():
            self._scroll_to_bottom()
        self._show_lock()

    def _show_lock(self) -> None:
        locked = self._follow.locked
        self.lock_button.setText("🔒" if locked else "🔓")
        self.lock_button.setToolTip(
            "Following the latest sentence. Scroll to stop; click to stay locked (Ctrl+Alt+L)."
            if locked
            else "Scrolling freely. Click to jump to the latest sentence and follow it (Ctrl+Alt+L)."
        )
        self.lock_button.setProperty("locked", locked)
        _repolish(self.lock_button)

    def _follow_target(self) -> int | None:
        """The scroll position that puts the newest bubble just above the bottom edge.

        Computed from that bubble's own geometry instead of the scroll bar's maximum: the
        maximum is the bottom of the whole scrolled widget, which can sit well below the
        last bubble (spare space left by wrapped text, the stretch below the rows), so
        scrolling to it leaves the newest sentences off the top of the view.
        """
        bar = self.transcript.verticalScrollBar()
        viewport = self.transcript.viewport()
        if bar is None or viewport is None or not self._order:
            return None
        self.rows_layout.activate()  # make sure the geometry below is current
        newest = self._rows[self._order[-1]]
        bottom = newest.y() + newest.height()  # in the scrolled widget's coordinates
        view = viewport.height()
        return max(0, min(bar.maximum(), bottom + _BOTTOM_GAP - view))

    def _scroll_to_bottom(self) -> None:
        """Jump (no animation) to the newest sentence."""
        self._glide.stop()
        bar = self.transcript.verticalScrollBar()
        target = self._follow_target()
        if bar is not None and target is not None:
            bar.setValue(target)

    def _on_range_changed(self, _low: int, _high: int) -> None:
        # The content changed size (new text, a new bubble, a resize). Moving only matters
        # when the newest bubble is not where it should be, and not on every change,
        # because live text grows many times a second. One timer batches those changes.
        if not self._follow.locked or self._settle.isActive():
            return
        bar = self.transcript.verticalScrollBar()
        target = self._follow_target()
        if bar is not None and target is not None and target != bar.value():
            self._settle.start()

    def _glide_to_bottom(self) -> None:
        bar = self.transcript.verticalScrollBar()
        target = self._follow_target()
        if bar is None or target is None or not self._follow.locked or target == bar.value():
            return
        self._glide.stop()
        self._glide.setStartValue(bar.value())
        self._glide.setEndValue(target)
        self._glide.start()

    def _on_user_scroll(self, _action: int = 0) -> None:
        self._glide.stop()
        self._settle.stop()
        if self._follow.user_scrolled():
            self._show_lock()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is not None and a1.type() == QEvent.Type.Wheel:
            bar = self.transcript.verticalScrollBar()
            if bar is not None and bar.maximum() > 0:  # nothing to scroll: stay locked
                self._on_user_scroll()
        return super().eventFilter(a0, a1)

    # ------------------------------------------------------------ rows
    def upsert_row(self, row: RowVM) -> None:
        widget = self._rows.get(row.id)
        if widget is None:
            widget = RowWidget()
            idx = bisect.bisect(self._order, row.id)  # placed by ID: spoken order
            self._order.insert(idx, row.id)
            self.rows_layout.insertWidget(idx, widget)
            self._rows[row.id] = widget
        widget.apply(row)

    def remove_row(self, utterance_id: int) -> None:
        widget = self._rows.pop(utterance_id, None)
        if widget is not None:
            self._order.remove(utterance_id)
            widget.setParent(None)
            widget.deleteLater()

    def clear_rows(self) -> None:
        for uid in list(self._rows):
            self.remove_row(uid)

    def show_suggestions(self, vm: SuggestionVM) -> None:
        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if not vm.visible:
            self.suggestions.hide()
            self._pane_dragged = False
            return
        self.reply_to.setText(f"Replying to: “{vm.reply_to}”" if vm.reply_to else "")
        self.reply_to.setVisible(bool(vm.reply_to))
        for opt in vm.options:
            self.options_layout.addWidget(OptionWidget(opt))
        message = "Thinking…" if vm.loading else (vm.message or "")
        self.suggest_message.setText(message)
        self.suggest_message.setVisible(bool(message))
        self.stale.setVisible(vm.stale and not vm.loading)
        if self.suggestions.isHidden():
            self.suggestions.show()
        QTimer.singleShot(0, self._fit_suggestions)  # after the layout has measured the text

    def _reply_gap(self) -> int:
        """Pixels between the 'Replying to' line and the first reply (for tests)."""
        first = self.options_layout.itemAt(0)
        widget = first.widget() if first is not None else None
        if widget is None:
            return 0
        return widget.y() - (self.reply_to.y() + self.reply_to.height())

    def _on_divider_moved(self, _pos: int, _index: int) -> None:
        self._pane_dragged = True

    def _fit_suggestions(self) -> None:
        """Size the suggestions panel to exactly what its replies need, so nothing scrolls
        and nothing is cut off. The transcript gets the rest, but never less than its
        minimum height. Not done once the user has dragged the divider themselves."""
        if self._pane_dragged or self.suggestions.isHidden():
            return
        layout = self.suggestions.layout()
        if layout is None:
            return
        # Wrapped text is as tall as the width allows, so use the width the panel really
        # has: the splitter's, not a not-yet-laid-out child's.
        width = self.panes.width()
        need = layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else 0
        need = need or layout.sizeHint().height()
        total = self.panes.height() - self.panes.handleWidth()
        room = total - self.transcript.minimumHeight()
        want = max(self.suggestions.minimumHeight(), min(need, room))
        self.panes.setSizes([total - want, want])

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        QTimer.singleShot(0, self._fit_suggestions)  # the text wraps differently now

    def set_status(self, vm: StatusVM) -> None:
        self.status.setText(vm.text)
        self.status.setToolTip(vm.text)
        self.status.setProperty("level", vm.level)
        _repolish(self.status)

    def set_listening(self, listening: bool, busy: bool) -> None:
        self.start.setText("Stop" if listening else "Start")
        self.start.setProperty("listening", listening)
        self.start.setEnabled(not busy)
        _repolish(self.start)

    def set_language(self, text: str) -> None:
        self.language.setText(text)

    def set_capture(self, vm: CaptureVM) -> None:
        self.capture.setText(("● " if not vm.hidden else "") + vm.text)
        self.capture.setProperty("hidden", vm.hidden)
        _repolish(self.capture)
