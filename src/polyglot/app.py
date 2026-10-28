"""Qt application: the composition root for the GUI.

Creates the controller, the presenter, the view and the one thread bridge, and connects
signals. Nothing else in the app knows all of these at once.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from polyglot.config import AppConfig, ConfigStore, Secrets
from polyglot.controller import PIPELINE_SECTIONS, AppController
from polyglot.core.errors import ConfigError
from polyglot.core.events import EventBus

log = logging.getLogger(__name__)


def _list_devices() -> list[dict[str, Any]]:
    from polyglot.adapters.audio.sounddevice_source import list_input_devices

    return list_input_devices()


def _tray_icon() -> Any:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap

    pix = QPixmap(32, 32)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#2f7de1"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, 28, 28)
    p.setPen(QColor("white"))
    font = p.font()
    font.setBold(True)
    font.setPointSize(14)
    p.setFont(font)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "P")
    p.end()
    return QIcon(pix)


def run_gui(config_path: Path | None, *, demo: bool = False) -> int:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

    from polyglot import platform
    from polyglot.adapters.openai_models import OpenAIModelCatalog
    from polyglot.adapters.secrets import KeyringSecretStore, MemorySecretStore
    from polyglot.ui.presenter import Presenter
    from polyglot.ui.qt.bridge import EventBridge
    from polyglot.ui.qt.hotkeys import GlobalHotkeys
    from polyglot.ui.qt.overlay import OverlayWindow
    from polyglot.ui.qt.settings_dialog import SettingsDialog
    from polyglot.ui.qt.shield_filter import CaptureShieldFilter

    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    app.setApplicationName("Polyglot")
    app.setQuitOnLastWindowClosed(False)

    _install_excepthook()

    store = ConfigStore(config_path)
    config_error = ""
    try:
        config = store.load()
    except ConfigError as exc:
        config_error = str(exc)
        config = AppConfig()  # run on defaults; store.save() refuses, so the file is kept

    secrets = Secrets(MemorySecretStore() if demo else KeyringSecretStore())
    bus = EventBus()
    controller = AppController(
        config,
        secrets,
        bus,
        demo=demo,
        save=None if demo else store.save,
    )

    window = OverlayWindow(config.ui)
    presenter = Presenter(window, controller, user_language=config.translation.target_language)
    bridge = EventBridge(bus, presenter.handle)  # the one thread bridge

    # Capture shield on every top-level window (overlay, dialogs, menus, tooltips).
    shield_filter: CaptureShieldFilter | None = None
    if config.ui.hide_from_capture:
        shield_filter = CaptureShieldFilter(platform.window_shield())
        shield_filter.status_changed.connect(presenter.on_shield_status)
        app.installEventFilter(shield_filter)
    else:
        presenter.on_shield_status(None)

    def refresh_flags() -> None:
        built = controller.built
        presenter.translation_enabled = bool(built and built.translation_enabled)

    # View intents -> presenter -> controller.
    def start_stop() -> None:
        presenter.on_start_stop()
        refresh_flags()  # the first successful build may happen here

    window.start_stop_clicked.connect(start_stop)
    window.suggest_clicked.connect(presenter.on_suggest)
    window.clear_clicked.connect(presenter.on_clear)
    window.language_clicked.connect(presenter.on_unlock_language)

    hotkeys = GlobalHotkeys()
    hotkeys.suggest.connect(presenter.on_suggest)
    hotkeys.toggle_window.connect(window.toggle_visible)
    hotkeys.bind(config.ui.hotkey_suggest, config.ui.hotkey_toggle)

    model_catalog = OpenAIModelCatalog()  # caches the account's model list per session

    def open_settings() -> None:
        nonlocal shield_filter
        dialog = SettingsDialog(
            controller.config, secrets, _list_devices, window, models=model_catalog
        )
        accepted = bool(dialog.exec())
        new, keys_changed = dialog.result_config, dialog.keys_changed
        dialog.deleteLater()
        if accepted and new is not None:
            changed = controller.apply_config(new)
            if keys_changed and not (changed & PIPELINE_SECTIONS):
                controller.rebuild()  # new API keys take effect now
            refresh_flags()
            presenter.on_user_language(new.translation.target_language)
            window.setWindowOpacity(new.ui.opacity)
            hotkeys.bind(new.ui.hotkey_suggest, new.ui.hotkey_toggle)
            if new.ui.hide_from_capture and shield_filter is None:
                shield_filter = CaptureShieldFilter(platform.window_shield())
                shield_filter.status_changed.connect(presenter.on_shield_status)
                app.installEventFilter(shield_filter)
                shield_filter.shield_widget(window)
            elif not new.ui.hide_from_capture and shield_filter is not None:
                # Windows stay hidden (and the footer says so) until restart.
                QMessageBox.information(
                    window,
                    "Restart needed",
                    "Windows stay hidden from screen sharing until Polyglot restarts.",
                )

    window.settings_clicked.connect(open_settings)

    def save_geometry() -> None:
        g = window.geometry()
        ui = controller.config.ui.model_copy(
            update={"x": g.x(), "y": g.y(), "width": g.width(), "height": g.height()}
        )
        controller.apply_config(controller.config.model_copy(update={"ui": ui}))

    def quit_app() -> None:
        if controller.listening:
            answer = QMessageBox.question(window, "Quit Polyglot", "Stop listening and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        app.quit()

    window.quit_clicked.connect(quit_app)

    tray: QSystemTrayIcon | None = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(_tray_icon(), app)
        menu = QMenu(window)  # parented: the tray does not own its menu
        menu.addAction("Show / hide", window.toggle_visible)
        menu.addAction("Start / stop listening", presenter.on_start_stop)
        menu.addAction("Suggest replies", presenter.on_suggest)
        menu.addAction("Settings…", open_settings)
        menu.addSeparator()
        menu.addAction("Quit", quit_app)
        tray.setContextMenu(menu)
        tray.setToolTip("Polyglot")
        tray.activated.connect(
            lambda reason: window.toggle_visible()
            if reason == QSystemTrayIcon.ActivationReason.Trigger
            else None
        )
        tray.show()

    def shutdown() -> None:
        # Stop everything that could call into Qt from another thread, then tear down
        # Qt objects in a defined order (not at interpreter exit).
        save_geometry()
        hotkeys.stop()
        controller.shutdown()
        bridge.close()
        # Before any widget is deleted: destroying windows makes the shield filter
        # recompute its status, which must not reach the (deleted) view.
        if shield_filter is not None:
            app.removeEventFilter(shield_filter)
            shield_filter.close()
        presenter.detach()
        if tray is not None:
            tray.hide()
            tray.setContextMenu(None)
            tray.deleteLater()
        window.hide()
        window.deleteLater()

    app.aboutToQuit.connect(shutdown)

    window.show()
    if config_error:
        presenter.view.set_status(presenter_status_error(config_error))

    def startup() -> None:
        # After the event loop starts, so the window paints before devices/models load.
        controller.ensure_built()
        refresh_flags()

    QTimer.singleShot(0, startup)
    presenter.view.set_language(
        presenter.language_label(config.asr.language, config.asr.language is not None)
    )
    return int(app.exec())


def _install_excepthook() -> None:
    """PyQt aborts the process on an exception escaping a slot unless an excepthook is
    installed. Log it and keep running instead."""

    def hook(exc_type: Any, exc: BaseException, tb: Any) -> None:
        log.critical("unhandled error", exc_info=(exc_type, exc, tb))

    sys.excepthook = hook


def presenter_status_error(message: str) -> Any:
    from polyglot.ui.presenter import StatusVM

    return StatusVM(f"Config error, using defaults: {message}", "error")
