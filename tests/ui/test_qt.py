"""Qt views, the thread bridge, the capture-shield filter and an app smoke test.

Runs offscreen in CI (QT_QPA_PLATFORM=offscreen). Skipped where PyQt6/pytest-qt are
not installed.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from polyglot.adapters.fakes import NoopShield
from polyglot.adapters.secrets import MemorySecretStore
from polyglot.config import AppConfig, Secrets, UiConfig, save_config
from polyglot.core.events import EventBus, PipelineState, PipelineStatus
from polyglot.core.ports import ShieldStatus
from polyglot.ui.presenter import CaptureVM, OptionVM, RowVM, StatusVM, SuggestionVM

pytestmark = pytest.mark.ui


@pytest.fixture
def window(qtbot: Any) -> Any:
    from polyglot.ui.qt.overlay import OverlayWindow

    w = OverlayWindow(UiConfig())
    qtbot.addWidget(w)
    return w


def test_stylesheet_ships_with_package() -> None:
    from polyglot.ui.qt.overlay import load_stylesheet

    assert "#Overlay" in load_stylesheet()


def test_rows_are_placed_in_spoken_order(window: Any) -> None:
    for uid in (3, 1, 2):
        window.upsert_row(RowVM(uid, f"text {uid}"))
    assert window._order == [1, 2, 3]
    widgets = [window.rows_layout.itemAt(i).widget() for i in range(3)]
    assert [w.original.text() for w in widgets] == ["text 1", "text 2", "text 3"]


def test_row_update_in_place_and_remove(window: Any) -> None:
    window.upsert_row(RowVM(1, "hola", note="translating…"))
    row = window._rows[1]
    assert not row.note.isHidden() and row.translation.isHidden()
    window.upsert_row(RowVM(1, "hola", translation="hello"))
    assert window._rows[1] is row  # updated, not duplicated
    assert row.translation.text() == "hello" and row.note.isHidden()
    window.remove_row(1)
    window.remove_row(1)  # removing twice is harmless
    assert window._rows == {} and window._order == []
    window.upsert_row(RowVM(2, "x"))
    window.clear_rows()
    assert window._rows == {}


def test_suggestions_panel(window: Any) -> None:
    window.show_suggestions(SuggestionVM(reply_to="¿Qué?", loading=True))
    assert not window.suggestions.isHidden()
    assert window.suggest_message.text() == "Thinking…"
    window.show_suggestions(
        SuggestionVM(
            reply_to="¿Qué?", options=(OptionVM("Sí", "Yes"), OptionVM("No", "No")), stale=True
        )
    )
    assert window.options_layout.count() == 2
    assert not window.stale.isHidden()
    assert "¿Qué?" in window.reply_to.text()
    window.show_suggestions(SuggestionVM(visible=False))
    assert window.suggestions.isHidden() and window.options_layout.count() == 0


def test_option_click_copies_text(window: Any, qtbot: Any) -> None:
    from PyQt6.QtCore import Qt

    window.show_suggestions(SuggestionVM(options=(OptionVM("Copiame", "Copy me"),)))
    opt = window.options_layout.itemAt(0).widget()
    qtbot.mouseClick(opt, Qt.MouseButton.LeftButton)
    clipboard = QApplication.clipboard()
    assert clipboard is not None and clipboard.text() == "Copiame"


def test_status_listening_language_capture(window: Any) -> None:
    window.set_status(StatusVM("Falling behind", "warn"))
    assert window.status.text() == "Falling behind" and window.status.property("level") == "warn"
    window.set_listening(True, busy=True)
    assert window.start.text() == "Stop" and not window.start.isEnabled()
    window.set_listening(False, busy=False)
    assert window.start.text() == "Start" and window.start.isEnabled()
    window.set_language("Spanish (locked)")
    assert window.language.text() == "Spanish (locked)"
    window.set_capture(CaptureVM(False, "Visible in screen share"))
    assert window.capture.text().endswith("Visible in screen share")


def test_buttons_emit_intents(window: Any, qtbot: Any) -> None:
    with qtbot.waitSignal(window.start_stop_clicked):
        window.start.click()
    with qtbot.waitSignal(window.suggest_clicked):
        window.suggest.click()


def test_bridge_delivers_on_gui_thread(qtbot: Any) -> None:
    from polyglot.ui.qt.bridge import EventBridge

    bus = EventBus()
    got: list[tuple[Any, bool]] = []
    app = QApplication.instance()
    assert app is not None
    gui_thread = app.thread()
    bridge = EventBridge(bus, lambda ev: got.append((ev, QThread.currentThread() is gui_thread)))
    ev = PipelineStatus(PipelineState.LISTENING)
    t = threading.Thread(target=bus.publish, args=(ev,))
    t.start()
    t.join()
    qtbot.waitUntil(lambda: bool(got), timeout=2000)
    assert got == [(ev, True)]
    bridge.close()
    bus.publish(ev)
    qtbot.wait(50)
    assert len(got) == 1


def test_shield_filter_covers_every_top_level_window(qtbot: Any) -> None:
    from polyglot.ui.qt.shield_filter import CaptureShieldFilter

    shield = NoopShield(ShieldStatus.HIDDEN)
    filt = CaptureShieldFilter(shield)
    app = QApplication.instance()
    assert app is not None
    app.installEventFilter(filt)
    statuses: list[Any] = []
    filt.status_changed.connect(statuses.append)
    try:
        main = QWidget()
        qtbot.addWidget(main)
        main.show()
        dialog = QDialog(main)
        dialog.show()
        qtbot.waitUntil(lambda: len(shield.calls) >= 2, timeout=2000)
        assert int(main.winId()) in shield.calls and int(dialog.winId()) in shield.calls
        main.hide()
        main.show()  # shown again: same handle, not re-applied
        assert shield.calls.count(int(main.winId())) == 1
        assert statuses == [ShieldStatus.HIDDEN]
        dialog.close()
    finally:
        app.removeEventFilter(filt)


@pytest.mark.skipif(sys.platform != "win32", reason="real SetWindowDisplayAffinity is Windows-only")
def test_real_windows_shield_hides_a_window(qtbot: Any) -> None:
    from polyglot.platform.windows import WindowsShield

    w = QWidget()
    qtbot.addWidget(w)
    w.show()
    status = WindowsShield().exclude_from_capture(int(w.winId()))
    assert status in (ShieldStatus.HIDDEN, ShieldStatus.UNSUPPORTED)


def test_settings_dialog_builds_config_and_stores_keys(qtbot: Any) -> None:
    from polyglot.ui.qt.settings_dialog import SettingsDialog

    secrets = Secrets(MemorySecretStore(), environ={})
    dlg = SettingsDialog(
        AppConfig(), secrets, lambda: [{"id": 3, "name": "BlackHole 2ch", "virtual": True}]
    )
    qtbot.addWidget(dlg)
    dlg.translate_on.setChecked(True)
    dlg.openai_key.setText("sk-test")
    dlg.end_silence.setValue(800)
    dlg.partials.setChecked(True)
    dlg.device.setCurrentIndex(1)
    dlg._save()
    cfg = dlg.result_config
    assert cfg is not None
    assert (
        cfg.translation.enabled
        and cfg.vad.end_silence_ms == 800
        and cfg.vad.partial_interval_ms == 1000
    )
    assert cfg.audio.device == "3"
    assert secrets.get(Secrets.OPENAI) == "sk-test" and dlg.keys_changed


def test_settings_dialog_survives_device_listing_error(qtbot: Any) -> None:
    from polyglot.ui.qt.settings_dialog import SettingsDialog

    def broken() -> list[dict[str, Any]]:
        raise OSError("PortAudio missing")

    dlg = SettingsDialog(AppConfig(), Secrets(MemorySecretStore(), environ={}), broken)
    qtbot.addWidget(dlg)
    dlg._save()
    assert dlg.result_config == AppConfig()


def test_app_smoke_demo_mode(qtbot: Any, tmp_path: Path) -> None:
    """Start the real GUI in demo mode, let it run briefly, quit cleanly."""
    from polyglot.app import run_gui

    cfg = AppConfig()
    cfg.ui.hotkey_suggest = ""
    cfg.ui.hotkey_toggle = ""
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    app = QApplication.instance()
    assert app is not None
    QTimer.singleShot(2500, app.quit)
    assert run_gui(path, demo=True) == 0
