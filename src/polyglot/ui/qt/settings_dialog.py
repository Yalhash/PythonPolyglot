"""Settings dialog built from ``AppConfig``. Returns a new, validated config; API keys go
straight to the secret store and are never part of the config."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from polyglot.config import AppConfig, Secrets
from polyglot.core import languages

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")


def _combo(items: list[tuple[str, Any]], current: Any) -> QComboBox:
    box = QComboBox()
    for label, value in items:
        box.addItem(label, value)
    idx = box.findData(current)
    box.setCurrentIndex(idx if idx >= 0 else 0)
    return box


def _language_items(with_auto: bool) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = [("Auto-detect", None)] if with_auto else []
    items += sorted(((languages.name(c), c) for c in languages.known_codes()), key=lambda x: x[0])
    return items


def _spin(lo: int, hi: int, value: int, suffix: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    return s


def _dspin(lo: float, hi: float, value: float, step: float, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(2)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    return s


class SettingsDialog(QDialog):
    def __init__(
        self,
        config: AppConfig,
        secrets: Secrets,
        list_devices: Callable[[], list[dict[str, Any]]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Polyglot settings")
        self.setObjectName("SettingsDialog")
        self._config, self._secrets = config, secrets
        self.result_config: AppConfig | None = None
        self.keys_changed = False
        tabs = QTabWidget()
        tabs.addTab(self._audio_tab(list_devices), "Audio")
        tabs.addTab(self._speech_tab(), "Speech")
        tabs.addTab(self._translation_tab(), "Translation")
        tabs.addTab(self._suggestions_tab(), "Suggestions")
        tabs.addTab(self._window_tab(), "Window")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    # ------------------------------------------------------------ tabs
    def _audio_tab(self, list_devices: Callable[[], list[dict[str, Any]]]) -> QWidget:
        a = self._config.audio
        w, form = QWidget(), QFormLayout()
        w.setLayout(form)
        self.source = _combo(
            [("Automatic (system audio on Windows)", "auto"), ("System audio (Windows loopback)", "loopback"),
             ("Input device", "device"), ("WAV file (testing)", "file")],
            a.source,
        )  # fmt: skip
        try:
            devices = list_devices()
            error = ""
        except Exception as exc:
            devices, error = [], str(exc)
        items: list[tuple[str, Any]] = [("Default input", None)]
        items += [
            (f"{d['name']}{'  (virtual)' if d.get('virtual') else ''}", str(d["id"]))
            for d in devices
        ]
        self.device = _combo(items, a.device)
        self.file = QLineEdit(a.file or "")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.file)
        row.addWidget(browse)
        form.addRow("Listen to", self.source)
        form.addRow("Input device", self.device)
        form.addRow("WAV file", row)
        if error:
            form.addRow(QLabel(f"Could not list devices: {error}"))
        return w

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a WAV file", "", "WAV files (*.wav)")
        if path:
            self.file.setText(path)

    def _speech_tab(self) -> QWidget:
        s, v = self._config.asr, self._config.vad
        w, form = QWidget(), QFormLayout()
        w.setLayout(form)
        self.model = _combo([(m, m) for m in WHISPER_MODELS], s.model)
        self.asr_device = _combo(
            [("Automatic", "auto"), ("CPU", "cpu"), ("NVIDIA GPU (CUDA)", "cuda")], s.device
        )
        self.source_lang = _combo(_language_items(with_auto=True), s.language)
        self.vad_engine = _combo(
            [("Silero (recommended)", "silero"), ("Energy (simple)", "energy")], v.engine
        )
        self.end_silence = _spin(100, 5000, v.end_silence_ms, " ms")
        self.partials = QCheckBox("Show text while the speaker is still talking")
        self.partials.setChecked(v.partial_interval_ms > 0)
        form.addRow("Whisper model", self.model)
        form.addRow("Run on", self.asr_device)
        form.addRow("Their language", self.source_lang)
        form.addRow("Speech detector", self.vad_engine)
        form.addRow("End of sentence after", self.end_silence)
        form.addRow("", self.partials)
        return w

    def _translation_tab(self) -> QWidget:
        t = self._config.translation
        w, form = QWidget(), QFormLayout()
        w.setLayout(form)
        self.translate_on = QCheckBox("Translate (sends transcribed text to the provider)")
        self.translate_on.setChecked(t.enabled)
        self.provider = _combo(
            [("OpenAI", "openai"), ("Google Cloud Translation", "google")], t.provider
        )
        self.target = _combo(_language_items(with_auto=False), t.target_language)
        self.translate_model = QLineEdit(t.model)
        self.openai_key = self._key_field(Secrets.OPENAI)
        self.google_key = self._key_field(Secrets.GOOGLE)
        form.addRow("", self.translate_on)
        form.addRow("Provider", self.provider)
        form.addRow("Your language", self.target)
        form.addRow("OpenAI model", self.translate_model)
        form.addRow("OpenAI API key", self.openai_key)
        form.addRow("Google API key", self.google_key)
        form.addRow(QLabel("Keys are stored in the system keychain, not in the config file."))
        return w

    def _key_field(self, name: str) -> QLineEdit:
        field = QLineEdit()
        field.setEchoMode(QLineEdit.EchoMode.Password)
        field.setPlaceholderText("stored" if self._secrets.get(name) else "not set")
        return field

    def _suggestions_tab(self) -> QWidget:
        s = self._config.suggestions
        w, form = QWidget(), QFormLayout()
        w.setLayout(form)
        self.suggest_on = QCheckBox("Suggest replies when I press the hotkey (uses OpenAI)")
        self.suggest_on.setChecked(s.enabled)
        self.suggest_model = QLineEdit(s.model)
        self.conversation = _combo(
            [("Meeting", "meeting"), ("Interview", "interview"), ("Casual", "casual")],
            s.conversation_type,
        )
        self.option_count = _spin(1, 3, s.options)
        form.addRow("", self.suggest_on)
        form.addRow("Model", self.suggest_model)
        form.addRow("Conversation", self.conversation)
        form.addRow("Replies to offer", self.option_count)
        return w

    def _window_tab(self) -> QWidget:
        u, p = self._config.ui, self._config.privacy
        w, form = QWidget(), QFormLayout()
        w.setLayout(form)
        self.opacity = _dspin(0.2, 1.0, u.opacity, 0.05)
        self.hide_capture = QCheckBox("Hide this window from screen sharing (Windows 10 2004+)")
        self.hide_capture.setChecked(u.hide_from_capture)
        self.hotkey_suggest = QLineEdit(u.hotkey_suggest)
        self.hotkey_toggle = QLineEdit(u.hotkey_toggle)
        self.dump_audio = QCheckBox("Save each sentence's audio for debugging")
        self.dump_audio.setChecked(p.dump_audio)
        form.addRow("Opacity", self.opacity)
        form.addRow("", self.hide_capture)
        form.addRow("Suggest hotkey", self.hotkey_suggest)
        form.addRow("Show/hide hotkey", self.hotkey_toggle)
        form.addRow("", self.dump_audio)
        return w

    # ------------------------------------------------------------ save
    def _save(self) -> None:
        c = self._config.model_dump()
        c["audio"].update(source=self.source.currentData(), device=self.device.currentData(),
                          file=self.file.text().strip() or None)  # fmt: skip
        c["asr"].update(model=self.model.currentData(), device=self.asr_device.currentData(),
                        language=self.source_lang.currentData())  # fmt: skip
        c["vad"].update(
            engine=self.vad_engine.currentData(),
            end_silence_ms=self.end_silence.value(),
            partial_interval_ms=(c["vad"]["partial_interval_ms"] or 1000)
            if self.partials.isChecked()
            else 0,
        )
        c["translation"].update(enabled=self.translate_on.isChecked(), provider=self.provider.currentData(),
                                target_language=self.target.currentData(),
                                model=self.translate_model.text().strip() or "gpt-4o-mini")  # fmt: skip
        c["suggestions"].update(enabled=self.suggest_on.isChecked(),
                                model=self.suggest_model.text().strip() or "gpt-4.1-mini",
                                conversation_type=self.conversation.currentData(),
                                options=self.option_count.value())  # fmt: skip
        c["ui"].update(opacity=self.opacity.value(), hide_from_capture=self.hide_capture.isChecked(),
                       hotkey_suggest=self.hotkey_suggest.text().strip(),
                       hotkey_toggle=self.hotkey_toggle.text().strip())  # fmt: skip
        c["privacy"].update(dump_audio=self.dump_audio.isChecked())
        try:
            new = AppConfig.model_validate(c)
        except ValidationError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        for field, name in ((self.openai_key, Secrets.OPENAI), (self.google_key, Secrets.GOOGLE)):
            if field.text().strip():
                try:
                    self._secrets.set(name, field.text().strip())
                except Exception as exc:
                    QMessageBox.warning(self, "Could not store key", f"{name}: {exc}")
                    return
                self.keys_changed = True
        self.result_config = new
        self.accept()
