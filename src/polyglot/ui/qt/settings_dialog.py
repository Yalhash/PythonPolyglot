"""Settings dialog built from ``AppConfig``. Returns a new, validated config; API keys go
straight to the secret store and are never part of the config."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Protocol

from pydantic import ValidationError
from PyQt6.QtCore import QObject, Qt, pyqtSignal
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

from polyglot.config import AppConfig, Secrets, SuggestionsConfig, TranslationConfig
from polyglot.core import languages

log = logging.getLogger(__name__)

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")


def _combo(items: list[tuple[str, Any]], current: Any) -> QComboBox:
    box = QComboBox()
    for label, value in items:
        box.addItem(label, value)
    idx = box.findData(current)
    if idx < 0 and current is not None:
        # A value the list does not know (custom model, regional language, device by
        # name) is kept as-is rather than silently replaced by the first entry.
        box.addItem(str(current), current)
        idx = box.count() - 1
    box.setCurrentIndex(max(idx, 0))
    return box


class ModelLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def note(self) -> str: ...


class ModelCatalog(Protocol):
    """Where model names come from (built by the composition root, so this module never
    imports an adapter)."""

    def known(self) -> Sequence[ModelLike]: ...
    def cached(self, api_key: str) -> Sequence[ModelLike] | None: ...
    def fetch(self, api_key: str) -> Sequence[ModelLike]: ...


def _fill_models(box: QComboBox, models: Sequence[ModelLike], current: str, note: str) -> None:
    """Editable list of model IDs (tooltips carry the notes); keeps ``current`` even if
    the list does not contain it, so a custom or newer model is never lost."""
    box.blockSignals(True)
    box.clear()
    for m in models:
        box.addItem(m.id)
        box.setItemData(box.count() - 1, m.note or m.id, Qt.ItemDataRole.ToolTipRole)
    if current and box.findText(current) < 0:
        box.addItem(current)
        box.setItemData(box.count() - 1, note, Qt.ItemDataRole.ToolTipRole)
    box.setCurrentText(current)
    box.blockSignals(False)


def _model_combo(models: Sequence[ModelLike], current: str) -> QComboBox:
    box = QComboBox()
    box.setEditable(True)  # any model ID can be typed
    box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    box.setMinimumContentsLength(18)
    _fill_models(box, models, current, "Custom model")
    return box


class _ModelLoader(QObject):
    """Fetches the live model list on a worker thread; results arrive as queued signals.
    Parentless and kept alive until it reports, so closing the dialog early is safe (Qt
    drops signals to a deleted receiver)."""

    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)
    _alive: ClassVar[set[_ModelLoader]] = set()

    def __init__(self, catalog: ModelCatalog, api_key: str) -> None:
        super().__init__()
        self._catalog, self._key = catalog, api_key

    def start(self) -> None:
        _ModelLoader._alive.add(self)
        threading.Thread(target=self._run, name="polyglot-models", daemon=True).start()

    def _run(self) -> None:
        try:
            models = list(self._catalog.fetch(self._key))
        except Exception as exc:
            log.info("could not list OpenAI models: %s", exc)
            self.failed.emit(str(exc))
        else:
            self.loaded.emit(models)
        finally:
            _ModelLoader._alive.discard(self)


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
        *,
        models: ModelCatalog | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Polyglot settings")
        self.setObjectName("SettingsDialog")
        self._config, self._secrets = config, secrets
        self._catalog = models
        self._known_models: Sequence[ModelLike] = models.known() if models else ()
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
        self._show_account_models(fetch=True)

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
        self.compute_type = _combo(
            [
                ("Automatic (fastest supported)", "auto"),
                ("int8 (fast on CPU)", "int8"),
                ("float16 (GPU)", "float16"),
                ("float32 (most precise, slowest)", "float32"),
            ],
            s.compute_type,
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
        form.addRow("Precision", self.compute_type)
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
        self.translate_model = _model_combo(self._known_models, t.model)
        self.translate_model.setToolTip("OpenAI model for translation. Type any model ID.")
        self.provider.currentIndexChanged.connect(self._provider_changed)
        self.openai_key = self._key_field(Secrets.OPENAI)
        self.google_key = self._key_field(Secrets.GOOGLE)
        form.addRow("", self.translate_on)
        form.addRow("Provider", self.provider)
        form.addRow("Your language", self.target)
        form.addRow("OpenAI model", self._with_refresh(self.translate_model))
        self.models_status = QLabel("")
        self.models_status.setWordWrap(True)
        form.addRow("", self.models_status)
        form.addRow("OpenAI API key", self.openai_key)
        form.addRow("Google API key", self.google_key)
        form.addRow(QLabel("Keys are stored in the system keychain, not in the config file."))
        self._provider_changed()
        return w

    def _with_refresh(self, box: QComboBox) -> QWidget:
        row, lay = QWidget(), QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        row.setLayout(lay)
        refresh = QPushButton("Refresh")
        refresh.setToolTip("Load the models your OpenAI API key can use")
        refresh.clicked.connect(lambda: self._show_account_models(fetch=True, force=True))
        refresh.setEnabled(self._catalog is not None)
        lay.addWidget(box, 1)
        lay.addWidget(refresh)
        return row

    def _provider_changed(self) -> None:
        self.translate_model.setEnabled(self.provider.currentData() == "openai")

    # ------------------------------------------------------------ model list
    def _openai_key(self) -> str | None:
        typed = self.openai_key.text().strip()
        return typed or self._secrets.get(Secrets.OPENAI)

    def _show_account_models(self, *, fetch: bool, force: bool = False) -> None:
        """Show the models the OpenAI key can use: from the session cache at once, or
        fetched in the background (on open, or when Refresh is pressed)."""
        if self._catalog is None:
            return
        key = self._openai_key()
        if not key:
            self.models_status.setText("Add an OpenAI API key to list the models it can use.")
            return
        cached = None if force else self._catalog.cached(key)
        if cached is not None:
            self._models_loaded(list(cached))
            return
        if not fetch:
            return
        self.models_status.setText("Loading the models your key can use…")
        loader = _ModelLoader(self._catalog, key)
        loader.loaded.connect(self._models_loaded)
        loader.failed.connect(self._models_failed)
        loader.start()

    def _models_loaded(self, models: list[ModelLike]) -> None:
        if not models:
            self.models_status.setText("Your key lists no chat models; showing the built-in list.")
            return
        note = "Not in your account's model list"
        for box in (self.translate_model, self.suggest_model):
            _fill_models(box, models, box.currentText().strip(), note)
        self.models_status.setText(f"{len(models)} models available to your key.")

    def _models_failed(self, error: str) -> None:
        self.models_status.setText(f"Could not load your models ({error}); showing the built-in list.")

    def _key_field(self, name: str) -> QLineEdit:
        field = QLineEdit()
        field.setEchoMode(QLineEdit.EchoMode.Password)
        source = self._secrets.source(name)
        if source is None:
            field.setPlaceholderText("not set")
        elif source == "keychain":
            field.setPlaceholderText("stored in the keychain")
        else:
            field.setPlaceholderText(f"from {source} (overrides the keychain)")
        return field

    def _suggestions_tab(self) -> QWidget:
        s = self._config.suggestions
        w, form = QWidget(), QFormLayout()
        w.setLayout(form)
        self.suggest_on = QCheckBox("Suggest replies when I press the hotkey (uses OpenAI)")
        self.suggest_on.setChecked(s.enabled)
        self.suggest_model = _model_combo(self._known_models, s.model)
        self.suggest_model.setToolTip("OpenAI model for reply suggestions. Type any model ID.")
        self.conversation = _combo(
            [("Meeting", "meeting"), ("Interview", "interview"), ("Casual", "casual")],
            s.conversation_type,
        )
        self.option_count = _spin(1, 3, s.options)
        form.addRow("", self.suggest_on)
        form.addRow("Model", self._with_refresh(self.suggest_model))
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
                        compute_type=self.compute_type.currentData(),
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
                                model=self.translate_model.currentText().strip()
                                or TranslationConfig.model_fields["model"].default)  # fmt: skip
        c["suggestions"].update(enabled=self.suggest_on.isChecked(),
                                model=self.suggest_model.currentText().strip()
                                or SuggestionsConfig.model_fields["model"].default,
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
