"""Presenter (Model-View-Presenter) and view model. No Qt here, so it is unit-tested.

The presenter receives core events on the GUI thread (the Qt bridge hands them over),
keeps a view model, formats it, and calls a passive view. It also turns user actions
into commands. The view never sees core objects and never makes decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from polyglot.core import languages
from polyglot.core.events import (
    ConversationCleared,
    Event,
    LanguageChanged,
    PipelineState,
    PipelineStatus,
    StageError,
    SuggestionsReady,
    SuggestionStarted,
    UtteranceDiscarded,
    UtteranceUpdated,
)
from polyglot.core.models import ErrorKind, Utterance, UtteranceStatus
from polyglot.core.ports import ShieldStatus

ERROR_TEXT: dict[ErrorKind, str] = {
    ErrorKind.TIMEOUT: "timed out",
    ErrorKind.UNAVAILABLE: "service unavailable",
    ErrorKind.QUOTA: "quota exhausted",
    ErrorKind.AUTH: "API key rejected",
    ErrorKind.BAD_RESPONSE: "unexpected response",
    ErrorKind.TRANSCRIPTION: "transcription failed",
    ErrorKind.OVERLOADED: "skipped: transcription fell behind",
    ErrorKind.NOT_CONFIGURED: "not set up",
    ErrorKind.NOTHING_TO_ANSWER: "nothing to reply to yet",
    ErrorKind.AUDIO: "audio problem",
    ErrorKind.INTERNAL: "internal error",
}


# ================================================================= view model
@dataclass(frozen=True)
class RowVM:
    id: int
    original: str
    translation: str | None = None
    note: str | None = None  # "listening…", "translating…", "translation failed: …"
    partial: bool = False
    failed: bool = False


@dataclass(frozen=True)
class OptionVM:
    say: str  # their language: what to say
    means: str  # your language: what it means


@dataclass(frozen=True)
class SuggestionVM:
    reply_to: str = ""
    options: tuple[OptionVM, ...] = ()
    loading: bool = False
    message: str | None = None  # an error or hint instead of options
    stale: bool = False  # newer sentences have been said since; offer a refresh
    visible: bool = True


@dataclass(frozen=True)
class StatusVM:
    text: str
    level: str = "info"  # "info" | "warn" | "error"


@dataclass(frozen=True)
class CaptureVM:
    hidden: bool
    text: str


class ViewPort(Protocol):
    def upsert_row(self, row: RowVM) -> None: ...
    def remove_row(self, utterance_id: int) -> None: ...
    def clear_rows(self) -> None: ...
    def show_suggestions(self, vm: SuggestionVM) -> None: ...
    def set_status(self, vm: StatusVM) -> None: ...
    def set_listening(self, listening: bool, busy: bool) -> None: ...
    def set_language(self, text: str) -> None: ...
    def set_capture(self, vm: CaptureVM) -> None: ...


class NullView:
    """A view that ignores every call. Swapped in by ``Presenter.detach()`` so that a
    signal arriving during shutdown cannot touch deleted widgets."""

    def upsert_row(self, row: RowVM) -> None:
        pass

    def remove_row(self, utterance_id: int) -> None:
        pass

    def clear_rows(self) -> None:
        pass

    def show_suggestions(self, vm: SuggestionVM) -> None:
        pass

    def set_status(self, vm: StatusVM) -> None:
        pass

    def set_listening(self, listening: bool, busy: bool) -> None:
        pass

    def set_language(self, text: str) -> None:
        pass

    def set_capture(self, vm: CaptureVM) -> None:
        pass


class Commands(Protocol):
    def toggle(self) -> None: ...
    def request_suggestion(self) -> None: ...
    def set_language(self, language: str | None) -> None: ...
    def clear(self) -> None: ...


# ================================================================= presenter
@dataclass
class Presenter:
    view: ViewPort
    commands: Commands
    user_language: str = "en"
    translation_enabled: bool = False
    rows: dict[int, Utterance] = field(default_factory=dict)
    suggestion_for: int | None = None
    _suggestion: SuggestionVM | None = None
    _state: PipelineState = PipelineState.STOPPED
    _cleared_upto: int = 0  # events about utterances <= this belong to a cleared conversation

    # ------------------------------------------------------------ events in
    def handle(self, event: Event) -> None:
        if self._is_cleared(event):
            return
        if isinstance(event, ConversationCleared):
            self._cleared_upto = max(self._cleared_upto, event.upto)
            self.rows.clear()
            self.suggestion_for = None
            self._suggestion = None
            self.view.clear_rows()
            self.view.show_suggestions(SuggestionVM(visible=False))
        elif isinstance(event, UtteranceUpdated):
            self._on_utterance(event.utterance)
        elif isinstance(event, UtteranceDiscarded):
            self.rows.pop(event.utterance_id, None)
            self.view.remove_row(event.utterance_id)
        elif isinstance(event, SuggestionStarted):
            self.suggestion_for = event.utterance_id
            self._show_suggestion(
                SuggestionVM(reply_to=self._text(event.utterance_id), loading=True)
            )
        elif isinstance(event, SuggestionsReady):
            s = event.suggestion
            self.suggestion_for = s.utterance_id
            if s.error is not None:
                vm = SuggestionVM(
                    reply_to=self._text(s.utterance_id),
                    message=f"Suggestions failed: {ERROR_TEXT[s.error]}",
                )
            else:
                vm = SuggestionVM(
                    reply_to=self._text(s.utterance_id),
                    options=tuple(OptionVM(o.source_text, o.user_text) for o in s.options),
                )
            self._show_suggestion(vm)
        elif isinstance(event, PipelineStatus):
            self._on_status(event)
        elif isinstance(event, StageError):
            self._on_error(event)
        elif isinstance(event, LanguageChanged):
            self.view.set_language(self.language_label(event.language, event.locked))

    def _on_utterance(self, u: Utterance) -> None:
        self.rows[u.id] = u
        self.view.upsert_row(self.row_vm(u))
        if self._suggestion is not None and self.suggestion_for is not None:
            stale = self._latest_final() > self.suggestion_for
            if stale != self._suggestion.stale:
                self._show_suggestion(self._suggestion, stale=stale)

    def _on_status(self, event: PipelineStatus) -> None:
        self._state = event.state
        busy = event.state is PipelineState.LOADING
        listening = event.state in (
            PipelineState.LOADING,
            PipelineState.LISTENING,
            PipelineState.FALLING_BEHIND,
        )
        self.view.set_listening(listening, busy)
        text = {
            PipelineState.STOPPED: "Stopped",
            PipelineState.LOADING: "Loading speech model…",
            PipelineState.LISTENING: "Listening",
            PipelineState.FALLING_BEHIND: "Falling behind",
            PipelineState.ERROR: "Error",
        }[event.state]
        if event.detail:
            text = f"{text}: {event.detail}"
        level = {
            PipelineState.ERROR: "error",
            PipelineState.FALLING_BEHIND: "warn",
        }.get(event.state, "info")
        self.view.set_status(StatusVM(text, level))

    def _on_error(self, event: StageError) -> None:
        if event.stage == "suggest" and event.kind in (
            ErrorKind.NOTHING_TO_ANSWER,
            ErrorKind.NOT_CONFIGURED,
        ):
            hint = (
                "Nothing has been said yet."
                if event.kind is ErrorKind.NOTHING_TO_ANSWER
                else "Suggestions are off. Enable them in Settings."
            )
            self._show_suggestion(SuggestionVM(message=hint))
            return
        if event.stage == "setup" and event.kind is ErrorKind.NOT_CONFIGURED:
            self.view.set_status(StatusVM(event.message, "warn"))
            return
        what = {"asr": "Transcription", "translate": "Translation", "suggest": "Suggestions",
                "audio": "Audio", "setup": "Setup", "settings": "Settings"}.get(event.stage, event.stage.title())  # fmt: skip
        level = "warn" if event.retryable else "error"
        self.view.set_status(StatusVM(f"{what}: {ERROR_TEXT[event.kind]} — {event.message}", level))

    # ------------------------------------------------------------ formatting
    def row_vm(self, u: Utterance) -> RowVM:
        failed = u.status is UtteranceStatus.FAILED
        note: str | None = None
        translation = u.translation
        if u.status is UtteranceStatus.LISTENING:
            note = "listening…"
        elif u.status is UtteranceStatus.PARTIAL:
            note = None
        elif failed:
            note = ERROR_TEXT.get(u.error or ErrorKind.INTERNAL, "failed")
        elif u.status is UtteranceStatus.FINAL:
            if u.error is not None:
                note = f"translation failed: {ERROR_TEXT[u.error]}"
            elif self.translation_enabled and not languages.same_language(
                u.source_lang, self.user_language
            ):
                note = "translating…"
        return RowVM(
            id=u.id,
            original=u.text,
            translation=translation,
            note=note,
            partial=u.status is UtteranceStatus.PARTIAL,
            failed=failed,
        )

    def language_label(self, language: str | None, locked: bool) -> str:
        if not language:
            return "Language: auto"
        return f"{languages.name(language)}{' (locked)' if locked else ''}"

    def capture_vm(self, status: ShieldStatus | None) -> CaptureVM:
        if status is ShieldStatus.HIDDEN:
            return CaptureVM(True, "Hidden from screen share")
        if status is None:
            return CaptureVM(False, "Visible in screen share (hiding is off)")
        if status is ShieldStatus.FAILED:
            return CaptureVM(False, "Visible in screen share (hiding failed)")
        return CaptureVM(False, "Visible in screen share")

    def _text(self, utterance_id: int) -> str:
        u = self.rows.get(utterance_id)
        return u.text if u is not None else ""

    def _latest_final(self) -> int:
        finals = [u.id for u in self.rows.values() if u.is_final and u.text]
        return max(finals, default=0)

    def _show_suggestion(self, vm: SuggestionVM, *, stale: bool | None = None) -> None:
        if stale is None:
            stale = self.suggestion_for is not None and self._latest_final() > self.suggestion_for
        vm = SuggestionVM(vm.reply_to, vm.options, vm.loading, vm.message, stale, True)
        self._suggestion = vm
        self.view.show_suggestions(vm)

    # ------------------------------------------------------------ user actions
    def on_start_stop(self) -> None:
        self.commands.toggle()

    def on_suggest(self) -> None:
        self.commands.request_suggestion()

    def on_clear(self) -> None:
        # The view clears when the core confirms (ConversationCleared), so results
        # already on their way cannot bring old rows back.
        self.commands.clear()

    def _is_cleared(self, event: Event) -> bool:
        if isinstance(event, UtteranceUpdated):
            uid = event.utterance.id
        elif isinstance(event, (UtteranceDiscarded, SuggestionStarted)):
            uid = event.utterance_id
        elif isinstance(event, SuggestionsReady):
            uid = event.suggestion.utterance_id
        else:
            return False
        return uid <= self._cleared_upto

    def on_unlock_language(self) -> None:
        self.commands.set_language(None)

    def detach(self) -> None:
        """Stop updating the view (shutdown). Later events and calls become no-ops."""
        self.view = NullView()

    def on_shield_status(self, status: ShieldStatus | None) -> None:
        self.view.set_capture(self.capture_vm(status))

    def on_user_language(self, code: str) -> None:
        self.user_language = code
        for u in self.rows.values():
            self.view.upsert_row(self.row_vm(u))
