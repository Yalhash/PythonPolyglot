from __future__ import annotations

from typing import Any

from polyglot.core.events import (
    ConversationCleared,
    LanguageChanged,
    PipelineState,
    PipelineStatus,
    StageError,
    SuggestionsReady,
    SuggestionStarted,
    UtteranceDiscarded,
    UtteranceUpdated,
)
from polyglot.core.models import (
    ErrorKind,
    ReplyOption,
    Speaker,
    Suggestion,
    Utterance,
    UtteranceStatus,
)
from polyglot.core.ports import ShieldStatus
from polyglot.ui.presenter import Presenter, RowVM, StatusVM, SuggestionVM


class FakeView:
    def __init__(self) -> None:
        self.rows: dict[int, RowVM] = {}
        self.calls: list[tuple[str, Any]] = []
        self.suggestion: SuggestionVM | None = None
        self.status: StatusVM | None = None

    def upsert_row(self, row: RowVM) -> None:
        self.rows[row.id] = row

    def remove_row(self, uid: int) -> None:
        self.rows.pop(uid, None)

    def clear_rows(self) -> None:
        self.rows.clear()

    def show_suggestions(self, vm: SuggestionVM) -> None:
        self.suggestion = vm

    def set_status(self, vm: StatusVM) -> None:
        self.status = vm

    def __getattr__(self, name: str) -> Any:
        return lambda *a: self.calls.append((name, a))


class FakeCommands:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *a: self.calls.append((name, a))


def u(uid: int, status: UtteranceStatus, text: str = "", **kw: Any) -> UtteranceUpdated:
    t_end = None if status in (UtteranceStatus.LISTENING, UtteranceStatus.PARTIAL) else 1.0
    return UtteranceUpdated(Utterance(uid, 0.0, t_end, status, text, **kw))


def make(translation: bool = True) -> tuple[Presenter, FakeView, FakeCommands]:
    view, cmds = FakeView(), FakeCommands()
    return Presenter(view, cmds, user_language="en", translation_enabled=translation), view, cmds


def test_row_lifecycle_notes() -> None:
    p, view, _ = make()
    p.handle(u(1, UtteranceStatus.LISTENING))
    assert view.rows[1].note == "listening…"
    p.handle(u(1, UtteranceStatus.PARTIAL, "hola qu"))
    assert view.rows[1].partial and view.rows[1].note is None
    p.handle(u(1, UtteranceStatus.FINAL, "hola", source_lang="es"))
    assert view.rows[1].note == "translating…"
    p.handle(u(1, UtteranceStatus.TRANSLATED, "hola", source_lang="es", translation="hello"))
    assert view.rows[1] == RowVM(1, "hola", "hello", None, False, False)


def test_no_translating_note_when_same_language_or_disabled() -> None:
    p, view, _ = make()
    p.handle(u(1, UtteranceStatus.FINAL, "hi", source_lang="en"))
    assert view.rows[1].note is None
    p2, view2, _ = make(translation=False)
    p2.handle(u(1, UtteranceStatus.FINAL, "hola", source_lang="es"))
    assert view2.rows[1].note is None


def test_failures_are_shown_not_hidden() -> None:
    p, view, _ = make()
    p.handle(u(1, UtteranceStatus.FINAL, "hola", source_lang="es", error=ErrorKind.TIMEOUT))
    assert view.rows[1].note == "translation failed: timed out"
    p.handle(u(2, UtteranceStatus.FAILED, "", error=ErrorKind.OVERLOADED))
    assert view.rows[2].failed and "fell behind" in (view.rows[2].note or "")


def test_discard_removes_row() -> None:
    p, view, _ = make()
    p.handle(u(1, UtteranceStatus.LISTENING))
    p.handle(UtteranceDiscarded(1))
    assert 1 not in view.rows and 1 not in p.rows


def test_suggestions_labelled_with_sentence_and_marked_stale() -> None:
    p, view, _ = make()
    p.handle(u(1, UtteranceStatus.FINAL, "¿Qué opina?", source_lang="es"))
    p.handle(SuggestionStarted(1))
    assert view.suggestion and view.suggestion.loading and view.suggestion.reply_to == "¿Qué opina?"
    p.handle(SuggestionsReady(Suggestion(1, (ReplyOption("Creo que sí", "I think so"),))))
    assert view.suggestion.options[0].say == "Creo que sí"
    assert not view.suggestion.stale
    p.handle(u(2, UtteranceStatus.FINAL, "Otra cosa", source_lang="es"))
    assert view.suggestion.stale  # newer sentence said since: offer a refresh
    assert view.suggestion.reply_to == "¿Qué opina?"  # still linked to what it answers


def test_suggestion_error_and_hints() -> None:
    p, view, _ = make()
    p.handle(StageError("suggest", ErrorKind.NOTHING_TO_ANSWER, "x"))
    assert view.suggestion and view.suggestion.message == "Nothing has been said yet."
    p.handle(u(1, UtteranceStatus.FINAL, "hola"))
    p.handle(SuggestionsReady(Suggestion(1, (), ErrorKind.QUOTA, "quota")))
    assert view.suggestion.message == "Suggestions failed: quota exhausted"


def test_status_and_listening_state() -> None:
    p, view, _ = make()
    p.handle(PipelineStatus(PipelineState.LOADING, "loading speech model"))
    assert ("set_listening", (True, True)) in view.calls
    p.handle(PipelineStatus(PipelineState.FALLING_BEHIND, "slow"))
    assert view.status == StatusVM("Falling behind: slow", "warn")
    p.handle(PipelineStatus(PipelineState.STOPPED))
    assert ("set_listening", (False, False)) in view.calls
    p.handle(StageError("translate", ErrorKind.AUTH, "bad key"))
    assert view.status is not None and view.status.level == "error"
    assert "Translation: API key rejected" in view.status.text
    p.handle(StageError("setup", ErrorKind.NOT_CONFIGURED, "No key"))
    assert view.status == StatusVM("No key", "warn")


def test_language_and_capture_labels() -> None:
    p, view, _ = make()
    p.handle(LanguageChanged("es", locked=True))
    assert ("set_language", ("Spanish (locked)",)) in view.calls
    assert p.capture_vm(ShieldStatus.HIDDEN).hidden
    assert p.capture_vm(ShieldStatus.UNSUPPORTED).text == "Visible in screen share"
    assert "failed" in p.capture_vm(ShieldStatus.FAILED).text
    assert "off" in p.capture_vm(None).text


def test_user_actions_become_commands() -> None:
    p, view, cmds = make()
    p.handle(u(1, UtteranceStatus.FINAL, "hola"))
    p.on_start_stop()
    p.on_suggest()
    p.on_unlock_language()
    p.on_clear()
    assert [c[0] for c in cmds.calls] == ["toggle", "request_suggestion", "set_language", "clear"]
    assert cmds.calls[2] == ("set_language", (None,))
    assert 1 in view.rows  # cleared only when the core confirms


def test_clear_waits_for_core_and_ignores_late_events_for_cleared_rows() -> None:
    p, view, _ = make()
    p.handle(u(1, UtteranceStatus.FINAL, "hola", source_lang="es"))
    p.handle(u(2, UtteranceStatus.LISTENING))
    p.handle(ConversationCleared(upto=2))
    assert view.rows == {} and view.suggestion == SuggestionVM(visible=False)
    # Results that were already queued when Clear was pressed must not resurrect rows.
    p.handle(u(1, UtteranceStatus.TRANSLATED, "hola", source_lang="es", translation="hello"))
    p.handle(SuggestionsReady(Suggestion(2, (ReplyOption("a", "b"),))))
    assert view.rows == {} and view.suggestion == SuggestionVM(visible=False)
    p.handle(u(3, UtteranceStatus.LISTENING))  # new conversation
    assert list(view.rows) == [3]


def test_detach_stops_all_view_updates() -> None:
    """Shutdown detaches the presenter first, so a late signal (e.g. the shield status
    recomputed while windows are destroyed) cannot reach deleted widgets."""
    p, view, _ = make()
    p.detach()
    p.on_shield_status(ShieldStatus.HIDDEN)
    p.handle(u(1, UtteranceStatus.FINAL, "hola"))
    p.handle(LanguageChanged("es", locked=True))
    assert view.calls == [] and view.rows == {}


# ------------------------------------------------------------------ the user's own lines
def test_the_users_own_lines_are_marked_and_never_translating() -> None:
    from polyglot.core.models import Speaker

    p, view, _ = make()
    p.handle(u(1, UtteranceStatus.FINAL, "hola", source_lang="es"))
    p.handle(u(2, UtteranceStatus.FINAL, "buenos dias", source_lang="es", speaker=Speaker.USER))
    assert view.rows[1].note == "translating…" and not view.rows[1].mine
    assert view.rows[2].mine and view.rows[2].note is None  # no translation will come


def keyed(n: int = 4) -> tuple[Presenter, FakeView, FakeCommands]:
    p, view, cmds = make()
    p.set_reply_keys(tuple(f"Ctrl+Alt+{d}" for d in "1234567890"[:n]))
    return p, view, cmds


def test_their_finished_sentences_get_reply_hotkeys_newest_first() -> None:
    p, view, _ = keyed(2)
    p.handle(u(1, UtteranceStatus.FINAL, "uno", source_lang="es"))
    assert view.rows[1].reply_key == "Ctrl+Alt+1"
    p.handle(u(2, UtteranceStatus.LISTENING))
    assert view.rows[2].reply_key is None  # not ready yet: no hotkey, no highlight
    p.handle(u(2, UtteranceStatus.FINAL, "dos", source_lang="es"))
    assert view.rows[2].reply_key == "Ctrl+Alt+1"
    assert view.rows[1].reply_key == "Ctrl+Alt+2"  # shifted down by the newer sentence
    p.handle(u(3, UtteranceStatus.FINAL, "tres", source_lang="es"))
    assert [view.rows[i].reply_key for i in (3, 2, 1)] == ["Ctrl+Alt+1", "Ctrl+Alt+2", None]


def test_mine_and_failed_sentences_get_no_reply_hotkey() -> None:
    p, view, _ = keyed()
    p.handle(u(1, UtteranceStatus.FINAL, "uno", source_lang="es"))
    p.handle(u(2, UtteranceStatus.FINAL, "my answer", speaker=Speaker.USER))
    p.handle(u(3, UtteranceStatus.FAILED, "", error=ErrorKind.TRANSCRIPTION))
    assert view.rows[1].reply_key == "Ctrl+Alt+1"
    assert view.rows[2].reply_key is None and view.rows[3].reply_key is None


def test_discarding_a_sentence_moves_the_hotkeys_up() -> None:
    p, view, _ = keyed()
    p.handle(u(1, UtteranceStatus.FINAL, "uno"))
    p.handle(u(2, UtteranceStatus.FINAL, "dos"))
    p.handle(UtteranceDiscarded(2))
    assert view.rows[1].reply_key == "Ctrl+Alt+1"


def test_no_hotkeys_bound_means_no_highlight() -> None:
    p, view, _ = make()
    p.handle(u(1, UtteranceStatus.FINAL, "uno"))
    assert view.rows[1].reply_key is None
    p.set_reply_keys(("Ctrl+Alt+1",))  # bound later (settings saved): labels appear
    assert view.rows[1].reply_key == "Ctrl+Alt+1"
    p.set_reply_keys(())
    assert view.rows[1].reply_key is None


def test_reply_hotkey_asks_for_that_sentence() -> None:
    p, view, cmds = keyed()
    for i, text in enumerate(["uno", "dos", "tres"], 1):
        p.handle(u(i, UtteranceStatus.FINAL, text))
    p.on_suggest_slot(0)
    p.on_suggest_slot(2)
    assert cmds.calls == [("request_suggestion", (3,)), ("request_suggestion", (1,))]
    p.on_suggest_slot(3)  # Ctrl+Alt+4: nothing that old yet
    assert len(cmds.calls) == 2
    assert view.suggestion and view.suggestion.message == "Nothing to reply to with Ctrl+Alt+4 yet."


def test_answering_an_older_sentence_is_not_stale_until_they_say_more() -> None:
    p, view, _ = keyed()
    p.handle(u(1, UtteranceStatus.FINAL, "uno"))
    p.handle(u(2, UtteranceStatus.FINAL, "dos"))
    p.handle(SuggestionStarted(1))
    p.handle(SuggestionsReady(Suggestion(1, (ReplyOption("a", "b"),))))
    assert view.suggestion and view.suggestion.reply_to == "uno" and not view.suggestion.stale
    p.handle(u(3, UtteranceStatus.FINAL, "tres"))
    assert view.suggestion.stale


def test_clear_drops_the_reply_hotkeys() -> None:
    p, view, _ = keyed()
    p.handle(u(1, UtteranceStatus.FINAL, "uno"))
    p.handle(ConversationCleared(1))
    p.handle(u(2, UtteranceStatus.FINAL, "dos"))
    assert view.rows[2].reply_key == "Ctrl+Alt+1"


def test_only_their_sentences_make_a_suggestion_stale() -> None:
    from polyglot.core.models import Speaker

    p, view, _ = make()
    p.handle(u(1, UtteranceStatus.FINAL, "¿Qué opina?", source_lang="es"))
    p.handle(SuggestionStarted(1))
    p.handle(SuggestionsReady(Suggestion(1, (ReplyOption("Creo que sí", "I think so"),))))
    assert view.suggestion is not None and not view.suggestion.stale
    p.handle(u(2, UtteranceStatus.FINAL, "Creo que sí", source_lang="es", speaker=Speaker.USER))
    assert not view.suggestion.stale  # the user answering is not something new to reply to
    p.handle(u(3, UtteranceStatus.FINAL, "Interesante", source_lang="es"))
    assert view.suggestion.stale
