"""End-to-end pipeline behaviour with fakes: ordering, pairing, staleness, backpressure,
failures, suggestions and lifecycle. Deterministic; no models, devices or network."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from polyglot.adapters.audio.playback import ArraySource
from polyglot.adapters.fakes import (
    AmplitudeVad,
    CannedSuggester,
    EchoTranslator,
    ScriptedTranscriber,
    silence,
    speech_like,
    tone,
)
from polyglot.core.errors import (
    AudioSourceError,
    AuthError,
    ProviderUnavailable,
    TranscriptionFailed,
)
from polyglot.core.events import (
    LanguageChanged,
    PipelineState,
    PipelineStatus,
    StageError,
    SuggestionsReady,
    SuggestionStarted,
    UtteranceDiscarded,
)
from polyglot.core.models import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    ErrorKind,
    IdSource,
    Speaker,
    Transcript,
    Utterance,
    UtteranceStatus,
)
from polyglot.core.pipeline import PipelineConfig
from polyglot.core.ports import AudioSink
from polyglot.core.profile import UserProfile
from polyglot.core.session import LanguagePolicy, Session
from tests.conftest import Recorder, utterances_audio, wait_for

TEXTS = ["uno", "dos", "tres", "cuatro"]


def run_to_end(pipeline: Any, timeout: float = 15.0) -> None:
    pipeline.start()
    assert wait_for(lambda: pipeline.source_ended, timeout), "source never ended"
    assert pipeline.wait_idle(timeout), "pipeline never became idle"


def test_transcripts_and_translations_stay_paired_when_translations_finish_out_of_order(
    make_pipeline: Any, recorder: Recorder
) -> None:
    # The first sentence's translation is the slowest, so translations complete 4,3,2,1.
    delays = {"uno": 0.4, "dos": 0.3, "tres": 0.2, "cuatro": 0.0}
    tr = EchoTranslator(delays=delays)
    p = make_pipeline(utterances_audio(4), transcriber=ScriptedTranscriber(TEXTS), translator=tr)
    run_to_end(p)
    finals = recorder.finals()
    assert [u.id for u in finals] == [1, 2, 3, 4]
    assert [u.text for u in finals] == TEXTS
    for u in finals:
        assert u.status is UtteranceStatus.TRANSLATED
        assert u.translation == f"[es→en] {u.text}"  # each translation on its own sentence


def test_exactly_one_asr_worker(make_pipeline: Any) -> None:
    asr = ScriptedTranscriber(TEXTS, delay_s=0.05)
    p = make_pipeline(utterances_audio(4), transcriber=asr)
    run_to_end(p)
    assert len(asr.calls) == 4
    assert asr.max_concurrency == 1


def test_language_locks_and_is_passed_to_whisper(make_pipeline: Any, recorder: Recorder) -> None:
    # All audio arrives at once here, so the worker may pick up sentence 3 a moment
    # before the coordinator locks the language from sentence 2. What is guaranteed:
    # no hint before two agreeing finals, the hint applies once locked, and never reverts.
    asr = ScriptedTranscriber(TEXTS * 2, language="es", probability=0.95, delay_s=0.05)
    cfg = PipelineConfig(ring_seconds=600, language_policy=LanguagePolicy(agree=2))
    p = make_pipeline(utterances_audio(6), transcriber=asr, config=cfg)
    run_to_end(p)
    hints = [hint for _, hint in asr.calls]
    assert hints[:2] == [None, None]
    first = hints.index("es")
    assert all(h == "es" for h in hints[first:])
    assert hints[-1] == "es"
    assert recorder.of(LanguageChanged) == [LanguageChanged("es", locked=True)]


def test_same_language_is_not_translated(make_pipeline: Any, recorder: Recorder) -> None:
    tr = EchoTranslator()
    p = make_pipeline(
        utterances_audio(2),
        transcriber=ScriptedTranscriber(["hi", "there"], language="en"),
        translator=tr,
    )
    run_to_end(p)
    assert tr.calls == []
    assert all(u.status is UtteranceStatus.FINAL for u in recorder.finals())


def test_retryable_translation_error_is_retried_once(
    make_pipeline: Any, recorder: Recorder
) -> None:
    class Flaky(EchoTranslator):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 1

        def translate(self, text: str, source: str | None, target: str, *, timeout: float) -> str:
            if self.failures:
                self.failures -= 1
                self.calls.append((text, source, target))
                raise ProviderUnavailable("blip")
            return super().translate(text, source, target, timeout=timeout)

    tr = Flaky()
    p = make_pipeline(utterances_audio(1), transcriber=ScriptedTranscriber(["hola"]), translator=tr)
    run_to_end(p)
    (u,) = recorder.finals()
    assert u.status is UtteranceStatus.TRANSLATED
    assert len(tr.calls) == 2


def test_non_retryable_translation_error_shows_on_the_row(
    make_pipeline: Any, recorder: Recorder
) -> None:
    tr = EchoTranslator(errors={"hola": AuthError("bad key")})
    p = make_pipeline(utterances_audio(1), transcriber=ScriptedTranscriber(["hola"]), translator=tr)
    run_to_end(p)
    (u,) = recorder.finals()
    assert u.status is UtteranceStatus.FINAL
    assert u.error is ErrorKind.AUTH
    assert u.translation is None  # never a made-up translation
    assert len(tr.calls) == 1
    errs = recorder.of(StageError)
    assert any(e.stage == "translate" and e.kind is ErrorKind.AUTH for e in errs)


def test_empty_transcript_is_discarded(make_pipeline: Any, recorder: Recorder) -> None:
    p = make_pipeline(utterances_audio(2), transcriber=ScriptedTranscriber(["", "hola"]))
    run_to_end(p)
    assert [e.utterance_id for e in recorder.of(UtteranceDiscarded)] == [1]
    assert [u.text for u in recorder.finals()] == ["hola"]


def test_transcription_failure_marks_row_failed(make_pipeline: Any, recorder: Recorder) -> None:
    asr = ScriptedTranscriber(["a", "b"], fail_on=[0], error=TranscriptionFailed("decoder"))
    p = make_pipeline(utterances_audio(2), transcriber=asr)
    run_to_end(p)
    us = recorder.utterances()
    assert us[1].status is UtteranceStatus.FAILED and us[1].error is ErrorKind.TRANSCRIPTION
    assert us[2].text == "b"


def test_falling_behind_drops_oldest_waiting_final_and_says_so(
    make_pipeline: Any, recorder: Recorder
) -> None:
    asr = ScriptedTranscriber(delay_s=0.3)
    cfg = PipelineConfig(ring_seconds=600, finals_queue_size=1)
    p = make_pipeline(utterances_audio(5, speech_s=0.5, gap_s=0.7), transcriber=asr, config=cfg)
    run_to_end(p)
    us = recorder.utterances()
    overloaded = [u for u in us.values() if u.error is ErrorKind.OVERLOADED]
    assert overloaded, "expected at least one dropped utterance"
    assert all(u.status is UtteranceStatus.FAILED for u in overloaded)
    assert any(s.state is PipelineState.FALLING_BEHIND for s in recorder.of(PipelineStatus))
    # Nothing is silently lost: every utterance ended up either transcribed or marked.
    assert all(u.is_final for u in us.values())


def test_suggestion_with_nothing_said(make_pipeline: Any, recorder: Recorder) -> None:
    p = make_pipeline(utterances_audio(0), suggester=CannedSuggester())
    p.request_suggestion()
    assert wait_for(lambda: recorder.of(StageError))
    assert recorder.of(StageError)[0].kind is ErrorKind.NOTHING_TO_ANSWER


def test_suggestion_not_configured(make_pipeline: Any, recorder: Recorder) -> None:
    p = make_pipeline(utterances_audio(0))
    p.request_suggestion()
    assert wait_for(lambda: recorder.of(StageError))
    assert recorder.of(StageError)[0].kind is ErrorKind.NOT_CONFIGURED


def test_suggestion_targets_latest_final_with_history(
    make_pipeline: Any, recorder: Recorder
) -> None:
    sug = CannedSuggester()
    p = make_pipeline(
        utterances_audio(3), transcriber=ScriptedTranscriber(TEXTS[:3]), suggester=sug
    )
    run_to_end(p)
    p.request_suggestion()
    assert wait_for(lambda: recorder.of(SuggestionsReady))
    (ready,) = recorder.of(SuggestionsReady)
    assert ready.suggestion.utterance_id == 3
    assert len(ready.suggestion.options) == 2
    ctx = sug.contexts[0]
    assert ctx.target.text == "tres"
    assert [u.text for u in ctx.history] == ["uno", "dos"]
    assert ctx.source_language == "Spanish" and ctx.user_language == "English"
    assert recorder.of(SuggestionStarted) == [SuggestionStarted(3)]


def test_pressing_suggest_again_supersedes_in_flight_request(
    make_pipeline: Any, recorder: Recorder
) -> None:
    p = make_pipeline(utterances_audio(1), suggester=CannedSuggester(delay_s=0.3))
    run_to_end(p)
    p.request_suggestion()
    time.sleep(0.05)
    p.request_suggestion()
    assert wait_for(lambda: len(recorder.of(SuggestionsReady)) >= 1, 3)
    time.sleep(0.5)  # let the first (superseded) request finish too
    assert len(recorder.of(SuggestionsReady)) == 1  # only the newer request is shown
    assert len(recorder.of(SuggestionStarted)) == 2


def test_late_suggestion_is_still_shown_linked_to_its_sentence(
    make_pipeline: Any, recorder: Recorder
) -> None:
    """Asked about sentence 1; sentence 2 finishes before the reply arrives. The reply is
    kept (not dropped) and stays linked to sentence 1."""
    audio = speech_like([("silence", 0.3), ("tone", 1.0), ("silence", 0.9)])
    later = speech_like([("tone", 1.0), ("silence", 0.9)])

    class TwoPart:
        sample_rate = SAMPLE_RATE

        def start(self, sink: AudioSink) -> None:
            self._sink = sink
            sink.on_audio(audio)

        def more(self) -> None:
            self._sink.on_audio(later)
            self._sink.on_end()

        def stop(self) -> None:
            pass

    src = TwoPart()
    sug = CannedSuggester(delay_s=0.6)
    p = make_pipeline(
        source=src, transcriber=ScriptedTranscriber(["primera", "segunda"]), suggester=sug
    )
    p.start()
    assert wait_for(lambda: len(recorder.finals()) == 1)
    p.request_suggestion()
    src.more()
    assert wait_for(lambda: len(recorder.finals()) == 2, 5)
    assert wait_for(lambda: recorder.of(SuggestionsReady), 5)
    (ready,) = recorder.of(SuggestionsReady)
    assert ready.suggestion.utterance_id == 1
    assert "primera" in ready.suggestion.options[0].source_text


def test_suggestion_failure_is_reported_not_replaced(
    make_pipeline: Any, recorder: Recorder
) -> None:
    p = make_pipeline(utterances_audio(1), suggester=CannedSuggester(error=AuthError("nope")))
    run_to_end(p)
    p.request_suggestion()
    assert wait_for(lambda: recorder.of(SuggestionsReady))
    s = recorder.of(SuggestionsReady)[0].suggestion
    assert s.options == () and s.error is ErrorKind.AUTH


def test_clear_ignores_late_results(make_pipeline: Any, recorder: Recorder) -> None:
    tr = EchoTranslator(delays={"uno": 0.4})
    p = make_pipeline(utterances_audio(1), transcriber=ScriptedTranscriber(["uno"]), translator=tr)
    p.start()
    assert wait_for(lambda: recorder.finals())
    p.clear()
    time.sleep(0.6)
    assert all(u.status is not UtteranceStatus.TRANSLATED for u in recorder.utterances().values())
    assert p.session.utterances == {}
    from polyglot.core.events import ConversationCleared

    assert recorder.of(ConversationCleared) == [ConversationCleared(1)]


def test_stop_flushes_utterance_in_progress_and_restart_continues_ids(
    make_pipeline: Any, recorder: Recorder
) -> None:
    class Endless:
        """Speech that never pauses; stops only when told to."""

        sample_rate = SAMPLE_RATE

        def __init__(self) -> None:
            self._stop = threading.Event()

        def start(self, sink: AudioSink) -> None:
            self._stop.clear()
            sink.on_audio(np.concatenate((np.zeros(4800, dtype=np.float32), tone(1.0))))

        def stop(self) -> None:
            self._stop.set()

    p = make_pipeline(source=Endless(), transcriber=ScriptedTranscriber(["a", "b"]))
    p.start()
    time.sleep(0.3)
    p.stop()  # mid-utterance: it must still be finished and transcribed
    assert [u.text for u in recorder.finals()] == ["a"]
    assert recorder.of(PipelineStatus)[-1].state is PipelineState.STOPPED or wait_for(
        lambda: recorder.of(PipelineStatus)[-1].state is PipelineState.STOPPED
    )
    p.start()
    time.sleep(0.3)
    p.stop()
    assert [u.id for u in recorder.finals()] == [1, 2]


def test_gap_fill_ends_utterance_when_loopback_goes_silent(
    make_pipeline: Any, recorder: Recorder
) -> None:
    """WASAPI loopback stops delivering data when nothing plays. The utterance must still
    end (by synthesized silence), without a stop or end-of-stream."""

    class GoesQuiet:
        sample_rate = SAMPLE_RATE

        def start(self, sink: AudioSink) -> None:
            sink.on_audio(np.concatenate((np.zeros(4800, dtype=np.float32), tone(1.0))))
            # ...and then nothing at all, like a loopback device with no audio playing.

        def stop(self) -> None:
            pass

    cfg = PipelineConfig(ring_seconds=60, gap_fill_after_s=0.1)
    p = make_pipeline(source=GoesQuiet(), transcriber=ScriptedTranscriber(["hola"]), config=cfg)
    p.start()
    assert wait_for(lambda: [u.text for u in recorder.finals()] == ["hola"], 3)


def test_stereo_and_resampled_input(bus: Any, recorder: Recorder) -> None:
    from polyglot.adapters.fakes import AmplitudeVad
    from polyglot.core.pipeline import Pipeline

    mono = utterances_audio(1)
    up = np.repeat(mono, 3)  # 48 kHz version of the 16 kHz signal
    stereo = np.stack((up, up), axis=1)

    class Decimate3:
        def process(self, block: Any) -> Any:
            return np.asarray(block, dtype=np.float32)[::3].copy()

        def flush(self) -> Any:
            return np.zeros(0, dtype=np.float32)

    rates: list[int] = []

    def factory(rate: int) -> Decimate3:
        rates.append(rate)
        return Decimate3()

    p = Pipeline(
        source=ArraySource(stereo, 48_000, block_ms=30),
        resampler_factory=factory,
        vad=AmplitudeVad(),
        transcriber=ScriptedTranscriber(["hola"]),
        translator=None,
        suggester=None,
        bus=bus,
        config=PipelineConfig(ring_seconds=60),
    )
    try:
        run_to_end(p)
        assert rates == [48_000]
        assert [u.text for u in recorder.finals()] == ["hola"]
    finally:
        p.close()


def test_source_start_failure_is_reported(make_pipeline: Any, recorder: Recorder) -> None:
    class Broken:
        sample_rate = SAMPLE_RATE

        def start(self, sink: AudioSink) -> None:
            raise OSError("device busy")

        def stop(self) -> None:
            pass

    p = make_pipeline(source=Broken())
    p.start()
    assert not p.running
    assert wait_for(lambda: any(e.kind is ErrorKind.AUDIO for e in recorder.of(StageError)))
    assert wait_for(lambda: p.state is PipelineState.ERROR)


def test_source_error_mid_stream_is_reported(make_pipeline: Any, recorder: Recorder) -> None:
    class Unplugged:
        sample_rate = SAMPLE_RATE

        def start(self, sink: AudioSink) -> None:
            threading.Timer(0.05, lambda: sink.on_error(OSError("unplugged"))).start()

        def stop(self) -> None:
            pass

    p = make_pipeline(source=Unplugged())
    p.start()
    assert wait_for(lambda: p.state is PipelineState.ERROR)
    assert any("unplugged" in e.message for e in recorder.of(StageError))


def test_model_load_failure_is_reported(make_pipeline: Any, recorder: Recorder) -> None:
    asr = ScriptedTranscriber(load_error=TranscriptionFailed("model download failed"))
    p = make_pipeline(utterances_audio(1), transcriber=asr)
    p.start()
    assert wait_for(lambda: p.state is PipelineState.ERROR)
    assert any(e.stage == "asr" and "download" in e.message for e in recorder.of(StageError))


def test_status_sequence(make_pipeline: Any, recorder: Recorder) -> None:
    p = make_pipeline(utterances_audio(1))
    run_to_end(p)
    p.stop()
    assert wait_for(lambda: recorder.of(PipelineStatus)[-1].state is PipelineState.STOPPED)
    states = [s.state for s in recorder.of(PipelineStatus)]
    assert states[:2] == [PipelineState.LOADING, PipelineState.LISTENING]


@pytest.mark.parametrize("partial_ms", [320])
def test_partials_flow_then_final_overrides(
    make_pipeline: Any, recorder: Recorder, partial_ms: int
) -> None:
    from polyglot.core.segmenter import SegmenterConfig

    cfg = PipelineConfig(
        ring_seconds=600, segmenter=SegmenterConfig(partial_interval_ms=partial_ms)
    )
    asr = ScriptedTranscriber(lambda audio: f"{len(audio)} samples")
    # Real-time pacing: if all audio arrived at once, the final would (correctly)
    # replace the queued partials before any were transcribed.
    src = ArraySource(utterances_audio(1, speech_s=1.5), SAMPLE_RATE, realtime=True, block_ms=32)
    p = make_pipeline(source=src, transcriber=asr, config=cfg)
    run_to_end(p)
    from polyglot.core.events import UtteranceUpdated

    statuses = [e.utterance.status for e in recorder.of(UtteranceUpdated)]
    assert UtteranceStatus.PARTIAL in statuses
    assert statuses[-1] is UtteranceStatus.FINAL
    versions = [
        e.utterance.partial_version
        for e in recorder.of(UtteranceUpdated)
        if e.utterance.status is UtteranceStatus.PARTIAL
    ]
    assert versions == sorted(versions)


# ---------------------------------------------------------------- lifecycle regressions
# Found by an independent review probing start/stop/failure interleavings.


def test_model_load_failure_stops_capture_and_keeps_error(
    make_pipeline: Any, recorder: Recorder
) -> None:
    asr = ScriptedTranscriber(load_error=TranscriptionFailed("no model"))
    src = ArraySource(utterances_audio(6), SAMPLE_RATE, realtime=True)
    p = make_pipeline(
        source=src, transcriber=asr, config=PipelineConfig(ring_seconds=60, finals_queue_size=1)
    )
    p.start()
    assert wait_for(lambda: not p.running, 5), "capture should stop when the model cannot load"
    assert wait_for(lambda: recorder.of(PipelineStatus)[-1].state is PipelineState.STOPPED, 5)
    states = [s.state for s in recorder.of(PipelineStatus)]
    assert PipelineState.ERROR in states
    assert PipelineState.FALLING_BEHIND not in states[states.index(PipelineState.ERROR) :]
    # nothing left "listening" forever
    assert all(u.status is not UtteranceStatus.LISTENING for u in recorder.utterances().values())


def test_model_loaded_after_stop_does_not_claim_listening(
    make_pipeline: Any, recorder: Recorder
) -> None:
    class SlowLoad(ScriptedTranscriber):
        def load(self) -> None:
            time.sleep(0.5)
            super().load()

    p = make_pipeline(utterances_audio(1), transcriber=SlowLoad())
    p.start()
    p.stop(timeout=0.1)
    time.sleep(0.8)
    assert p.state is PipelineState.STOPPED
    assert recorder.of(PipelineStatus)[-1].state is PipelineState.STOPPED


def test_restart_never_runs_two_asr_workers(make_pipeline: Any) -> None:
    asr = ScriptedTranscriber(delay_s=0.5)
    p = make_pipeline(utterances_audio(4), transcriber=asr)
    p.start()
    time.sleep(0.3)
    p.stop(timeout=0.05)  # the worker is mid-transcription and outlives stop()
    p.start()
    time.sleep(1.5)
    p.stop()
    assert asr.max_concurrency == 1


def test_suggestion_skips_a_failed_latest_sentence(make_pipeline: Any, recorder: Recorder) -> None:
    asr = ScriptedTranscriber(["uno", "dos"], fail_on=[1], error=TranscriptionFailed("x"))
    sug = CannedSuggester()
    p = make_pipeline(utterances_audio(2), transcriber=asr, suggester=sug)
    run_to_end(p)
    p.request_suggestion()
    assert wait_for(lambda: recorder.of(SuggestionsReady))
    assert recorder.of(SuggestionsReady)[0].suggestion.utterance_id == 1


def test_retranslate_pending_after_rebuild(bus: Any, recorder: Recorder) -> None:
    from polyglot.adapters.audio.resample import IdentityResampler
    from polyglot.adapters.fakes import AmplitudeVad
    from polyglot.core.models import Utterance
    from polyglot.core.pipeline import Pipeline
    from polyglot.core.session import Session

    session = Session("en")
    session.put(Utterance(1, 0.0, 1.0, UtteranceStatus.FINAL, "hola", "es"))
    p = Pipeline(
        source=ArraySource(utterances_audio(0), SAMPLE_RATE),
        resampler_factory=lambda _r: IdentityResampler(),
        vad=AmplitudeVad(),
        transcriber=ScriptedTranscriber(),
        translator=EchoTranslator(),
        suggester=None,
        bus=bus,
        session=session,
    )
    try:
        p.retranslate_pending()
        assert wait_for(lambda: session.get(1).status is UtteranceStatus.TRANSLATED)  # type: ignore[union-attr]
        assert session.get(1).translation == "[es→en] hola"  # type: ignore[union-attr]
    finally:
        p.close()


# ------------------------------------------------------------------ context and profile
def test_suggestion_context_carries_the_profile_from_config(
    make_pipeline: Any, recorder: Recorder
) -> None:
    sug = CannedSuggester()
    profile = UserProfile(name="Sam", hobbies=("chess",))
    p = make_pipeline(
        utterances_audio(1),
        transcriber=ScriptedTranscriber(TEXTS[:1]),
        suggester=sug,
        config=PipelineConfig(ring_seconds=600.0, profile=profile),
    )
    run_to_end(p)
    p.request_suggestion()
    assert wait_for(lambda: recorder.of(SuggestionsReady))
    assert sug.contexts[0].profile == profile


def test_a_profile_change_applies_to_the_next_suggestion(
    make_pipeline: Any, recorder: Recorder
) -> None:
    sug = CannedSuggester()
    p = make_pipeline(
        utterances_audio(1), transcriber=ScriptedTranscriber(TEXTS[:1]), suggester=sug
    )
    run_to_end(p)
    p.request_suggestion()
    assert wait_for(lambda: len(recorder.of(SuggestionsReady)) == 1)
    p.set_profile(UserProfile(role="Engineer"))
    p.request_suggestion()
    assert wait_for(lambda: len(recorder.of(SuggestionsReady)) == 2)
    assert sug.contexts[0].profile.is_empty
    assert sug.contexts[1].profile.role == "Engineer"


def test_the_users_own_turns_are_context_but_never_the_target(
    make_pipeline: Any, recorder: Recorder
) -> None:
    session = Session("en")
    session.put(
        Utterance(
            id=1,
            t_start=0.0,
            t_end=1.0,
            status=UtteranceStatus.FINAL,
            text="I led a team of five",
            speaker=Speaker.USER,
        )
    )
    session.put(  # spoken "later" than everything the audio will produce
        Utterance(
            id=500,
            t_start=0.0,
            t_end=1.0,
            status=UtteranceStatus.FINAL,
            text="Thanks",
            speaker=Speaker.USER,
        )
    )
    sug = CannedSuggester()
    p = make_pipeline(
        utterances_audio(2),
        transcriber=ScriptedTranscriber(TEXTS[:2]),
        suggester=sug,
        session=session,
        ids=IdSource(2),
    )
    run_to_end(p)
    p.request_suggestion()
    assert wait_for(lambda: recorder.of(SuggestionsReady))
    ctx = sug.contexts[0]
    assert ctx.target.text == "dos" and ctx.target.speaker is Speaker.OTHER
    assert [(u.text, u.speaker) for u in ctx.history] == [
        ("I led a team of five", Speaker.USER),
        ("uno", Speaker.OTHER),
    ]


# ------------------------------------------------------------------ the user's microphone
class ByLoudness:
    """Quiet speech is "them", loud speech is "me"; both in Spanish, so a missing guard
    shows up as the user's own lines being translated or moving the language lock."""

    def __init__(self, language: str = "es") -> None:
        self.language = language

    def load(self) -> None:
        pass

    def transcribe(self, audio: Any, language: str | None) -> Transcript:
        loud = float(np.max(np.abs(audio))) > 0.5
        return Transcript("me" if loud else "them", self.language, 0.95)


def bursts(n: int, amp: float) -> Any:
    parts: list[Any] = [silence(0.3)]
    for _ in range(n):
        parts += [tone(1.0, amp=amp), silence(0.9)]
    audio = np.concatenate(parts)
    return np.concatenate((audio, np.zeros((-audio.size) % FRAME_SAMPLES, dtype=np.float32)))


class LiveFake:
    """A source the test feeds by hand, like a real device."""

    sample_rate = SAMPLE_RATE

    def __init__(self, fail_on_start: Exception | None = None) -> None:
        self.sink: Any = None
        self.started = self.stopped = 0
        self.fail_on_start = fail_on_start

    def start(self, sink: Any) -> None:
        if self.fail_on_start is not None:
            raise self.fail_on_start
        self.sink, self.started = sink, self.started + 1

    def stop(self) -> None:
        self.stopped += 1


def two_sources(make_pipeline: Any, *, others: int, mine: int, **kw: Any) -> Any:
    return make_pipeline(
        bursts(others, 0.3),
        transcriber=ByLoudness(),
        user_source=ArraySource(bursts(mine, 0.8), SAMPLE_RATE),
        user_vad=AmplitudeVad(),
        **kw,
    )


def run_both(p: Any) -> None:
    p.start()
    assert wait_for(lambda: p.source_ended and p.user_source_ended, 15)
    assert p.wait_idle(15), "pipeline never became idle"


def test_microphone_utterances_are_tagged_user_and_share_one_id_sequence(
    make_pipeline: Any, recorder: Recorder
) -> None:
    p = two_sources(make_pipeline, others=3, mine=2)
    run_both(p)
    finals = recorder.finals()
    assert sorted(u.text for u in finals) == ["me", "me", "them", "them", "them"]
    assert {u.text: u.speaker for u in finals} == {"me": Speaker.USER, "them": Speaker.OTHER}
    assert len({u.id for u in finals}) == 5  # one IdSource: no clashes between streams


def test_the_users_own_lines_are_not_translated(make_pipeline: Any, recorder: Recorder) -> None:
    p = two_sources(make_pipeline, others=2, mine=2, translator=EchoTranslator())
    run_both(p)
    finals = recorder.finals()
    mine = [u for u in finals if u.speaker is Speaker.USER]
    theirs = [u for u in finals if u.speaker is Speaker.OTHER]
    assert len(mine) == 2 and len(theirs) == 2
    assert all(u.status is UtteranceStatus.FINAL and u.translation is None for u in mine)
    assert all(u.status is UtteranceStatus.TRANSLATED for u in theirs)


def test_the_users_speech_does_not_move_the_language_lock(
    make_pipeline: Any, recorder: Recorder
) -> None:
    # Two agreeing confident finals lock the language. One from them plus three from the
    # user must not.
    p = two_sources(make_pipeline, others=1, mine=3)
    run_both(p)
    assert p.session.locked_language is None
    assert not recorder.of(LanguageChanged)


def test_suggestions_reply_to_them_with_both_speakers_as_history(
    make_pipeline: Any, recorder: Recorder
) -> None:
    sug = CannedSuggester()
    p = two_sources(make_pipeline, others=2, mine=2, suggester=sug)
    run_both(p)
    p.request_suggestion()
    assert wait_for(lambda: recorder.of(SuggestionsReady))
    ctx = sug.contexts[0]
    finals = recorder.finals()
    latest_theirs = max(u.id for u in finals if u.speaker is Speaker.OTHER)
    assert ctx.target.id == latest_theirs and ctx.target.speaker is Speaker.OTHER
    assert [(u.id, u.speaker) for u in ctx.history] == [
        (u.id, u.speaker) for u in finals if u.id < latest_theirs
    ]


def test_a_microphone_that_will_not_start_does_not_stop_listening_to_them(
    make_pipeline: Any, recorder: Recorder
) -> None:
    p = make_pipeline(
        bursts(1, 0.3),
        transcriber=ByLoudness(),
        user_source=LiveFake(fail_on_start=AudioSourceError("device busy")),
        user_vad=AmplitudeVad(),
    )
    p.start()
    assert wait_for(lambda: p.source_ended and p.user_source_ended, 10)
    assert p.wait_idle(10)
    assert [u.text for u in recorder.finals()] == ["them"]
    (err,) = [e for e in recorder.of(StageError) if e.stage == "audio"]
    assert "microphone" in err.message and "device busy" in err.message and err.retryable
    assert all(s.state is not PipelineState.ERROR for s in recorder.of(PipelineStatus))
    assert p.running


def test_a_live_microphone_is_transcribed_and_stopped_with_the_pipeline(
    make_pipeline: Any, recorder: Recorder
) -> None:
    other, mic = LiveFake(), LiveFake()
    p = make_pipeline(
        source=other, transcriber=ByLoudness(), user_source=mic, user_vad=AmplitudeVad()
    )
    p.start()
    assert other.started == 1 and mic.started == 1
    mic.sink.on_audio(bursts(1, 0.8))  # the user speaks into the microphone
    assert wait_for(lambda: any(u.text == "me" for u in recorder.finals()), 10)
    (mine,) = recorder.finals()
    assert mine.speaker is Speaker.USER
    p.stop()
    assert other.stopped == 1 and mic.stopped == 1


def test_user_source_and_vad_must_come_together(make_pipeline: Any) -> None:
    with pytest.raises(ValueError, match="go together"):
        make_pipeline(utterances_audio(1), user_source=LiveFake())
