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
    CannedSuggester,
    EchoTranslator,
    ScriptedTranscriber,
    speech_like,
    tone,
)
from polyglot.core.errors import AuthError, ProviderUnavailable, TranscriptionFailed
from polyglot.core.events import (
    LanguageChanged,
    PipelineState,
    PipelineStatus,
    StageError,
    SuggestionsReady,
    SuggestionStarted,
    UtteranceDiscarded,
)
from polyglot.core.models import SAMPLE_RATE, ErrorKind, UtteranceStatus
from polyglot.core.pipeline import PipelineConfig
from polyglot.core.ports import AudioSink
from polyglot.core.session import LanguagePolicy
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
    asr = ScriptedTranscriber(TEXTS, language="es", probability=0.95)
    cfg = PipelineConfig(ring_seconds=600, language_policy=LanguagePolicy(agree=2))
    p = make_pipeline(utterances_audio(4), transcriber=asr, config=cfg)
    run_to_end(p)
    assert [hint for _, hint in asr.calls] == [None, None, "es", "es"]
    assert recorder.of(LanguageChanged) == [LanguageChanged("es", locked=True)]


def test_same_language_is_not_translated(make_pipeline: Any, recorder: Recorder) -> None:
    tr = EchoTranslator()
    p = make_pipeline(
        utterances_audio(2), transcriber=ScriptedTranscriber(["hi", "there"], language="en"), translator=tr
    )
    run_to_end(p)
    assert tr.calls == []
    assert all(u.status is UtteranceStatus.FINAL for u in recorder.finals())


def test_retryable_translation_error_is_retried_once(make_pipeline: Any, recorder: Recorder) -> None:
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


def test_non_retryable_translation_error_shows_on_the_row(make_pipeline: Any, recorder: Recorder) -> None:
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


def test_suggestion_targets_latest_final_with_history(make_pipeline: Any, recorder: Recorder) -> None:
    sug = CannedSuggester()
    p = make_pipeline(utterances_audio(3), transcriber=ScriptedTranscriber(TEXTS[:3]), suggester=sug)
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
    p = make_pipeline(source=src, transcriber=ScriptedTranscriber(["primera", "segunda"]), suggester=sug)
    p.start()
    assert wait_for(lambda: len(recorder.finals()) == 1)
    p.request_suggestion()
    src.more()
    assert wait_for(lambda: len(recorder.finals()) == 2, 5)
    assert wait_for(lambda: recorder.of(SuggestionsReady), 5)
    (ready,) = recorder.of(SuggestionsReady)
    assert ready.suggestion.utterance_id == 1
    assert "primera" in ready.suggestion.options[0].source_text


def test_suggestion_failure_is_reported_not_replaced(make_pipeline: Any, recorder: Recorder) -> None:
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
def test_partials_flow_then_final_overrides(make_pipeline: Any, recorder: Recorder, partial_ms: int) -> None:
    from polyglot.core.segmenter import SegmenterConfig

    cfg = PipelineConfig(ring_seconds=600, segmenter=SegmenterConfig(partial_interval_ms=partial_ms))
    asr = ScriptedTranscriber(lambda audio: f"{len(audio)} samples")
    p = make_pipeline(utterances_audio(1, speech_s=2.0), transcriber=asr, config=cfg)
    run_to_end(p)
    from polyglot.core.events import UtteranceUpdated

    statuses = [e.utterance.status for e in recorder.of(UtteranceUpdated)]
    assert UtteranceStatus.PARTIAL in statuses
    assert statuses[-1] is UtteranceStatus.FINAL
    versions = [e.utterance.partial_version for e in recorder.of(UtteranceUpdated) if e.utterance.status is UtteranceStatus.PARTIAL]
    assert versions == sorted(versions)
