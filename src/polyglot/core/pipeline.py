"""The pipeline: stages on their own threads, joined by bounded queues.

    audio callback -> RingBuffer -> FrontEnd (resample, VAD, segmenter)
        -> AsrQueue (finals, max N; latest partial per utterance) -> AsrWorker
        -> Coordinator (owns Session) <-> network pool (translate / suggest)
        -> EventBus

Rules this module enforces (see the design doc, "Pipeline"):

* The audio callback only downmixes and copies into the ring buffer.
* Exactly one ASR worker; finals are transcribed in utterance-ID order.
* Only the coordinator thread mutates the Session. Everything else posts messages.
* Every result carries its utterance ID and is placed by it. Nothing is dropped for being
  late; the only discards are superseded partials and superseded suggestion requests.
* Queues are bounded. When ASR falls behind, the oldest waiting final is dropped, marked
  ``FAILED(overloaded)``, and a FALLING_BEHIND status is published.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from polyglot.core import languages
from polyglot.core.context import ContextBuilder
from polyglot.core.errors import PolyglotError, ProviderError
from polyglot.core.events import (
    ConversationCleared,
    EventBus,
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
    FRAME_SAMPLES,
    FRAME_SECONDS,
    ConversationType,
    ErrorKind,
    IdSource,
    ReplyOption,
    Speaker,
    SpeechSegment,
    Suggestion,
    Transcript,
    Utterance,
    UtteranceStatus,
)
from polyglot.core.ports import (
    AudioSource,
    RawAudio,
    Resampler,
    Suggester,
    Transcriber,
    Translator,
    VoiceActivityModel,
)
from polyglot.core.profile import UserProfile
from polyglot.core.ringbuffer import RingBuffer
from polyglot.core.segmenter import (
    Discarded,
    Segmenter,
    SegmenterConfig,
    SegmentReady,
    SpeechStarted,
)
from polyglot.core.session import LanguagePolicy, Session

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    target_language: str = "en"
    language_policy: LanguagePolicy = field(default_factory=LanguagePolicy)
    translate_timeout_s: float = 3.0
    suggest_timeout_s: float = 6.0
    suggestion_history: int = 6
    suggestion_history_chars: int = 3000
    suggestion_options: int = 2
    conversation_type: ConversationType = ConversationType.MEETING
    profile: UserProfile = field(default_factory=UserProfile)
    ring_seconds: float = 30.0
    finals_queue_size: int = 8
    gap_fill: bool = True
    gap_fill_after_s: float = 0.25
    network_workers: int = 4
    network_retries: int = 1


# ============================================================== ASR queue
class AsrQueue:
    """Finals in arrival (= ID) order, bounded; plus one latest-only partial slot."""

    def __init__(self, max_finals: int) -> None:
        self._finals: deque[SpeechSegment] = deque()
        self._max = max_finals
        self._partials: dict[int, SpeechSegment] = {}
        self._cv = threading.Condition()
        self._closed = False
        self._in_flight = 0  # taken by the worker, result not yet posted

    def put_final(self, seg: SpeechSegment) -> SpeechSegment | None:
        """Queue a final. Returns the dropped oldest final if the queue was full."""
        with self._cv:
            dropped = self._finals.popleft() if len(self._finals) >= self._max else None
            self._finals.append(seg)
            self._partials.pop(seg.utterance_id, None)
            self._cv.notify()
            return dropped

    def put_partial(self, seg: SpeechSegment) -> None:
        with self._cv:
            self._partials[seg.utterance_id] = seg  # replaces an older, unprocessed partial
            self._cv.notify()

    def get(self, timeout: float) -> SpeechSegment | None:
        with self._cv:
            if not self._finals and not self._partials and not self._closed:
                self._cv.wait(timeout)
            if self._finals:
                self._in_flight += 1
                return self._finals.popleft()
            if self._partials:
                uid = min(self._partials)
                self._in_flight += 1
                return self._partials.pop(uid)
            return None

    def done(self) -> None:
        """The worker has posted the result for the segment it took."""
        with self._cv:
            self._in_flight -= 1

    def depth(self) -> int:
        """Queued plus in-hand segments: 0 means every result has been posted."""
        with self._cv:
            return len(self._finals) + len(self._partials) + self._in_flight

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


# ============================================================== coordinator messages
@dataclass(frozen=True)
class _Started:
    uid: int
    t_start: float
    speaker: Speaker = Speaker.OTHER


@dataclass(frozen=True)
class _Discarded:
    uid: int


@dataclass(frozen=True)
class _Overloaded:
    uid: int


@dataclass(frozen=True)
class _Overrun:
    samples: int


@dataclass(frozen=True)
class _TranscriptDone:
    uid: int
    is_final: bool
    version: int
    t_start: float
    t_end: float
    transcript: Transcript
    asr_ms: float
    speaker: Speaker = Speaker.OTHER


@dataclass(frozen=True)
class _TranscriptFailed:
    uid: int
    is_final: bool
    t_end: float
    error: PolyglotError
    speaker: Speaker = Speaker.OTHER


@dataclass(frozen=True)
class _TranslationDone:
    uid: int
    text: str | None
    error: PolyglotError | None
    ms: float


@dataclass(frozen=True)
class _SuggestionDone:
    generation: int
    uid: int
    options: tuple[ReplyOption, ...]
    error: PolyglotError | None


@dataclass(frozen=True)
class _AsrState:
    ok: bool
    detail: str = ""
    error: PolyglotError | None = None


@dataclass(frozen=True)
class _SourceEnded:
    speaker: Speaker = Speaker.OTHER


@dataclass(frozen=True)
class _SourceError:
    error: Exception
    speaker: Speaker = Speaker.OTHER


@dataclass(frozen=True)
class _Command:
    name: str
    arg: Any = None


@dataclass(frozen=True)
class _Stop:
    pass


# ============================================================== front end
class FrontEnd:
    """Ring buffer -> resample -> 32 ms frames -> VAD -> segmenter -> ASR queue."""

    def __init__(
        self,
        ring: RingBuffer,
        resampler: Resampler,
        vad: VoiceActivityModel,
        segmenter: Segmenter,
        asr_queue: AsrQueue,
        post: Callable[[object], None],
        cfg: PipelineConfig,
        segment_tap: Callable[[SpeechSegment], None] | None = None,
        wall_clock: Callable[[], float] = time.monotonic,
        speaker: Speaker = Speaker.OTHER,
    ) -> None:
        self._speaker = speaker  # stamped on everything this stream produces
        self._ring, self._resampler, self._vad = ring, resampler, vad
        self._seg, self._q, self._post, self._cfg = segmenter, asr_queue, post, cfg
        self._tap = segment_tap
        self._clock = wall_clock
        self._pending: RawAudio = np.zeros(0, dtype=np.float32)
        self._t = 0.0  # stream time of the next frame, seconds
        self._stop = threading.Event()
        self._eof = threading.Event()
        self._thread: threading.Thread | None = None
        self._overrun_seen = 0
        self._last_data = wall_clock()
        self.finished = threading.Event()

    def start(self) -> None:
        name = "polyglot-frontend" + ("" if self._speaker is Speaker.OTHER else "-user")
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def signal_eof(self) -> None:
        self._eof.set()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    # -------------------------------------------------------------------- loop
    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._ring.wait(0.02)
                data = self._ring.read()
                self._check_overrun()
                if data.size:
                    self._last_data = self._clock()
                    self._feed(self._resampler.process(data))
                elif self._eof.is_set():
                    break
                elif self._cfg.gap_fill and self._seg.in_speech:
                    self._fill_gap()
            # End of stream or stop: finish the utterance in progress.
            self._feed(self._resampler.flush())
            if self._pending.size:
                pad = np.zeros(FRAME_SAMPLES - self._pending.size, dtype=np.float32)
                self._feed(pad)
            self._route(self._seg.flush())
            if self._eof.is_set():
                self._post(_SourceEnded(self._speaker))
        except Exception as exc:
            log.exception("front end crashed")
            self._post(_SourceError(exc, self._speaker))
        finally:
            self.finished.set()

    def _feed(self, samples: RawAudio) -> None:
        if samples.size == 0:
            return
        buf = np.concatenate((self._pending, samples)) if self._pending.size else samples
        n_frames = buf.size // FRAME_SAMPLES
        for i in range(n_frames):
            frame = np.ascontiguousarray(buf[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
            prob = float(self._vad.speech_prob(frame))
            self._route(self._seg.push(frame, prob, self._t))
            self._t += FRAME_SECONDS
        self._pending = buf[n_frames * FRAME_SAMPLES :].copy()

    def _fill_gap(self) -> None:
        """Loopback devices deliver nothing while the output is silent. Without this, an
        utterance whose speaker stops talking would never reach its end-of-speech
        silence. Feed real-time-equivalent silence while in speech."""
        idle = self._clock() - self._last_data
        if idle < self._cfg.gap_fill_after_s:
            return
        n = int(idle / FRAME_SECONDS)
        silent = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        for _ in range(n):
            if not self._seg.in_speech:
                break
            self._route(self._seg.push(silent, 0.0, self._t))
            self._t += FRAME_SECONDS
        self._last_data = self._clock()

    def _check_overrun(self) -> None:
        current = self._ring.overrun_samples
        if current > self._overrun_seen:
            self._post(_Overrun(current - self._overrun_seen))
            self._overrun_seen = current

    def _route(self, outputs: list[Any]) -> None:
        for out in outputs:
            if isinstance(out, SpeechStarted):
                self._post(_Started(out.utterance_id, out.t_start, self._speaker))
            elif isinstance(out, Discarded):
                self._post(_Discarded(out.utterance_id))
            elif isinstance(out, SegmentReady):
                seg = replace(out.segment, speaker=self._speaker)
                if seg.is_final:
                    if self._tap is not None:
                        try:
                            self._tap(seg)
                        except Exception:
                            log.exception("segment tap failed")
                    dropped = self._q.put_final(seg)
                    if dropped is not None:
                        self._post(_Overloaded(dropped.utterance_id))
                else:
                    self._q.put_partial(seg)


# ============================================================== ASR worker
class AsrWorker:
    """The one thread that runs the Transcriber."""

    def __init__(
        self,
        transcriber: Transcriber,
        asr_queue: AsrQueue,
        post: Callable[[object], None],
        language: Callable[[], str | None],
    ) -> None:
        self._tr, self._q, self._post, self._language = transcriber, asr_queue, post, language
        self._thread: threading.Thread | None = None
        self.busy = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="polyglot-asr", daemon=True)
        self._thread.start()

    def join(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            self._tr.load()
        except PolyglotError as exc:
            self._post(_AsrState(ok=False, detail=str(exc), error=exc))
            return
        except Exception as exc:
            log.exception("model load failed")
            err = PolyglotError(f"model load failed: {exc}")
            self._post(_AsrState(ok=False, detail=str(err), error=err))
            return
        self._post(_AsrState(ok=True))
        while True:
            seg = self._q.get(timeout=0.1)
            if seg is None:
                if self._q.closed:
                    return
                continue
            self.busy = True
            try:
                self._transcribe(seg)
            finally:
                self.busy = False
                self._q.done()

    def _transcribe(self, seg: SpeechSegment) -> None:
        t0 = time.monotonic()
        try:
            tr = self._tr.transcribe(seg.audio, self._language())
        except PolyglotError as exc:
            self._post(
                _TranscriptFailed(seg.utterance_id, seg.is_final, seg.t_end, exc, seg.speaker)
            )
            return
        except Exception as exc:
            log.exception("transcriber raised an unexpected error")
            err = PolyglotError(f"unexpected transcriber error: {exc}")
            err.kind = ErrorKind.TRANSCRIPTION
            self._post(
                _TranscriptFailed(seg.utterance_id, seg.is_final, seg.t_end, err, seg.speaker)
            )
            return
        ms = (time.monotonic() - t0) * 1000
        self._post(
            _TranscriptDone(
                seg.utterance_id,
                seg.is_final,
                seg.version,
                seg.t_start,
                seg.t_end,
                tr,
                ms,
                seg.speaker,
            )
        )


# ============================================================== coordinator
class Coordinator:
    """Owns the Session; applies results; dispatches network work; publishes events."""

    def __init__(
        self,
        session: Session,
        bus: EventBus,
        cfg: PipelineConfig,
        translator: Translator | None,
        suggester: Suggester | None,
        asr_queue_depth: Callable[[], int],
    ) -> None:
        self.session, self._bus, self._cfg = session, bus, cfg
        self._translator, self._suggester = translator, suggester
        self._asr_depth = asr_queue_depth
        self.inbox: queue.Queue[object] = queue.Queue(maxsize=10_000)
        self._pool = ThreadPoolExecutor(cfg.network_workers, thread_name_prefix="polyglot-net")
        self._thread: threading.Thread | None = None
        self._suggest_gen = 0
        self._context = ContextBuilder(
            max_utterances=cfg.suggestion_history,
            max_chars=cfg.suggestion_history_chars,
            conversation_type=cfg.conversation_type,
            max_options=cfg.suggestion_options,
            profile=cfg.profile,
        )
        self._ignore_upto = 0  # utterance IDs <= this were cleared by the user
        self._pending_network = 0
        self.state = PipelineState.STOPPED
        self.source_ended = threading.Event()  # the other person's audio source
        self.user_source_ended = threading.Event()  # the user's microphone
        self.on_fatal: Callable[[], None] | None = None  # set by Pipeline
        self._count_lock = threading.Lock()
        self._posted = 0  # messages accepted into the inbox
        self._handled = 0  # messages fully handled (written by the coordinator thread)
        self._timings: dict[int, float] = {}

    # ------------------------------------------------------------------ plumbing
    def post(self, msg: object) -> None:
        try:
            with self._count_lock:
                self.inbox.put_nowait(msg)
                self._posted += 1
        except queue.Full:  # pragma: no cover - would need 10k messages backed up
            log.error("coordinator inbox full; dropping %s", type(msg).__name__)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="polyglot-coord", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self.post(_Stop())
        if self._thread is not None:
            self._thread.join(timeout)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def is_idle(self) -> bool:
        """Every posted message handled and no network job outstanding."""
        with self._count_lock:
            caught_up = self._handled >= self._posted
        return caught_up and self._pending_network == 0

    def drain(self, timeout: float) -> bool:
        """Wait until every message posted before this call has been handled (network
        jobs may still be in flight). Returns False on timeout."""
        with self._count_lock:
            target = self._posted
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._handled >= target:
                return True
            time.sleep(0.005)
        return False

    def _run(self) -> None:
        while True:
            msg = self.inbox.get()
            try:
                if isinstance(msg, _Stop):
                    return
                self._handle(msg)
            except Exception as exc:
                log.exception("coordinator failed handling %s", type(msg).__name__)
                self._bus.publish(StageError("coordinator", ErrorKind.INTERNAL, str(exc)))
            finally:
                with self._count_lock:
                    self._handled += 1

    def _set_state(self, state: PipelineState, detail: str = "") -> None:
        if self.state is PipelineState.ERROR and state in (
            PipelineState.LISTENING,
            PipelineState.FALLING_BEHIND,
        ):
            return  # an error stays visible until the user stops or restarts
        if state != self.state or detail:
            self.state = state
            self._bus.publish(PipelineStatus(state, detail))

    def _publish(self, u: Utterance) -> None:
        self.session.put(u)
        self._bus.publish(UtteranceUpdated(u))

    # ------------------------------------------------------------------ dispatch
    def _handle(self, msg: object) -> None:
        if isinstance(msg, _Started):
            if msg.uid > self._ignore_upto:
                self._publish(Utterance(id=msg.uid, t_start=msg.t_start, speaker=msg.speaker))
        elif isinstance(msg, _Discarded):
            self.session.remove(msg.uid)
            self._bus.publish(UtteranceDiscarded(msg.uid))
        elif isinstance(msg, _TranscriptDone):
            self._on_transcript(msg)
        elif isinstance(msg, _TranscriptFailed):
            self._on_transcript_failed(msg)
        elif isinstance(msg, _TranslationDone):
            self._pending_network -= 1
            self._on_translation(msg)
        elif isinstance(msg, _SuggestionDone):
            self._pending_network -= 1
            self._on_suggestion(msg)
        elif isinstance(msg, _Overloaded):
            self._on_overloaded(msg.uid)
        elif isinstance(msg, _Overrun):
            self._set_state(
                PipelineState.FALLING_BEHIND,
                f"audio buffer overrun: {msg.samples} samples lost",
            )
        elif isinstance(msg, _AsrState):
            self._on_asr_state(msg)
        elif isinstance(msg, _SourceEnded):
            (self.user_source_ended if msg.speaker is Speaker.USER else self.source_ended).set()
        elif isinstance(msg, _SourceError):
            self._on_source_error(msg)
        elif isinstance(msg, _Command):
            self._on_command(msg)
        else:  # pragma: no cover
            log.warning("unknown coordinator message %r", msg)

    def _on_source_error(self, msg: _SourceError) -> None:
        if msg.speaker is Speaker.USER:
            # The other person's audio is the point of the app: losing the microphone is
            # reported (as a warning), and listening carries on without it.
            self._bus.publish(
                StageError("audio", ErrorKind.AUDIO, f"microphone: {msg.error}", retryable=True)
            )
            self.user_source_ended.set()
            return
        self._set_state(PipelineState.ERROR, f"audio source failed: {msg.error}")
        self._bus.publish(StageError("audio", ErrorKind.AUDIO, str(msg.error)))
        self.source_ended.set()

    # ------------------------------------------------------------------ ASR results
    def _on_asr_state(self, msg: _AsrState) -> None:
        if msg.ok:
            # A model that finishes loading after the user pressed Stop must not flip
            # the status back to "listening".
            if self.state is PipelineState.LOADING:
                self._set_state(PipelineState.LISTENING)
            return
        self._set_state(PipelineState.ERROR, msg.detail)
        kind = msg.error.kind if msg.error else ErrorKind.TRANSCRIPTION
        self._bus.publish(StageError("asr", kind, msg.detail))
        if self.on_fatal is not None:
            self.on_fatal()  # nothing can be transcribed: stop capturing

    def _on_transcript(self, msg: _TranscriptDone) -> None:
        if msg.uid <= self._ignore_upto:
            return
        current = self.session.get(msg.uid)
        tr = msg.transcript
        if not msg.is_final:
            if current is None or current.is_final or msg.version <= current.partial_version:
                return  # stale partial: a newer partial or the final already covers it
            self._publish(
                current.evolve(
                    status=UtteranceStatus.PARTIAL,
                    text=tr.text.strip(),
                    partial_version=msg.version,
                    source_lang=tr.language or current.source_lang,
                )
            )
            return

        text = tr.text.strip()
        if not text:  # VAD heard something, Whisper found no words: not speech
            self.session.remove(msg.uid)
            self._bus.publish(UtteranceDiscarded(msg.uid))
            self._maybe_recover()
            return
        lang = self.session.locked_language or tr.language
        base = current or Utterance(id=msg.uid, t_start=msg.t_start, speaker=msg.speaker)
        u = base.evolve(
            status=UtteranceStatus.FINAL, text=text, source_lang=lang, t_end=msg.t_end, error=None
        )
        self._publish(u)
        log.info("utt=%d final asr_ms=%.0f audio_s=%.2f", u.id, msg.asr_ms, msg.t_end - msg.t_start)
        # Only the other person's speech decides their language; the user's own turns
        # (in whatever language they happen to speak) must not move the lock.
        if u.speaker is Speaker.OTHER and self.session.observe_language(
            tr.language, tr.language_probability
        ):
            self._bus.publish(LanguageChanged(self.session.locked_language, locked=True))
        self._maybe_recover()
        self._dispatch_translation(u)

    def _on_transcript_failed(self, msg: _TranscriptFailed) -> None:
        if not msg.is_final or msg.uid <= self._ignore_upto:
            log.warning("partial transcription failed for utt=%d: %s", msg.uid, msg.error)
            return
        current = self.session.get(msg.uid) or Utterance(
            id=msg.uid, t_start=msg.t_end, speaker=msg.speaker
        )
        self._publish(
            current.evolve(status=UtteranceStatus.FAILED, error=msg.error.kind, t_end=msg.t_end)
        )
        self._bus.publish(StageError("asr", msg.error.kind, str(msg.error)))

    def _on_overloaded(self, uid: int) -> None:
        current = self.session.get(uid)
        if current is not None:
            self._publish(
                current.evolve(
                    status=UtteranceStatus.FAILED,
                    error=ErrorKind.OVERLOADED,
                    t_end=current.t_start,
                )
            )
        self._set_state(
            PipelineState.FALLING_BEHIND,
            "transcription is falling behind; consider a smaller Whisper model",
        )

    def _maybe_recover(self) -> None:
        if self.state is PipelineState.FALLING_BEHIND and self._asr_depth() == 0:
            self._set_state(PipelineState.LISTENING)

    # ------------------------------------------------------------------ translation
    def _dispatch_translation(self, u: Utterance) -> None:
        target = self.session.target_language
        if u.speaker is Speaker.USER:
            return  # the user knows what they said; no point paying to translate it
        if self._translator is None or languages.same_language(u.source_lang, target):
            return
        translator, timeout = self._translator, self._cfg.translate_timeout_s
        text, source = u.text, u.source_lang
        self._pending_network += 1
        self._timings[u.id] = time.monotonic()

        def job() -> _TranslationDone:
            t0 = time.monotonic()
            try:
                out = self._with_retry(
                    lambda: translator.translate(text, source, target, timeout=timeout)
                )
                return _TranslationDone(u.id, out, None, (time.monotonic() - t0) * 1000)
            except PolyglotError as exc:
                return _TranslationDone(u.id, None, exc, (time.monotonic() - t0) * 1000)

        self._submit(job)

    def _on_translation(self, msg: _TranslationDone) -> None:
        current = self.session.get(msg.uid)
        if current is None or msg.uid <= self._ignore_upto:
            return
        if msg.error is not None:
            self._publish(current.evolve(error=msg.error.kind))
            self._bus.publish(
                StageError("translate", msg.error.kind, str(msg.error), msg.error.retryable)
            )
            return
        self._publish(
            current.evolve(status=UtteranceStatus.TRANSLATED, translation=msg.text, error=None)
        )
        log.info("utt=%d translated translate_ms=%.0f", msg.uid, msg.ms)

    # ------------------------------------------------------------------ suggestions
    def _request_suggestion(self) -> None:
        if self._suggester is None:
            self._bus.publish(
                StageError("suggest", ErrorKind.NOT_CONFIGURED, "suggestions are not enabled")
            )
            return
        target = self.session.latest_answerable()
        if target is None:
            self._bus.publish(
                StageError("suggest", ErrorKind.NOTHING_TO_ANSWER, "nothing has been said yet")
            )
            return
        self._suggest_gen += 1
        gen = self._suggest_gen
        ctx = self._context.build(self.session, target)
        suggester, timeout = self._suggester, self._cfg.suggest_timeout_s
        self._pending_network += 1
        self._bus.publish(SuggestionStarted(target.id))

        def job() -> _SuggestionDone:
            try:
                opts = self._with_retry(lambda: suggester.suggest(ctx, timeout=timeout))
                return _SuggestionDone(gen, target.id, tuple(opts), None)
            except PolyglotError as exc:
                return _SuggestionDone(gen, target.id, (), exc)

        self._submit(job)

    def _on_suggestion(self, msg: _SuggestionDone) -> None:
        if msg.generation != self._suggest_gen:
            return  # the user pressed the hotkey again; that newer request wins
        if msg.error is not None:
            sug = Suggestion(msg.uid, (), msg.error.kind, str(msg.error))
            self._bus.publish(
                StageError("suggest", msg.error.kind, str(msg.error), msg.error.retryable)
            )
        else:
            sug = Suggestion(msg.uid, msg.options)
        self._bus.publish(SuggestionsReady(sug))

    # ------------------------------------------------------------------ commands
    def _on_command(self, cmd: _Command) -> None:
        if cmd.name == "suggest":
            self._request_suggestion()
        elif cmd.name == "set_language":
            self.session.set_language(cmd.arg)
            self._bus.publish(LanguageChanged(cmd.arg, locked=cmd.arg is not None))
        elif cmd.name == "set_target":
            self.session.target_language = cmd.arg
        elif cmd.name == "set_profile":
            self._context = self._context.with_profile(cmd.arg)
        elif cmd.name == "abandon_open":
            self._abandon_open()
        elif cmd.name == "retranslate":
            for u in self.session.ordered():
                if u.status is UtteranceStatus.FINAL and u.translation is None and u.error is None:
                    self._dispatch_translation(u)
        elif cmd.name == "status":
            state, detail = cmd.arg
            self._set_state(state, detail)
        elif cmd.name == "clear":
            known = list(self.session.utterances)
            self._ignore_upto = max([self._ignore_upto, *known])
            self.session.clear()
            self._suggest_gen += 1
            self._bus.publish(ConversationCleared(self._ignore_upto))
        else:
            log.warning("unknown command %s", cmd.name)

    def _abandon_open(self) -> None:
        """After stopping: utterances that will never get a final transcript (the model
        failed, or the worker could not finish in time) are marked failed, not left
        "listening" forever."""
        for u in self.session.ordered():
            if u.status in (UtteranceStatus.LISTENING, UtteranceStatus.PARTIAL):
                self._publish(
                    u.evolve(
                        status=UtteranceStatus.FAILED,
                        error=ErrorKind.TRANSCRIPTION,
                        t_end=u.t_start,
                    )
                )

    # ------------------------------------------------------------------ network helpers
    def _submit(self, job: Callable[[], object]) -> None:
        def done(fut: Future[object]) -> None:
            exc = fut.exception()
            if exc is not None:  # pragma: no cover - jobs catch PolyglotError themselves
                log.error("network job crashed: %s", exc)
                return
            self.post(fut.result())

        try:
            self._pool.submit(job).add_done_callback(done)
        except RuntimeError:  # pool shut down during stop
            self._pending_network -= 1

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        attempts = self._cfg.network_retries + 1
        for attempt in range(attempts):
            try:
                return fn()
            except ProviderError as exc:
                if not exc.retryable or attempt == attempts - 1:
                    raise
                time.sleep(random.uniform(0.2, 0.5))
            except PolyglotError:
                raise
            except Exception as exc:
                log.exception("provider raised an unexpected error")
                raise ProviderError(
                    f"unexpected provider error: {exc}", kind=ErrorKind.INTERNAL, retryable=False
                ) from exc
        raise AssertionError("unreachable")  # pragma: no cover


# ============================================================== facade
class _Sink:
    """AudioSink for the pipeline. Runs on the audio library's thread: copy only."""

    def __init__(
        self,
        ring: RingBuffer,
        frontend: Callable[[], FrontEnd | None],
        post: Callable[[object], None],
        speaker: Speaker = Speaker.OTHER,
    ) -> None:
        self._ring, self._frontend, self._post = ring, frontend, post
        self._speaker = speaker
        self._errored = False

    def on_audio(self, block: RawAudio) -> None:
        if block.ndim == 2:
            block = block.mean(axis=1, dtype=np.float32)
        self._ring.write(block)

    def on_end(self) -> None:
        fe = self._frontend()
        if fe is not None:
            fe.signal_eof()

    def on_error(self, error: Exception) -> None:
        if not self._errored:
            self._errored = True
            self._post(_SourceError(error, self._speaker))
            fe = self._frontend()
            if fe is not None:
                fe.signal_eof()


class Pipeline:
    """Builds and runs the stages. All public methods are safe to call from any thread.

    ``start``/``stop`` control listening (audio source, front end, ASR worker). The
    coordinator and the session outlive them, so the conversation, suggestions and
    language commands keep working while stopped. ``close`` shuts everything down.
    """

    def __init__(
        self,
        *,
        source: AudioSource,
        resampler_factory: Callable[[int], Resampler],
        vad: VoiceActivityModel,
        transcriber: Transcriber,
        translator: Translator | None,
        suggester: Suggester | None,
        bus: EventBus,
        config: PipelineConfig | None = None,
        segment_tap: Callable[[SpeechSegment], None] | None = None,
        session: Session | None = None,
        ids: IdSource | None = None,
        user_source: AudioSource | None = None,
        user_vad: VoiceActivityModel | None = None,
    ) -> None:
        if (user_source is None) != (user_vad is None):
            raise ValueError("user_source and user_vad go together: a VAD model keeps state")
        self.cfg = config or PipelineConfig()
        self._source = source
        self._user_source, self._user_vad = user_source, user_vad
        self._resampler_factory = resampler_factory
        self._vad = vad
        self._transcriber = transcriber
        self._bus = bus
        self._tap = segment_tap
        # A rebuilt pipeline (after a settings change) can continue an existing
        # conversation: pass the previous pipeline's ``session`` and ``ids``.
        self.session = session or Session(self.cfg.target_language, self.cfg.language_policy)
        self.session.target_language = self.cfg.target_language
        self.session.policy = self.cfg.language_policy  # lock rules follow the new config
        self.ids = ids or IdSource()
        self._lock = threading.Lock()
        self._running = False
        self._frontend: FrontEnd | None = None
        self._user_frontend: FrontEnd | None = None
        self._asr: AsrWorker | None = None
        self._asr_queue: AsrQueue | None = None
        self._coord = Coordinator(
            self.session, bus, self.cfg, translator, suggester, self._asr_depth
        )
        self._coord.on_fatal = self._stop_in_background
        self._coord.start()

    def _stop_in_background(self) -> None:
        # Called on the coordinator thread; stop() drains the coordinator, so it must
        # not run there.
        threading.Thread(target=self.stop, name="polyglot-fatal-stop", daemon=True).start()

    def _asr_depth(self) -> int:
        q = self._asr_queue
        return q.depth() if q is not None else 0

    # ------------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self._running

    @property
    def state(self) -> PipelineState:
        return self._coord.state

    @property
    def source_ended(self) -> bool:
        """True once a finite source (file) has been fully consumed and flushed."""
        return self._coord.source_ended.is_set()

    @property
    def user_source_ended(self) -> bool:
        """The same for the user's microphone (also set if the microphone failed). A
        live microphone never ends, so only a finite source is worth waiting for."""
        return self._coord.user_source_ended.is_set()

    @property
    def has_user_source(self) -> bool:
        return self._user_source is not None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            cfg, coord = self.cfg, self._coord
            previous = self._asr
            if previous is not None and previous.is_alive():
                previous.join(30.0)  # never two ASR workers: wait for the old one
                if previous.is_alive():
                    self._bus.publish(
                        StageError(
                            "asr",
                            ErrorKind.INTERNAL,
                            "previous transcription still running; try again",
                        )
                    )
                    return
            coord.source_ended.clear()
            coord.user_source_ended.clear()
            asr_q = AsrQueue(cfg.finals_queue_size)
            rate = self._source.sample_rate
            ring = RingBuffer(max(1, int(cfg.ring_seconds * rate)))
            self._vad.reset()
            frontend = FrontEnd(
                ring,
                self._resampler_factory(rate),
                self._vad,
                Segmenter(cfg.segmenter, self.ids),
                asr_q,
                coord.post,
                cfg,
                self._tap,
            )
            session = self.session
            asr = AsrWorker(self._transcriber, asr_q, coord.post, lambda: session.locked_language)
            self._asr_queue, self._frontend, self._asr = asr_q, frontend, asr

            user_ring: RingBuffer | None = None
            if self._user_source is not None and self._user_vad is not None:
                # The user's microphone: its own ring, resampler, VAD state and segmenter,
                # but the same ID sequence, ASR queue/worker and coordinator.
                user_rate = self._user_source.sample_rate
                user_ring = RingBuffer(max(1, int(cfg.ring_seconds * user_rate)))
                self._user_vad.reset()
                self._user_frontend = FrontEnd(
                    user_ring,
                    self._resampler_factory(user_rate),
                    self._user_vad,
                    Segmenter(cfg.segmenter, self.ids),
                    asr_q,
                    coord.post,
                    cfg,
                    self._tap,
                    speaker=Speaker.USER,
                )

            coord.post(_Command("status", (PipelineState.LOADING, "loading speech model")))
            asr.start()
            frontend.start()
            try:
                self._source.start(_Sink(ring, lambda: self._frontend, coord.post))
            except Exception as exc:
                log.exception("audio source failed to start")
                self._user_frontend = None
                self._teardown(flush=False)
                coord.post(_SourceError(exc))
                return
            self._running = True
            if (
                self._user_source is not None
                and user_ring is not None
                and self._user_frontend is not None
            ):
                self._user_frontend.start()
                try:
                    self._user_source.start(
                        _Sink(user_ring, lambda: self._user_frontend, coord.post, Speaker.USER)
                    )
                except Exception as exc:
                    # Listening to the other person still works without the microphone.
                    log.exception("microphone failed to start")
                    self._stop_user_frontend(flush=False)
                    coord.post(_SourceError(exc, Speaker.USER))

    def stop(self, timeout: float = 10.0) -> None:
        """Stop listening. The utterance in progress is finished and transcribed."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            try:
                self._source.stop()
            except Exception:
                log.exception("audio source failed to stop cleanly")
            if self._user_source is not None:
                try:
                    self._user_source.stop()
                except Exception:
                    log.exception("microphone failed to stop cleanly")
            self._teardown(flush=True, timeout=timeout)
            self._coord.post(_Command("abandon_open"))
            self._coord.post(_Command("status", (PipelineState.STOPPED, "")))
            # When stop() returns, the last utterance's transcript has been applied.
            self._coord.drain(timeout=2.0)

    def close(self) -> None:
        """Stop listening and shut down the coordinator. The pipeline cannot be reused."""
        self.stop()
        self._coord.stop()

    def _stop_user_frontend(self, *, flush: bool) -> None:
        fe = self._user_frontend
        if fe is not None:
            if flush:
                fe.signal_eof()
            fe.stop(timeout=2.0)

    def _teardown(self, *, flush: bool, timeout: float = 5.0) -> None:
        if self._frontend is not None:
            if flush:
                self._frontend.signal_eof()
            self._frontend.stop(timeout=2.0)
        self._stop_user_frontend(flush=flush)  # before the queue closes: it may enqueue
        if self._asr_queue is not None:
            self._asr_queue.close()
        if self._asr is not None:
            self._asr.join(timeout)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """For file sources and tests: wait until the source ended and all work finished.
        With a user source that must have ended too, so only wait on a finite one."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            fe, asr, q, coord = self._frontend, self._asr, self._asr_queue, self._coord
            ufe = self._user_frontend
            if (
                fe is not None
                and (ufe is None or ufe.finished.is_set())
                and asr is not None
                and q is not None
                and fe.finished.is_set()
                and q.depth() == 0
                and not asr.busy
                and coord.is_idle()
            ):
                time.sleep(0.05)  # let any just-posted message land, then re-check
                if q.depth() == 0 and not asr.busy and coord.is_idle():
                    return True
            time.sleep(0.01)
        return False

    # ------------------------------------------------------------------ commands
    def request_suggestion(self) -> None:
        """Ask for replies to the latest final utterance (bound to the suggest hotkey)."""
        self._coord.post(_Command("suggest"))

    def set_language(self, language: str | None) -> None:
        """Force the source language, or ``None`` to unlock and auto-detect again."""
        self._coord.post(_Command("set_language", language))

    def set_target_language(self, language: str) -> None:
        self._coord.post(_Command("set_target", language))

    def set_profile(self, profile: UserProfile) -> None:
        """Use a new user profile for the next suggestion. Requests already in flight keep
        the profile they were built with."""
        self._coord.post(_Command("set_profile", profile))

    def retranslate_pending(self) -> None:
        """Translate final utterances still waiting for one (e.g. after a rebuild
        dropped in-flight requests)."""
        self._coord.post(_Command("retranslate"))

    def clear(self) -> None:
        """Forget the conversation. Late results for cleared utterances are ignored."""
        self._coord.post(_Command("clear"))
