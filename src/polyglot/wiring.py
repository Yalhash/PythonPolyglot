"""Composition: config -> concrete adapters -> Pipeline. No Qt here.

This and ``polyglot.app`` are the only modules that know every concrete class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from polyglot.config import AppConfig, Secrets, data_dir
from polyglot.core.errors import ConfigError, PolyglotError
from polyglot.core.events import EventBus
from polyglot.core.models import ConversationType, IdSource
from polyglot.core.pipeline import Pipeline, PipelineConfig
from polyglot.core.ports import AudioSource, Suggester, Transcriber, Translator, VoiceActivityModel
from polyglot.core.segmenter import SegmenterConfig
from polyglot.core.session import LanguagePolicy, Session

log = logging.getLogger(__name__)

DEMO_LINES = (
    "Buenos días, gracias por venir hoy.",
    "¿Puede contarme un poco sobre su experiencia?",
    "Interesante. ¿Por qué quiere trabajar con nosotros?",
    "¿Tiene alguna pregunta para nosotros?",
)


@dataclass
class Built:
    pipeline: Pipeline
    warnings: list[str] = field(default_factory=list)
    translation_enabled: bool = False
    suggestions_enabled: bool = False
    source_name: str = ""


def pipeline_config(cfg: AppConfig) -> PipelineConfig:
    v = cfg.vad
    return PipelineConfig(
        segmenter=SegmenterConfig(
            start_threshold=v.start_threshold,
            end_threshold=v.end_threshold,
            start_frames=v.start_frames,
            end_silence_ms=v.end_silence_ms,
            pre_roll_ms=v.pre_roll_ms,
            max_utterance_s=v.max_utterance_s,
            min_utterance_ms=v.min_utterance_ms,
            partial_interval_ms=v.partial_interval_ms,
        ),
        target_language=cfg.translation.target_language,
        language_policy=LanguagePolicy(
            agree=cfg.asr.lock_after,
            min_probability=cfg.asr.lock_min_probability,
            forced=cfg.asr.language,
        ),
        translate_timeout_s=cfg.translation.timeout_s,
        suggest_timeout_s=cfg.suggestions.timeout_s,
        suggestion_history=cfg.suggestions.history,
        suggestion_options=cfg.suggestions.options,
        conversation_type=ConversationType(cfg.suggestions.conversation_type),
        ring_seconds=cfg.audio.ring_seconds,
        finals_queue_size=cfg.asr.queue_size,
        gap_fill=cfg.audio.gap_fill,
    )


def build_source(cfg: AppConfig, warnings: list[str]) -> tuple[AudioSource, str]:
    from polyglot import platform

    a = cfg.audio
    kind = a.source
    if kind == "auto":
        kind = "loopback" if platform.current().has_loopback else "device"
        if kind == "device":
            warnings.append(
                "System audio needs a virtual device on this OS (e.g. BlackHole on macOS); "
                "using the default input device."
            )
    if kind == "file":
        if not a.file:
            raise ConfigError("audio.source is 'file' but audio.file is not set")
        from polyglot.adapters.audio.playback import WavFileSource

        return WavFileSource(a.file, realtime=True), f"file {a.file}"
    if kind == "loopback":
        if not platform.current().has_loopback:
            raise ConfigError("loopback capture is only available on Windows")
        from polyglot.adapters.audio.wasapi_loopback import WasapiLoopbackSource

        src = WasapiLoopbackSource()
        return src, f"loopback: {src.device_name}"
    from polyglot.adapters.audio.sounddevice_source import SoundDeviceSource

    device: int | str | None = a.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    sd_src = SoundDeviceSource(device)
    return sd_src, sd_src.device_name


def build_vad(cfg: AppConfig, warnings: list[str]) -> VoiceActivityModel:
    from polyglot.adapters.vad.energy import EnergyVad

    if cfg.vad.engine == "energy":
        return EnergyVad()
    try:
        from polyglot.adapters.vad.silero import SileroVad

        return SileroVad(cfg.vad.model_path)
    except (PolyglotError, ImportError, OSError) as exc:
        warnings.append(f"Silero VAD unavailable ({exc}); using the simpler energy detector.")
        return EnergyVad()


def build_transcriber(cfg: AppConfig) -> Transcriber:
    from polyglot.adapters.asr.faster_whisper import FasterWhisperTranscriber

    return FasterWhisperTranscriber(
        cfg.asr.model,
        device=cfg.asr.device,
        compute_type=cfg.asr.compute_type,
        beam_size=cfg.asr.beam_size,
    )


def build_translator(cfg: AppConfig, secrets: Secrets, warnings: list[str]) -> Translator | None:
    t = cfg.translation
    if not t.enabled:
        return None
    if t.provider == "google":
        key = secrets.get(Secrets.GOOGLE)
        if not key:
            warnings.append("Translation is on but no Google API key is set.")
            return None
        from polyglot.adapters.translate.google import GoogleTranslator

        return GoogleTranslator(key)
    key = secrets.get(Secrets.OPENAI)
    if not key:
        warnings.append("Translation is on but no OpenAI API key is set.")
        return None
    from polyglot.adapters.translate.openai import OpenAITranslator

    return OpenAITranslator(key, model=t.model)


def build_suggester(cfg: AppConfig, secrets: Secrets, warnings: list[str]) -> Suggester | None:
    if not cfg.suggestions.enabled:
        return None
    key = secrets.get(Secrets.OPENAI)
    if not key:
        warnings.append("Suggestions are on but no OpenAI API key is set.")
        return None
    from polyglot.adapters.suggest.openai import OpenAISuggester

    return OpenAISuggester(key, model=cfg.suggestions.model)


def build(
    cfg: AppConfig,
    secrets: Secrets,
    bus: EventBus,
    *,
    session: Session | None = None,
    ids: IdSource | None = None,
    source: AudioSource | None = None,
) -> Built:
    """Build the real pipeline from config. Raises ConfigError for unusable settings.

    ``source`` overrides the configured audio source (the CLI uses it for fast replay).
    """
    from polyglot.adapters.audio.resample import make_resampler

    warnings: list[str] = []
    if source is None:
        source, source_name = build_source(cfg, warnings)
    else:
        source_name = type(source).__name__
    translator = build_translator(cfg, secrets, warnings)
    suggester = build_suggester(cfg, secrets, warnings)
    tap = None
    if cfg.privacy.dump_audio:
        from polyglot.adapters.dump import SegmentDumper

        tap = SegmentDumper(data_dir() / "dumps")
    pipeline = Pipeline(
        source=source,
        resampler_factory=make_resampler,
        vad=build_vad(cfg, warnings),
        transcriber=build_transcriber(cfg),
        translator=translator,
        suggester=suggester,
        bus=bus,
        config=pipeline_config(cfg),
        segment_tap=tap,
        session=session,
        ids=ids,
    )
    return Built(pipeline, warnings, translator is not None, suggester is not None, source_name)


def build_demo(
    cfg: AppConfig,
    bus: EventBus,
    *,
    session: Session | None = None,
    ids: IdSource | None = None,
) -> Built:
    """Everything fake: synthetic audio, scripted transcripts, echo translation, canned
    suggestions. Runs with no models, keys, audio devices or network."""
    from polyglot.adapters.audio.playback import ArraySource
    from polyglot.adapters.audio.resample import IdentityResampler
    from polyglot.adapters.fakes import (
        AmplitudeVad,
        CannedSuggester,
        EchoTranslator,
        ScriptedTranscriber,
        speech_like,
    )
    from polyglot.core.models import SAMPLE_RATE

    pattern: list[tuple[str, float]] = []
    for _ in DEMO_LINES:
        pattern += [("silence", 1.0), ("tone", 1.6)]
    pattern.append(("silence", 1.0))
    audio = speech_like(pattern)
    pcfg = pipeline_config(cfg)
    built = Built(
        Pipeline(
            source=ArraySource(audio, SAMPLE_RATE, realtime=True),
            resampler_factory=lambda _rate: IdentityResampler(),
            vad=AmplitudeVad(),
            transcriber=ScriptedTranscriber(list(DEMO_LINES), language="es", delay_s=0.3),
            translator=EchoTranslator(),
            suggester=CannedSuggester(delay_s=0.5),
            bus=bus,
            config=pcfg,
            session=session,
            ids=ids,
        ),
        ["Demo mode: synthetic audio, scripted text, no network."],
        translation_enabled=True,
        suggestions_enabled=True,
        source_name="demo",
    )
    return built
