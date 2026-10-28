"""Command line entry point: ``polyglot`` / ``python -m polyglot``."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

from polyglot import __version__


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polyglot", description=__doc__)
    p.add_argument("--config", type=Path, help="config file (default: user config dir)")
    p.add_argument("--demo", action="store_true", help="run with synthetic audio and fakes")
    p.add_argument("--headless", action="store_true", help="no window; print events")
    p.add_argument("--file", type=Path, help="headless: transcribe this WAV file")
    p.add_argument("--fast", action="store_true", help="headless --file: do not pace in real time")
    p.add_argument("--list-devices", action="store_true", help="list audio input devices")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--version", action="version", version=f"polyglot {__version__}")
    return p


def list_devices() -> int:
    from polyglot import platform
    from polyglot.core.errors import PolyglotError

    try:
        from polyglot.adapters.audio.sounddevice_source import list_input_devices

        for d in list_input_devices():
            flag = "  (virtual)" if d["virtual"] else ""
            print(f"[{d['id']:>3}] {d['name']}  {d['rate']} Hz{flag}")
    except PolyglotError as exc:
        print(f"input devices unavailable: {exc}", file=sys.stderr)
    if platform.current().has_loopback:
        from polyglot.adapters.audio.wasapi_loopback import list_loopback_devices

        print("\nLoopback (system audio):")
        for d in list_loopback_devices():
            print(f"[{d['id']:>3}] {d['name']}  {d['rate']} Hz")
    return 0


def run_headless(config_path: Path | None, wav: Path | None, *, demo: bool, fast: bool) -> int:
    from polyglot.adapters.secrets import KeyringSecretStore
    from polyglot.config import Secrets, load_config
    from polyglot.core.errors import PolyglotError
    from polyglot.core.events import (
        Event,
        EventBus,
        PipelineStatus,
        StageError,
        SuggestionsReady,
        UtteranceUpdated,
    )
    from polyglot.core.models import Speaker, UtteranceStatus
    from polyglot.wiring import build, build_demo

    cfg = load_config(config_path)
    if wav is not None:
        cfg = cfg.model_copy(
            update={"audio": cfg.audio.model_copy(update={"source": "file", "file": str(wav)})}
        )
    bus = EventBus()

    def show(ev: Event) -> None:
        if isinstance(ev, UtteranceUpdated):
            u = ev.utterance
            if u.status in (
                UtteranceStatus.FINAL,
                UtteranceStatus.TRANSLATED,
                UtteranceStatus.FAILED,
            ):
                extra = f"  ->  {u.translation}" if u.translation else ""
                err = f"  [{u.error.value}]" if u.error else ""
                who = "you: " if u.speaker is Speaker.USER else ""
                print(f"#{u.id} ({u.source_lang or '?'}) {who}{u.text}{extra}{err}", flush=True)
        elif isinstance(ev, SuggestionsReady):
            for o in ev.suggestion.options:
                print(f"   suggest: {o.source_text}  ({o.user_text})", flush=True)
        elif isinstance(ev, (PipelineStatus, StageError)):
            print(f"-- {ev}", file=sys.stderr, flush=True)

    bus.subscribe(show)
    try:
        if demo:
            built = build_demo(cfg, bus)
        else:
            secrets = Secrets(KeyringSecretStore())
            source = None
            if wav is not None and fast:
                from polyglot.adapters.audio.playback import WavFileSource

                source = WavFileSource(wav, realtime=False)
                ring = min(600.0, source.duration_s + 5)
                cfg = cfg.model_copy(
                    update={"audio": cfg.audio.model_copy(update={"ring_seconds": ring})}
                )
            built = build(cfg, secrets, bus, source=source)
    except PolyglotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for w in built.warnings:
        print(f"warning: {w}", file=sys.stderr)

    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    built.pipeline.start()
    if wav is not None or demo:
        while not done.is_set():
            if built.pipeline.source_ended and built.pipeline.wait_idle(timeout=0.5):
                break
            done.wait(0.1)
    else:
        done.wait()
    built.pipeline.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from polyglot import config, logging_setup

    logging_setup.configure(
        args.log_level, None if args.headless else config.log_dir() / "polyglot.log"
    )
    if args.list_devices:
        return list_devices()
    if args.headless or args.file:
        return run_headless(args.config, args.file, demo=args.demo, fast=args.fast)
    from polyglot.app import run_gui

    return run_gui(args.config, demo=args.demo)


if __name__ == "__main__":
    sys.exit(main())
