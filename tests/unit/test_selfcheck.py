from __future__ import annotations

from pathlib import Path

import pytest

from polyglot import selfcheck
from polyglot.__main__ import main


def _ok() -> str:
    return "fine"


def _boom() -> str:
    raise ImportError("missing.dll")


def test_all_required_pass_gives_exit_0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(selfcheck, "REQUIRED", [("a", _ok), ("b", _ok)])
    monkeypatch.setattr(selfcheck, "OPTIONAL", [("engine", _boom)])
    report = tmp_path / "r.txt"
    assert selfcheck.run(report) == 0
    text = report.read_text(encoding="utf-8")
    assert "ok    a: fine" in text
    assert "--    engine: not installed (ImportError)" in text  # optional never fails
    assert "all required checks passed" in text


def test_a_failed_required_check_gives_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(selfcheck, "REQUIRED", [("a", _ok), ("dll", _boom)])
    monkeypatch.setattr(selfcheck, "OPTIONAL", [])
    assert selfcheck.run() == 1
    out = capsys.readouterr().out
    assert "FAIL  dll: ImportError('missing.dll')" in out
    assert "1 required check(s) failed" in out


def test_cli_flag_runs_the_self_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(selfcheck, "REQUIRED", [("a", _ok)])
    monkeypatch.setattr(selfcheck, "OPTIONAL", [])
    report = tmp_path / "r.txt"
    assert main(["--self-check", str(report)]) == 0
    assert report.exists()
