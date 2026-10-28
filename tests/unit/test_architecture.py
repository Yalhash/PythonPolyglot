"""The dependency rule, enforced: imports point inward toward the core.

Parses every module's imports with ``ast`` (no import-time side effects), so it runs
without Qt, models or devices installed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "polyglot"
STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg = module_name(path).rsplit(".", 1)[0] if path.name != "__init__.py" else module_name(path)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)]
                found.add(".".join([*base, node.module] if node.module else base))
            elif node.module:
                found.add(node.module)
    return found


def modules(subpackage: str) -> list[Path]:
    root = SRC / subpackage if subpackage else SRC
    return sorted(root.rglob("*.py")) if root.is_dir() else [root.with_suffix(".py")]


def top(name: str) -> str:
    return name.split(".")[0]


@pytest.mark.parametrize("path", modules("core"), ids=lambda p: p.name)
def test_core_imports_only_stdlib_numpy_and_core(path: Path) -> None:
    bad = {
        m
        for m in imports_of(path)
        if not (top(m) in STDLIB or top(m) == "numpy" or m.startswith("polyglot.core"))
    }
    assert not bad, f"{path.name} imports outside the core: {sorted(bad)}"


@pytest.mark.parametrize("path", modules("adapters") + modules("platform"), ids=lambda p: p.name)
def test_adapters_and_platform_do_not_depend_on_ui(path: Path) -> None:
    bad = {
        m
        for m in imports_of(path)
        if top(m) == "PyQt6" or m.startswith(("polyglot.ui", "polyglot.app", "polyglot.controller"))
    }
    assert not bad, f"{path.name} depends on the UI layer: {sorted(bad)}"


def test_presenter_is_qt_free_and_adapter_free() -> None:
    imports = imports_of(SRC / "ui" / "presenter.py")
    assert not {m for m in imports if top(m) == "PyQt6" or m.startswith("polyglot.adapters")}


@pytest.mark.parametrize("path", modules("ui"), ids=lambda p: p.name)
def test_ui_does_not_reach_into_adapters(path: Path) -> None:
    assert not {
        m for m in imports_of(path) if m.startswith(("polyglot.adapters", "polyglot.wiring"))
    }


def test_controller_and_wiring_are_qt_free() -> None:
    for name in ("controller.py", "wiring.py", "config.py"):
        assert not {m for m in imports_of(SRC / name) if top(m) == "PyQt6"}, name


def test_no_platform_checks_outside_platform_layer() -> None:
    allowed = {"platform"}
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.relative_to(SRC).parts[0] in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "platform.system()" in text or "sys.platform ==" in text:
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"OS checks belong in polyglot.platform: {offenders}"
