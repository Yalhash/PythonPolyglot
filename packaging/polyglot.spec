# PyInstaller spec for the Polyglot desktop app (Windows and macOS).
#
# Build from the repository root:
#
#   uv sync --locked --no-dev --extra parakeet
#   uv pip install "pyinstaller>=6.11"
#   uv run --no-sync pyinstaller --noconfirm --clean packaging/polyglot.spec
#
# Output: dist/Polyglot/Polyglot.exe (Windows, a folder to zip) or dist/Polyglot.app (macOS).
# Then check it: the packaged app runs `--self-check`, which loads every native piece.
#
# What goes in: the app, Whisper (faster-whisper + CTranslate2), Silero VAD and, when the
# parakeet extra is installed, Parakeet (onnx-asr). Speech models are NOT bundled: they
# download on first use into the user's Hugging Face cache, as in a source install.
# Qwen3-ASR is left out on purpose: PyTorch would add about 1 GB to every download.
#
# The build is a folder ("onedir"), not a single exe: a one-file build unpacks a few
# hundred MB to a temp folder on every launch, which makes start-up slow.

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).parent  # SPECPATH: the folder of this file, set by PyInstaller
ASSETS = Path(SPECPATH) / "assets"
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


def optional(collect, package):
    """Collect from a package only if it is installed (optional engines)."""
    try:
        return collect(package)
    except Exception:
        return []


def installed(package):
    import importlib.util

    return importlib.util.find_spec(package) is not None


datas = []
binaries = []
hiddenimports = []

# The app itself: adapters are imported lazily by wiring.py, so take every module, plus
# theme.qss.
hiddenimports += collect_submodules("polyglot")
datas += collect_data_files("polyglot")

# faster-whisper ships the Silero VAD model in faster_whisper/assets; CTranslate2 loads
# its native libraries itself.
datas += collect_data_files("faster_whisper")
binaries += collect_dynamic_libs("ctranslate2")

# Packages that read their own version (importlib.metadata) at import time.
for dist in ("tqdm", "huggingface_hub", "tokenizers", "filelock", "packaging", "requests",
             "numpy", "faster_whisper", "keyring", "onnx_asr"):
    datas += optional(copy_metadata, dist)

# Parakeet (optional extra). onnx-asr finds its preprocessors as package data.
if installed("onnx_asr"):
    hiddenimports += collect_submodules("onnx_asr")
    datas += collect_data_files("onnx_asr")

# Keychain and global-hotkey backends are chosen at runtime, so name them.
hiddenimports += ["keyring.backends"]
if IS_WIN:
    hiddenimports += [
        "keyring.backends.Windows",
        "pynput.keyboard._win32",
        "pynput.mouse._win32",
        "pyaudiowpatch",
    ]
elif IS_MAC:
    hiddenimports += [
        "keyring.backends.macOS",
        "pynput.keyboard._darwin",
        "pynput.mouse._darwin",
    ]

excludes = [
    "torch", "torchaudio", "qwen_asr", "transformers",  # Qwen3 extra: too big to bundle
    "tkinter", "matplotlib", "IPython", "pytest", "mypy", "ruff",
    "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets", "PyQt6.QtQml", "PyQt6.QtQuick",
    "PyQt6.Qt3DCore", "PyQt6.QtMultimedia", "PyQt6.QtBluetooth", "PyQt6.QtPdf",
]

a = Analysis(
    [str(Path(SPECPATH) / "launch.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Polyglot",
    console=False,  # a windowed app: no terminal pops up behind it
    icon=str(ASSETS / ("polyglot.ico" if IS_WIN else "polyglot.icns")),
    upx=False,  # UPX corrupts some Qt and ONNX Runtime DLLs
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="Polyglot",
    upx=False,
)

if IS_MAC:
    __version__ = re.search(
        r'__version__ = "([^"]+)"', (ROOT / "src/polyglot/__init__.py").read_text()
    ).group(1)

    app = BUNDLE(
        coll,
        name="Polyglot.app",
        icon=str(ASSETS / "polyglot.icns"),
        bundle_identifier="io.github.yalhash.polyglot",
        version=__version__,
        info_plist={
            "CFBundleDisplayName": "Polyglot",
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            # Shown by macOS when the app first opens the microphone or BlackHole.
            "NSMicrophoneUsageDescription": (
                "Polyglot transcribes the conversation on this Mac. "
                "Audio is never sent anywhere."
            ),
            "LSMinimumSystemVersion": "12.0",
        },
    )
