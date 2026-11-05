# Polyglot

A small always-on-top window that listens to the other side of a call, transcribes it
locally, translates it into your language, and suggests replies in theirs when you
press a hotkey.

* **Local speech recognition**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  by default, or [Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) or
  [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (see [Speech engines](#speech-engines)).
  Audio never leaves your machine.
* **Sentence-aware**: voice-activity detection (Silero VAD) cuts audio at the speaker's
  pauses, not every N seconds, so the recognizer and the translator see whole sentences.
* **Translation** (OpenAI or Google) and **reply suggestions** (OpenAI) are opt-in.
  Only text is sent, and only once you enable them.
* **Hidden from screen sharing on Windows 10 2004+**: the window stays visible to you
  but is left out of captures. Not possible on macOS 15+ (see
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#platform-layer)), so the window shows a
  "visible in screen share" indicator there.

> Status: v1 rewrite (`1.0.0.dev0`). The v0 prototype is tagged `v0.1.0`.

## Quick start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # creates .venv and installs the exact locked versions
uv run polyglot --demo  # try the UI with synthetic audio: no models, keys or network
uv run polyglot         # the real thing
```

On first real run Whisper downloads its model (`base` is about 150 MB).

### Speech engines

Settings → Speech → Speech engine picks the local recognizer (`asr.engine` in
`config.toml`). All three run without an NVIDIA GPU; install the optional ones first:

| Engine | Install | Languages | Runs on | Notes |
| --- | --- | --- | --- | --- |
| Whisper (default) | included | 99 | CPU, CUDA | Model size under "Whisper model". |
| Parakeet v3 | `uv sync --extra parakeet` | 25 European (incl. French) | CPU (int8), CUDA, DirectML | Fastest and most accurate on a CPU. ~670 MB download. Does not report the language: set "Their language" so translation and suggestions know it. |
| Qwen3-ASR | `uv sync --extra qwen3` | 30, plus Chinese dialects | CPU (float32), CUDA | 0.6B suits a CPU; 1.7B wants a GPU. Pulls in PyTorch (~1 GB). Detects the language. |

Both extras together: `uv sync --extra parakeet --extra qwen3`. The model downloads on the
first run with that engine. "Run on: Automatic" uses a GPU when the engine's runtime can
use one and the CPU otherwise. For Parakeet on an AMD or Intel GPU, replace ONNX Runtime
with its DirectML build (`uv pip uninstall onnxruntime && uv pip install
onnxruntime-directml`, redone after each `uv sync`); Automatic then picks it up.

### Hotkeys (system-wide)

| Action | Default |
| --- | --- |
| Suggest replies to the last sentence | `Ctrl+Alt+S` |
| Suggest replies to one of their recent sentences | `Ctrl+Alt+1` (newest) … `Ctrl+Alt+4` |
| Show / hide the window | `Ctrl+Alt+H` |
| Lock / unlock following the latest sentence | `Ctrl+Alt+L` |

Inside the window: `Ctrl+Enter` suggests, `Esc` hides. Click a suggestion to copy it.

Each of the other person's finished sentences that a numbered hotkey can reach is outlined
in orange, with its hotkey in the top-right corner: `1` is the newest, `2` the one before,
and so on. Settings → Window sets how many sentences get one (1 to 10, following the
number row: `1`–`9`, then `0`) and which keys are held with the number.

The transcript follows the newest sentence (🔒, next to Suggest). Scrolling by hand
unlocks it (🔓) so the view stays where you are reading; it follows again only when you
click the lock button or press `Ctrl+Alt+L`, which also jumps to the latest sentence.

### Audio source

* **Windows**: captures system audio directly (WASAPI loopback). No virtual cable is needed.
* **macOS**: install [BlackHole](https://github.com/ExistentialAudio/BlackHole), route
  the call's output to it (a Multi-Output Device lets you still hear it), and pick it in
  Settings → Audio.

`uv run polyglot --list-devices` lists input and loopback devices.

### API keys

Settings → Translation stores keys in the OS keychain (Windows Credential Manager or
macOS Keychain), never in a file. For development, `POLYGLOT_OPENAI_API_KEY` /
`OPENAI_API_KEY` and `POLYGLOT_GOOGLE_API_KEY` override the keychain.

### Models

Settings → Translation and Settings → Suggestions each have a model dropdown. It starts
with a short built-in list and, once an OpenAI key is set, loads the chat models your key
can actually use (Refresh reloads it). Any model ID can also be typed in. The default for
both is `gpt-6-luna`, the replacement OpenAI names for `gpt-4o-mini` and `gpt-4.1-mini`,
which it retires on 2027-01-14. A saved setting that still holds one of those two is moved
to `gpt-6-luna` when the config is loaded.

Translation and suggestions are separate settings, so they can use different models:
a larger model for suggestions (for example `gpt-6.1-sol`) leaves translation fast.
Requests reuse one HTTPS connection per worker thread, so only the first one pays for
the connection setup.

Older models get a `temperature`; reasoning models get the lowest `reasoning_effort`
that fits the job instead (`none` for translation, `low` for suggestions). If a model
rejects a parameter or value, the request is retried once without it, and that is
remembered for the session.

### Your microphone

Settings → Audio → "Also listen to my microphone" adds a second input for your own voice
(off by default). What you say is transcribed on this computer, shown in the window as a
"You" line, and used as context, so suggestions know what you already said. It is never
translated and never replied to. Use headphones: with speakers, the microphone also hears
the other person and those lines would be tagged as yours. If the microphone cannot be
opened you get a warning and listening to the other person carries on.

The **Test** button next to the microphone picker records 5 seconds from the selected device
and plays them back, then tells you if it was too quiet, too loud, or silent. Stop listening
first: the playback would otherwise be picked up as the other person's audio.

### About you

Settings → About you lets suggestions talk about *you*: your name, role, skills, languages,
experience, hobbies and interests, how you want replies to sound, and what you want from the
conversation. Fill it in once; it is saved in `config.toml` (the `[profile]` section) and
used every time. Suggestions also see the conversation around the sentence they answer:
up to `suggestions.history` utterances before it and, when a reply hotkey answers an
earlier sentence, up to `suggestions.history_after` after it, within
`suggestions.history_chars` characters in all (the nearest are kept first). The prompt
numbers the lines and marks the one to reply to.

Settings → Suggestions → Extra instructions adds your own rules to every request
(`suggestions.instructions`, up to 1000 characters), for example "Don't ask questions back
unless the conversation needs one." Like the profile, it is sent to OpenAI and applies to the
next suggestion without restarting listening.

Everything in your profile is sent to the AI provider with each suggestion request, so it only has
fields for things you would say out loud to the other person. There is deliberately no field
for health, politics, religion, money or other private details, and the free-text fields are
yours to keep clean. Profile edits take effect on the next suggestion with no restart.

## Configuration

Settings are saved to `config.toml` in your user config directory
(`%LOCALAPPDATA%\polyglot` on Windows, `~/Library/Application Support/polyglot` on
macOS). [`config.example.toml`](config.example.toml) lists every option with its
default. Precedence, lowest to highest: defaults, then `config.toml`, then
`POLYGLOT_<SECTION>__<FIELD>` environment variables (e.g. `POLYGLOT_ASR__MODEL=small`),
then changes made in the settings dialog.

## Headless use

```bash
uv run polyglot --file recording.wav --fast    # transcribe (and translate) a WAV file
uv run polyglot --headless                     # listen live, print to the terminal
```

## Development

```bash
uv sync                          # dev tools included
uv run pytest                    # fast suite: no models, devices or network
uv run pytest -m slow            # real Whisper tiny + Silero (needs espeak-ng); Parakeet/Qwen3 too if installed
uv run ruff check . && uv run ruff format .
uv run mypy
```

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. Add one with
`uv add <package>` (or `uv add --dev` for tooling) and commit both files.

CI (GitHub Actions) runs lint, type checks and tests on Windows, macOS and Linux with
Python 3.10 and 3.12, plus a real-model job. On feature branches it also commits lockfile
updates and ruff autofixes.

Architecture, the pipeline and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
