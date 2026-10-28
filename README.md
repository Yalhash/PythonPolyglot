# Polyglot

A small always-on-top window that listens to the other side of a call, transcribes it
locally, translates it into your language, and suggests replies in theirs when you
press a hotkey.

* **Local speech recognition** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
  Audio never leaves your machine.
* **Sentence-aware**: voice-activity detection (Silero VAD) cuts audio at the speaker's
  pauses, not every N seconds, so Whisper and the translator see whole sentences.
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

### Hotkeys (system-wide)

| Action | Default |
| --- | --- |
| Suggest replies to the last sentence | `Ctrl+Alt+S` |
| Show / hide the window | `Ctrl+Alt+H` |

Inside the window: `Ctrl+Enter` suggests, `Esc` hides. Click a suggestion to copy it.

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
can actually use (Refresh reloads it). Any model ID can also be typed in. Defaults:
`gpt-4o-mini` for translation and `gpt-4.1-mini` for suggestions. OpenAI retires both on
2027-01-14 and names `gpt-6-luna` as the replacement.

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

### About you

Settings → About you lets suggestions talk about *you*: your name, role, skills, languages,
experience, hobbies and interests, how you want replies to sound, and what you want from the
conversation. Fill it in once; it is saved in `config.toml` (the `[profile]` section) and
used every time. Suggestions also see what was said earlier in the conversation, up to
`suggestions.history` utterances or `suggestions.history_chars` characters.

Everything in your profile is sent to OpenAI with each suggestion request, so it only has
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
uv run pytest -m slow            # real Whisper tiny + Silero (needs espeak-ng)
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
