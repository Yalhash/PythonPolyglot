# Polyglot user guide

The [README](../README.md) gets you running. This page covers every setting in more
detail.

* [The window](#the-window)
* [Hotkeys](#hotkeys)
* [Audio](#audio)
* [Speech engines](#speech-engines)
* [Translation and suggestion models](#translation-and-suggestion-models)
* [About you](#about-you)
* [Configuration file](#configuration-file)

## The window

* **Orange-outlined lines** are what the other person said: their words on top, the
  translation underneath. The badge in the corner (`Ctrl+Alt+1`, `Ctrl+Alt+2`, …) is the
  hotkey that asks for replies to that sentence: `1` is the newest, `2` the one before,
  and so on.
* **"You" lines** are what you said, when your microphone is on.
* **The bottom panel** shows reply suggestions in their language, with your language
  underneath. Click one to copy it.
* **The lock button** (🔒, next to Suggest) keeps the view on the newest sentence.
  Scrolling up by hand unlocks it (🔓) so the view stays where you are reading; click the
  lock or press `Ctrl+Alt+L` to follow again.
* **The language label** in the title bar (for example "French (locked)") is the language
  Polyglot settled on. Click it to let it detect the language again.

## Hotkeys

These work system-wide, even while another app has focus.

| Action | Default |
| --- | --- |
| Suggest replies to the last sentence | `Ctrl+Alt+S` |
| Suggest replies to one of their recent sentences | `Ctrl+Alt+1` (newest) … `Ctrl+Alt+4` |
| Show / hide the window | `Ctrl+Alt+H` |
| Lock / unlock following the latest sentence | `Ctrl+Alt+L` |

Inside the window: `Ctrl+Enter` suggests, `Esc` hides.

Settings → Window sets how many recent sentences get a numbered hotkey (1 to 10, following
the number row: `1`–`9`, then `0`) and which keys are held with the number.

On macOS, global hotkeys need permission: System Settings → Privacy & Security →
Accessibility (and Input Monitoring), then turn on Polyglot.

## Audio

### What Polyglot listens to

* **Windows**: Polyglot hears whatever your computer is playing (the call, a video), with
  no extra setup. Technically this is WASAPI loopback capture.
* **macOS**: macOS does not let apps hear other apps' sound directly, so you need a free
  virtual audio device:
  1. Install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole).
  2. In **Audio MIDI Setup**, create a **Multi-Output Device** that includes both your
     speakers or headphones and BlackHole, and set it as your output. You still hear the
     call, and BlackHole gets a copy.
  3. In Polyglot, Settings → Audio → Listen to → BlackHole.

From a source install, `uv run polyglot --list-devices` lists every input and loopback
device.

### Your microphone

Settings → Audio → "Also listen to my microphone" adds a second input for your own voice
(off by default). What you say is transcribed on this computer, shown as a "You" line, and
used as context, so suggestions know what you already said. It is never translated and
never replied to.

Use headphones: with speakers, the microphone also hears the other person and those lines
would be tagged as yours. If the microphone cannot be opened you get a warning and
listening to the other person carries on.

The **Test** button next to the microphone picker records 5 seconds and plays them back,
then tells you if it was too quiet, too loud, or silent. Stop listening first: the
playback would otherwise be picked up as the other person's audio.

## Speech engines

Settings → Speech → Speech engine picks the speech recognizer. All of them run on your
computer, and none needs an NVIDIA graphics card.

| Engine | In the download? | Languages | Notes |
| --- | --- | --- | --- |
| Whisper (default) | yes | 99 | Good all-rounder. Bigger models (Settings → Speech → Whisper model) are more accurate and slower. `base` is about 150 MB. |
| Parakeet v3 | yes | 25 European, including French | Fastest and most accurate on a CPU. About 670 MB. It does not detect the language, so set "Their language". |
| Qwen3-ASR | no, source install only | 30, plus Chinese dialects | Detects the language. 0.6B suits a CPU; 1.7B wants a GPU. Needs PyTorch (about 1 GB). |

Each engine downloads its model the first time you use it.

From source, install the optional engines with `uv sync --extra parakeet`,
`uv sync --extra qwen3`, or both: `uv sync --extra parakeet --extra qwen3`.

**Run on: Automatic** uses a graphics card when the engine can use one, and the CPU
otherwise. For Parakeet on an AMD or Intel graphics card (source install), replace ONNX
Runtime with its DirectML build: `uv pip uninstall onnxruntime && uv pip install
onnxruntime-directml` (redo it after each `uv sync`).

## Translation and suggestion models

Settings → Translation and Settings → Suggestions each have a model dropdown. It starts
with a short built-in list and, once an OpenAI key is set, loads the chat models your key
can use (Refresh reloads it). Any model ID can also be typed in. The default for both is
`gpt-6-luna`; a saved setting that still names a retired model (`gpt-4o-mini`,
`gpt-4.1-mini`) is moved to it when the config is loaded.

Translation and suggestions can use different models: a larger model for suggestions
(for example `gpt-6.1-sol`) leaves translation fast.

<details>
<summary>How requests are made</summary>

Requests reuse one HTTPS connection per worker thread, so only the first one pays for the
connection setup. Older models get a `temperature`; reasoning models get the lowest
`reasoning_effort` that fits the job instead (`none` for translation, `low` for
suggestions). If a model rejects a parameter or value, the request is retried once without
it, and that is remembered for the session.

</details>

### API keys

Settings → Translation stores keys in your system's keychain (Windows Credential Manager
or macOS Keychain), never in a file. For development, `POLYGLOT_OPENAI_API_KEY` /
`OPENAI_API_KEY` and `POLYGLOT_GOOGLE_API_KEY` environment variables override the keychain.

## About you

Settings → About you lets suggestions talk about *you*: your name, role, skills, languages,
experience, hobbies and interests, how you want replies to sound, and what you want from
the conversation. Fill it in once; it is saved in `config.toml` (the `[profile]` section)
and used every time, with no restart.

Everything in your profile is sent to the AI provider with each suggestion request, so it
only has fields for things you would say out loud to the other person. There is
deliberately no field for health, politics, religion, money or other private details, and
the free-text fields are yours to keep clean.

Settings → Suggestions → Extra instructions adds your own rules to every request (up to
1000 characters), for example "Don't ask questions back unless the conversation needs one."

Suggestions also see the conversation around the sentence they answer: up to
`suggestions.history` lines before it and, when a numbered hotkey answers an earlier
sentence, up to `suggestions.history_after` after it, within `suggestions.history_chars`
characters in all (the nearest are kept first).

## Configuration file

Settings are saved to `config.toml` in your user config folder:

* Windows: `%LOCALAPPDATA%\polyglot`
* macOS: `~/Library/Application Support/polyglot`

[`config.example.toml`](../config.example.toml) lists every option with its default.
Precedence, lowest to highest: defaults, then `config.toml`, then
`POLYGLOT_<SECTION>__<FIELD>` environment variables (e.g. `POLYGLOT_ASR__MODEL=small`),
then changes made in the settings dialog.

Logs are written to the user log folder and contain no transcript text unless
`privacy.log_content` is turned on.
