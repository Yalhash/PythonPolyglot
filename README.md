# Polyglot - Real-time Interview Translation Assistant

A cross-platform desktop application that captures audio from video call services (Zoom, Microsoft Teams, Google Meet, etc.) and provides real-time transcription, translation, and AI-generated response suggestions.

## Features

- **Real-time Audio Capture**: Captures system audio from any call service
- **Live Transcription**: Uses local Whisper model (free, no API costs!)
- **Instant Translation**: Translates between multiple languages in real-time (requires API key)
- **AI Response Suggestions**: Generates contextually appropriate responses in both languages (requires OpenAI API key)
- **Transparent Overlay**: Non-intrusive floating window that stays on top
- **Screen Share Invisible**: Window visible to you but invisible in screen shares
- **Cross-platform**: Works on Windows and macOS

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` (optional - only needed for translation/response features):
   ```bash
   cp .env.example .env
   ```
4. (Optional) Edit `.env` and add API keys:
   - **Transcription works locally for free!** No API key needed.
   - Add `OPENAI_API_KEY` only if you want translation and response generation
   - Or add `GOOGLE_TRANSLATE_API_KEY` for translation only

## Usage

### List Available Audio Devices
First, check what audio devices are available:
```bash
python list_audio_devices.py
```

### Run the Application
Run the application from the project root:
```bash
python src/main.py
```

Or run as a module:
```bash
python -m src.main
```

The application will:
- Start capturing system audio
- Display a transparent overlay window
- Show original text, translations, and response suggestions
- Automatically hide when screen sharing is detected

## Audio Device Setup

**Important for Windows users**: Capturing system audio requires a loopback device or virtual audio cable.

### Option 1: Use Windows WASAPI Loopback (if available)
Some Windows systems have built-in loopback devices. Run `python list_audio_devices.py` to see available devices.

### Option 2: Install a Virtual Audio Device (Recommended)
Install one of these free tools:
- **VB-Audio Cable**: https://vb-audio.com/Cable/
- **OBS Virtual Audio**: Comes with OBS Studio
- **Voicemeeter**: https://vb-audio.com/Voicemeeter/

After installation:
1. Set your call application (Zoom, Teams, etc.) to output to the virtual device
2. Run `python list_audio_devices.py` to find the device ID
3. Set `AUDIO_DEVICE_ID=<device_id>` in your `.env` file

## Configuration

Edit the `.env` file to configure:
- API keys (OpenAI required, Google Translate optional)
- Source and target languages
- Audio device selection (use `list_audio_devices.py` to find device IDs)
- Window position and opacity

## Requirements

- Python 3.8+
- **No API key needed for transcription!** (runs locally)
- OpenAI API key (optional - only for translation and response generation)
- Windows 10+ or macOS 10.14+

## Model Sizes

The local Whisper model can be configured in `.env`:
- `tiny` - Fastest, least accurate (~39MB)
- `base` - Good balance (default, ~74MB)
- `small` - Better accuracy (~244MB)
- `medium` - High accuracy (~769MB)
- `large-v2` - Best accuracy (~1.5GB)
- `large-v3` - Latest best accuracy (~1.5GB)

First run will download the model automatically.

## License

MIT

