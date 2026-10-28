"""Language codes (as Whisper reports them) to English names, for prompts and the UI."""

from __future__ import annotations

_NAMES: dict[str, str] = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan",
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "gl": "Galician", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian",
    "is": "Icelandic", "it": "Italian", "ja": "Japanese", "ka": "Georgian", "kk": "Kazakh",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian",
    "ml": "Malayalam", "mr": "Marathi", "ms": "Malay", "ne": "Nepali", "nl": "Dutch",
    "no": "Norwegian", "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", "sr": "Serbian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Tagalog",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}  # fmt: skip


def primary(code: str | None) -> str | None:
    """'pt-BR' -> 'pt'; None stays None."""
    if not code:
        return None
    return code.replace("_", "-").split("-")[0].lower()


def name(code: str | None) -> str:
    """Human name for a language code; unknown codes are returned as given."""
    if not code:
        return "Unknown"
    return _NAMES.get(primary(code) or "", code)


def same_language(a: str | None, b: str | None) -> bool:
    pa, pb = primary(a), primary(b)
    return pa is not None and pa == pb


def known_codes() -> list[str]:
    return sorted(_NAMES)
