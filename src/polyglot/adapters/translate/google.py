"""Translation with Google Cloud Translation (Basic, v2 REST API, API key)."""

from __future__ import annotations

from polyglot.adapters.http import HttpClient, UrllibHttpClient, raise_for_status
from polyglot.core import languages
from polyglot.core.errors import BadResponse

URL = "https://translation.googleapis.com/language/translate/v2"


class GoogleTranslator:
    def __init__(self, api_key: str, *, http: HttpClient | None = None) -> None:
        self._key = api_key
        self._http = http or UrllibHttpClient()

    def translate(self, text: str, source: str | None, target: str, *, timeout: float) -> str:
        if not text.strip():
            return ""
        payload: dict[str, str] = {
            "q": text,
            "target": languages.primary(target) or target,
            "format": "text",  # without this, v2 returns HTML-escaped text (&#39; etc.)
        }
        src = languages.primary(source)
        if src:
            payload["source"] = src
        resp = self._http.post_json(URL, payload, params={"key": self._key}, timeout=timeout)
        raise_for_status(resp, "Google Translate")
        try:
            out = resp.json()["data"]["translations"][0]["translatedText"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BadResponse("Google Translate response has no translation") from exc
        if not isinstance(out, str):
            raise BadResponse("Google Translate returned a non-text translation")
        return out
