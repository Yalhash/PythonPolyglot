"""OpenAI model choice: a built-in list, the live list from the API, and the request
options each model family accepts.

Why per-model options: older chat models (``gpt-4o``, ``gpt-4.1``) take ``temperature``;
reasoning models (``gpt-5`` onward, ``o``-series) reject it, and instead take
``reasoning_effort``, which decides most of their latency. Families differ in which
effort values they accept, so a request starts with the fastest setting for the job and,
if the API rejects a parameter or value (HTTP 400 naming the parameter), retries once
with the next fallback. What worked is remembered per model for the rest of the session.

Built-in list and retirement dates checked against OpenAI's model and deprecation pages
on 2026-10-28. The live list (``list_chat_models``) is what an account can actually use.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any

from polyglot.adapters.http import HttpClient, HttpResponse, UrllibHttpClient, raise_for_status
from polyglot.core.errors import BadResponse

DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    note: str = ""  # shown as a tooltip in Settings


#: Shutdown dates announced by OpenAI (deprecations page).
RETIRING: dict[str, str] = {
    "gpt-4o": "2027-01-14",
    "gpt-4o-mini": "2027-01-14",
    "gpt-4.1-mini": "2027-01-14",
    "gpt-5-mini": "2026-12-11",
}

_KNOWN: tuple[tuple[str, str], ...] = (
    ("gpt-6-luna", "GPT-6, most efficient: fast and low cost. OpenAI's named replacement "
                   "for gpt-4o-mini and gpt-4.1-mini."),  # fmt: skip
    ("gpt-6.1-terra", "GPT-6.1 family."),
    ("gpt-6.1-sol", "GPT-6.1 family, more capable; slower and costlier than luna."),
    ("gpt-6-astra", "GPT-6 flagship: most capable, slowest and most expensive."),
    ("gpt-4o-mini", "Fast, non-reasoning."),
    ("gpt-4.1-mini", "Fast, non-reasoning."),
    ("gpt-4o", "Non-reasoning."),
)


def _with_retirement(model_id: str, note: str) -> ModelInfo:
    date = RETIRING.get(model_id)
    return ModelInfo(model_id, f"{note} Retires {date}.".strip() if date else note)


KNOWN_MODELS: tuple[ModelInfo, ...] = tuple(_with_retirement(m, n) for m, n in _KNOWN)
_KNOWN_NOTES = {m.id: m.note for m in KNOWN_MODELS}

# ------------------------------------------------------------------ request options
_SAMPLING_PREFIXES = ("gpt-4o", "gpt-4.1", "gpt-4-", "gpt-3.5", "chatgpt-4o")
#: A rejected effort value falls back to the next one; after the last, it is dropped.
_EFFORT_FALLBACK = {"none": "minimal", "minimal": "low", "low": None}


def uses_temperature(model: str) -> bool:
    """Older, non-reasoning chat models take ``temperature``; reasoning models do not."""
    return model.startswith(_SAMPLING_PREFIXES) or model == "gpt-4"


class _Memory:
    """What each model turned out to accept, learned from rejections (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dropped: dict[str, set[str]] = {}
        self._effort: dict[str, str] = {}

    def apply(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            out = {k: v for k, v in payload.items() if k not in self._dropped.get(model, ())}
            if "reasoning_effort" in out and model in self._effort:
                out["reasoning_effort"] = self._effort[model]
            return out

    def adjust(self, model: str, payload: dict[str, Any], param: str) -> dict[str, Any] | None:
        """The payload to retry with after ``param`` was rejected, or None to give up."""
        if param not in payload or param not in ("temperature", "reasoning_effort"):
            return None
        out = dict(payload)
        with self._lock:
            if param == "reasoning_effort":
                nxt = _EFFORT_FALLBACK.get(str(payload[param]))
                if nxt is not None:
                    out[param] = self._effort[model] = nxt
                    return out
            self._dropped.setdefault(model, set()).add(param)
            if param == "reasoning_effort":
                self._effort.pop(model, None)
        del out[param]
        return out

    def clear(self) -> None:
        with self._lock:
            self._dropped.clear()
            self._effort.clear()


MEMORY = _Memory()


def chat_options(model: str, *, temperature: float, effort: str) -> dict[str, Any]:
    """The tuning parameters for ``model``: ``temperature`` for older chat models,
    ``reasoning_effort`` for reasoning models."""
    return {"temperature": temperature} if uses_temperature(model) else {"reasoning_effort": effort}


def rejected_param(resp: HttpResponse) -> str | None:
    """The parameter an HTTP 400 complains about, if the API names one."""
    if resp.status != 400:
        return None
    try:
        err = json.loads(resp.body).get("error") or {}
    except (json.JSONDecodeError, AttributeError):
        return None
    param = err.get("param") if isinstance(err, dict) else None
    return param if isinstance(param, str) else None


# ------------------------------------------------------------------ live model list
_INCLUDE = re.compile(r"^(gpt-|chatgpt-|o\d)")
_EXCLUDE = (
    "audio", "realtime", "tts", "transcribe", "whisper", "image", "embedding", "moderation",
    "search", "live", "codex", "cyber", "rosalind", "computer-use", "dall-e", "sora",
    "instruct", "translate", "deep-research", "oss",
)  # fmt: skip
_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{4}$")


def is_text_chat_model(model_id: str) -> bool:
    """A general text model usable for translation and replies (no audio, image,
    realtime or specialist models, and no dated snapshots)."""
    mid = model_id.lower()
    return (
        bool(_INCLUDE.match(mid))
        and not any(word in mid for word in _EXCLUDE)
        and not _SNAPSHOT.search(mid)
    )


def sort_models(ids: list[str]) -> list[ModelInfo]:
    """Built-in models first, in their order; then the rest alphabetically."""
    order = {m.id: i for i, m in enumerate(KNOWN_MODELS)}
    unique = sorted(set(ids), key=lambda m: (order.get(m, len(order)), m))
    return [_with_retirement(m, _KNOWN_NOTES.get(m, "")) for m in unique]


def list_chat_models(
    api_key: str,
    *,
    http: HttpClient | None = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 5.0,
) -> list[ModelInfo]:
    """The text chat models this API key can use (``GET /v1/models``, filtered)."""
    resp = (http or UrllibHttpClient()).get_json(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    raise_for_status(resp, "OpenAI")
    data = resp.json()
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise BadResponse("OpenAI model list has no 'data'")
    ids = [m["id"] for m in items if isinstance(m, dict) and isinstance(m.get("id"), str)]
    return sort_models([m for m in ids if is_text_chat_model(m)])


class OpenAIModelCatalog:
    """What the settings dialog shows: the built-in list at once, the live list on
    request (cached for the session, per key)."""

    def __init__(self, http: HttpClient | None = None, base_url: str = DEFAULT_BASE_URL) -> None:
        self._http, self._base = http, base_url
        self._cache: dict[str, list[ModelInfo]] = {}
        self._lock = threading.Lock()

    def known(self) -> list[ModelInfo]:
        return list(KNOWN_MODELS)

    def cached(self, api_key: str) -> list[ModelInfo] | None:
        with self._lock:
            return self._cache.get(api_key)

    def fetch(self, api_key: str) -> list[ModelInfo]:
        """Blocking; call off the GUI thread. Raises ProviderError subclasses."""
        models = list_chat_models(api_key, http=self._http, base_url=self._base)
        with self._lock:
            self._cache[api_key] = models
        return models
