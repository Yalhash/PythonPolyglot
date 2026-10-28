"""Translation with an OpenAI chat model (Chat Completions REST API)."""

from __future__ import annotations

import logging
from typing import Any

from polyglot.adapters.http import HttpClient, UrllibHttpClient, raise_for_status
from polyglot.adapters.openai_models import MEMORY, chat_options, rejected_param
from polyglot.core import languages
from polyglot.core.errors import BadResponse

DEFAULT_BASE_URL = "https://api.openai.com/v1"

log = logging.getLogger(__name__)


def chat_text(
    http: HttpClient,
    *,
    api_key: str,
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
    provider: str = "OpenAI",
) -> str:
    model = str(payload.get("model", ""))
    payload = MEMORY.apply(model, payload)
    for _ in range(4):  # at most a few fallbacks (effort value steps, then dropping it)
        resp = http.post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        param = rejected_param(resp)
        retry = MEMORY.adjust(model, payload, param) if param else None
        if retry is None:
            break
        log.info("%s rejected %s for %s; retrying without it", provider, param, model)
        payload = retry
    raise_for_status(resp, provider)
    data = resp.json()
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BadResponse(f"{provider} response has no choices") from exc
    if message.get("refusal"):
        raise BadResponse(f"{provider} refused: {message['refusal']}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise BadResponse(f"{provider} returned an empty message")
    return content.strip()


class OpenAITranslator:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = DEFAULT_BASE_URL,
        http: HttpClient | None = None,
    ) -> None:
        self._key, self.model, self._base = api_key, model, base_url
        self._http = http or UrllibHttpClient()

    def translate(self, text: str, source: str | None, target: str, *, timeout: float) -> str:
        if not text.strip():
            return ""
        src = f" from {languages.name(source)}" if source else ""
        payload = {
            "model": self.model,
            # Translation must be fast: no reasoning where the model allows it.
            **chat_options(self.model, temperature=0.2, effort="none"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional interpreter. Translate the user's message"
                        f"{src} into {languages.name(target)}. Keep the meaning, tone and "
                        "register. Reply with the translation only: no quotes, notes or "
                        "explanations."
                    ),
                },
                {"role": "user", "content": text},
            ],
        }
        return chat_text(
            self._http, api_key=self._key, base_url=self._base, payload=payload, timeout=timeout
        )
