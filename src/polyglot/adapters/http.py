"""Minimal JSON-over-HTTPS client on the standard library, behind an interface.

Providers depend on ``HttpClient``, so tests inject a fake transport and never touch the
network. Transport failures become ``ProviderTimeout``/``ProviderUnavailable``; HTTP error
statuses are returned (not raised) so each provider maps them to its own meaning.
"""

from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from polyglot.core.errors import (
    AuthError,
    BadResponse,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    QuotaExceeded,
)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: str

    def json(self) -> Any:
        try:
            return json.loads(self.body)
        except json.JSONDecodeError as exc:
            raise BadResponse(f"response is not JSON (HTTP {self.status})") from exc


class HttpClient(Protocol):
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        timeout: float,
    ) -> HttpResponse: ...


class UrllibHttpClient:
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        timeout: float,
    ) -> HttpResponse:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        if not url.startswith("https://"):
            raise ProviderError("refusing to send a request over plain HTTP", retryable=False)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return HttpResponse(resp.status, resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace") if exc.fp else ""
            return HttpResponse(exc.code, body)
        except TimeoutError as exc:
            raise ProviderTimeout(f"no response within {timeout:.1f}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ProviderTimeout(f"no response within {timeout:.1f}s") from exc
            if isinstance(exc.reason, ssl.SSLError):
                raise ProviderUnavailable(f"TLS error: {exc.reason}", retryable=False) from exc
            raise ProviderUnavailable(f"network error: {exc.reason}") from exc
        except OSError as exc:
            raise ProviderUnavailable(f"network error: {exc}") from exc


def raise_for_status(resp: HttpResponse, provider: str) -> None:
    """Map HTTP status codes to typed provider errors."""
    if 200 <= resp.status < 300:
        return
    detail = resp.body[:300].replace("\n", " ")
    if resp.status in (401, 403):
        raise AuthError(f"{provider} rejected the API key (HTTP {resp.status})")
    if resp.status == 429:
        if "quota" in resp.body.lower() or "billing" in resp.body.lower():
            raise QuotaExceeded(f"{provider} quota exhausted: {detail}")
        raise ProviderUnavailable(f"{provider} rate limited (HTTP 429)")
    if resp.status in (408, 409) or resp.status >= 500:
        raise ProviderUnavailable(f"{provider} unavailable (HTTP {resp.status})")
    raise ProviderError(
        f"{provider} request failed (HTTP {resp.status}): {detail}", retryable=False
    )
