"""Minimal JSON-over-HTTPS client on the standard library, behind an interface.

Providers depend on ``HttpClient``, so tests inject a fake transport and never touch the
network. Transport failures become ``ProviderTimeout``/``ProviderUnavailable``; HTTP error
statuses are returned (not raised) so each provider maps them to its own meaning.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import ssl
import threading
import urllib.parse
from collections.abc import Callable
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

    def get_json(
        self, url: str, *, headers: dict[str, str] | None = None, timeout: float
    ) -> HttpResponse: ...


class HttpsClient:
    """HTTPS over ``http.client`` with keep-alive.

    A connection is kept per thread and host, so only the first request from a worker pays
    for DNS, TCP and TLS setup (often 100-400 ms, which matters for a translation that has
    a budget of a few seconds). A connection the server closed while idle is replaced and
    the request sent once more; a timeout is never retried.
    """

    def __init__(self, connect: Connect | None = None) -> None:
        self._connect = connect or _connect
        self._local = threading.local()

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
        _require_https(url)
        return self._send(
            "POST",
            url,
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json", **(headers or {})},
            timeout,
        )

    def get_json(
        self, url: str, *, headers: dict[str, str] | None = None, timeout: float
    ) -> HttpResponse:
        _require_https(url)
        return self._send("GET", url, None, dict(headers or {}), timeout)

    def _pool(self) -> dict[tuple[str, int], http.client.HTTPSConnection]:
        pool: dict[tuple[str, int], http.client.HTTPSConnection] | None
        pool = getattr(self._local, "pool", None)
        if pool is None:
            pool = self._local.pool = {}
        return pool

    def _send(
        self, method: str, url: str, body: bytes | None, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        parts = urllib.parse.urlsplit(url)
        key = (parts.hostname or "", parts.port or 443)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        pool = self._pool()
        for attempt in (0, 1):
            conn = pool.get(key)
            reused = conn is not None
            if conn is None:
                conn = pool[key] = self._connect(key[0], key[1], timeout)
            else:
                conn.timeout = timeout
                if conn.sock is not None:
                    conn.sock.settimeout(timeout)
            try:
                conn.request(method, target, body=body, headers=headers)
                resp = conn.getresponse()
                text = resp.read().decode("utf-8", "replace")
            except TimeoutError as exc:
                self._drop(pool, key)
                raise ProviderTimeout(f"no response within {timeout:.1f}s") from exc
            except ssl.SSLError as exc:
                self._drop(pool, key)
                raise ProviderUnavailable(f"TLS error: {exc}", retryable=False) from exc
            except (http.client.HTTPException, OSError) as exc:
                self._drop(pool, key)
                if reused and attempt == 0:
                    continue  # the idle connection had been closed by the server
                if isinstance(exc, http.client.HTTPException):
                    raise ProviderUnavailable(f"connection dropped: {exc!r}") from exc
                raise ProviderUnavailable(f"network error: {exc}") from exc
            if resp.will_close:
                self._drop(pool, key)
            return HttpResponse(resp.status, text)
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _drop(
        pool: dict[tuple[str, int], http.client.HTTPSConnection], key: tuple[str, int]
    ) -> None:
        conn = pool.pop(key, None)
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()


Connect = Callable[[str, int, float], "http.client.HTTPSConnection"]


def _connect(host: str, port: int, timeout: float) -> http.client.HTTPSConnection:
    return http.client.HTTPSConnection(
        host, port, timeout=timeout, context=ssl.create_default_context()
    )


def _require_https(url: str) -> None:
    if not url.startswith("https://"):
        raise ProviderError("refusing to send a request over plain HTTP", retryable=False)


def raise_for_status(resp: HttpResponse, provider: str) -> None:
    """Map HTTP status codes to typed provider errors."""
    if 200 <= resp.status < 300:
        return
    detail = resp.body[:300].replace("\n", " ")
    body = resp.body.lower()
    if resp.status == 403:
        # Google returns 403 for quota and rate limits too, not only for a bad key.
        if "dailylimitexceeded" in body or "quota" in body:
            raise QuotaExceeded(f"{provider} quota exhausted: {detail}")
        if "ratelimitexceeded" in body:  # rateLimitExceeded, userRateLimitExceeded
            raise ProviderUnavailable(f"{provider} rate limited (HTTP 403)")
    if resp.status in (401, 403):
        raise AuthError(f"{provider} rejected the API key (HTTP {resp.status})")
    if resp.status == 429:
        if "quota" in body or "billing" in body:
            raise QuotaExceeded(f"{provider} quota exhausted: {detail}")
        raise ProviderUnavailable(f"{provider} rate limited (HTTP 429)")
    if resp.status in (408, 409) or resp.status >= 500:
        raise ProviderUnavailable(f"{provider} unavailable (HTTP {resp.status})")
    raise ProviderError(
        f"{provider} request failed (HTTP {resp.status}): {detail}", retryable=False
    )
