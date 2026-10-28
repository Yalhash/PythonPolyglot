"""Translation/suggestion providers and the HTTP layer, against a fake transport."""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from polyglot.adapters.http import HttpResponse, UrllibHttpClient, raise_for_status
from polyglot.adapters.suggest.openai import SCHEMA, OpenAISuggester, build_messages, parse_options
from polyglot.adapters.translate.google import GoogleTranslator
from polyglot.adapters.translate.openai import OpenAITranslator
from polyglot.core.errors import (
    AuthError,
    BadResponse,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    QuotaExceeded,
)
from polyglot.core.models import (
    ConversationType,
    SuggestionContext,
    Utterance,
    UtteranceStatus,
)


class FakeHttp:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def post_json(self, url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None,
                  params: dict[str, str] | None = None, timeout: float) -> HttpResponse:  # fmt: skip
        self.requests.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "params": params,
                "timeout": timeout,
            }
        )
        return self.responses.pop(0)


def chat(content: Any, **message: Any) -> HttpResponse:
    return HttpResponse(
        200, json.dumps({"choices": [{"message": {"content": content, **message}}]})
    )


# ------------------------------------------------------------------ status mapping
@pytest.mark.parametrize(
    ("status", "body", "exc", "retryable"),
    [
        (401, "", AuthError, False),
        (403, "", AuthError, False),
        (403, '{"error":{"errors":[{"reason":"dailyLimitExceeded"}]}}', QuotaExceeded, False),
        (403, '{"error":{"errors":[{"reason":"userRateLimitExceeded"}]}}', ProviderUnavailable, True),
        (429, '{"error":{"code":"insufficient_quota"}}', QuotaExceeded, False),
        (429, "slow down", ProviderUnavailable, True),
        (500, "", ProviderUnavailable, True),
        (503, "", ProviderUnavailable, True),
        (400, "bad request", ProviderError, False),
    ],
)
def test_status_mapping(status: int, body: str, exc: type, retryable: bool) -> None:
    with pytest.raises(exc) as info:
        raise_for_status(HttpResponse(status, body), "X")
    assert info.value.retryable is retryable


def test_2xx_passes() -> None:
    raise_for_status(HttpResponse(200, "{}"), "X")


def test_non_json_body_is_bad_response() -> None:
    with pytest.raises(BadResponse):
        HttpResponse(200, "<html>").json()


# ------------------------------------------------------------------ urllib transport
def test_refuses_plain_http() -> None:
    with pytest.raises(ProviderError):
        UrllibHttpClient().post_json("http://example.com", {}, timeout=1)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (TimeoutError("t"), ProviderTimeout),
        (urllib.error.URLError(TimeoutError("t")), ProviderTimeout),
        (urllib.error.URLError(ConnectionRefusedError("no")), ProviderUnavailable),
        (OSError("reset"), ProviderUnavailable),
    ],
)
def test_transport_errors_are_typed(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, expected: type
) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise raised

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(expected):
        UrllibHttpClient().post_json("https://example.com", {}, timeout=1)


def test_http_error_status_is_returned_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    def fail(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.HTTPError("https://x", 401, "no", {}, io.BytesIO(b'{"error":"bad key"}'))  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", fail)
    resp = UrllibHttpClient().post_json(
        "https://example.com", {"a": 1}, params={"k": "v"}, timeout=1
    )
    assert resp.status == 401 and "bad key" in resp.body


def test_success_path_sends_json(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Resp:
        status = 200

        def read(self) -> bytes:
            return b'{"ok": true}'

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    def fake_urlopen(req: Any, timeout: float) -> Resp:
        seen.update(
            url=req.full_url,
            data=json.loads(req.data),
            headers=dict(req.header_items()),
            timeout=timeout,
        )
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    resp = UrllibHttpClient().post_json(
        "https://h/x",
        {"a": 1},
        headers={"Authorization": "Bearer k"},
        params={"key": "q"},
        timeout=2.5,
    )
    assert resp.json() == {"ok": True}
    assert (
        seen["url"] == "https://h/x?key=q" and seen["data"] == {"a": 1} and seen["timeout"] == 2.5
    )
    assert seen["headers"]["Authorization"] == "Bearer k"


# ------------------------------------------------------------------ OpenAI translator
def test_openai_translate_request_and_result() -> None:
    http = FakeHttp(chat("  Good morning  "))
    out = OpenAITranslator("sk-test", model="m", http=http).translate(
        "Buenos días", "es", "en", timeout=3
    )
    assert out == "Good morning"
    req = http.requests[0]
    assert req["url"] == "https://api.openai.com/v1/chat/completions"
    assert req["headers"] == {"Authorization": "Bearer sk-test"}
    assert req["timeout"] == 3
    assert req["payload"]["model"] == "m"
    system = req["payload"]["messages"][0]["content"]
    assert "from Spanish" in system and "into English" in system  # names, not codes
    assert req["payload"]["messages"][1] == {"role": "user", "content": "Buenos días"}


@pytest.mark.parametrize(
    "resp",
    [
        chat(""),
        chat(None),
        chat("x", refusal="no"),
        HttpResponse(200, '{"choices": []}'),
        HttpResponse(200, "{}"),
    ],
)
def test_openai_bad_responses(resp: HttpResponse) -> None:
    with pytest.raises(BadResponse):
        OpenAITranslator("k", http=FakeHttp(resp)).translate("hola", None, "en", timeout=1)


def test_openai_auth_error() -> None:
    with pytest.raises(AuthError):
        OpenAITranslator("k", http=FakeHttp(HttpResponse(401, ""))).translate(
            "hola", None, "en", timeout=1
        )


def test_empty_text_makes_no_request() -> None:
    http = FakeHttp()
    assert OpenAITranslator("k", http=http).translate("  ", "es", "en", timeout=1) == ""
    assert GoogleTranslator("k", http=http).translate("", "es", "en", timeout=1) == ""
    assert http.requests == []


# ------------------------------------------------------------------ Google
def test_google_request_uses_text_format_and_codes() -> None:
    body = {"data": {"translations": [{"translatedText": "It's fine"}]}}
    http = FakeHttp(HttpResponse(200, json.dumps(body)))
    out = GoogleTranslator("gkey", http=http).translate("Está bien", "es-MX", "en-US", timeout=2)
    assert out == "It's fine"
    req = http.requests[0]
    assert req["params"] == {"key": "gkey"}
    assert req["payload"] == {"q": "Está bien", "target": "en", "source": "es", "format": "text"}


def test_google_without_source_and_bad_body() -> None:
    http = FakeHttp(
        HttpResponse(200, json.dumps({"data": {"translations": [{"translatedText": "x"}]}}))
    )
    GoogleTranslator("k", http=http).translate("hola", None, "en", timeout=1)
    assert "source" not in http.requests[0]["payload"]
    with pytest.raises(BadResponse):
        GoogleTranslator("k", http=FakeHttp(HttpResponse(200, "{}"))).translate(
            "a", None, "en", timeout=1
        )


# ------------------------------------------------------------------ Suggester
def ctx(**kw: Any) -> SuggestionContext:
    def utt(i: int, t: str, tr: str | None = None) -> Utterance:
        return Utterance(i, 0, 1, UtteranceStatus.TRANSLATED, t, "es", tr)

    return SuggestionContext(
        target=utt(3, "¿Qué opina?", "What do you think?"),
        history=(utt(1, "Hola", "Hello"), utt(2, "Bienvenido")),
        source_language="Spanish",
        user_language="English",
        conversation_type=kw.get("conversation_type", ConversationType.INTERVIEW),
        max_options=kw.get("max_options", 2),
    )


def test_suggester_uses_structured_output_and_context() -> None:
    content = json.dumps({"options": [{"source_text": "Creo que sí", "user_text": "I think so"},
                                      {"source_text": "No estoy seguro", "user_text": "I'm not sure"},
                                      {"source_text": "Tal vez", "user_text": "Maybe"}]})  # fmt: skip
    http = FakeHttp(chat(content))
    opts = OpenAISuggester("k", http=http).suggest(ctx(max_options=2), timeout=6)
    assert [o.source_text for o in opts] == ["Creo que sí", "No estoy seguro"]  # capped at max
    payload = http.requests[0]["payload"]
    assert payload["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
    system, user = (m["content"] for m in payload["messages"])
    assert "job interview" in system and "in Spanish" in system and "English" in system
    assert "Hola  (Hello)" in user and "Bienvenido" in user and "¿Qué opina?" in user


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "{}",
        '{"options": [{"source_text": "x"}]}',
        '{"options": []}',
        '{"options": [{"source_text": "", "user_text": "y"}]}',
    ],
)
def test_suggester_rejects_malformed_output(content: str) -> None:
    with pytest.raises(BadResponse):
        parse_options(content, 2)


def test_build_messages_without_history() -> None:
    c = ctx()
    c2 = SuggestionContext(c.target, (), "Spanish", "English", ConversationType.CASUAL, 1)
    _, user = (m["content"] for m in build_messages(c2))
    assert user.startswith("They just said:")
    assert "casual" in build_messages(c2)[0]["content"]


def test_truncated_response_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client

    def boom(*_a: Any, **_k: Any) -> Any:
        raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(ProviderUnavailable):
        UrllibHttpClient().post_json("https://example.com", {}, timeout=1)


def test_error_body_read_failure_keeps_status(monkeypatch: pytest.MonkeyPatch) -> None:
    class Body:
        def read(self, *_a: Any) -> bytes:
            raise TimeoutError("slow body")

        def close(self) -> None:
            pass

    def fail(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.HTTPError("https://x", 503, "busy", {}, Body())  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", fail)
    assert UrllibHttpClient().post_json("https://example.com", {}, timeout=1).status == 503
