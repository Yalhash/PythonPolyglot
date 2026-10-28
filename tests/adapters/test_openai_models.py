"""Model list, per-model request options and the unsupported-parameter fallback."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from polyglot.adapters.http import HttpResponse, UrllibHttpClient
from polyglot.adapters.openai_models import (
    KNOWN_MODELS,
    MEMORY,
    OpenAIModelCatalog,
    chat_options,
    is_text_chat_model,
    list_chat_models,
    uses_temperature,
)
from polyglot.adapters.suggest.openai import OpenAISuggester
from polyglot.adapters.translate.openai import OpenAITranslator
from polyglot.core.errors import AuthError, BadResponse, ProviderError
from polyglot.core.models import ConversationType, SuggestionContext, Utterance, UtteranceStatus


class FakeHttp:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    def post_json(self, url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None,
                  params: dict[str, str] | None = None, timeout: float) -> HttpResponse:  # fmt: skip
        self.posts.append({"url": url, "payload": dict(payload)})
        return self.responses.pop(0)

    def get_json(
        self, url: str, *, headers: dict[str, str] | None = None, timeout: float
    ) -> HttpResponse:
        self.gets.append({"url": url, "headers": headers, "timeout": timeout})
        return self.responses.pop(0)


def ok(content: str = "Hello") -> HttpResponse:
    return HttpResponse(200, json.dumps({"choices": [{"message": {"content": content}}]}))


def rejected(param: str, code: str = "unsupported_value") -> HttpResponse:
    body = {"error": {"message": f"bad {param}", "param": param, "code": code}}
    return HttpResponse(400, json.dumps(body))


@pytest.fixture(autouse=True)
def _fresh_memory() -> Iterator[None]:
    MEMORY.clear()
    yield
    MEMORY.clear()


# ------------------------------------------------------------------ request options
@pytest.mark.parametrize(
    ("model", "legacy"),
    [("gpt-4o-mini", True), ("gpt-4.1-mini", True), ("gpt-4o", True), ("gpt-6-luna", False),
     ("gpt-5-mini", False), ("o4-mini", False), ("some-future-model", False)],
)  # fmt: skip
def test_options_follow_the_model_family(model: str, legacy: bool) -> None:
    assert uses_temperature(model) is legacy
    opts = chat_options(model, temperature=0.2, effort="none")
    assert opts == ({"temperature": 0.2} if legacy else {"reasoning_effort": "none"})


def test_translator_sends_temperature_only_to_older_models() -> None:
    http = FakeHttp(ok(), ok())
    OpenAITranslator("k", model="gpt-4o-mini", http=http).translate("hola", "es", "en", timeout=3)
    OpenAITranslator("k", model="gpt-6-luna", http=http).translate("hola", "es", "en", timeout=3)
    old, new = (r["payload"] for r in http.posts)
    assert old["temperature"] == 0.2 and "reasoning_effort" not in old
    assert new["reasoning_effort"] == "none" and "temperature" not in new


def test_suggester_uses_low_effort_on_reasoning_models() -> None:
    content = json.dumps({"options": [{"source_text": "Sí", "user_text": "Yes"}]})
    http = FakeHttp(ok(content))
    target = Utterance(1, 0, 1, UtteranceStatus.TRANSLATED, "¿Listo?", "es", "Ready?")
    ctx = SuggestionContext(
        target=target, history=(), source_language="Spanish", user_language="English",
        conversation_type=ConversationType.MEETING, max_options=1,
    )  # fmt: skip
    OpenAISuggester("k", model="gpt-6-luna", http=http).suggest(ctx, timeout=6)
    payload = http.posts[0]["payload"]
    assert payload["reasoning_effort"] == "low" and "temperature" not in payload


def test_rejected_effort_steps_down_then_is_remembered() -> None:
    http = FakeHttp(rejected("reasoning_effort"), ok("one"), ok("two"))
    t = OpenAITranslator("k", model="gpt-5-mini", http=http)
    assert t.translate("uno", "es", "en", timeout=3) == "one"
    assert [r["payload"]["reasoning_effort"] for r in http.posts] == ["none", "minimal"]
    assert t.translate("dos", "es", "en", timeout=3) == "two"
    assert http.posts[2]["payload"]["reasoning_effort"] == "minimal"  # no second rejection


def test_effort_is_dropped_after_the_last_fallback() -> None:
    http = FakeHttp(*(rejected("reasoning_effort") for _ in range(3)), ok())
    OpenAITranslator("k", model="odd-model", http=http).translate("x", None, "en", timeout=3)
    efforts = [r["payload"].get("reasoning_effort") for r in http.posts]
    assert efforts == ["none", "minimal", "low", None]


def test_rejected_temperature_is_dropped_and_remembered() -> None:
    http = FakeHttp(rejected("temperature", "unsupported_parameter"), ok(), ok())
    t = OpenAITranslator("k", model="gpt-4o-mini", http=http)
    t.translate("a", None, "en", timeout=3)
    t.translate("b", None, "en", timeout=3)
    assert "temperature" in http.posts[0]["payload"]
    assert "temperature" not in http.posts[1]["payload"]
    assert "temperature" not in http.posts[2]["payload"]


def test_other_bad_requests_are_not_retried() -> None:
    http = FakeHttp(rejected("messages", "invalid_type"))
    with pytest.raises(ProviderError):
        OpenAITranslator("k", model="gpt-6-luna", http=http).translate("x", None, "en", timeout=3)
    assert len(http.posts) == 1


# ------------------------------------------------------------------ model list
def test_model_list_keeps_general_text_models_known_first() -> None:
    ids = [
        "gpt-7-nova", "gpt-6-luna", "gpt-4o-mini", "o4-mini", "gpt-4o-mini-2024-07-18",
        "gpt-realtime-2", "gpt-4o-mini-tts", "gpt-transcribe", "text-embedding-3-small",
        "gpt-image-2.5-flare", "dall-e-3", "gpt-5.3-codex", "whisper-1", "gpt-6.1-sol",
        "omni-moderation-latest", "gpt-live-1",
    ]  # fmt: skip
    http = FakeHttp(HttpResponse(200, json.dumps({"data": [{"id": i} for i in ids]})))
    models = list_chat_models("sk", http=http, timeout=4)
    assert [m.id for m in models] == ["gpt-6-luna", "gpt-6.1-sol", "gpt-4o-mini", "gpt-7-nova", "o4-mini"]
    assert http.gets[0]["url"] == "https://api.openai.com/v1/models"
    assert http.gets[0]["headers"] == {"Authorization": "Bearer sk"}
    assert "Retires 2027-01-14" in next(m.note for m in models if m.id == "gpt-4o-mini")


@pytest.mark.parametrize("model_id", ["gpt-4o-audio-preview", "gpt-4o-search-preview", "babbage-002"])
def test_specialist_models_are_filtered(model_id: str) -> None:
    assert not is_text_chat_model(model_id)


def test_model_list_errors_are_typed() -> None:
    with pytest.raises(AuthError):
        list_chat_models("bad", http=FakeHttp(HttpResponse(401, "{}")))
    with pytest.raises(BadResponse):
        list_chat_models("k", http=FakeHttp(HttpResponse(200, '{"object": "list"}')))


def test_catalog_caches_per_key() -> None:
    body = json.dumps({"data": [{"id": "gpt-6-luna"}]})
    http = FakeHttp(HttpResponse(200, body))
    catalog = OpenAIModelCatalog(http=http)
    assert catalog.known() == list(KNOWN_MODELS)
    assert catalog.cached("k") is None
    assert [m.id for m in catalog.fetch("k")] == ["gpt-6-luna"]
    assert [m.id for m in catalog.cached("k") or []] == ["gpt-6-luna"]
    assert catalog.cached("other") is None


def test_default_models_are_in_the_built_in_list() -> None:
    from polyglot.config import SuggestionsConfig, TranslationConfig

    ids = {m.id for m in KNOWN_MODELS}
    assert TranslationConfig().model in ids and SuggestionsConfig().model in ids


# ------------------------------------------------------------------ GET transport
def test_get_json_refuses_plain_http() -> None:
    with pytest.raises(ProviderError):
        UrllibHttpClient().get_json("http://api.openai.com/v1/models", timeout=1)


def test_get_json_sends_a_get_without_body(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Resp:
        status = 200

        def read(self) -> bytes:
            return b'{"data": []}'

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> Resp:
        seen.update(method=req.get_method(), data=req.data, auth=req.get_header("Authorization"))
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    resp = UrllibHttpClient().get_json(
        "https://api.openai.com/v1/models", headers={"Authorization": "Bearer k"}, timeout=2
    )
    assert resp.status == 200 and seen == {"method": "GET", "data": None, "auth": "Bearer k"}
