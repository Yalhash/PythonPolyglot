"""Translation/suggestion providers and the HTTP layer, against a fake transport."""

from __future__ import annotations

import json
from typing import Any

import pytest

from polyglot.adapters.http import HttpResponse
from polyglot.adapters.suggest.openai import SCHEMA, OpenAISuggester, build_messages, parse_options
from polyglot.adapters.translate.google import GoogleTranslator
from polyglot.adapters.translate.openai import OpenAITranslator
from polyglot.core.errors import (
    AuthError,
    BadResponse,
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
    assert ">>> [1] Them: ¿Qué opina?  (What do you think?)" in user
    assert "Suggest replies to message [1] (marked >>>)." in user
    assert "context only" not in user  # nothing came after it
    assert "casual" in build_messages(c2)[0]["content"]


def test_build_messages_labels_who_spoke() -> None:
    from polyglot.core.models import Speaker

    base = ctx()
    user_turn = Utterance(
        9, 0, 1, UtteranceStatus.FINAL, "Lidero un equipo", "es", speaker=Speaker.USER
    )
    c = SuggestionContext(base.target, (*base.history, user_turn), "Spanish", "English")
    _, user = (m["content"] for m in build_messages(c))
    assert "    [1] Them: Hola  (Hello)" in user
    assert "    [3] User: Lidero un equipo" in user
    assert "User is the person you are helping" in user


def test_build_messages_includes_the_profile_when_set() -> None:
    from polyglot.core.profile import UserProfile

    profile = UserProfile(
        name="Sam", role="Backend developer", hobbies=("cycling", "chess"), tone="polite"
    )
    c = SuggestionContext(ctx().target, (), "Spanish", "English", profile=profile)
    system = build_messages(c)[0]["content"]
    assert "About the user" in system
    assert "- Name: Sam" in system and "- Role: Backend developer" in system
    assert (
        "- Hobbies: cycling; chess" in system and "- Preferred tone for replies: polite" in system
    )
    assert "never state personal facts that are not listed" in system


def test_build_messages_without_a_profile_has_no_profile_section() -> None:
    assert "About the user" not in build_messages(ctx())[0]["content"]


def test_build_messages_marks_the_target_among_what_came_before_and_after() -> None:
    from polyglot.core.models import Speaker

    base = ctx()
    later = (
        Utterance(4, 0, 1, UtteranceStatus.FINAL, "Bueno, sigamos", "es", "Well, let's go on"),
        Utterance(5, 0, 1, UtteranceStatus.FINAL, "Me gusta", "es", speaker=Speaker.USER),
    )
    c = SuggestionContext(base.target, base.history, "Spanish", "English", following=later)
    _, user = (m["content"] for m in build_messages(c))
    lines = user.splitlines()
    assert lines[1:6] == [
        "    [1] Them: Hola  (Hello)",
        "    [2] Them: Bienvenido",
        ">>> [3] Them: ¿Qué opina?  (What do you think?)",
        "    [4] Them: Bueno, sigamos  (Well, let's go on)",
        "    [5] User: Me gusta",
    ]
    assert "Suggest replies to message [3]" in user
    assert "Messages after [3] are context only" in user
    assert "not repeat what the User has already said" in user


def test_build_messages_adds_the_users_instructions() -> None:
    base = ctx()
    assert "own instructions" not in build_messages(base)[0]["content"]
    c = SuggestionContext(
        base.target, base.history, "Spanish", "English",
        instructions="Don't ask questions back unless the conversation needs one.",
    )  # fmt: skip
    system = build_messages(c)[0]["content"]
    assert "The user's own instructions for the replies:\nDon't ask questions back" in system
    assert "except where they conflict with the rules above" in system
    assert system.index("own instructions") > system.index("Do not invent facts")
