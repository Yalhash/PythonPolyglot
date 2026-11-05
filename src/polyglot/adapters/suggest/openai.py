"""Reply suggestions with an OpenAI chat model, using structured (JSON-schema) output.

One parse path: the response must match the schema, or it is a ``BadResponse`` error.
There is no free-text fallback and no made-up reply on failure.
"""

from __future__ import annotations

import json
from typing import Any

from polyglot.adapters.http import HttpClient, HttpsClient
from polyglot.adapters.openai_models import chat_options
from polyglot.adapters.translate.openai import DEFAULT_BASE_URL, chat_text
from polyglot.core.errors import BadResponse
from polyglot.core.models import (
    ConversationType,
    ReplyOption,
    Speaker,
    SuggestionContext,
    Utterance,
)

SCHEMA: dict[str, Any] = {
    "name": "reply_options",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["options"],
        "properties": {
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source_text", "user_text"],
                    "properties": {
                        "source_text": {"type": "string"},
                        "user_text": {"type": "string"},
                    },
                },
            }
        },
    },
}

_SETTING = {
    ConversationType.INTERVIEW: "a job interview, where the user is the candidate",
    ConversationType.MEETING: "a work meeting",
    ConversationType.CASUAL: "a casual conversation",
}


_SPEAKER = {Speaker.OTHER: "Them", Speaker.USER: "User"}


def _line(n: int, u: Utterance, *, target: bool = False) -> str:
    marker = ">>>" if target else "   "
    text = f"{marker} [{n}] {_SPEAKER[u.speaker]}: {u.text}"
    return text + (f"  ({u.translation})" if u.translation else "")


def build_messages(ctx: SuggestionContext) -> list[dict[str, str]]:
    """The chat request. The conversation is one numbered transcript, oldest first, with
    the sentence to reply to marked ``>>>`` and named by its number, so the model knows
    which one to answer even when more was said after it."""
    convo = (*ctx.history, ctx.target, *ctx.following)
    n = len(ctx.history) + 1  # the target's number
    transcript = "\n".join(_line(i, u, target=i == n) for i, u in enumerate(convo, start=1))
    system = (
        f"You help the user reply in {ctx.source_language} during {_SETTING[ctx.conversation_type]}. "
        f"The user is more comfortable in {ctx.user_language}. Suggest {ctx.max_options} short, "
        "natural replies (one or two sentences each) to the one message you are asked about, "
        "each a different approach. For each, give `source_text` in "
        f"{ctx.source_language} (what to say) and `user_text` in {ctx.user_language} "
        "(what it means). Do not invent facts about the user; if the reply needs a fact you "
        "do not know, keep it general."
    )
    if facts := ctx.profile.facts():
        system += (
            "\n\nAbout the user (public details they chose to share):\n"
            + "\n".join(f"- {label}: {text}" for label, text in facts)
            + "\nUse these only where they genuinely fit what was said, and match the "
            "preferred tone if one is given. Do not bring up unrelated personal details, and "
            "never state personal facts that are not listed here."
        )
    user = (
        "Conversation (User is the person you are helping, Them is the other speaker; "
        "translations in parentheses):\n"
        f"{transcript}\n\n"
        f"Suggest replies to message [{n}] (marked >>>)."
    )
    if ctx.following:
        user += (
            f" Messages after [{n}] are context only: the conversation has moved on since "
            f"then, so the replies must answer [{n}] while still making sense now, and must "
            "not repeat what the User has already said."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_options(content: str, max_options: int) -> tuple[ReplyOption, ...]:
    try:
        data = json.loads(content)
        raw = data["options"]
        opts = tuple(
            ReplyOption(str(o["source_text"]).strip(), str(o["user_text"]).strip()) for o in raw
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BadResponse("suggestions did not match the expected format") from exc
    opts = tuple(o for o in opts if o.source_text)
    if not opts:
        raise BadResponse("no suggestions returned")
    return opts[:max_options]


class OpenAISuggester:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-6-luna",
        base_url: str = DEFAULT_BASE_URL,
        http: HttpClient | None = None,
    ) -> None:
        self._key, self.model, self._base = api_key, model, base_url
        self._http = http or HttpsClient()

    def suggest(self, context: SuggestionContext, *, timeout: float) -> tuple[ReplyOption, ...]:
        payload = {
            "model": self.model,
            # A little reasoning helps replies; more would not fit the 6 s budget.
            **chat_options(self.model, temperature=0.7, effort="low"),
            "messages": build_messages(context),
            "response_format": {"type": "json_schema", "json_schema": SCHEMA},
        }
        content = chat_text(
            self._http, api_key=self._key, base_url=self._base, payload=payload, timeout=timeout
        )
        return parse_options(content, context.max_options)
