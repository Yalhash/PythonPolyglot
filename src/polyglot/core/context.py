"""Decides what the suggester gets to see.

``ContextBuilder`` is the one place that turns the conversation (``Session``) plus what we
know about the user (``UserProfile``) into a ``SuggestionContext``. The window policy
lives here, not in the coordinator and not in a provider adapter, so a different
provider, a longer window, or a rolling summary later changes this module only.

It is pure and immutable. The coordinator thread owns the ``Session`` and calls
``build``; the result is a snapshot that is safe to hand to a network thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from polyglot.core import languages
from polyglot.core.models import ConversationType, SuggestionContext, Utterance
from polyglot.core.profile import UserProfile
from polyglot.core.session import Session


@dataclass(frozen=True)
class ContextBuilder:
    #: Most utterances of earlier conversation to include (either speaker).
    max_utterances: int = 6
    #: Soft cap on the characters of history (text plus translation). The window is cut
    #: from the oldest end; the most recent utterance is always kept.
    max_chars: int = 3000
    conversation_type: ConversationType = ConversationType.MEETING
    max_options: int = 2
    profile: UserProfile = field(default_factory=UserProfile)

    def with_profile(self, profile: UserProfile) -> ContextBuilder:
        return replace(self, profile=profile)

    def build(self, session: Session, target: Utterance) -> SuggestionContext:
        return SuggestionContext(
            target=target,
            history=self._history(session, target),
            source_language=languages.name(target.source_lang),
            user_language=languages.name(session.target_language),
            conversation_type=self.conversation_type,
            max_options=self.max_options,
            profile=self.profile,
        )

    def _history(self, session: Session, target: Utterance) -> tuple[Utterance, ...]:
        recent = session.history_before(target.id, self.max_utterances)
        kept: list[Utterance] = []
        used = 0
        for u in reversed(recent):  # newest first: the nearest context matters most
            cost = len(u.text) + len(u.translation or "")
            if kept and used + cost > self.max_chars:
                break
            kept.append(u)
            used += cost
        return tuple(reversed(kept))
