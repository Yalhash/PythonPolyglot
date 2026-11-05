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
    #: Soft cap on the characters of context (text plus translation), before and after
    #: the target together. The window is cut from the far ends, so the utterances
    #: nearest the target are kept first; the nearest one is always kept.
    max_chars: int = 3000
    #: Most utterances said after the target to include (when replying to an earlier
    #: sentence). 0 = only what came before.
    max_after: int = 4
    conversation_type: ConversationType = ConversationType.MEETING
    max_options: int = 2
    profile: UserProfile = field(default_factory=UserProfile)
    instructions: str = ""  # the user's extra instructions for replies

    def with_profile(self, profile: UserProfile) -> ContextBuilder:
        return replace(self, profile=profile)

    def with_instructions(self, instructions: str) -> ContextBuilder:
        return replace(self, instructions=instructions)

    def build(self, session: Session, target: Utterance) -> SuggestionContext:
        history, following = self._window(session, target)
        return SuggestionContext(
            target=target,
            history=history,
            following=following,
            source_language=languages.name(target.source_lang),
            user_language=languages.name(session.target_language),
            conversation_type=self.conversation_type,
            max_options=self.max_options,
            profile=self.profile,
            instructions=self.instructions,
        )

    def _window(
        self, session: Session, target: Utterance
    ) -> tuple[tuple[Utterance, ...], tuple[Utterance, ...]]:
        """Context on both sides of the target, nearest first: one step back, one step
        forward, and so on, until the character budget runs out on that side."""
        before = list(reversed(session.history_before(target.id, self.max_utterances)))
        after = list(session.history_after(target.id, self.max_after))
        kept_before: list[Utterance] = []
        kept_after: list[Utterance] = []
        used = 0
        open_before, open_after = True, True
        for step in range(max(len(before), len(after))):
            for side, kept, is_open in (
                (before, kept_before, open_before),
                (after, kept_after, open_after),
            ):
                if not is_open or step >= len(side):
                    continue
                u = side[step]
                cost = len(u.text) + len(u.translation or "")
                if (kept_before or kept_after) and used + cost > self.max_chars:
                    if side is before:
                        open_before = False
                    else:
                        open_after = False
                    continue
                kept.append(u)
                used += cost
        return tuple(reversed(kept_before)), tuple(kept_after)
