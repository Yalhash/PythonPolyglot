"""Conversation state. Owned and mutated by the coordinator thread only.

Other threads receive immutable ``Utterance`` snapshots through events, so nothing here
needs a lock.
"""

from __future__ import annotations

from dataclasses import dataclass

from polyglot.core.models import Speaker, Utterance, UtteranceStatus


@dataclass(frozen=True)
class LanguagePolicy:
    """Lock the source language once ``agree`` consecutive finals agree confidently."""

    agree: int = 2
    min_probability: float = 0.8
    forced: str | None = None  # a language set in config; never auto-changed


class Session:
    def __init__(self, target_language: str, policy: LanguagePolicy | None = None) -> None:
        self.target_language = target_language
        self.policy = policy or LanguagePolicy()
        self.utterances: dict[int, Utterance] = {}
        self.latest_final_id: int | None = None
        self.locked_language: str | None = self.policy.forced
        self._manual_lock = self.policy.forced is not None
        self._streak_lang: str | None = None
        self._streak = 0

    # ------------------------------------------------------------- utterances
    def get(self, utterance_id: int) -> Utterance | None:
        return self.utterances.get(utterance_id)

    def put(self, utterance: Utterance) -> Utterance:
        self.utterances[utterance.id] = utterance
        if utterance.is_final and (
            self.latest_final_id is None or utterance.id > self.latest_final_id
        ):
            self.latest_final_id = utterance.id
        return utterance

    def remove(self, utterance_id: int) -> None:
        self.utterances.pop(utterance_id, None)
        if self.latest_final_id == utterance_id:
            finals = [u.id for u in self.utterances.values() if u.is_final]
            self.latest_final_id = max(finals) if finals else None

    def latest_answerable(self) -> Utterance | None:
        """The most recent final utterance by the other person that has text (failed ones
        are skipped). The user's own turns are never something to reply to."""
        for u in reversed(self.ordered()):
            if (
                u.speaker is Speaker.OTHER
                and u.is_final
                and u.text
                and u.status is not UtteranceStatus.FAILED
            ):
                return u
        return None

    def ordered(self) -> list[Utterance]:
        return [self.utterances[k] for k in sorted(self.utterances)]

    def history_before(self, utterance_id: int, limit: int) -> tuple[Utterance, ...]:
        """Final utterances with text, spoken before ``utterance_id``, oldest first. Both
        speakers: what the user said is context for what to say next."""
        earlier = [
            u
            for u in self.ordered()
            if u.id < utterance_id
            and u.is_final
            and u.text
            and u.status is not UtteranceStatus.FAILED
        ]
        return tuple(earlier[-limit:]) if limit > 0 else ()

    def clear(self) -> None:
        self.utterances.clear()
        self.latest_final_id = None

    # --------------------------------------------------------------- language
    def observe_language(self, language: str | None, probability: float) -> bool:
        """Feed the detected language of a final transcript. Returns True if it locked."""
        if self.locked_language is not None or not language:
            return False
        if probability < self.policy.min_probability:
            self._streak_lang, self._streak = None, 0
            return False
        if language == self._streak_lang:
            self._streak += 1
        else:
            self._streak_lang, self._streak = language, 1
        if self._streak >= self.policy.agree:
            self.locked_language = language
            return True
        return False

    def set_language(self, language: str | None) -> None:
        """Manual override. ``None`` unlocks and restarts auto-detection."""
        self.locked_language = language
        self._manual_lock = language is not None
        self._streak_lang, self._streak = None, 0

    @property
    def language_is_manual(self) -> bool:
        return self._manual_lock
