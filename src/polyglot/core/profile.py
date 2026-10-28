"""What the user has chosen to tell the suggester about themselves.

The profile exists so replies can be about the user ("I have five years of backend
experience", "I like cycling") instead of generic. Everything in it is sent to the
suggestion provider on every request, so it is an **allowlist of public, shareable
details**: the sort of thing a person says out loud in an interview or at a meeting.

There is deliberately no field for health, politics, religion, finances, relationships,
or anything else private, and ``tests/unit/test_profile.py`` pins the field list so that
adding one is a conscious decision. The free-text fields can hold whatever the user
types; the settings dialog tells them what to leave out.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class UserProfile:
    name: str = ""  # what to be called
    role: str = ""  # current job title, or what they do
    location: str = ""  # city or country, as much as they would say out loud
    summary: str = ""  # a short "about me"
    goals: str = ""  # what they want from the conversation (e.g. "get this job")
    tone: str = ""  # how replies should sound (e.g. "polite, concise")
    languages: tuple[str, ...] = ()  # e.g. "English (native)", "Spanish (beginner)"
    skills: tuple[str, ...] = ()
    experience: tuple[str, ...] = ()  # short lines, e.g. "5 years of backend work at Acme"
    education: tuple[str, ...] = ()
    hobbies: tuple[str, ...] = ()
    interests: tuple[str, ...] = ()
    talking_points: tuple[str, ...] = ()  # other facts they are happy to mention

    @property
    def is_empty(self) -> bool:
        return not any(getattr(self, f.name) for f in fields(self))

    def facts(self) -> tuple[tuple[str, str], ...]:
        """(label, text) for every field that is set, in a fixed order. Lists are joined
        with "; ". This is what gets rendered into the prompt."""
        out: list[tuple[str, str]] = []
        for attr, label in _LABELS:
            value = getattr(self, attr)
            text = "; ".join(value) if isinstance(value, tuple) else value
            if text:
                out.append((label, text))
        return tuple(out)


_LABELS: tuple[tuple[str, str], ...] = (
    ("name", "Name"),
    ("role", "Role"),
    ("location", "Location"),
    ("summary", "About"),
    ("goals", "Goal"),
    ("tone", "Preferred tone for replies"),
    ("languages", "Languages"),
    ("skills", "Skills"),
    ("experience", "Experience"),
    ("education", "Education"),
    ("hobbies", "Hobbies"),
    ("interests", "Interests"),
    ("talking_points", "Also happy to mention"),
)
