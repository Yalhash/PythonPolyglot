from __future__ import annotations

from dataclasses import fields

from polyglot.core.profile import UserProfile

# The profile is sent to the suggestion provider with every request, so it is an
# allowlist of public, shareable details. If this test fails because you added a field,
# stop and decide whether it is something a person would say out loud in an interview.
# Health, politics, religion, finances and relationships do not belong here.
PUBLIC_FIELDS = {
    "name",
    "role",
    "location",
    "summary",
    "goals",
    "tone",
    "languages",
    "skills",
    "experience",
    "education",
    "hobbies",
    "interests",
    "talking_points",
}


def test_profile_fields_are_the_reviewed_public_allowlist() -> None:
    assert {f.name for f in fields(UserProfile)} == PUBLIC_FIELDS


def test_empty_profile() -> None:
    p = UserProfile()
    assert p.is_empty and p.facts() == ()
    assert not UserProfile(hobbies=("chess",)).is_empty
    assert not UserProfile(name="Sam").is_empty


def test_facts_are_labelled_ordered_and_skip_unset_fields() -> None:
    p = UserProfile(
        hobbies=("cycling", "chess"), name="Sam", role="Backend developer", tone="polite"
    )
    assert p.facts() == (
        ("Name", "Sam"),
        ("Role", "Backend developer"),
        ("Preferred tone for replies", "polite"),
        ("Hobbies", "cycling; chess"),
    )


def test_every_field_has_a_label() -> None:
    full = UserProfile(
        **{f.name: ("x",) if f.type != "str" else "x" for f in fields(UserProfile)}  # type: ignore[arg-type]
    )
    assert len(full.facts()) == len(PUBLIC_FIELDS)
