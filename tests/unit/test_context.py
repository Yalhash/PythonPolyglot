from __future__ import annotations

from polyglot.core.context import ContextBuilder
from polyglot.core.models import ConversationType, Speaker, Utterance, UtteranceStatus
from polyglot.core.profile import UserProfile
from polyglot.core.session import Session


def final(
    uid: int,
    text: str = "x",
    speaker: Speaker = Speaker.OTHER,
    translation: str | None = None,
    lang: str | None = "es",
) -> Utterance:
    return Utterance(
        id=uid,
        t_start=0.0,
        t_end=1.0,
        status=UtteranceStatus.FINAL,
        text=text,
        source_lang=lang,
        translation=translation,
        speaker=speaker,
    )


def session_with(*utterances: Utterance) -> Session:
    s = Session("en")
    for u in utterances:
        s.put(u)
    return s


def test_builds_languages_settings_and_profile() -> None:
    target = final(2, "¿Qué opina?")
    s = session_with(final(1, "Hola"), target)
    profile = UserProfile(name="Sam")
    ctx = ContextBuilder(
        conversation_type=ConversationType.INTERVIEW, max_options=3, profile=profile
    ).build(s, target)
    assert ctx.target is target
    assert [u.text for u in ctx.history] == ["Hola"]
    assert ctx.source_language == "Spanish" and ctx.user_language == "English"
    assert ctx.conversation_type is ConversationType.INTERVIEW and ctx.max_options == 3
    assert ctx.profile is profile


def test_history_includes_both_speakers_oldest_first() -> None:
    target = final(4, "¿Y usted?")
    s = session_with(
        final(1, "Cuénteme de usted"),
        final(2, "I led a team of five", Speaker.USER),
        final(3, "Interesante"),
        target,
    )
    ctx = ContextBuilder().build(s, target)
    assert [(u.id, u.speaker) for u in ctx.history] == [
        (1, Speaker.OTHER),
        (2, Speaker.USER),
        (3, Speaker.OTHER),
    ]


def test_history_is_capped_by_utterance_count() -> None:
    target = final(10)
    s = session_with(*[final(i, f"t{i}") for i in range(1, 10)], target)
    ctx = ContextBuilder(max_utterances=3).build(s, target)
    assert [u.text for u in ctx.history] == ["t7", "t8", "t9"]
    assert ContextBuilder(max_utterances=0).build(s, target).history == ()


def test_history_is_cut_from_the_oldest_end_by_the_character_budget() -> None:
    target = final(5)
    s = session_with(
        final(1, "a" * 100),
        final(2, "b" * 100),
        final(3, "c" * 60, translation="d" * 40),  # costs 100 with its translation
        final(4, "e" * 100),
        target,
    )
    ctx = ContextBuilder(max_utterances=10, max_chars=250).build(s, target)
    assert [u.id for u in ctx.history] == [3, 4]  # 100 + 100 fits, a third (100) does not
    ctx = ContextBuilder(max_utterances=10, max_chars=300).build(s, target)
    assert [u.id for u in ctx.history] == [2, 3, 4]


def test_the_most_recent_utterance_is_kept_even_if_over_budget() -> None:
    target = final(3)
    s = session_with(final(1, "a" * 50), final(2, "b" * 500), target)
    ctx = ContextBuilder(max_chars=200).build(s, target)
    assert [u.id for u in ctx.history] == [2]


def test_with_profile_returns_a_new_builder() -> None:
    a = ContextBuilder()
    b = a.with_profile(UserProfile(role="Engineer"))
    assert a.profile.is_empty and b.profile.role == "Engineer"
    assert b.max_utterances == a.max_utterances


def test_failed_and_unfinished_utterances_are_not_context() -> None:
    target = final(4)
    s = session_with(
        final(1, "ok"),
        final(2, "bad").evolve(status=UtteranceStatus.FAILED),
        Utterance(id=3, t_start=0, status=UtteranceStatus.PARTIAL, text="partial"),
        target,
    )
    assert [u.text for u in ContextBuilder().build(s, target).history] == ["ok"]
