from __future__ import annotations

import pytest

from polyglot.core import languages
from polyglot.core.models import IdSource, Speaker, Utterance, UtteranceStatus
from polyglot.core.session import LanguagePolicy, Session


def final(uid: int, text: str = "x") -> Utterance:
    return Utterance(id=uid, t_start=0.0, t_end=1.0, status=UtteranceStatus.FINAL, text=text)


def test_latest_final_tracks_highest_final_id() -> None:
    s = Session("en")
    s.put(Utterance(id=1, t_start=0))
    assert s.latest_final_id is None
    s.put(final(2))
    s.put(final(1))  # an earlier utterance finishing late does not move "latest" back
    assert s.latest_final_id == 2
    s.remove(2)
    assert s.latest_final_id == 1
    s.remove(1)
    assert s.latest_final_id is None


def test_history_before_excludes_failed_partial_and_later() -> None:
    s = Session("en")
    for uid in range(1, 6):
        s.put(final(uid, f"t{uid}"))
    s.put(Utterance(id=6, t_start=0, status=UtteranceStatus.PARTIAL, text="p"))
    s.put(final(3).evolve(status=UtteranceStatus.FAILED))
    hist = s.history_before(5, limit=3)
    assert [u.id for u in hist] == [1, 2, 4][-3:]
    assert s.history_before(5, limit=0) == ()
    assert [u.id for u in s.ordered()] == [1, 2, 3, 4, 5, 6]


def test_language_locks_after_agreeing_confident_finals() -> None:
    s = Session("en", LanguagePolicy(agree=2, min_probability=0.8))
    assert not s.observe_language("es", 0.95)
    assert not s.observe_language("fr", 0.95)  # disagreement restarts the streak
    assert not s.observe_language("es", 0.95)
    assert s.observe_language("es", 0.9)
    assert s.locked_language == "es"
    assert not s.observe_language("fr", 0.99)  # locked: ignored
    assert s.locked_language == "es"


def test_low_confidence_breaks_the_streak() -> None:
    s = Session("en", LanguagePolicy(agree=2, min_probability=0.8))
    s.observe_language("es", 0.95)
    s.observe_language("es", 0.5)
    assert not s.observe_language("es", 0.95)
    assert s.observe_language("es", 0.95)


def test_manual_and_forced_language() -> None:
    s = Session("en", LanguagePolicy(forced="de"))
    assert s.locked_language == "de" and s.language_is_manual
    s.set_language(None)
    assert s.locked_language is None and not s.language_is_manual
    s.set_language("it")
    assert s.locked_language == "it"


def test_clear() -> None:
    s = Session("en")
    s.put(final(1))
    s.clear()
    assert s.utterances == {} and s.latest_final_id is None


def test_id_source_is_monotonic() -> None:
    ids = IdSource()
    assert [ids(), ids(), ids()] == [1, 2, 3]
    assert IdSource(10)() == 10


def test_utterance_is_final_property() -> None:
    assert not Utterance(1, 0).is_final
    assert final(1).is_final
    assert final(1).evolve(status=UtteranceStatus.TRANSLATED).is_final
    assert Utterance(1, 0, t_end=1, status=UtteranceStatus.FAILED).is_final
    assert not Utterance(1, 0, status=UtteranceStatus.FAILED).is_final


@pytest.mark.parametrize(
    ("code", "name"), [("es", "Spanish"), ("pt-BR", "Portuguese"), ("zz", "zz"), (None, "Unknown")]
)
def test_language_names(code: str | None, name: str) -> None:
    assert languages.name(code) == name


def test_same_language() -> None:
    assert languages.same_language("en", "en-US")
    assert languages.same_language("pt_BR", "pt")
    assert not languages.same_language("en", "es")
    assert not languages.same_language(None, None)
    assert "es" in languages.known_codes()


def test_utterances_default_to_the_other_speaker() -> None:
    assert Utterance(1, 0).speaker is Speaker.OTHER
    assert final(1).evolve(speaker=Speaker.USER).speaker is Speaker.USER


def test_latest_answerable_ignores_the_users_own_turns() -> None:
    s = Session("en")
    s.put(final(1, "their question"))
    s.put(final(2, "my answer").evolve(speaker=Speaker.USER))
    target = s.latest_answerable()
    assert target is not None and target.id == 1
    s.remove(1)
    assert s.latest_answerable() is None  # only the user has spoken


def test_history_before_includes_the_users_turns() -> None:
    s = Session("en")
    s.put(final(1, "q1"))
    s.put(final(2, "a1").evolve(speaker=Speaker.USER))
    s.put(final(3, "q2"))
    assert [(u.id, u.speaker) for u in s.history_before(3, limit=5)] == [
        (1, Speaker.OTHER),
        (2, Speaker.USER),
    ]


def test_id_source_hands_out_unique_ids_to_concurrent_streams() -> None:
    import threading

    ids = IdSource()
    out: list[list[int]] = [[] for _ in range(8)]

    def draw(bucket: list[int]) -> None:
        bucket.extend(ids() for _ in range(500))

    threads = [threading.Thread(target=draw, args=(b,)) for b in out]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    drawn = [i for bucket in out for i in bucket]
    assert len(drawn) == len(set(drawn)) == 4000
    assert all(bucket == sorted(bucket) for bucket in out)  # each stream sees increasing ids
