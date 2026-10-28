from polyglot.ui.follow import FollowLock


def test_starts_locked() -> None:
    assert FollowLock().locked


def test_scrolling_unlocks_and_stays_unlocked() -> None:
    f = FollowLock()
    assert f.user_scrolled() is True  # this scroll released the lock
    assert not f.locked
    assert f.user_scrolled() is False  # already free: nothing changed
    assert not f.locked


def test_only_the_toggle_relocks() -> None:
    f = FollowLock()
    f.user_scrolled()
    assert f.toggle() is True and f.locked
    assert f.toggle() is False and not f.locked
