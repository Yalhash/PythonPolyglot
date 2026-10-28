"""Whether the transcript follows the newest sentence.

Locked (the default): the view stays on the latest utterance as text arrives. Scrolling
by hand unlocks it, so what the user is reading never jumps away. It locks again only on
request: the lock button or its hotkey. Pure Python so the rule can be tested without Qt.
"""

from __future__ import annotations


class FollowLock:
    def __init__(self, locked: bool = True) -> None:
        self.locked = locked

    def user_scrolled(self) -> bool:
        """The user moved the view. Returns True if that released the lock."""
        was, self.locked = self.locked, False
        return was

    def toggle(self) -> bool:
        self.locked = not self.locked
        return self.locked
