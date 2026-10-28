"""The one thread bridge: core events (any thread) -> presenter (GUI thread)."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from polyglot.core.events import Event, EventBus


class EventBridge(QObject):
    """Subscribes to the bus and re-emits each event as a queued Qt signal.

    Create it on the GUI thread: the connection then delivers to ``handler`` on
    the GUI thread no matter which pipeline thread published the event.
    """

    _event = pyqtSignal(object)

    def __init__(self, bus: EventBus, handler: Callable[[Event], None]) -> None:
        super().__init__()
        self._handler = handler
        # AutoConnection: queued when emitted from a pipeline thread (the receiver lives
        # on the GUI thread), direct when emitted on the GUI thread itself.
        self._event.connect(self._deliver)
        self._unsubscribe = bus.subscribe(self._event.emit)

    @pyqtSlot(object)
    def _deliver(self, event: Event) -> None:
        self._handler(event)

    def close(self) -> None:
        self._unsubscribe()
