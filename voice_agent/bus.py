from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Event:
    type: str
    payload: dict[str, Any]


class EventBus:
    def __init__(self) -> None:
        self._queue: queue.Queue[Event] = queue.Queue()
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()

    def publish(self, event: Event) -> None:
        self._queue.put(event)

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)

    def poll(self, timeout: float = 0.1) -> Event | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def dispatch(self, event: Event) -> None:
        handlers: list[Callable[[Event], None]] = []
        with self._lock:
            handlers = list(self._subscribers.get(event.type, []))
            handlers += self._subscribers.get("*", [])
        for handler in handlers:
            handler(event)
