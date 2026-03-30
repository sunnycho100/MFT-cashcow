"""Starter signal transport layer for v3."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Protocol

from ..strategy.types import SignalEnvelope


class SignalBus(Protocol):
    """Minimal interface for a signal transport."""

    def publish(self, envelope: SignalEnvelope) -> None:
        """Store a signal envelope for later consumption."""

    def drain(self) -> Iterable[SignalEnvelope]:
        """Yield all currently buffered signal envelopes."""


class InMemorySignalBus:
    """Simple local queue used before we switch to Redis or NATS."""

    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self._queue: Deque[SignalEnvelope] = deque()

    def publish(self, envelope: SignalEnvelope) -> None:
        """Append a new signal and drop the oldest one if the queue is full."""
        if len(self._queue) >= self.max_size:
            self._queue.popleft()
        self._queue.append(envelope)

    def drain(self) -> Iterable[SignalEnvelope]:
        """Yield all currently queued signals in FIFO order."""
        while self._queue:
            yield self._queue.popleft()
