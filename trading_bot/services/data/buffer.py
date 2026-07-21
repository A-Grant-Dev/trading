"""
Live Data Buffer — In-memory ring buffer for real-time market data.

Stores the most recent N data points from Binance WebSocket streams
in memory for fast access. Older data is periodically persisted to
the database.

This serves as a Redis replacement for local-only operation.
When Redis is available, this should be replaced with Redis Streams.

Data flow:
  Binance WS → LiveDataBuffer (in-memory) → periodic DB flush
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LiveDataBuffer:
    """
    Thread-safe in-memory ring buffer for live market data.

    Stores the most recent N entries per symbol per data type.
    Provides fast lookups for dashboards and strategy signals.

    Data types:
      - 'trade': Recent trades (aggTrade)
      - 'depth': Latest order book snapshot
      - 'kline': Latest kline (candle) update
      - 'funding': Latest funding rate / mark price
      - 'ticker': Latest 24hr ticker stats
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._lock = threading.Lock()

        # Deques per symbol per data type
        self._buffers: dict[str, dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_size))
        )

        # Latest snapshot per symbol per data type (most recent state)
        self._latest: dict[str, dict[str, Any]] = defaultdict(dict)

        # Statistics
        self._stats: dict[str, Any] = {
            "total_messages": 0,
            "started_at": None,
            "symbols": set(),
            "data_types": set(),
            "errors": 0,
        }

        # Callbacks for new data
        self._callbacks: list[callable] = []

    # ── Data Ingestion ─────────────────────────────────────────────

    def push(self, data_type: str, symbol: str, data: dict[str, Any]) -> None:
        """
        Push a new data point into the buffer.

        Args:
            data_type: One of 'trade', 'depth', 'kline', 'funding', 'ticker'
            symbol: Trading pair (e.g., BTCUSDT)
            data: Data dictionary from WebSocket stream
        """
        with self._lock:
            self._buffers[symbol][data_type].append(data)
            self._latest[symbol][data_type] = data
            self._stats["total_messages"] += 1
            self._stats["symbols"].add(symbol)
            self._stats["data_types"].add(data_type)
            if self._stats["started_at"] is None:
                self._stats["started_at"] = datetime.now(timezone.utc)

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(data_type, symbol, data)
            except Exception as e:
                logger.error("Callback error: %s", e)

    def push_error(self, error_msg: str) -> None:
        """Record a stream error."""
        with self._lock:
            self._stats["errors"] += 1
        logger.error("Live stream error: %s", error_msg)

    # ── Data Access ────────────────────────────────────────────────

    def get_latest(self, symbol: str, data_type: str) -> Optional[dict[str, Any]]:
        """Get the latest data point for a symbol + data type."""
        with self._lock:
            return self._latest.get(symbol, {}).get(data_type)

    def get_recent(self, symbol: str, data_type: str, n: int = 10) -> list[dict[str, Any]]:
        """Get the N most recent data points for a symbol + data type."""
        with self._lock:
            buf = self._buffers.get(symbol, {}).get(data_type, deque())
            return list(buf)[-n:]

    def get_all_latest(self) -> dict[str, dict[str, Any]]:
        """Get latest snapshot for every symbol × data type."""
        with self._lock:
            return dict(self._latest)

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "total_messages": self._stats["total_messages"],
                "started_at": self._stats["started_at"].isoformat() if self._stats["started_at"] else None,
                "symbols": sorted(self._stats["symbols"]),
                "data_types": sorted(self._stats["data_types"]),
                "errors": self._stats["errors"],
                "uptime_seconds": (
                    (datetime.now(timezone.utc) - self._stats["started_at"]).total_seconds()
                    if self._stats["started_at"] else 0
                ),
            }

    def get_orderbook(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get latest order book snapshot for a symbol."""
        return self.get_latest(symbol, "depth")

    def get_trades(self, symbol: str, n: int = 20) -> list[dict[str, Any]]:
        """Get recent trades for a symbol."""
        return self.get_recent(symbol, "trade", n)

    def get_funding(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get latest funding rate / mark price for a symbol."""
        return self.get_latest(symbol, "funding")

    # ── Callbacks ──────────────────────────────────────────────────

    def add_callback(self, callback: callable) -> None:
        """Register a callback function called on each new data point."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: callable) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    # ── Lifecycle ──────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all buffers."""
        with self._lock:
            self._buffers.clear()
            self._latest.clear()
            self._stats["total_messages"] = 0
            self._stats["errors"] = 0
            self._stats["symbols"] = set()
            self._stats["data_types"] = set()

    def flush_to_db(self) -> int:
        """
        Persist buffered data to database models.

        Currently a no-op — data is stored incrementally in the
        consumer instead of batched here. This method exists for
        future Redis Streams integration where batch flushing
        is more efficient.

        Returns:
            Number of rows flushed
        """
        # TODO: Implement batch DB flush when using Redis Streams
        return 0


# ── Module-level singleton ─────────────────────────────────────────

_buffer: Optional[LiveDataBuffer] = None
_buffer_lock = threading.Lock()


def get_buffer() -> LiveDataBuffer:
    """Get or create the global live data buffer singleton."""
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                _buffer = LiveDataBuffer()
                logger.info("Created global LiveDataBuffer")
    return _buffer


def reset_buffer() -> None:
    """Reset the global buffer (for testing)."""
    global _buffer
    with _buffer_lock:
        _buffer = None
