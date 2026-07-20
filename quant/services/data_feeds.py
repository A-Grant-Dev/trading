"""
Live Data Feed Service

Manages WebSocket connections to Binance for real-time market data.
Each stream runs in its own daemon thread and writes directly to the
appropriate model (MarketData, OrderBookSnapshot, TradeRecord, etc.).

Streams managed:
  - <symbol>@kline_<interval> → MarketData (live candles)
  - <symbol>@depth20@100ms → OrderBookSnapshot (depth snapshots)
  - <symbol>@aggTrade → TradeRecord (every trade)
  - !miniTicker@arr → Price cache (all tickers, lightweight)

Architecture:
  Each WebSocket runs in a daemon thread with auto-reconnect.
  A thread-safe cache is used for passing latest data to the Django views.

Renaissance principle: Data must flow continuously and reliably.
If the data feed stops, the models are blind.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable

import requests
from websocket import WebSocketApp

logger = logging.getLogger(__name__)

BINANCE_WSS = "wss://stream.binance.com:9443/ws"
BINANCE_API = "https://api.binance.com"


# ── Thread-Safe Cache ──────────────────────────────────────────────

_cache: dict[str, any] = {}
_cache_lock = threading.Lock()


def set_cache(key: str, value):
    with _cache_lock:
        _cache[key] = value


def get_cache(key: str, default=None):
    with _cache_lock:
        return _cache.get(key, default)


def pop_cache(key: str, default=None):
    with _cache_lock:
        return _cache.pop(key, default)


# ── WebSocket Thread Base ──────────────────────────────────────────


class WebSocketThread(threading.Thread):
    """
    Base class for WebSocket data feed threads.

    Handles connection, reconnection, and message dispatch.
    Subclasses override _on_message() to process specific stream types.
    """

    STREAM_URL = BINANCE_WSS

    def __init__(self, stream_name: str, name: str = None):
        super().__init__(daemon=True)
        self.stream_name = stream_name
        self._stop_event = threading.Event()
        self.callbacks: list[Callable] = []

    def run(self):
        """Main loop — connect and listen with auto-reconnect."""
        while not self._stop_event.is_set():
            try:
                self._connect_and_listen()
            except Exception as e:
                logger.warning(f"WebSocket {self.stream_name} error: {e}. Reconnecting in 3s...")
                self._stop_event.wait(3)

    def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages."""

        url = f"{self.STREAM_URL}/{self.stream_name}"

        def on_message(ws, raw_message):
            if self._stop_event.is_set():
                ws.close()
                return
            try:
                data = json.loads(raw_message)
                self._on_message(data)
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(f"Error processing {self.stream_name} msg: {e}")

        def on_error(ws, error):
            logger.warning(f"WebSocket {self.stream_name} error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.debug(f"WebSocket {self.stream_name} closed")

        ws = WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        # Run in a nested thread so the main reconnect loop stays responsive
        ws.run_forever(ping_interval=30, ping_timeout=10)

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def _on_message(self, data: dict):
        """Override in subclasses to handle specific stream data."""
        raise NotImplementedError

    def add_callback(self, callback: Callable):
        """Add a callback that receives parsed data on each message."""
        self.callbacks.append(callback)


# ── Kline Stream ───────────────────────────────────────────────────


class KlineStream(WebSocketThread):
    """
    Websocket stream: <symbol>@kline_<interval>

    Receives live candle updates and writes them to MarketData.
    The candle is only saved to the database when it closes.
    While open, it's stored in the live cache for current view display.
    """

    def __init__(self, symbol: str, interval: str = "1m"):
        stream_name = f"{symbol.lower()}@kline_{interval}"
        super().__init__(stream_name, name=f"kline-{symbol}-{interval}")
        self.symbol = symbol.upper()
        self.interval = interval

    def _on_message(self, data: dict):
        k = data.get("k", {})
        if not k:
            return

        is_closed = k.get("x", False)
        open_time = datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc)
        close_price = Decimal(str(k["c"]))
        volume = Decimal(str(k["v"]))

        # Store latest price in cache
        set_cache(f"price:{self.symbol}", {
            "price": float(close_price),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Store live candle in cache
        set_cache(f"live_candle:{self.symbol}:{self.interval}", {
            "open_time": open_time.isoformat(),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(close_price),
            "volume": float(volume),
            "is_closed": is_closed,
        })

        # Only write to DB when candle closes
        if is_closed:
            from quant.models import MarketData

            try:
                MarketData.objects.update_or_create(
                    symbol=self.symbol,
                    interval=self.interval,
                    open_time=open_time,
                    defaults={
                        "open": Decimal(str(k["o"])),
                        "high": Decimal(str(k["h"])),
                        "low": Decimal(str(k["l"])),
                        "close": close_price,
                        "volume": volume,
                        "quote_asset_volume": Decimal(str(k.get("q", 0))),
                        "taker_buy_base_vol": Decimal(str(k.get("V", 0))),
                        "taker_buy_quote_vol": Decimal(str(k.get("Q", 0))),
                        "trades": int(k.get("n", 0)),
                        "closed": True,
                    },
                )
            except Exception as e:
                logger.error(f"Failed to save kline for {self.symbol}: {e}")

        # Notify callbacks
        for cb in self.callbacks:
            try:
                cb(self.symbol, k)
            except Exception:
                pass


# ── Depth Stream ───────────────────────────────────────────────────


class DepthStream(WebSocketThread):
    """
    WebSocket stream: <symbol>@depth20@100ms

    Receives order book depth snapshots and writes to OrderBookSnapshot.
    Also stores in cache for the live order book view.
    """

    def __init__(self, symbol: str):
        stream_name = f"{symbol.lower()}@depth20@100ms"
        super().__init__(stream_name, name=f"depth-{symbol}")
        self.symbol = symbol.upper()

    def _on_message(self, data: dict):
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if not bids or not asks:
            return

        # Parse bids and asks
        bids_parsed = [[Decimal(b[0]), Decimal(b[1])] for b in bids]
        asks_parsed = [[Decimal(a[0]), Decimal(a[1])] for a in asks]

        bid_vol = sum(b[1] for b in bids_parsed)
        ask_vol = sum(a[1] for a in asks_parsed)
        total_vol = bid_vol + ask_vol

        imbalance = float(bid_vol / total_vol * 100) if total_vol > 0 else 50.0

        # Store in cache for live views
        set_cache(f"depth:{self.symbol}", {
            "bids": [[float(b[0]), float(b[1])] for b in bids_parsed],
            "asks": [[float(a[0]), float(a[1])] for a in asks_parsed],
            "bid_vol": float(bid_vol),
            "ask_vol": float(ask_vol),
            "imbalance": imbalance,
            "spread": float(asks_parsed[0][0] - bids_parsed[0][0]) if asks_parsed and bids_parsed else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Periodically write to DB (every ~10 seconds by sampling)
        timestamp = datetime.now(timezone.utc)
        if int(timestamp.timestamp()) % 10 < 1:  # ~once per 10 seconds
            try:
                from quant.models import OrderBookSnapshot

                OrderBookSnapshot.objects.create(
                    symbol=self.symbol,
                    timestamp=timestamp,
                    bids_json=[[str(b[0]), str(b[1])] for b in bids_parsed],
                    asks_json=[[str(a[0]), str(a[1])] for a in asks_parsed],
                    bid_vol=bid_vol,
                    ask_vol=ask_vol,
                    spread=asks_parsed[0][0] - bids_parsed[0][0],
                    imbalance_pct=imbalance,
                    first_bid_price=bids_parsed[0][0],
                    first_ask_price=asks_parsed[0][0],
                )
            except Exception as e:
                logger.error(f"Failed to save depth snapshot for {self.symbol}: {e}")

        for cb in self.callbacks:
            try:
                cb(self.symbol, {"bids": bids_parsed, "asks": asks_parsed, "imbalance": imbalance})
            except Exception:
                pass


# ── Trade Stream ───────────────────────────────────────────────────


class TradeStream(WebSocketThread):
    """
    WebSocket stream: <symbol>@aggTrade

    Receives every individual trade and writes to TradeRecord.
    """

    def __init__(self, symbol: str):
        stream_name = f"{symbol.lower()}@aggTrade"
        super().__init__(stream_name, name=f"trade-{symbol}")
        self.symbol = symbol.upper()
        self._batch_buffer: list = []
        self._last_flush = time.time()

    def _on_message(self, data: dict):
        trade = {
            "symbol": self.symbol,
            "trade_id": data.get("a"),
            "price": Decimal(str(data["p"])),
            "qty": Decimal(str(data["q"])),
            "quote_qty": Decimal(str(data.get("q", 0))) * Decimal(str(data.get("p", 0))),
            "is_buyer_maker": data.get("m", False),
            "is_best_match": data.get("M", None),
            "time": datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc),
        }

        # Cache the last trade
        set_cache(f"last_trade:{self.symbol}", {
            "price": float(trade["price"]),
            "qty": float(trade["qty"]),
            "time": trade["time"].isoformat(),
            "is_buyer_maker": trade["is_buyer_maker"],
        })

        # Update price cache
        set_cache(f"price:{self.symbol}", {
            "price": float(trade["price"]),
            "timestamp": trade["time"].isoformat(),
        })

        # Buffer for batch DB writes
        self._batch_buffer.append(trade)
        now = time.time()

        # Flush to DB every 5 seconds or every 50 trades
        if len(self._batch_buffer) >= 50 or (now - self._last_flush) > 5:
            self._flush_buffer()

        for cb in self.callbacks:
            try:
                cb(self.symbol, trade)
            except Exception:
                pass

    def _flush_buffer(self):
        """Batch insert buffered trades."""
        if not self._batch_buffer:
            return

        from quant.models import TradeRecord

        objects = []
        for t in self._batch_buffer:
            objects.append(TradeRecord(**t))

        try:
            TradeRecord.objects.bulk_create(objects, ignore_conflicts=True, batch_size=200)
        except Exception as e:
            logger.error(f"Failed to batch insert trades for {self.symbol}: {e}")

        self._batch_buffer.clear()
        self._last_flush = time.time()


# ── Feed Manager ───────────────────────────────────────────────────


class DataFeedManager:
    """
    Manages all WebSocket data feed threads for tracked symbols.

    Provides a simple API to subscribe/unsubscribe streams and
    access the latest cached data.

    Usage:
        manager = DataFeedManager.get_instance()
        manager.subscribe("BTCUSDT")
        latest_price = manager.get_price("BTCUSDT")
        manager.unsubscribe("BTCUSDT")
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.streams: dict[str, WebSocketThread] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "DataFeedManager":
        """Singleton accessor."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def subscribe(self, symbol: str, interval: str = "1m") -> None:
        """
        Subscribe to all data streams for a symbol.

        Starts kline, depth, and trade threads.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Kline interval (default: 1m)
        """
        symbol = symbol.upper()
        if symbol in self.streams:
            return  # Already subscribed

        with self._lock:
            threads = [
                KlineStream(symbol, interval),
                DepthStream(symbol),
                TradeStream(symbol),
            ]
            for t in threads:
                t.start()
                self.streams[f"{symbol}:{t.__class__.__name__}"] = t

            logger.info(f"Subscribed to data feeds for {symbol}")

    def unsubscribe(self, symbol: str) -> None:
        """
        Unsubscribe all streams for a symbol.

        Args:
            symbol: Trading pair
        """
        symbol = symbol.upper()
        with self._lock:
            keys_to_remove = [k for k in self.streams if k.startswith(f"{symbol}:")]
            for key in keys_to_remove:
                thread = self.streams.pop(key)
                thread.stop()
            logger.info(f"Unsubscribed from data feeds for {symbol}")

    def get_price(self, symbol: str) -> dict | None:
        """Get the latest cached price for a symbol."""
        return get_cache(f"price:{symbol.upper()}")

    def get_live_candle(self, symbol: str, interval: str = "1m") -> dict | None:
        """Get the current (possibly unclosed) candle."""
        return get_cache(f"live_candle:{symbol.upper()}:{interval}")

    def get_depth(self, symbol: str) -> dict | None:
        """Get the latest cached order book depth."""
        return get_cache(f"depth:{symbol.upper()}")

    def get_last_trade(self, symbol: str) -> dict | None:
        """Get the most recent trade."""
        return get_cache(f"last_trade:{symbol.upper()}")

    def stop_all(self) -> None:
        """Stop all data feed threads."""
        for key in list(self.streams.keys()):
            self.unsubscribe(key.split(":")[0])
        logger.info("All data feeds stopped")


# ── Symbol Discovery ───────────────────────────────────────────────


def get_tradable_symbols(quote_asset: str = "USDT") -> list[str]:
    """
    Fetch all currently tradable USDT pairs from Binance.

    Uses the exchangeInfo endpoint (cached, updated every 5 minutes).

    Returns:
        List of symbol strings (e.g., ['BTCUSDT', 'ETHUSDT', ...])
    """
    cache_key = f"tradable_symbols_{quote_asset}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    try:
        resp = requests.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        symbols = []
        for s in data.get("symbols", []):
            if s.get("status") == "TRADING" and s.get("quoteAsset") == quote_asset:
                symbols.append(s["symbol"])

        set_cache(cache_key, symbols)
        return symbols

    except Exception as e:
        logger.error(f"Failed to fetch tradable symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # Fallback defaults
