"""
Autonomous Trading Bot — Binance WebSocket Consumer

Connects to Binance WebSocket streams to ingest real-time market data:
- Order book depth (depth20@100ms)
- Aggregated trades (aggTrade)
- Kline/candle updates (kline_1m)
- Mark price & funding rate (markPrice@1s)

Data flows: Binance WS → Consumer → LiveDataBuffer → Dashboard/Strategies

Uses Django Channels for the WebSocket interface to the frontend
and asyncio/websockets for the Binance connection.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import websockets
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from trading_bot.services.data.buffer import get_buffer

logger = logging.getLogger(__name__)

# ── Binance Stream URL Templates ──────────────────────────────────

SPOT_STREAM = "wss://stream.binance.com:9443/ws"
FUTURES_STREAM = "wss://fstream.binance.com/ws"


def _build_stream_name(symbol: str, stream_type: str) -> str:
    """Build the full stream name for a Binance WebSocket subscription.

    Args:
        symbol: Trading pair lowercase (e.g., 'btcusdt')
        stream_type: Stream type (depth20, aggTrade, kline_1m, markPrice)
    """
    return f"{symbol.lower()}@{stream_type}"


# ── Data Model Mappers ─────────────────────────────────────────────


def _map_depth(data: dict) -> dict:
    """Map Binance depthUpdate event to standardized format (b/a format)."""
    bids = [[float(p), float(q)] for p, q in data.get("b", [])]
    asks = [[float(p), float(q)] for p, q in data.get("a", [])]

    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    total_vol = bid_vol + ask_vol

    best_bid = bids[0][0] if bids else 0
    best_ask = asks[0][0] if asks else 0
    spread = best_ask - best_bid if best_ask and best_bid else 0

    return {
        "symbol": data.get("s", ""),
        "timestamp": datetime.fromtimestamp(data.get("E", 0) / 1000, tz=timezone.utc).isoformat(),
        "bids": bids,
        "asks": asks,
        "bid_volume": bid_vol,
        "ask_volume": ask_vol,
        "spread": spread,
        "spread_pct": (spread / best_ask * 100) if best_ask > 0 else 0,
        "imbalance_pct": (bid_vol / total_vol * 100) if total_vol > 0 else 50.0,
        "depth_pressure": (
            (bid_vol - ask_vol) / (bid_vol + ask_vol)
            if (bid_vol + ask_vol) > 0 else 0
        ),
        "first_bid": best_bid,
        "first_ask": best_ask,
        "type": "depth",
    }


def _map_trade(data: dict) -> dict:
    """Map Binance aggTrade event to standardized format."""
    return {
        "symbol": data.get("s", ""),
        "trade_id": data.get("t", 0),
        "price": float(data.get("p", 0)),
        "quantity": float(data.get("q", 0)),
        "quote_quantity": float(data.get("q", 0)) * float(data.get("p", 0)),
        "time": datetime.fromtimestamp(data.get("T", 0) / 1000, tz=timezone.utc).isoformat(),
        "is_buyer_maker": data.get("m", True),
        "is_best_match": data.get("M", True),
        "type": "trade",
    }


def _map_kline(data: dict) -> dict:
    """Map Binance kline event to standardized format."""
    k = data.get("k", {})
    return {
        "symbol": data.get("s", ""),
        "interval": k.get("i", ""),
        "timestamp": datetime.fromtimestamp(k.get("t", 0) / 1000, tz=timezone.utc).isoformat(),
        "open": float(k.get("o", 0)),
        "high": float(k.get("h", 0)),
        "low": float(k.get("l", 0)),
        "close": float(k.get("c", 0)),
        "volume": float(k.get("v", 0)),
        "quote_volume": float(k.get("q", 0)),
        "trades": k.get("n", 0),
        "is_final": k.get("x", False),
        "type": "kline",
    }


def _map_mark_price(data: dict) -> dict:
    """Map Binance mark price event to standardized format."""
    return {
        "symbol": data.get("s", ""),
        "timestamp": datetime.fromtimestamp(data.get("E", 0) / 1000, tz=timezone.utc).isoformat(),
        "mark_price": float(data.get("p", 0)),
        "index_price": float(data.get("i", 0)),
        "funding_rate": float(data.get("r", 0)),
        "next_funding_time": datetime.fromtimestamp(
            data.get("T", 0) / 1000, tz=timezone.utc
        ).isoformat() if data.get("T") else None,
        "type": "funding",
    }


def _map_depth_snapshot(data: dict) -> dict:
    """
    Map Binance depth20 snapshot to standardized format (bids/asks format).

    The depth20@100ms stream sends snapshots with 'bids' and 'asks' arrays
    directly (no 'depthUpdate' event type key). This handles that format.
    """
    bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
    asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    total_vol = bid_vol + ask_vol

    best_bid = bids[0][0] if bids else 0
    best_ask = asks[0][0] if asks else 0
    spread = best_ask - best_bid if best_ask and best_bid else 0

    return {
        "symbol": data.get("s", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bids": bids,
        "asks": asks,
        "bid_volume": bid_vol,
        "ask_volume": ask_vol,
        "spread": spread,
        "spread_pct": (spread / best_ask * 100) if best_ask > 0 else 0,
        "imbalance_pct": (bid_vol / total_vol * 100) if total_vol > 0 else 50.0,
        "depth_pressure": (
            (bid_vol - ask_vol) / (bid_vol + ask_vol)
            if (bid_vol + ask_vol) > 0 else 0
        ),
        "first_bid": best_bid,
        "first_ask": best_ask,
        "type": "depth",
    }


# ═══════════════════════════════════════════════════════════════════
#  Channels Consumer — Frontend WebSocket
# ═══════════════════════════════════════════════════════════════════


class BinanceStreamConsumer(AsyncJsonWebsocketConsumer):
    """
    Django Channels WebSocket consumer for real-time Binance data.

    Frontend clients connect to ws://host/ws/trading-bot/live/BTCUSDT/
    to receive live streaming data for that symbol.

    The consumer subscribes to Binance WebSocket streams and forwards
    parsed data to both:
      1. The connected frontend client (real-time dashboard)
      2. The in-memory LiveDataBuffer (for strategy signals)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbol: str = ""
        self._binance_task: Optional[asyncio.Task] = None
        self._funding_task: Optional[asyncio.Task] = None
        self._running = False
        self._streams: list[str] = []
        self._buffer = get_buffer()

    async def connect(self):
        """Accept WebSocket connection and start Binance stream."""
        self.symbol = self.scope["url_route"]["kwargs"].get("symbol", "BTCUSDT").upper()
        await self.accept()
        logger.info("Frontend WS connected for %s", self.symbol)

        # Send initial status
        await self.send_json({
            "type": "status",
            "message": f"Connected to live stream for {self.symbol}",
            "symbol": self.symbol,
        })

        # Start Binance stream tasks
        self._running = True
        self._binance_task = asyncio.create_task(
            self._run_binance_streams()
        )
        self._funding_task = asyncio.create_task(
            self._run_funding_stream()
        )

    async def disconnect(self, close_code):
        """Clean up on disconnect."""
        self._running = False
        for task_name in ["_binance_task", "_funding_task"]:
            task = getattr(self, task_name, None)
            if task:
                task.cancel()
                setattr(self, task_name, None)
        logger.info("Frontend WS disconnected for %s (code=%s)", self.symbol, close_code)

    async def receive_json(self, content):
        """Handle incoming messages from the frontend client."""
        msg_type = content.get("type", "")

        if msg_type == "subscribe":
            symbols = content.get("symbols", [self.symbol])
            streams = content.get("streams", ["depth20", "aggTrade", "kline_1m", "markPrice"])
            logger.info("Client subscribed to %s streams for %s", streams, symbols)
            await self.send_json({
                "type": "status",
                "message": f"Subscribed to {len(streams)} stream types for {len(symbols)} symbols",
            })

        elif msg_type == "ping":
            await self.send_json({"type": "pong"})

        elif msg_type == "get_stats":
            stats = self._buffer.get_stats()
            await self.send_json({"type": "stats", "data": stats})

    async def _run_binance_streams(self):
        """
        Connect to Binance WebSocket and forward data to the client.

        Subscribes to multiple streams for the configured symbol:
        - depth20@100ms: Partial order book (top 20 levels)
        - aggTrade: Aggregated trade stream
        - kline_1m: 1-minute kline/candle updates
        - markPrice@1s: Mark price & funding rate (futures stream)
        """
        symbol_lower = self.symbol.lower()
        streams = [
            f"{symbol_lower}@depth20@100ms",
            f"{symbol_lower}@aggTrade",
            f"{symbol_lower}@kline_1m",
        ]

        import websockets

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1,
        }

        retry_count = 0
        max_retries = 10

        while self._running and retry_count < max_retries:
            try:
                async with websockets.connect(
                    SPOT_STREAM,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    # Subscribe to streams
                    await ws.send(json.dumps(subscribe_msg))
                    response = await asyncio.wait_for(ws.recv(), timeout=10)
                    sub_result = json.loads(response)
                    if sub_result.get("result") is None:
                        logger.info(
                            "Subscribed to Binance streams for %s: %s",
                            self.symbol, streams,
                        )
                    else:
                        logger.warning(
                            "Subscription result for %s: %s",
                            self.symbol, sub_result,
                        )

                    retry_count = 0  # Reset on successful connection

                    # Read messages in a loop
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        await self._process_binance_message(raw_msg)

            except asyncio.CancelledError:
                logger.info("Binance stream task cancelled for %s", self.symbol)
                break
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(
                    "Binance WS disconnected for %s (code=%s, retry=%d/%d)",
                    self.symbol, e.code, retry_count + 1, max_retries,
                )
            except Exception as e:
                logger.error(
                    "Binance WS error for %s: %s (retry=%d/%d)",
                    self.symbol, e, retry_count + 1, max_retries,
                )

            retry_count += 1
            if self._running and retry_count < max_retries:
                wait = min(2 ** retry_count, 30)  # Exponential backoff
                logger.info("Reconnecting in %ds...", wait)
                await asyncio.sleep(wait)

        if retry_count >= max_retries:
            error_msg = f"Max retries reached for {self.symbol}"
            logger.error(error_msg)
            self._buffer.push_error(error_msg)
            await self.send_json({
                "type": "error",
                "message": error_msg,
            })

    async def _run_funding_stream(self):
        """
        Connect to Binance Futures WebSocket for mark price & funding rate.

        Uses a separate connection to the Futures stream endpoint since
        mark price data is only available on the USDT-M futures stream.
        """
        symbol_lower = self.symbol.lower()
        stream = f"{symbol_lower}@markPrice@1s"

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream],
            "id": 2,
        }

        retry_count = 0
        max_retries = 10

        while self._running and retry_count < max_retries:
            try:
                async with websockets.connect(
                    FUTURES_STREAM,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    response = await asyncio.wait_for(ws.recv(), timeout=10)
                    logger.info(
                        "Funding stream subscribed for %s: %s",
                        self.symbol, json.loads(response),
                    )
                    retry_count = 0

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(raw_msg)
                            if data.get("e") == "markPriceUpdate":
                                mapped = _map_mark_price(data)
                                self._buffer.push("funding", self.symbol, mapped)
                                await self.send_json(mapped)
                        except json.JSONDecodeError:
                            continue

            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(
                    "Funding WS disconnected for %s (retry %d/%d)",
                    self.symbol, retry_count + 1, max_retries,
                )
            except Exception as e:
                logger.error(
                    "Funding WS error for %s: %s", self.symbol, e,
                )

            retry_count += 1
            if self._running and retry_count < max_retries:
                await asyncio.sleep(min(2 ** retry_count, 30))

    async def _process_binance_message(self, raw_msg: bytes):
        """Parse and forward a Binance WebSocket message."""
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Binance message: %s", e)
            return

        event_type = data.get("e", "")

        # Map the event to a standardized format
        mapped = None
        data_type = None

        if event_type == "depthUpdate":
            # Full depth stream uses b/a keys
            if "b" in data and "a" in data:
                mapped = _map_depth(data)
                data_type = "depth"
        elif "bids" in data and "asks" in data and "lastUpdateId" in data:
            # Partial depth snapshot (depth20@100ms) uses bids/asks keys
            mapped = _map_depth_snapshot(data)
            data_type = "depth"
        elif event_type == "aggTrade":
            mapped = _map_trade(data)
            data_type = "trade"
        elif event_type == "kline":
            mapped = _map_kline(data)
            data_type = "kline"
        elif event_type == "markPriceUpdate":
            mapped = _map_mark_price(data)
            data_type = "funding"

        if mapped and data_type:
            # Push to buffer
            self._buffer.push(data_type, self.symbol, mapped)

            # Forward to frontend client
            await self.send_json(mapped)
