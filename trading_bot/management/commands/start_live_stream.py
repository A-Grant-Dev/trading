"""
Management command to start the live Binance WebSocket stream.

Connects to Binance real-time data streams and pipes data into the
in-memory LiveDataBuffer for dashboard display and strategy signals.

Usage:
    # Start live stream for a specific symbol
    python manage.py start_live_stream --symbol BTCUSDT

    # Monitor the live data buffer state
    python manage.py start_live_stream --status

    # Clear the live data buffer
    python manage.py start_live_stream --clear
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets
from django.core.management.base import BaseCommand

from trading_bot.services.data.buffer import get_buffer, reset_buffer

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Start live Binance WebSocket stream for real-time market data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbol",
            type=str,
            default="BTCUSDT",
            help="Trading pair to stream (default: BTCUSDT)",
        )
        parser.add_argument(
            "--status",
            action="store_true",
            default=False,
            help="Show live data buffer status and exit",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            default=False,
            help="Clear the live data buffer",
        )
        parser.add_argument(
            "--dump",
            type=str,
            default=None,
            help="Dump latest data for a symbol (e.g., --dump BTCUSDT)",
        )

    def handle(self, *args, **options):
        if options["status"]:
            self._show_status()
            return

        if options["clear"]:
            reset_buffer()
            self.stdout.write(self.style.SUCCESS("✅ Live data buffer cleared"))
            return

        if options["dump"]:
            self._dump_data(options["dump"])
            return

        symbol = options["symbol"].upper()
        self._start_stream(symbol)

    def _show_status(self):
        """Display live data buffer status."""
        buffer = get_buffer()
        stats = buffer.get_stats()

        self.stdout.write(self.style.SUCCESS(
            f"\n📊 Live Data Buffer Status\n"
            f"{'='*60}"
        ))

        if stats["started_at"]:
            self.stdout.write(f"  Started:     {stats['started_at']}")
            self.stdout.write(f"  Uptime:      {stats['uptime_seconds']:.0f}s")
            self.stdout.write(f"  Messages:    {stats['total_messages']:,}")
            self.stdout.write(f"  Errors:      {stats['errors']}")
            self.stdout.write(f"  Symbols:     {', '.join(stats['symbols'])}")
            self.stdout.write(f"  Data types:  {', '.join(stats['data_types'])}")
        else:
            self.stdout.write(self.style.WARNING("  No data received yet"))

        # Show latest data per symbol
        latest = buffer.get_all_latest()
        if latest:
            self.stdout.write("\nLatest Snapshots:\n")
            for symbol, types in latest.items():
                for data_type, data in types.items():
                    ts = data.get("timestamp", "?")
                    if isinstance(ts, str):
                        ts = ts[:19]
                    self.stdout.write(
                        f"  {symbol:<10} {data_type:<10} @ {ts}"
                    )

        self.stdout.write("")

    def _dump_data(self, symbol: str):
        """Dump latest data for a specific symbol."""
        symbol = symbol.upper()
        buffer = get_buffer()

        self.stdout.write(self.style.SUCCESS(f"\n📡 Live Data for {symbol}\n{'='*60}"))

        # Latest order book
        ob = buffer.get_orderbook(symbol)
        if ob:
            self.stdout.write(f"\nOrder Book (top 5 bids/asks):")
            self.stdout.write(f"  Spread:  {ob.get('spread', '?'):.2f} ({ob.get('spread_pct', '?'):.4f}%)")
            self.stdout.write(f"  Imbalance: {ob.get('imbalance_pct', '?'):.1f}%")
            self.stdout.write(f"  Bids:     {json.dumps(ob.get('bids', [])[:5])}")
            self.stdout.write(f"  Asks:     {json.dumps(ob.get('asks', [])[:5])}")
        else:
            self.stdout.write(self.style.WARNING("\n  No order book data yet"))

        # Recent trades
        trades = buffer.get_trades(symbol, n=5)
        if trades:
            self.stdout.write(f"\nRecent Trades (last {len(trades)}):")
            for t in trades[:5]:
                side = "SELL" if t.get("is_buyer_maker") else "BUY"
                self.stdout.write(
                    f"  {side:<5} {t.get('price', 0):<12.2f} qty={t.get('quantity', 0):<.6f}"
                )
        else:
            self.stdout.write(self.style.WARNING("\n  No trade data yet"))

        # Funding rate
        funding = buffer.get_funding(symbol)
        if funding:
            self.stdout.write(f"\nFunding / Mark Price:")
            self.stdout.write(f"  Mark Price:  {funding.get('mark_price', '?'):.2f}")
            self.stdout.write(f"  Funding:     {funding.get('funding_rate', 0) * 100:.4f}%")
        else:
            self.stdout.write(self.style.WARNING("\n  No funding data yet"))

        self.stdout.write("")

    def _start_stream(self, symbol: str):
        """
        Start the live Binance WebSocket stream in CLI mode.

        This runs the stream directly (without Channels) for CLI monitoring.
        The Channels consumer is used when connecting via a browser.
        """
        SPOT_STREAM = "wss://stream.binance.com:9443/ws"
        streams = [
            f"{symbol.lower()}@depth20@100ms",
            f"{symbol.lower()}@aggTrade",
            f"{symbol.lower()}@kline_1m",
        ]

        self.stdout.write(self.style.SUCCESS(
            f"\n📡 Starting live Binance stream for {symbol}\n"
            f"  Streams: {', '.join(streams)}\n"
            f"  Press Ctrl+C to stop\n"
            f"{'='*60}\n"
        ))

        buffer = get_buffer()

        async def _run():
            async with websockets.connect(SPOT_STREAM, ping_interval=20, ping_timeout=10) as ws:
                # Subscribe
                subscribe = {
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": 1,
                }
                await ws.send(json.dumps(subscribe))

                # Wait for subscription confirmation
                resp = await asyncio.wait_for(ws.recv(), timeout=10)
                self.stdout.write(f"  ✅ Subscribed: {resp}\n")

                # Listen for messages
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                        event = data.get("e", "")

                        # Map and push to buffer
                        # depth20@100ms sends snaps with bids/asks (no event type)
                        if "bids" in data and "asks" in data and "lastUpdateId" in data:
                            buffer.push("depth", symbol, _summarize_depth_snapshot(data))
                            self.stdout.write(
                                f"  📊 Depth @ {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
                            )
                        elif event == "depthUpdate":
                            buffer.push("depth", symbol, _summarize_depth(data))
                            self.stdout.write(
                                f"  📊 Depth @ {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
                            )
                        elif event == "aggTrade":
                            buffer.push("trade", symbol, {
                                "price": float(data.get("p", 0)),
                                "quantity": float(data.get("q", 0)),
                                "is_buyer_maker": data.get("m", True),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            side = "🔴 SELL" if data.get("m") else "🟢 BUY"
                            self.stdout.write(
                                f"  {side} {float(data.get('p', 0)):.2f} x {float(data.get('q', 0)):.6f}"
                            )
                        elif event == "kline":
                            k = data.get("k", {})
                            self.stdout.write(
                                f"  🕯️  Kline {k.get('i', '')} "
                                f"O:{float(k.get('o', 0)):.2f} "
                                f"C:{float(k.get('c', 0)):.2f} "
                                f"V:{float(k.get('v', 0)):.2f}"
                            )

                    except json.JSONDecodeError:
                        pass

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\n\n⏹️  Stream stopped by user"))
            stats = buffer.get_stats()
            self.stdout.write(
                f"\n📊 Session stats: {stats['total_messages']:,} messages, "
                f"{stats['errors']} errors, "
                f"{stats['uptime_seconds']:.0f}s uptime\n"
            )


def _summarize_depth(data: dict) -> dict:
    """Create a summary of depthUpdate data (b/a format)."""
    bids = [[float(p), float(q)] for p, q in data.get("b", [])[:5]]
    asks = [[float(p), float(q)] for p, q in data.get("a", [])[:5]]
    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bids": bids,
        "asks": asks,
        "bid_volume": bid_vol,
        "ask_volume": ask_vol,
        "spread": asks[0][0] - bids[0][0] if asks and bids else 0,
        "imbalance_pct": (bid_vol / (bid_vol + ask_vol) * 100) if (bid_vol + ask_vol) > 0 else 50,
    }


def _summarize_depth_snapshot(data: dict) -> dict:
    """Create a summary of depth20 snapshot data (bids/asks format)."""
    bids = [[float(p), float(q)] for p, q in data.get("bids", [])[:5]]
    asks = [[float(p), float(q)] for p, q in data.get("asks", [])[:5]]
    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bids": bids,
        "asks": asks,
        "bid_volume": bid_vol,
        "ask_volume": ask_vol,
        "spread": asks[0][0] - bids[0][0] if asks and bids else 0,
        "imbalance_pct": (bid_vol / (bid_vol + ask_vol) * 100) if (bid_vol + ask_vol) > 0 else 50,
    }
