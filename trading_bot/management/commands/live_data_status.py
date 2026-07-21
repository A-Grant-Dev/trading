"""
Management command to monitor the live data stream status.

Shows the current state of the in-memory LiveDataBuffer including:
- Connected symbols and data types
- Message counts and error rates
- Latest data snapshots per symbol
- Stream uptime

Usage:
    python manage.py live_data_status
    python manage.py live_data_status --symbol BTCUSDT
    python manage.py live_data_status --watch
"""

import json
import logging
import time
from datetime import datetime, timezone

from django.core.management.base import BaseCommand

from trading_bot.services.data.buffer import get_buffer

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Monitor live data stream status"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbol",
            type=str,
            default=None,
            help="Filter by symbol (e.g., BTCUSDT)",
        )
        parser.add_argument(
            "--watch",
            action="store_true",
            default=False,
            help="Continuously watch status (refresh every 5s)",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Output as JSON",
        )
        parser.add_argument(
            "--n",
            type=int,
            default=5,
            help="Number of recent trades to show per symbol (default: 5)",
        )

    def handle(self, *args, **options):
        symbol_filter = options["symbol"].upper() if options["symbol"] else None
        as_json = options["json"]
        watch = options["watch"]
        n_trades = options["n"]

        if as_json:
            self._output_json(symbol_filter, n_trades)
            return

        if watch:
            self._watch_loop(symbol_filter, n_trades)
            return

        self._show_table(symbol_filter, n_trades)

    def _show_table(self, symbol_filter: str | None, n_trades: int):
        """Display live data status as a table."""
        buffer = get_buffer()
        stats = buffer.get_stats()

        self.stdout.write(self.style.SUCCESS(
            f"\n📡 Live Data Stream Status\n"
            f"{'='*60}"
        ))

        # Header stats
        if stats["started_at"]:
            self.stdout.write(f"  Uptime:      {stats['uptime_seconds']:.0f}s")
            self.stdout.write(f"  Messages:    {stats['total_messages']:,}")
            self.stdout.write(f"  Errors:      {stats['errors']}")
            self.stdout.write(f"  Symbols:     {', '.join(stats['symbols'])}")
            self.stdout.write(f"  Data types:  {', '.join(stats['data_types'])}")
            self.stdout.write("")
        else:
            self.stdout.write(self.style.WARNING("  No data — stream not started"))
            self.stdout.write("  Run: python manage.py start_live_stream --symbol BTCUSDT")
            self.stdout.write("")

        # Per-symbol detail
        latest = buffer.get_all_latest()
        filtered_symbols = (
            [s for s in latest.keys() if symbol_filter in s.upper()]
            if symbol_filter
            else list(latest.keys())
        )

        for symbol in sorted(filtered_symbols):
            self.stdout.write(f"\n{'─'*40}")
            self.stdout.write(f"  {symbol}")

            # Order book
            ob = buffer.get_orderbook(symbol)
            if ob:
                self.stdout.write(
                    f"    Order Book: spread={ob.get('spread', 0):.2f} "
                    f"({ob.get('spread_pct', 0):.4f}%) "
                    f"imbalance={ob.get('imbalance_pct', 0):.1f}%"
                )

            # Recent trades
            trades = buffer.get_trades(symbol, n_trades)
            if trades:
                self.stdout.write(f"    Recent Trades ({len(trades)}):")
                for t in trades[-5:]:
                    side = "SELL" if t.get("is_buyer_maker") else "BUY"
                    price = t.get("price", 0)
                    qty = t.get("quantity", 0)
                    self.stdout.write(f"      {side:<5} {price:>10.2f} x {qty:<.6f}")

            # Funding
            funding = buffer.get_funding(symbol)
            if funding:
                fr = funding.get("funding_rate", 0) * 100
                mp = funding.get("mark_price", 0)
                self.stdout.write(f"    Funding: {fr:.4f}% | Mark: ${mp:.2f}")

        if not filtered_symbols:
            self.stdout.write(self.style.WARNING("\n  No data for the specified filter"))

        self.stdout.write("")

    def _watch_loop(self, symbol_filter: str | None, n_trades: int):
        """Continuously watch live data updates."""
        try:
            while True:
                self._show_table(symbol_filter, n_trades)
                time.sleep(5)
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\n⏹️  Stopped"))

    def _output_json(self, symbol_filter: str | None, n_trades: int):
        """Output as JSON for programmatic consumption."""
        buffer = get_buffer()
        stats = buffer.get_stats()

        output = {
            "status": "running" if stats["started_at"] else "idle",
            "stats": stats,
            "symbols": {},
        }

        for symbol, types in buffer.get_all_latest().items():
            if symbol_filter and symbol_filter not in symbol.upper():
                continue
            output["symbols"][symbol] = {
                "orderbook": buffer.get_orderbook(symbol),
                "recent_trades": buffer.get_trades(symbol, n_trades),
                "funding": buffer.get_funding(symbol),
            }

        self.stdout.write(json.dumps(output, indent=2, default=str))
