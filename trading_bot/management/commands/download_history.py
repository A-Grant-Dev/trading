"""
Management command to download historical OHLCV data.

Downloads years of market data from Binance (or other exchanges)
into the OHLCV model for backtesting and feature engineering.

Usage:
    # Download 2 years of 1-hour BTC data
    python manage.py download_history --symbol BTCUSDT --interval 1h --days 730

    # Download multiple symbols
    python manage.py download_history --symbol BTCUSDT,ETHUSDT,SOLUSDT --interval 1m --days 30

    # Force re-download even if data exists
    python manage.py download_history --symbol BTCUSDT --force

    # Quick test (just 10 days)
    python manage.py download_history --symbol BTCUSDT --interval 1h --days 10

    # See all available symbols on Binance
    python manage.py download_history --list-symbols
"""

import logging
from datetime import datetime, timezone

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Max, Min

from trading_bot.models import OHLCV
from trading_bot.services.data.historical import download_history

logger = logging.getLogger(__name__)

# Popular symbols to show in --list-symbols
POPULAR_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT",
    "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ARBUSDT",
    "OPUSDT", "APTUSDT", "FILUSDT", "NEARUSDT", "AAVEUSDT",
    "CRVUSDT", "ALGOUSDT", "HBARUSDT", "VETUSDT", "XTZUSDT",
    "ICPUSDT", "RUNEUSDT", "SANDUSDT", "AXSUSDT", "MANAUSDT",
    "GRTUSDT", "CHZUSDT", "EGLDUSDT", "KAVAUSDT", "QTUMUSDT",
    "ZECUSDT", "DASHUSDT", "WAVESUSDT", "FTMUSDT", "1INCHUSDT",
    "YFIUSDT", "MKRUSDT", "SNXUSDT", "SUSHIUSDT", "COMPUSDT",
    "BALUSDT", "LRCUSDT", "BATUSDT", "ZRXUSDT", "BANDUSDT",
]


class Command(BaseCommand):
    help = "Download historical OHLCV data from exchange into the database"

    def add_arguments(self, parser):  # noqa: PLW0221
        parser.add_argument(
            "--symbol",
            type=str,
            default="BTCUSDT",
            help="Trading pair symbol(s), comma-separated (default: BTCUSDT)",
        )
        parser.add_argument(
            "--interval",
            type=str,
            default="1h",
            choices=[
                "1s", "5s", "15s", "1m", "3m", "5m", "15m",
                "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w",
            ],
            help="Candle interval (default: 1h)",
        )
        parser.add_argument(
            "--days",
            type=int,
            default=365,
            help="Number of days of history to download (default: 365)",
        )
        parser.add_argument(
            "--exchange",
            type=str,
            default="binance",
            help="Exchange identifier (default: binance)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Force re-download even if data exists in DB",
        )
        parser.add_argument(
            "--list-symbols",
            action="store_true",
            default=False,
            help="Print popular tradable symbols and exit",
        )
        parser.add_argument(
            "--status",
            action="store_true",
            default=False,
            help="Show download status summary and exit",
        )

    def handle(self, *args, **options):
        if options["list_symbols"]:
            self._list_symbols()
            return

        if options["status"]:
            self._show_status()
            return

        symbols = [s.strip().upper() for s in options["symbol"].split(",")]
        interval = options["interval"]
        days = options["days"]
        exchange = options["exchange"]
        force = options["force"]

        total_fetched = 0
        total_stored = 0
        total_duration = 0.0
        results = []

        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"  Historical Data Download\n"
            f"  Symbols: {', '.join(symbols)}\n"
            f"  Interval: {interval}\n"
            f"  Days: {days}\n"
            f"  Exchange: {exchange}\n"
            f"  Force: {force}\n"
            f"{'='*60}\n"
        ))

        for symbol in symbols:
            self.stdout.write(f"\n📥 Downloading {symbol} {interval}...")
            self.stdout.flush()

            result = download_history(
                symbol=symbol,
                interval=interval,
                days=days,
                exchange=exchange,
                force=force,
            )

            results.append(result)
            total_fetched += result.get("candles_fetched", 0)
            total_stored += result.get("candles_stored", 0)
            total_duration += result.get("duration_seconds", 0)

            status = result.get("status", "unknown")
            stored = result.get("candles_stored", 0)
            duration = result.get("duration_seconds", 0)

            if status == "success":
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✅ {symbol} {interval}: stored {stored} new candles ({duration:.1f}s)"
                    )
                )
            elif status == "up_to_date":
                self.stdout.write(
                    self.style.WARNING(f"  ⏺ {symbol} {interval}: already up to date")
                )
            elif status == "no_data":
                self.stdout.write(
                    self.style.ERROR(f"  ⚠️  {symbol} {interval}: no data returned")
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f"  ❌ {symbol} {interval}: {result}")
                )

        # Summary
        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"  Download Complete\n"
            f"  Total symbols: {len(symbols)}\n"
            f"  Total fetched: {total_fetched} candles\n"
            f"  Total stored: {total_stored} candles\n"
            f"  Total time: {total_duration:.1f}s\n"
            f"{'='*60}\n"
        ))

    def _list_symbols(self):
        """Print popular tradable symbols."""
        self.stdout.write(self.style.SUCCESS("\nPopular Binance Spot Symbols:\n"))
        for i in range(0, len(POPULAR_SYMBOLS), 5):
            chunk = POPULAR_SYMBOLS[i:i + 5]
            self.stdout.write("  " + "  ".join(f"{s:<12}" for s in chunk))

        self.stdout.write(self.style.WARNING(
            "\nUse any of these as --symbol argument."
            "\nAll USDT trading pairs are supported.\n"
        ))

    def _show_status(self):
        """Show download status summary."""
        total_candles = OHLCV.objects.count()
        symbols = (
            OHLCV.objects.values_list("symbol", flat=True)
            .distinct()
            .order_by("symbol")
        )
        intervals = (
            OHLCV.objects.values_list("interval", flat=True)
            .distinct()
            .order_by("interval")
        )

        now = datetime.now(timezone.utc)

        self.stdout.write(self.style.SUCCESS(
            f"\n📊 OHLCV Data Status\n"
            f"{'='*60}\n"
            f"  Total candles stored: {total_candles:,}\n"
            f"  Symbols: {', '.join(symbols) if symbols else 'none'}\n"
            f"  Intervals: {', '.join(intervals) if intervals else 'none'}\n"
        ))

        if symbols:
            # Single annotated query instead of N+1
            stats = (
                OHLCV.objects.values("symbol")
                .annotate(
                    total=Count("id"),
                    latest=Max("timestamp"),
                    earliest=Min("timestamp"),
                )
                .order_by("symbol")
            )

            self.stdout.write("\nPer-Symbol Breakdown:\n")
            for row in stats:
                sym = row["symbol"]
                count = row["total"]
                latest_ts = row["latest"]
                earliest_ts = row["earliest"]

                latest_str = latest_ts.isoformat() if latest_ts else "—"
                earliest_str = earliest_ts.isoformat() if earliest_ts else "—"

                if latest_ts:
                    gap_hours = (now - latest_ts).total_seconds() / 3600
                    freshness = f"{gap_hours:.1f}h ago" if gap_hours < 48 else "⚠️ stale"
                else:
                    freshness = "—"

                self.stdout.write(
                    f"  {sym:<12} {count:>8,} candles  "
                    f"from {earliest_str[:10]}  to {latest_str[:10]}  ({freshness})"
                )

            self.stdout.write("")
