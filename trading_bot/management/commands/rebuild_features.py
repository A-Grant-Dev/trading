"""
Management command to rebuild feature snapshots for ML models.

Computes all features from raw OHLCV data using the feature engine
and stores them as FeatureSnapshot records.

Usage:
    # Build features for default symbol (BTCUSDT, 1h, 365 days)
    python manage.py rebuild_features

    # Build features for a specific symbol and interval
    python manage.py rebuild_features --symbol ETHUSDT --interval 1h --days 90

    # Build features for multiple symbols (dry run, no store)
    python manage.py rebuild_features --symbol BTCUSDT,ETHUSDT,SOLUSDT --no-store

    # Build features from specific number of candles (not days)
    python manage.py rebuild_features --symbol BTCUSDT --limit 1000

    # Show current feature engine status
    python manage.py rebuild_features --status
"""

import logging
from datetime import datetime, timezone

from django.core.management.base import BaseCommand
from django.db.models import Count

from trading_bot.models import FeatureSnapshot, OHLCV
from trading_bot.services.features import engine as feature_engine

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Rebuild feature snapshots from raw market data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbol",
            type=str,
            default="BTCUSDT",
            help="Trading pair(s), comma-separated (default: BTCUSDT)",
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
            help="Number of days of history (default: 365)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Max candles (overrides --days if set)",
        )
        parser.add_argument(
            "--store",
            action="store_true",
            default=True,
            help="Store features in DB (default: True)",
        )
        parser.add_argument(
            "--no-store",
            action="store_false",
            dest="store",
            help="Dry run — compute but don't store",
        )
        parser.add_argument(
            "--status",
            action="store_true",
            default=False,
            help="Show feature store status and exit",
        )
        parser.add_argument(
            "--feature-version",
            type=str,
            default=None,
            help="Override feature set version (default: auto)",
        )

    def handle(self, *args, **options):
        if options["status"]:
            self._show_status()
            return

        symbols = [s.strip().upper() for s in options["symbol"].split(",")]
        interval = options["interval"]
        days = options["days"]
        limit = options["limit"]
        store = options["store"]
        version = options.get("feature_version") or options.get("feature-version")

        total_features = 0
        total_rows = 0
        total_stored = 0
        total_duration = 0.0

        self.stdout.write(self.style.SUCCESS(
            f"\n🧠 Feature Engine — {feature_engine.FEATURE_SOURCES.keys() | set()}\n"
            f"{'='*60}\n"
            f"  Symbols:  {', '.join(symbols)}\n"
            f"  Interval: {interval}\n"
            f"  Days:     {days if not limit else f'n/a (limit={limit})'}\n"
            f"  Store:    {store}\n"
            f"  Version:  {version or feature_engine.get_feature_version()}\n"
            f"{'='*60}\n"
        ))

        for symbol in symbols:
            self.stdout.write(f"\n📊 Building features for {symbol} {interval}...")
            self.stdout.flush()

            result = feature_engine.rebuild_features(
                symbol=symbol,
                interval=interval,
                days=None if limit else days,
                limit=limit,
                store=store,
                version=version,
            )

            total_features += result.get("n_features", 0)
            total_rows += result.get("n_rows", 0)
            total_stored += result.get("n_stored", 0)
            total_duration += result.get("duration_seconds", 0)

            n_f = result.get("n_features", 0)
            n_r = result.get("n_rows", 0)
            n_s = result.get("n_stored", 0)
            dur = result.get("duration_seconds", 0)

            if result.get("status") == "success":
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✅ {symbol}: {n_f} features × {n_r} rows "
                        f"({n_s} stored) in {dur:.1f}s"
                    )
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f"  ⚠️  {symbol}: {result.get('status', 'error')}")
                )

        # Summary
        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"  Feature Rebuild Complete\n"
            f"  Total symbols:    {len(symbols)}\n"
            f"  Total features:   {total_features}\n"
            f"  Total rows:       {total_rows:,}\n"
            f"  Total stored:     {total_stored:,}\n"
            f"  Total time:       {total_duration:.1f}s\n"
            f"{'='*60}\n"
        ))

    def _show_status(self):
        """Display feature engine status."""
        total_snapshots = FeatureSnapshot.objects.count()
        symbols = (
            FeatureSnapshot.objects.values_list("symbol", flat=True)
            .distinct()
            .order_by("symbol")
        )
        versions = (
            FeatureSnapshot.objects.values_list("feature_set_version", flat=True)
            .distinct()
            .order_by("-feature_set_version")
        )

        self.stdout.write(self.style.SUCCESS(
            f"\n📊 Feature Engine Status\n"
            f"{'='*60}\n"
            f"  Feature snapshots stored: {total_snapshots:,}\n"
            f"  Feature set version:      {feature_engine.get_feature_version()}\n"
            f"  Symbols:                  {', '.join(symbols) if symbols else 'none'}\n"
            f"  Versions in DB:           {', '.join(versions) if versions else 'none'}\n"
        ))

        if symbols:
            self.stdout.write("\nPer-Symbol Breakdown:\n")
            for sym in symbols:
                count = FeatureSnapshot.objects.filter(symbol=sym).count()
                latest = FeatureSnapshot.objects.filter(symbol=sym).order_by("-timestamp").first()
                n_features = len(latest.features) if latest and latest.features else 0
                latest_str = latest.timestamp.isoformat()[:19] if latest else "—"
                self.stdout.write(
                    f"  {sym:<12} {count:>8,} snapshots  "
                    f"{n_features:>3} features each  latest: {latest_str}"
                )
            self.stdout.write("")

        # Available sources
        self.stdout.write("Feature Sources:\n")
        for key, desc in feature_engine.FEATURE_SOURCES.items():
            self.stdout.write(f"  ✅ {key:<20} {desc}")
        self.stdout.write("")
