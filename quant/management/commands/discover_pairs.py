"""
Management Command: Discover Cointegrated Pairs

Scans the Binance USDT market for cointegrated trading pairs using
the Engle-Granger test. Stores discovered pairs in the Pair model
for live monitoring and signal generation.

Usage:
    python manage.py discover_pairs
    python manage.py discover_pairs --limit 50
    python manage.py discover_pairs --symbols BTCUSDT,ETHUSDT,SOLUSDT
    python manage.py discover_pairs --min-pvalue 0.01 --days 60

Renaissance principle: Finding hidden statistical relationships between
seemingly unrelated assets is the foundation of statistical arbitrage.
"""

import logging
from datetime import datetime, timezone

import pandas as pd
from django.core.management.base import BaseCommand

from quant.models import Pair
from quant.services.cointegration import (
    PairsFinder,
    fetch_daily_close_prices,
    DEFAULT_P_THRESHOLD,
    DEFAULT_LOOKBACK_DAYS,
)
from quant.services.data_feeds import get_tradable_symbols, set_cache

logger = logging.getLogger(__name__)

# Pre-filter: only test symbols with these quote assets
QUOTE_ASSETS = ["USDT", "USDC", "FDUSD"]


class Command(BaseCommand):
    help = "Discover cointegrated trading pairs from Binance market data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            default="",
            help="Comma-separated symbols to scan (default: all USDT pairs)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=100,
            help="Max symbols to scan (default: 100, use 0 for all)",
        )
        parser.add_argument(
            "--days",
            type=int,
            default=DEFAULT_LOOKBACK_DAYS,
            help=f"Days of historical data (default: {DEFAULT_LOOKBACK_DAYS})",
        )
        parser.add_argument(
            "--min-pvalue",
            type=float,
            default=DEFAULT_P_THRESHOLD,
            help=f"Maximum p-value threshold (default: {DEFAULT_P_THRESHOLD})",
        )
        parser.add_argument(
            "--save",
            action="store_true",
            default=True,
            help="Save discovered pairs to database (default: True)",
        )
        parser.add_argument(
            "--no-save",
            action="store_false",
            dest="save",
            help="Print results without saving to database",
        )
        parser.add_argument(
            "--backtest",
            action="store_true",
            help="Run quick backtest on discovered pairs",
        )

    def handle(self, *args, **options):
        limit = options["limit"]
        days = options["days"]
        p_threshold = options["min_pvalue"]
        do_save = options["save"]
        do_backtest = options.get("backtest", False)

        # Get symbols to scan
        if options["symbols"]:
            symbols = [s.strip().upper() for s in options["symbols"].split(",")]
            self.stdout.write(f"Scanning {len(symbols)} specified symbols...")
        else:
            self.stdout.write(f"Fetching tradable USDT pairs from Binance...")
            all_symbols = get_tradable_symbols("USDT")

            # Filter for quality: exclude leveraged and lower-case pairs
            symbols = [
                s for s in all_symbols
                if s.endswith("USDT") and not any(x in s for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])
            ]

            if limit and limit > 0:
                # Take top N by volume (or alphabetically as a proxy)
                symbols = sorted(symbols)[:limit]

            self.stdout.write(f"Testing {len(symbols)} USDT pairs...")

        if len(symbols) < 2:
            self.stderr.write("Need at least 2 symbols to find pairs")
            return

        # Fetch price data
        self.stdout.write(f"Fetching {days} days of daily close prices...")
        price_data = fetch_daily_close_prices(symbols, days=days)

        if len(price_data) < 2:
            self.stderr.write("Could not fetch enough price data")
            return

        self.stdout.write(f"Got data for {len(price_data)} symbols")

        # Run cointegration scan
        self.stdout.write("Running Engle-Granger cointegration tests...")
        finder = PairsFinder(list(price_data.keys()))
        discovered = finder.find_cointegrated_pairs(
            price_data,
            p_threshold=p_threshold,
        )

        if not discovered:
            self.stdout.write(self.style.WARNING("No cointegrated pairs found"))
            return

        self.stdout.write(
            self.style.SUCCESS(f"Found {len(discovered)} cointegrated pairs!")
        )

        # Display results
        self._print_results(discovered)

        # Save to database
        if do_save:
            saved_count = 0
            for pair_result in discovered:
                pair = PairsFinder.update_pair_model(pair_result)

                # Run quick backtest if requested
                if do_backtest:
                    try:
                        from quant.services.pairs_signals import PairsSignalGenerator
                        from quant.services.data_ingestion import get_market_data_df
                        from quant.services.data_utils import ohlcv_to_dataframe

                        # Try to get historical data for backtest
                        df_a = ohlcv_to_dataframe(pair.symbol_a, "1h", 500)
                        df_b = ohlcv_to_dataframe(pair.symbol_b, "1h", 500)

                        if not df_a.empty and not df_b.empty:
                            combined = pd.DataFrame({
                                pair.symbol_a: df_a["close"],
                                pair.symbol_b: df_b["close"],
                            }).dropna()

                            generator = PairsSignalGenerator()
                            bt_results = generator.backtest_pair(
                                pair_result,
                                combined,
                            )

                            if bt_results and "total_trades" in bt_results:
                                pair.total_trades = bt_results["total_trades"]
                                pair.win_rate = bt_results.get("win_rate")
                                pair.save(update_fields=["total_trades", "win_rate"])

                                self.stdout.write(
                                    f"  Backtest: {bt_results['total_trades']} trades, "
                                    f"{bt_results['win_rate']:.1%} win rate, "
                                    f"Sharpe {bt_results['sharpe_ratio']}"
                                )
                    except Exception as e:
                        logger.debug(f"Backtest skipped for {pair}: {e}")

                saved_count += 1

            self.stdout.write(self.style.SUCCESS(f"Saved {saved_count} pairs to database"))

        # Cache results for dashboard
        set_cache("discovered_pairs", discovered)

    def _print_results(self, pairs: list[dict]):
        """Print discovered pairs in a formatted table."""
        self.stdout.write("\n" + "=" * 90)
        self.stdout.write(f"{'Pair':<25} {'P-Value':<10} {'Half-Life':<12} {'Z-Score':<10} {'Hedge β':<12} {'N':<8}")
        self.stdout.write("=" * 90)

        for p in pairs[:20]:  # Show top 20
            pair_name = f"{p['symbol_a']}/{p['symbol_b']}"
            half_life = f"{p.get('half_life', 'N/A')}h" if p.get('half_life') else "N/A"
            zscore = f"{p.get('current_zscore', 0):.2f}"
            hedge = f"{p.get('hedge_ratio', 0):.4f}"
            pval = f"{p['p_value']:.6f}"

            self.stdout.write(
                f"{pair_name:<25} {pval:<10} {half_life:<12} {zscore:<10} {hedge:<12} {p['n_data_points']:<8}"
            )

        if len(pairs) > 20:
            self.stdout.write(f"... and {len(pairs) - 20} more pairs")

        self.stdout.write("=" * 90)
        self.stdout.write("")
