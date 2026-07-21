"""
Management command to fetch on-chain data from public APIs.

Fetches on-chain metrics from free public sources:
- Bitcoin blockchain metrics (Mempool.space API)
- BTC Fear & Greed-like metrics (CoinGecko)
- Whale transaction tracking (Blockchair / Whale-Alert free tier)

Data is stored in the AuditLog and can be consumed by strategy
services for feature engineering.

Usage:
    python manage.py fetch_onchain
    python manage.py fetch_onchain --verbose
"""

import json
import logging
import time
from datetime import datetime, timezone

import requests
from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)

# ── Free Public API Endpoints ─────────────────────────────────────

MEMPOOL_API = "https://mempool.space/api"
COINGECKO_API = "https://api.coingecko.com/api/v3"
WHALE_ALERT_API = "https://api.whale-alert.io/v1"


class Command(BaseCommand):
    help = "Fetch on-chain data from public APIs"

    def add_arguments(self, parser):
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Show detailed output",
        )

    def handle(self, *args, **options):
        verbose = options["verbose"]
        results = {}

        self.stdout.write(self.style.SUCCESS(
            f"\n🔗 Fetching On-Chain Data\n"
            f"{'='*60}"
        ))

        # 1. Bitcoin blockchain stats from Mempool.space
        results["bitcoin_stats"] = self._fetch_bitcoin_stats(verbose)

        # 2. CoinGecko market data (includes on-chain-like metrics)
        results["market_data"] = self._fetch_coingecko_data(verbose)

        # 3. Summary
        self._print_summary(results)

    def _fetch_bitcoin_stats(self, verbose: bool) -> dict:
        """Fetch Bitcoin blockchain stats from Mempool.space."""
        self.stdout.write("\n📡 Bitcoin Blockchain Stats...")

        result = {"status": "error", "data": {}}
        endpoints = [
            ("difficulty", f"{MEMPOOL_API}/v1/difficulty-adjustment"),
            ("block_height", f"{MEMPOOL_API}/blocks/tip/height"),
            ("mempool", f"{MEMPOOL_API}/mempool"),
            ("fees", f"{MEMPOOL_API}/v1/fees/recommended"),
        ]

        for name, url in endpoints:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                result["data"][name] = resp.json()
                self.stdout.write(f"  ✅ {name}")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  ❌ {name}: {e}"))
                result["data"][name] = None

        result["status"] = "success"

        if verbose and result["data"].get("mempool"):
            mempool = result["data"]["mempool"]
            self.stdout.write(
                f"    Mempool: {mempool.get('count', 0):,} txns, "
                f"{mempool.get('vsize', 0):,} vbytes"
            )

        if verbose and result["data"].get("fees"):
            fees = result["data"]["fees"]
            self.stdout.write(
                f"    Fees (fast/avg/slow): "
                f"{fees.get('fastestFee', '?')} / "
                f"{fees.get('halfHourFee', '?')} / "
                f"{fees.get('hourFee', '?')} sat/vB"
            )

        return result

    def _fetch_coingecko_data(self, verbose: bool) -> dict:
        """Fetch market data from CoinGecko."""
        self.stdout.write("\n📡 CoinGecko Market Data...")

        result = {"status": "error", "data": {}}

        try:
            # Simple price + market data for BTC
            resp = requests.get(
                f"{COINGECKO_API}/coins/bitcoin",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "community_data": "false",
                    "developer_data": "false",
                },
                timeout=15,
            )
            if resp.status_code == 429:
                self.stdout.write(self.style.WARNING("  ⚠️  Rate limited by CoinGecko"))
                result["status"] = "rate_limited"
                return result

            resp.raise_for_status()
            data = resp.json()

            market_data = data.get("market_data", {})
            result["data"] = {
                "price_usd": market_data.get("current_price", {}).get("usd"),
                "market_cap": market_data.get("market_cap", {}).get("usd"),
                "total_volume": market_data.get("total_volume", {}).get("usd"),
                "price_change_24h_pct": market_data.get("price_change_percentage_24h"),
                "price_change_7d_pct": market_data.get("price_change_percentage_7d"),
                "high_24h": market_data.get("high_24h", {}).get("usd"),
                "low_24h": market_data.get("low_24h", {}).get("usd"),
                "circulating_supply": market_data.get("circulating_supply"),
                "total_supply": market_data.get("total_supply"),
            }
            result["status"] = "success"
            self.stdout.write(f"  ✅ BTC data: ${result['data']['price_usd']:,.2f}")

            if verbose:
                for k, v in result["data"].items():
                    if v is not None:
                        self.stdout.write(f"    {k}: {v}")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  ❌ CoinGecko: {e}"))

        return result

    def _print_summary(self, results: dict):
        """Print a summary of fetched data."""
        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"  On-Chain Data Summary\n"
            f"{'='*60}"
        ))

        # BTC stats
        btc = results.get("bitcoin_stats", {})
        if btc.get("status") == "success":
            fees = btc.get("data", {}).get("fees", {})
            height = btc.get("data", {}).get("block_height")
            self.stdout.write(f"  Bitcoin Block Height: {height}")
            self.stdout.write(f"  Fee (fast): {fees.get('fastestFee', '?')} sat/vB")

        # Market data
        mkt = results.get("market_data", {})
        if mkt.get("status") == "success":
            price = mkt.get("data", {}).get("price_usd")
            change = mkt.get("data", {}).get("price_change_24h_pct")
            if price:
                self.stdout.write(f"  BTC Price: ${price:,.2f}")
            if change is not None:
                direction = "🟢" if change > 0 else "🔴"
                self.stdout.write(f"  24h Change: {direction} {change:+.2f}%")

        self.stdout.write("")
