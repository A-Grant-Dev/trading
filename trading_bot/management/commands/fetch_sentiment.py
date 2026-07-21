"""
Management command to fetch market sentiment data from public APIs.

Fetches sentiment indicators from free sources:
- Crypto Fear & Greed Index (alternative.me)
- Market overview from CoinGecko
- Simple sentiment scoring from price action

This data flows into the feature engine and strategy signals.

Usage:
    python manage.py fetch_sentiment
    python manage.py fetch_sentiment --verbose
"""

import json
import logging
from datetime import datetime, timezone

import requests
from django.core.management.base import BaseCommand

from trading_bot.models import AuditLog

logger = logging.getLogger(__name__)

# ── Free Public API Endpoints ─────────────────────────────────────

FEAR_GREED_API = "https://api.alternative.me/fng/"
COINGECKO_API = "https://api.coingecko.com/api/v3"


class Command(BaseCommand):
    help = "Fetch market sentiment data from public APIs"

    def add_arguments(self, parser):
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Show detailed output",
        )
        parser.add_argument(
            "--store",
            action="store_true",
            default=True,
            help="Store results in AuditLog (default: True)",
        )

    def handle(self, *args, **options):
        verbose = options["verbose"]
        store = options["store"]

        self.stdout.write(self.style.SUCCESS(
            f"\n📊 Fetching Market Sentiment\n"
            f"{'='*60}"
        ))

        results = {}

        # 1. Fear & Greed Index
        results["fear_greed"] = self._fetch_fear_greed(verbose)

        # 2. Market sentiment overview
        results["market_sentiment"] = self._fetch_market_sentiment(verbose)

        # 3. Store in audit log
        if store:
            self._store_results(results)

        # 4. Display summary
        self._print_summary(results)

    def _fetch_fear_greed(self, verbose: bool) -> dict:
        """Fetch Crypto Fear & Greed Index."""
        self.stdout.write("\n📡 Fear & Greed Index...")

        result = {"status": "error", "data": {}}

        try:
            resp = requests.get(
                FEAR_GREED_API,
                params={"limit": 1, "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("data") and len(data["data"]) > 0:
                entry = data["data"][0]
                result["data"] = {
                    "value": int(entry.get("value", 50)),
                    "classification": entry.get("value_classification", "Neutral"),
                    "timestamp": datetime.fromtimestamp(
                        int(entry.get("timestamp", 0)), tz=timezone.utc
                    ).isoformat() if entry.get("timestamp") else None,
                }
                result["status"] = "success"
                self.stdout.write(
                    f"  ✅ Fear & Greed: {result['data']['value']}/100 "
                    f"({result['data']['classification']})"
                )
            else:
                self.stdout.write(self.style.ERROR("  ❌ No data returned"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  ❌ Fear & Greed API: {e}"))

        return result

    def _fetch_market_sentiment(self, verbose: bool) -> dict:
        """Fetch market sentiment overview from CoinGecko."""
        self.stdout.write("\n📡 Market Sentiment Overview...")

        result = {"status": "error", "data": {}}

        try:
            # BTC price and 24h change as a simple sentiment signal
            resp = requests.get(
                f"{COINGECKO_API}/simple/price",
                params={
                    "ids": "bitcoin,ethereum",
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_market_cap": "true",
                    "include_24hr_vol": "true",
                },
                timeout=15,
            )

            if resp.status_code == 429:
                self.stdout.write(self.style.WARNING("  ⚠️  Rate limited by CoinGecko"))
                result["status"] = "rate_limited"
                return result

            resp.raise_for_status()
            data = resp.json()

            result["data"] = {
                "btc_price_usd": data.get("bitcoin", {}).get("usd"),
                "btc_24h_change": data.get("bitcoin", {}).get("usd_24h_change"),
                "btc_market_cap": data.get("bitcoin", {}).get("usd_market_cap"),
                "eth_price_usd": data.get("ethereum", {}).get("usd"),
                "eth_24h_change": data.get("ethereum", {}).get("usd_24h_change"),
            }
            result["status"] = "success"

            btc_change = result["data"].get("btc_24h_change")
            if btc_change is not None:
                emoji = "🟢" if btc_change > 0 else "🔴"
                self.stdout.write(
                    f"  ✅ BTC: ${result['data']['btc_price_usd']:,.2f} "
                    f"{emoji} {btc_change:+.2f}%"
                )

            if verbose:
                for k, v in result["data"].items():
                    if v is not None:
                        self.stdout.write(f"    {k}: {v}")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  ❌ Market sentiment: {e}"))

        return result

    def _store_results(self, results: dict):
        """Store sentiment data in AuditLog."""
        fg = results.get("fear_greed", {}).get("data", {})
        mkt = results.get("market_sentiment", {}).get("data", {})

        details = {
            "fear_greed_value": fg.get("value"),
            "fear_greed_classification": fg.get("classification"),
            "btc_price": mkt.get("btc_price_usd"),
            "btc_24h_change": mkt.get("btc_24h_change"),
            "eth_price": mkt.get("eth_price_usd"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        AuditLog.objects.create(
            action="info",
            message=f"Sentiment data: F&G={fg.get('value', '?')}/100 ({fg.get('classification', '?')}), "
            f"BTC=${mkt.get('btc_price_usd', '?'):,}",
            details=details,
        )

    def _print_summary(self, results: dict):
        """Print sentiment summary."""
        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"  Sentiment Summary\n"
            f"{'='*60}"
        ))

        fg = results.get("fear_greed", {}).get("data", {})
        if fg:
            value = fg.get("value", "?")
            classification = fg.get("classification", "?")
            self.stdout.write(f"  Fear & Greed: {value}/100 ({classification})")

        mkt = results.get("market_sentiment", {}).get("data", {})
        if mkt:
            btc = mkt.get("btc_price_usd")
            if btc:
                self.stdout.write(f"  BTC: ${btc:,.2f}")

        self.stdout.write("")
