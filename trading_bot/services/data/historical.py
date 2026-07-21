"""
Historical Data Pipeline — CCXT OHLCV Downloader

Downloads and stores OHLCV market data from Binance (and other
exchanges) into the OHLCV model.

Key design decisions:
- Supports incremental updates — only fetches what's missing
- Circuit breaker protection on every external call (circuit_breaker.py)
- All timestamps timezone-aware UTC (Coding Agent Rule #4)
- Uses ccxt for exchange abstraction (supports 100+ exchanges)
- Logs every batch download for audit trail

Usage:
    python manage.py download_history --symbol BTCUSDT --interval 1h --days 730
    python manage.py download_history --symbol BTCUSDT,ETHUSDT --interval 1m --days 30
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import ccxt

from trading_bot.models import AuditLog, OHLCV
from trading_bot.services.circuit_breaker import (
    CircuitBreakerOpenError,
    circuit_breaker,
)
from trading_bot.services.config import get_config

logger = logging.getLogger(__name__)

# ── Exchange Instance Cache ──────────────────────────────────────────

_exchange_cache: dict[str, ccxt.Exchange] = {}


def _get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    """
    Get or create a cached CCXT exchange instance.

    Caches the exchange instance to avoid repeated load_markets()
    calls, which hit the exchange API each time.

    Args:
        exchange_id: Exchange identifier (default: binance)

    Returns:
        Configured ccxt Exchange instance
    """
    if exchange_id in _exchange_cache:
        return _exchange_cache[exchange_id]

    exchange_class = getattr(ccxt, exchange_id)
    exchange_cfg = get_config().get("exchange", {})
    exchange = exchange_class({
        "enableRateLimit": exchange_cfg.get("rate_limit_enabled", True),
        "timeout": exchange_cfg.get("timeout_ms", 30000),
        "options": {"defaultType": "spot"},
    })
    exchange.load_markets()
    _exchange_cache[exchange_id] = exchange
    return exchange

def _fetch_page(
    exchange: ccxt.Exchange,
    symbol: str,
    interval: str,
    since_ms: Optional[int],
    limit: int,
) -> list[list]:
    """Raw CCXT OHLCV fetch call (wrapped by circuit breaker)."""
    return exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=interval,
        since=since_ms,
        limit=limit,
    )


@circuit_breaker(
    name="ccxt_ohlcv",
    max_retries=3,
    retry_delay=2.0,
    failure_threshold=3,
    cooldown_seconds=60.0,
    exceptions=(ccxt.NetworkError, ccxt.RateLimitExceeded, ConnectionError),
)
def fetch_ohlcv_range(
    symbol: str,
    interval: str = "1h",
    since: Optional[datetime] = None,
    limit: int = 1000,
    exchange_id: str = "binance",
) -> list[dict]:
    """
    Fetch OHLCV data from exchange using CCXT.

    Protected by circuit breaker (circuit_breaker.py) — after 3
    consecutive failures, blocks requests for 60 seconds.

    Handles pagination automatically when since is provided.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
        since: Start datetime (None = earliest available)
        limit: Max candles per API call (CCXT max is typically 1000)
        exchange_id: Exchange identifier

    Returns:
        List of dicts with keys: timestamp, open, high, low, close, volume
    """
    exchange = _get_exchange(exchange_id)
    exchange_cfg = get_config().get("exchange", {})
    retry_attempts = exchange_cfg.get("retry_attempts", 3)

    since_ms = int(since.timestamp() * 1000) if since else None

    all_candles: list[list] = []
    max_pages = 1000
    page = 0
    last_timestamp = since_ms

    while page < max_pages:
        try:
            candles = _fetch_page(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                since_ms=last_timestamp,
                limit=limit,
            )
        except CircuitBreakerOpenError:
            raise  # Re-raise — caller should handle
        except Exception as e:
            logger.error("Fatal error fetching %s: %s", symbol, e)
            break

        if not candles or len(candles) < 2:
            break  # No more data

        all_candles.extend(candles)
        last_timestamp = candles[-1][0] + 1  # Next page starts after last candle

        if len(candles) < limit:
            break

        page += 1
        time.sleep(0.5 / retry_attempts)  # Adaptive politeness delay

    # Convert to dict format
    results = []
    for c in all_candles:
        ts, o, h, l, cl, v = c[0], c[1], c[2], c[3], c[4], c[5]
        results.append({
            "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            "open": o,
            "high": h,
            "low": l,
            "close": cl,
            "volume": v,
        })

    # Deduplicate by timestamp (CCXT can occasionally return overlapping pages)
    seen_timestamps: set[int] = set()
    unique_results = []
    for r in results:
        ts_key = int(r["timestamp"].timestamp() * 1000)
        if ts_key not in seen_timestamps:
            seen_timestamps.add(ts_key)
            unique_results.append(r)

    logger.info(
        "Fetched %d unique candles for %s %s (from %s)",
        len(unique_results),
        symbol,
        interval,
        since.isoformat() if since else "earliest",
    )
    return unique_results


# ── Database Storage ───────────────────────────────────────────────


def store_ohlcv_batch(
    candles: list[dict],
    symbol: str,
    interval: str,
    exchange: str = "binance",
    batch_size: int = 500,
) -> int:
    """
    Store OHLCV candles in the database using bulk_create with
    conflict handling for incremental updates.

    Args:
        candles: List of candle dicts from fetch_ohlcv_range()
        symbol: Trading pair
        interval: Candle interval
        exchange: Exchange identifier
        batch_size: Number of candles to insert per bulk operation

    Returns:
        Number of new rows inserted
    """
    if not candles:
        return 0

    objs: list[OHLCV] = []
    existing_keys: set[tuple] = set()

    # Pre-fetch existing timestamps for deduplication
    existing = OHLCV.objects.filter(
        exchange=exchange,
        symbol=symbol.upper(),
        interval=interval,
    ).values_list("timestamp", flat=True)
    existing_set = set(
        int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else ts
        for ts in existing
    )

    inserted = 0
    for c in candles:
        ts = c["timestamp"]
        ts_key = int(ts.timestamp() * 1000)

        if ts_key in existing_set or ts_key in existing_keys:
            continue

        existing_keys.add(ts_key)
        objs.append(
            OHLCV(
                exchange=exchange,
                symbol=symbol.upper(),
                interval=interval,
                timestamp=ts,
                open=Decimal(str(c["open"])),
                high=Decimal(str(c["high"])),
                low=Decimal(str(c["low"])),
                close=Decimal(str(c["close"])),
                volume=Decimal(str(c["volume"])),
            )
        )

        if len(objs) >= batch_size:
            try:
                OHLCV.objects.bulk_create(objs, ignore_conflicts=True)
                inserted += len(objs)
                logger.debug("Inserted batch of %d OHLCV rows for %s", len(objs), symbol)
            except Exception as e:
                logger.error("Bulk insert failed for %s: %s", symbol, e)
            objs = []

    # Final batch
    if objs:
        try:
            OHLCV.objects.bulk_create(objs, ignore_conflicts=True)
            inserted += len(objs)
        except Exception as e:
            logger.error("Final bulk insert failed for %s: %s", symbol, e)

    if inserted > 0:
        logger.info("Stored %d new OHLCV rows for %s %s", inserted, symbol, interval)

    return inserted


# ── High-level Orchestrator ────────────────────────────────────────


def download_history(
    symbol: str,
    interval: str = "1h",
    days: int = 365,
    exchange: str = "binance",
    force: bool = False,
) -> dict:
    """
    Download and store historical OHLCV data for a symbol.

    Supports incremental updates — only fetches data that's not
    already in the database, unless force=True.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candle interval (1m, 5m, 1h, 1d, etc.)
        days: Number of days of history to fetch
        exchange: Exchange identifier
        force: Re-download even if data exists

    Returns:
        Dict with keys: symbol, interval, candles_fetched, candles_stored,
        duration_seconds, start_date, end_date
    """
    start_time = time.time()
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)

    logger.info(
        "Starting download for %s %s (last %d days from %s)",
        symbol, interval, days, since.date(),
    )

    # Check what we already have (for incremental support)
    if not force:
        latest = (
            OHLCV.objects.filter(
                exchange=exchange,
                symbol=symbol.upper(),
                interval=interval,
            )
            .order_by("-timestamp")
            .first()
        )
        if latest:
            # Only fetch what's missing
            since = max(since, latest.timestamp + timedelta(seconds=1))
            logger.info(
                "Incremental update — newest candle is %s, fetching from %s",
                latest.timestamp, since,
            )
            if since >= now:
                logger.info("Data is already up to date for %s %s", symbol, interval)
                return {
                    "symbol": symbol,
                    "interval": interval,
                    "candles_fetched": 0,
                    "candles_stored": 0,
                    "duration_seconds": 0,
                    "start_date": None,
                    "end_date": None,
                    "status": "up_to_date",
                }

    # Fetch from exchange
    candles = fetch_ohlcv_range(
        symbol=symbol,
        interval=interval,
        since=since,
        exchange_id=exchange,
    )

    if not candles:
        logger.warning("No data returned for %s %s", symbol, interval)
        return {
            "symbol": symbol,
            "interval": interval,
            "candles_fetched": 0,
            "candles_stored": 0,
            "duration_seconds": time.time() - start_time,
            "start_date": None,
            "end_date": None,
            "status": "no_data",
        }

    # Store in database
    stored = store_ohlcv_batch(
        candles=candles,
        symbol=symbol,
        interval=interval,
        exchange=exchange,
    )

    duration = time.time() - start_time
    start_date = candles[0]["timestamp"] if candles else None
    end_date = candles[-1]["timestamp"] if candles else None

    # Log to audit trail
    AuditLog.objects.create(
        action="info",
        message=f"Downloaded {stored} new candles for {symbol} {interval} ({days}d)",
        details={
            "symbol": symbol,
            "interval": interval,
            "days": days,
            "fetched": len(candles),
            "stored": stored,
            "duration_seconds": round(duration, 2),
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
        },
    )

    logger.info(
        "Download complete for %s %s — fetched %d, stored %d (%.1fs)",
        symbol, interval, len(candles), stored, duration,
    )

    return {
        "symbol": symbol,
        "interval": interval,
        "candles_fetched": len(candles),
        "candles_stored": stored,
        "duration_seconds": round(duration, 2),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "status": "success",
    }
