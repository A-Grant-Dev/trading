"""
Feature Engine — Master Orchestrator

The central feature computation pipeline that combines all data
sources into a unified, versioned feature matrix.

Architecture:
    OHLCV (DB) ──→ Technical Features (polars) ──┐
    Order Book ──→ Microstructure Features ───────┤
    Sentiment ───→ Sentiment Features ────────────┼→ Feature Matrix ──→ FeatureSnapshot (DB)
    On-Chain ────→ On-Chain Features ─────────────┤
    Derivatives ─→ Derivatives Features ──────────┘

The pipeline is designed to be:
- Incremental: Only recompute features for new data
- Versioned: Feature_set_version tracks the pipeline version
- Pure: Side effects only in the store_to_db function
- Fast: Uses polars LazyFrame for vectorized computation
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import polars as pl

from trading_bot.models import FeatureSnapshot, OHLCV
from trading_bot.services.data.buffer import get_buffer
from trading_bot.services.features.orderbook import (
    compute_microstructure_features,
    compute_orderbook_features,
)
from trading_bot.services.features.sentiment_features import (
    compute_derivatives_features,
    compute_market_context_features,
    compute_onchain_features,
    compute_sentiment_features,
    get_feature_version,
)
from trading_bot.services.features.technical import add_technical_features

logger = logging.getLogger(__name__)

# ── Feature Source Names ──────────────────────────────────────────

FEATURE_SOURCES = {
    "technical": "OHLCV Technical Indicators",
    "orderbook": "Order Book Microstructure",
    "sentiment": "Sentiment & Fear/Greed",
    "market_context": "Market Context & Cross-Asset",
    "onchain": "On-Chain Metrics",
    "derivatives": "Derivatives & Funding",
}


def build_feature_matrix(
    symbol: str,
    interval: str = "1h",
    days: Optional[int] = None,
    limit: Optional[int] = None,
    include_sources: Optional[list[str]] = None,
) -> pl.LazyFrame:
    """
    Build a complete feature matrix for a symbol.

    Combines technical indicators, order book features, sentiment,
    on-chain, and derivatives data into a single LazyFrame.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candle interval (1m, 5m, 1h, 1d, etc.)
        days: Number of days of history (mutually exclusive with limit)
        limit: Max candles to include (mutually exclusive with days)
        include_sources: List of feature sources to include.
            Default: all sources (None)

    Returns:
        Polars LazyFrame with feature columns and 'timestamp' index.
        NaN values for warmup periods.
    """
    if include_sources is None:
        include_sources = list(FEATURE_SOURCES.keys())

    # ── Fetch OHLCV Data ───────────────────────────────────────
    t0 = time.time()

    qs = OHLCV.objects.filter(
        exchange="binance",
        symbol=symbol.upper(),
        interval=interval,
    ).order_by("timestamp")

    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        qs = qs.filter(timestamp__gte=cutoff)

    if limit:
        # Get the most recent N entries
        qs = qs.order_by("-timestamp")[:limit]

    candles = list(qs.values(
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_volume",
    ))

    if not candles:
        logger.warning("No OHLCV data for %s %s", symbol, interval)
        return pl.LazyFrame()

    # Convert to polars
    df = pl.DataFrame(
        [
            {
                "timestamp": c["timestamp"],
                "open": float(c["open"]) if c["open"] is not None else 0.0,
                "high": float(c["high"]) if c["high"] is not None else 0.0,
                "low": float(c["low"]) if c["low"] is not None else 0.0,
                "close": float(c["close"]) if c["close"] is not None else 0.0,
                "volume": float(c["volume"]) if c["volume"] else 0.0,
            }
            for c in candles
        ],
        schema={
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    ).sort("timestamp")

    lf = df.lazy()
    n_rows_initial = len(df)
    t1 = time.time()
    logger.info("Loaded %d OHLCV rows for %s %s in %.2fs", n_rows_initial, symbol, interval, t1 - t0)

    # ── Technical Features ──────────────────────────────────────
    if "technical" in include_sources:
        lf = add_technical_features(lf)
        logger.info("Added technical features")

    # ── Static Features (added as constant columns) ─────────────
    # These are the same for every row in the window

    static_features = {}

    # Sentiment features
    if "sentiment" in include_sources:
        sent_features = compute_sentiment_features()
        static_features.update(sent_features)
        logger.info("Added sentiment features")

    # Market context features
    if "market_context" in include_sources:
        ctx_features = compute_market_context_features(symbol, interval, limit or 100)
        static_features.update(ctx_features)
        logger.info("Added market context features")

    # On-chain features
    if "onchain" in include_sources:
        onchain_features = compute_onchain_features()
        static_features.update(onchain_features)
        logger.info("Added on-chain features")

    # Derivatives features
    if "derivatives" in include_sources:
        deriv_features = compute_derivatives_features()
        static_features.update(deriv_features)
        logger.info("Added derivatives features")

    # Add static features as constant columns
    for feat_name, feat_value in static_features.items():
        lf = lf.with_columns(pl.lit(feat_value).alias(feat_name))

    # ── Order Book Features (from live buffer) ─────────────────
    if "orderbook" in include_sources:
        try:
            buffer = get_buffer()
            ob_data = buffer.get_orderbook(symbol)
            if ob_data and "bids" in ob_data and "asks" in ob_data:
                ob_features = compute_orderbook_features(
                    ob_data["bids"], ob_data["asks"]
                )
                for feat_name, feat_value in ob_features.items():
                    lf = lf.with_columns(pl.lit(feat_value).alias(feat_name))
                logger.info("Added live order book features")

                # Microstructure features from recent trades
                trades = buffer.get_trades(symbol, n=100)
                micro_features = compute_microstructure_features(trades)
                for feat_name, feat_value in micro_features.items():
                    lf = lf.with_columns(pl.lit(feat_value).alias(feat_name))
                logger.info("Added microstructure features")
        except Exception as e:
            logger.warning("Failed to add order book features: %s", e)

    total_time = time.time() - t0
    logger.info(
        "Feature matrix built for %s %s: %d cols × %d rows in %.2fs",
        symbol, interval,
        len(lf.columns),
        n_rows_initial,
        total_time,
    )

    return lf


# ── Storage ────────────────────────────────────────────────────────


def store_feature_snapshot(
    lf: pl.LazyFrame,
    symbol: str,
    interval: str,
    version: Optional[str] = None,
    batch_size: int = 500,
) -> int:
    """
    Store the feature matrix as FeatureSnapshot records in the DB.

    Each row in the LazyFrame becomes one FeatureSnapshot record
    with the row's features stored as JSON.

    Args:
        lf: Polars LazyFrame with feature columns (must include 'timestamp')
        symbol: Trading pair
        interval: Candle interval
        version: Feature set version (default: from sentiment_features)
        batch_size: Number of rows per bulk_create batch

    Returns:
        Number of snapshots stored
    """
    if version is None:
        version = get_feature_version()

    try:
        df = lf.collect()
    except Exception as e:
        logger.error("Failed to collect LazyFrame: %s", e)
        return 0

    if df.is_empty():
        return 0

    # Compute source hash for reproducibility
    source_hash = hashlib.sha256(
        json.dumps(
            {k: str(v) for k, v in zip(df.columns, [df[k].mean() for k in df.columns])},
            default=str,
        ).encode()
    ).hexdigest()[:64]

    objs: list[FeatureSnapshot] = []
    inserted = 0
    exclude_cols = {"timestamp", "open", "high", "low", "close", "volume"}

    for row in df.iter_rows(named=True):
        ts = row.get("timestamp")
        if ts is None:
            continue

        # Build features dict (exclude raw OHLCV columns)
        features = {
            k: float(v) if v is not None else 0.0
            for k, v in row.items()
            if k not in exclude_cols
        }

        objs.append(
            FeatureSnapshot(
                timestamp=ts,
                symbol=symbol.upper(),
                interval=interval,
                feature_set_version=version,
                features=features,
                source_hash=source_hash,
            )
        )

        if len(objs) >= batch_size:
            try:
                FeatureSnapshot.objects.bulk_create(objs, ignore_conflicts=True)
                inserted += len(objs)
            except Exception as e:
                logger.error("Bulk create failed: %s", e)
            objs = []

    if objs:
        try:
            FeatureSnapshot.objects.bulk_create(objs, ignore_conflicts=True)
            inserted += len(objs)
        except Exception as e:
            logger.error("Final bulk create failed: %s", e)

    logger.info(
        "Stored %d feature snapshots for %s %s (v%s)",
        inserted, symbol, interval, version,
    )
    return inserted


# ── High-level Orchestrator ────────────────────────────────────────


def rebuild_features(
    symbol: str,
    interval: str = "1h",
    days: Optional[int] = None,
    limit: Optional[int] = None,
    store: bool = True,
    version: Optional[str] = None,
) -> dict[str, Any]:
    """
    Full feature rebuild pipeline for a single symbol.

    Fetches raw data → computes features → stores snapshots.

    This is the main entry point called by the management command,
    Celery tasks, and the optimizer.

    Args:
        symbol: Trading pair
        interval: Candle interval
        days: History in days (mutually exclusive with limit)
        limit: Max candles (mutually exclusive with days)
        store: Whether to persist to FeatureSnapshot table
        version: Feature set version override

    Returns:
        Dict with keys: symbol, interval, n_features, n_rows,
        duration_seconds, version, n_stored
    """
    t0 = time.time()

    lf = build_feature_matrix(
        symbol=symbol,
        interval=interval,
        days=days,
        limit=limit,
    )

    try:
        collected = lf.collect()
        n_rows = len(collected)
        n_features = len(collected.columns) - 6  # Exclude OHLCV columns
    except Exception:
        n_rows = 0
        n_features = 0

    n_stored = 0
    if store and n_rows > 0:
        n_stored = store_feature_snapshot(
            lf=lf,
            symbol=symbol,
            interval=interval,
            version=version,
        )

    duration = time.time() - t0

    return {
        "symbol": symbol,
        "interval": interval,
        "n_features": n_features,
        "n_rows": n_rows,
        "n_stored": n_stored,
        "duration_seconds": round(duration, 2),
        "version": version or get_feature_version(),
        "status": "success" if n_rows > 0 else "no_data",
    }
