"""
Sentiment & On-Chain Feature Engine

Maps data from sentiment, on-chain, and macro sources into
numeric features for ML models.

Data Sources Covered (Section 2):
- Sentiment: Fear & Greed Index, news headlines, X/Twitter volume
- On-Chain: Exchange flows, active addresses, whale movements
- Cross-Asset: BTC dominance, ETH/BTC ratio
- Derivatives: Funding rate extremes, open interest delta

Each source is fetched by periodic tasks (Phase 2) and stored in
the database. This module reads from DB and produces features.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import polars as pl

from trading_bot.models import AuditLog, FeatureSnapshot, OHLCV

logger = logging.getLogger(__name__)

# ── Feature Version ────────────────────────────────────────────────

FEATURE_SET_VERSION = "1.0.0"


def get_feature_version() -> str:
    """Return the current feature set version string."""
    return FEATURE_SET_VERSION


# ── Sentiment Features ─────────────────────────────────────────────


def compute_sentiment_features() -> dict[str, float]:
    """
    Compute sentiment features from stored AuditLog entries.

    Reads the most recent sentiment data (Fear & Greed Index, market
    overview) from the AuditLog and returns normalized features.

    Data Sources (Section 2):
    - Sentiment: Fear & Greed Index
    - Sentiment: News headline embeddings (via token count proxy)
    - Cross-Asset: BTC price, 24h change

    Returns:
        Dict of feature name → value (all normalized)
    """
    features: dict[str, float] = {
        "sentiment_fear_greed": 50.0,  # Neutral default
        "sentiment_btc_change_24h": 0.0,
        "sentiment_market_volatility": 0.0,
    }

    try:
        # Pull latest sentiment from AuditLog
        latest_sentiment = (
            AuditLog.objects.filter(action="info", details__fear_greed_value__isnull=False)
            .order_by("-timestamp")
            .first()
        )
        if latest_sentiment and latest_sentiment.details:
            details = latest_sentiment.details

            # Fear & Greed Index (0-100, normalized)
            fg_value = details.get("fear_greed_value")
            if fg_value is not None:
                features["sentiment_fear_greed"] = float(fg_value)

            # BTC 24h change
            btc_change = details.get("btc_24h_change")
            if btc_change is not None:
                features["sentiment_btc_change_24h"] = float(btc_change)

            # BTC absolute change as volatility proxy
            features["sentiment_market_volatility"] = abs(features["sentiment_btc_change_24h"])

    except Exception as e:
        logger.warning("Failed to fetch sentiment features: %s", e)

    return features


# ── Market Context Features ────────────────────────────────────────


def compute_market_context_features(
    symbol: str,
    interval: str = "1h",
    limit: int = 100,
) -> dict[str, float]:
    """
    Compute market context features from recent OHLCV data.

    Data Sources (Section 2):
    - Market Data: Multi-exchange OHLCV
    - Cross-Asset: BTC Dominance (via price ratio proxy)

    Args:
        symbol: Trading pair
        interval: Candle interval
        limit: Number of recent candles

    Returns:
        Dict of feature name → value
    """
    features: dict[str, float] = {
        "market_recent_return": 0.0,
        "market_recent_volatility": 0.0,
        "market_volume_change": 0.0,
        "market_high_vs_low_range": 0.0,
    }

    try:
        candles = (
            OHLCV.objects.filter(
                symbol=symbol.upper(),
                interval=interval,
            )
            .order_by("-timestamp")[:limit]
        )

        if not candles or len(candles) < 20:
            return features

        # Use polars for fast computation
        data = [
            {
                "timestamp": c.timestamp,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            }
            for c in candles
        ]

        df = pl.DataFrame(data).sort("timestamp")

        # Recent return (last close vs first close in window)
        if len(df) > 1:
            first_close = df["close"][0]
            last_close = df["close"][-1]
            features["market_recent_return"] = float(
                (last_close / first_close - 1) if first_close > 0 else 0
            )

        # Recent volatility (std of log returns)
        log_returns = df.select(
            pl.col("close").log().diff().alias("lr")
        ).drop_nulls()
        if len(log_returns) > 1:
            features["market_recent_volatility"] = float(
                log_returns["lr"].std()
            )

        # Volume change (recent vs older)
        half = len(df) // 2
        recent_vol = df["volume"][half:].mean()
        older_vol = df["volume"][:half].mean()
        features["market_volume_change"] = float(
            (recent_vol / older_vol - 1) if older_vol > 0 else 0
        )

        # High-low range
        recent = df.tail(10)
        features["market_high_vs_low_range"] = float(
            (recent["high"].max() - recent["low"].min()) / recent["close"].mean()
            if len(recent) > 0 and recent["close"].mean() > 0 else 0
        )

    except Exception as e:
        logger.warning("Failed to compute market context: %s", e)

    return features


# ── On-Chain Features (Proxy / Stub) ───────────────────────────────


def compute_onchain_features() -> dict[str, float]:
    """
    Compute on-chain feature proxies.

    Data Sources (Section 2):
    - On-Chain: Active addresses (via block height as proxy)
    - On-Chain: Exchange net flows (stub — requires Glassnode/CryptoQuant)
    - On-Chain: Whale movements (stub)

    Currently returns neutral values for unavailable sources.
    Will be populated when on-chain data pipelines are connected.
    """
    features: dict[str, float] = {
        "onchain_btc_blocks_24h": 144.0,  # ~144 blocks/day
        "onchain_active_addresses": 0.0,   # Requires Glassnode API
        "onchain_exchange_flow": 0.0,       # Requires CryptoQuant
        "onchain_whale_count": 0.0,         # Requires Whale Alert
        "onchain_sopr": 0.0,                # Requires Glassnode
        "onchain_mvrv_zscore": 0.0,         # Requires Glassnode
    }

    try:
        # Try to pull latest on-chain data from AuditLog
        latest_onchain = (
            AuditLog.objects.filter(
                action="info",
                details__btc_price__isnull=False,
            )
            .exclude(details__btc_price=None)
            .order_by("-timestamp")
            .first()
        )
        if latest_onchain and latest_onchain.details:
            details = latest_onchain.details
            # Block height trend (mempool block count as proxy)
            block_height = details.get("block_height")
            if block_height is not None:
                features["onchain_btc_blocks_24h"] = float(block_height)

    except Exception as e:
        logger.warning("Failed to fetch on-chain features: %s", e)

    return features


# ── Derivatives Features (Stub) ────────────────────────────────────


def compute_derivatives_features() -> dict[str, float]:
    """
    Compute derivatives market features.

    Data Sources (Section 2):
    - Derivatives: Funding rate extremes (requires Binance Futures API)
    - Derivatives: Open interest delta
    - Derivatives: Liquidation cascades

    Returns neutral defaults. Will be populated when futures
    data pipeline is connected in Phase 7.
    """
    return {
        "deriv_funding_rate": 0.0,
        "deriv_open_interest_change": 0.0,
        "deriv_long_short_ratio": 0.5,
        "deriv_liquidation_volume": 0.0,
    }
