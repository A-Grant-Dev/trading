"""
Order Book Feature Engine — Market Microstructure Features

Computes order book features for ML models from real-time/database
order book snapshots.

Data Sources Covered (Section 2):
- Microstructure: Bid-ask spread, order book imbalance, depth imbalance,
  trade aggressor side (buy vs sell volume)
- Market Data: Full L2/L3 order book depth

All functions are pure — accept a snapshot dict, return feature dict.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def compute_orderbook_features(
    bids: list[list[float]],
    asks: list[list[float]],
    timestamp: Optional[datetime] = None,
    top_n: int = 10,
) -> dict[str, float]:
    """
    Compute order book microstructure features from a snapshot.

    Args:
        bids: List of [price, quantity] sorted best bid first
        asks: List of [price, quantity] sorted best ask first
        timestamp: Snapshot timestamp (optional)
        top_n: Number of levels to use for depth features

    Returns:
        Dict of feature name → value

    Features:
        ob_imbalance_pct: bid_volume / total_volume * 100
        ob_spread: best_ask - best_bid
        ob_spread_pct: spread / best_ask * 100
        ob_depth_pressure: (bid_vol_top5 - ask_vol_top5) / (bid_vol_top5 + ask_vol_top5)
        ob_bid_concentration: % of bid volume in top 3 levels
        ob_ask_concentration: % of ask volume in top 3 levels
        ob_bid_volume: total bid volume
        ob_ask_volume: total ask volume
        ob_bid_levels: number of bid levels
        ob_ask_levels: number of ask levels
        ob_micro_price: (best_bid + best_ask) / 2
    """
    features: dict[str, float] = {}

    if not bids or not asks:
        return _empty_features()

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    spread = best_ask - best_bid

    bid_vol_total = sum(q for _, q in bids)
    ask_vol_total = sum(q for _, q in asks)
    total_vol = bid_vol_total + ask_vol_total

    bid_vol_top5 = sum(q for _, q in bids[:5])
    ask_vol_top5 = sum(q for _, q in asks[:5])
    bid_vol_top3 = sum(q for _, q in bids[:3])
    ask_vol_top3 = sum(q for _, q in asks[:3])

    features["ob_imbalance_pct"] = (
        (bid_vol_total / total_vol * 100) if total_vol > 0 else 50.0
    )
    features["ob_spread"] = spread
    features["ob_spread_pct"] = (spread / best_ask * 100) if best_ask > 0 else 0.0
    features["ob_depth_pressure"] = (
        (bid_vol_top5 - ask_vol_top5) / (bid_vol_top5 + ask_vol_top5)
        if (bid_vol_top5 + ask_vol_top5) > 0 else 0.0
    )
    features["ob_bid_concentration"] = (
        (bid_vol_top3 / bid_vol_total * 100) if bid_vol_total > 0 else 0.0
    )
    features["ob_ask_concentration"] = (
        (ask_vol_top3 / ask_vol_total * 100) if ask_vol_total > 0 else 0.0
    )
    features["ob_bid_volume"] = bid_vol_total
    features["ob_ask_volume"] = ask_vol_total
    features["ob_bid_levels"] = float(len(bids))
    features["ob_ask_levels"] = float(len(asks))
    features["ob_micro_price"] = (best_bid + best_ask) / 2.0
    features["ob_best_bid"] = best_bid
    features["ob_best_ask"] = best_ask

    return features


def compute_microstructure_features(
    trades: list[dict[str, Any]],
    window: int = 100,
) -> dict[str, float]:
    """
    Compute market microstructure features from recent trades.

    Data Source (Section 2): Trade aggressor side (buy vs sell volume)

    Args:
        trades: List of trade dicts with keys: price, quantity, is_buyer_maker
        window: Number of recent trades to analyze

    Returns:
        Dict of feature name → value

    Features:
        micro_buy_ratio: fraction of buy trades in window
        micro_buy_volume_ratio: fraction of buy volume in window
        micro_trade_count: number of trades in window
        micro_avg_trade_size: average trade size in window
        micro_volume_imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol)
    """
    features: dict[str, float] = {}

    if not trades:
        features.update({
            "micro_buy_ratio": 0.5,
            "micro_buy_volume_ratio": 0.5,
            "micro_trade_count": 0.0,
            "micro_avg_trade_size": 0.0,
            "micro_volume_imbalance": 0.0,
        })
        return features

    recent = trades[-window:] if len(trades) > window else trades
    total = len(recent)

    buy_trades = [t for t in recent if not t.get("is_buyer_maker", True)]
    buy_volume = sum(t.get("quantity", 0) for t in buy_trades)
    sell_volume = sum(t.get("quantity", 0) for t in recent) - buy_volume
    total_volume = buy_volume + sell_volume

    features["micro_buy_ratio"] = len(buy_trades) / total if total > 0 else 0.5
    features["micro_buy_volume_ratio"] = buy_volume / total_volume if total_volume > 0 else 0.5
    features["micro_trade_count"] = float(total)
    features["micro_avg_trade_size"] = total_volume / total if total > 0 else 0.0
    features["micro_volume_imbalance"] = (
        (buy_volume - sell_volume) / (buy_volume + sell_volume)
        if (buy_volume + sell_volume) > 0 else 0.0
    )

    return features


def _empty_features() -> dict[str, float]:
    """Return empty/neutral feature values."""
    return {
        "ob_imbalance_pct": 50.0,
        "ob_spread": 0.0,
        "ob_spread_pct": 0.0,
        "ob_depth_pressure": 0.0,
        "ob_bid_concentration": 0.0,
        "ob_ask_concentration": 0.0,
        "ob_bid_volume": 0.0,
        "ob_ask_volume": 0.0,
        "ob_bid_levels": 0.0,
        "ob_ask_levels": 0.0,
        "ob_micro_price": 0.0,
        "ob_best_bid": 0.0,
        "ob_best_ask": 0.0,
    }
