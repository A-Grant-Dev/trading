"""
Data Export & Transformation Utilities

Transforms raw market data into model-ready feature matrices.
Provides technical indicators, resampling, and feature extraction.

Note: We implement indicators manually using numpy/pandas instead of
pandas-ta because numba (pandas-ta's dependency) doesn't support
Python 3.14 yet.

Renaissance principle: Feature engineering is where quant models
win or lose. The right features reveal hidden statistical patterns.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import requests

from quant.models import MarketData

logger = logging.getLogger(__name__)

BINANCE_API = "https://api.binance.com"

# ── DataFrame Conversion ──────────────────────────────────────────


def ohlcv_to_dataframe(
    symbol: str,
    interval: str = "1h",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from MarketData table and return as a DataFrame.

    Falls back to Binance REST API if no data in DB.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Kline interval
        limit: Max rows

    Returns:
        DataFrame with columns: open, high, low, close, volume,
        quote_asset_volume, trades, taker_buy_base_vol,
        taker_buy_quote_vol. Index: open_time (datetime)
    """
    qs = MarketData.objects.filter(
        symbol=symbol.upper(),
        interval=interval,
    ).order_by("-open_time")[:limit]

    # Cannot reorder after slicing, so reverse in Python
    data = list(reversed(qs.values(
        "open_time", "open", "high", "low", "close",
        "volume", "quote_asset_volume", "trades",
        "taker_buy_base_vol", "taker_buy_quote_vol",
    )))

    if data and len(data) >= 20:
        df = pd.DataFrame(data)
        df.set_index("open_time", inplace=True)
        return _convert_decimals(df)

    # Fallback: fetch from Binance REST
    logger.info(f"No cached data for {symbol} {interval}, fetching from Binance REST")
    return _fetch_from_binance_rest(symbol, interval, limit)


def _convert_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Decimal columns to float for numpy compatibility."""
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = df[col].astype(float)
        except (ValueError, TypeError):
            pass
    return df


def _fetch_from_binance_rest(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch klines from Binance REST API as DataFrame."""
    try:
        resp = requests.get(
            f"{BINANCE_API}/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch from Binance REST: {e}")
        return pd.DataFrame()

    rows = []
    for k in raw:
        rows.append({
            "open_time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc),
            "quote_asset_volume": float(k[7]),
            "trades": int(k[8]),
            "taker_buy_base_vol": float(k[9]),
            "taker_buy_quote_vol": float(k[10]),
        })

    df = pd.DataFrame(rows)
    df.set_index("open_time", inplace=True)
    return df


# ── Technical Indicators ───────────────────────────────────────────


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to a OHLCV DataFrame.

    Implements indicators using numpy/pandas directly since pandas-ta
    is not available on Python 3.14.

    Added columns:
        rsi_14, macd, macd_signal, macd_hist,
        bb_upper, bb_middle, bb_lower, bb_width,
        atr_14, atr_pct,
        ema_9, ema_21, ema_50,
        volume_sma_20, volume_ratio,
        obv (On-Balance Volume),
        vwap (Volume-Weighted Average Price)

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with additional indicator columns (NaN for warmup periods)
    """
    if df.empty:
        return df
    df = df.copy()
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float) if "volume" in df.columns else None

    # ── RSI (14) ───────────────────────────────────────────────────
    df["rsi_14"] = compute_rsi(close, 14)

    # ── MACD (12, 26, 9) ──────────────────────────────────────────
    ema12 = _compute_ema(close, 12)
    ema26 = _compute_ema(close, 26)
    macd_line = ema12 - ema26
    signal = _compute_ema(macd_line, 9)
    df["macd"] = macd_line
    df["macd_signal"] = signal
    df["macd_hist"] = macd_line - signal

    # ── Bollinger Bands (20, 2) ──────────────────────────────────
    sma20 = _compute_sma(close, 20)
    std20 = _compute_rolling_std(close, 20)
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_middle"] = sma20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # ── ATR (14) ──────────────────────────────────────────────────
    df["atr_14"] = compute_atr(high, low, close, 14)
    df["atr_pct"] = df["atr_14"] / df["close"] * 100

    # ── EMAs ─────────────────────────────────────────────────────
    df["ema_9"] = _compute_ema(close, 9)
    df["ema_21"] = _compute_ema(close, 21)
    df["ema_50"] = _compute_ema(close, 50)

    # ── Volume Indicators ─────────────────────────────────────────
    if volume is not None:
        df["volume_sma_20"] = _compute_sma(volume, 20)
        df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)
        df["obv"] = _compute_obv(close, volume)

    # ── VWAP ──────────────────────────────────────────────────────
    if volume is not None:
        typical_price = (high + low + close) / 3
        cum_pv = np.cumsum(typical_price * volume)
        cum_vol = np.cumsum(volume)
        df["vwap"] = cum_pv / cum_vol

    # ── Log Returns ──────────────────────────────────────────────
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_15"] = df["close"].pct_change(15)

    # ── Volatility ──────────────────────────────────────────────
    df["volatility_14"] = df["log_return"].rolling(14).std()
    df["volatility_30"] = df["log_return"].rolling(30).std()

    return df


def _compute_sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(period - 1, len(values)):
        result[i] = np.mean(values[i - period + 1 : i + 1])
    return result


def _compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average with tolerance for leading NaN values."""
    result = np.full_like(values, np.nan, dtype=float)
    if len(values) < period:
        return result
    alpha = 2 / (period + 1)

    # Find first position with `period` consecutive non-NaN values
    mask = ~np.isnan(values)
    start = period - 1
    while start < len(values):
        if mask[start - period + 1:start + 1].all():
            result[start] = np.nanmean(values[start - period + 1:start + 1])
            break
        start += 1

    if start >= len(values) or np.isnan(result[start]):
        return result  # No valid segment found

    for i in range(start + 1, len(values)):
        if not mask[i]:
            break  # Stop at next gap
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def compute_rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    result = np.full_like(values, np.nan, dtype=float)
    if len(values) < period + 1:
        return result

    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.full_like(values, np.nan, dtype=float)
    avg_loss = np.full_like(values, np.nan, dtype=float)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(values)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = avg_gain / np.where(avg_loss == 0, 0.001, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    result = rsi
    return result


def _compute_rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(period - 1, len(values)):
        result[i] = np.std(values[i - period + 1 : i + 1], ddof=1)
    return result


def compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average True Range."""
    result = np.full_like(close, np.nan, dtype=float)
    if len(close) < period + 1:
        return result

    tr = np.full_like(close, np.nan, dtype=float)
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    result[period] = np.mean(tr[1 : period + 1])
    for i in range(period + 1, len(close)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def _compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    result = np.zeros_like(volume)
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            result[i] = result[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            result[i] = result[i - 1] - volume[i]
        else:
            result[i] = result[i - 1]
    return result


# ── Resampling ─────────────────────────────────────────────────────


def resample_ohlcv(
    df: pd.DataFrame, target_interval: str
) -> pd.DataFrame:
    """
    Resample OHLCV data from a lower interval to a higher one.

    E.g., 1m → 5m, 5m → 1h, 1h → 1d

    Args:
        df: DataFrame with datetime index and OHLCV columns
        target_interval: Target pandas offset string (e.g., '5min', '1h', '1D')

    Returns:
        Resampled DataFrame with OHLCV aggregation
    """
    if df.empty:
        return df

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Add optional columns if they exist
    if "quote_asset_volume" in df.columns:
        ohlc_dict["quote_asset_volume"] = "sum"
    if "trades" in df.columns:
        ohlc_dict["trades"] = "sum"
    if "taker_buy_base_vol" in df.columns:
        ohlc_dict["taker_buy_base_vol"] = "sum"
    if "taker_buy_quote_vol" in df.columns:
        ohlc_dict["taker_buy_quote_vol"] = "sum"

    return df.resample(target_interval).agg(ohlc_dict).dropna()


# ── Order Book Features ────────────────────────────────────────────


def compute_order_book_features(symbol: str) -> dict:
    """
    Compute order book features from snapshot.

    Fetches the latest snapshot from Binance REST API and computes
    features useful for ML models.

    Features:
        imbalance_pct: bid_vol / (bid_vol + ask_vol) * 100
        spread_pct: spread / best_ask * 100
        depth_pressure: (bid_vol_top5 - ask_vol_top5) / (bid_vol_top5 + ask_vol_top5)
        bid_concentration: % of total bid volume in top 3 levels
        ask_concentration: % of total ask volume in top 3 levels

    Returns:
        Dict of features or empty dict on failure
    """
    try:
        resp = requests.get(
            f"{BINANCE_API}/api/v3/depth",
            params={"symbol": symbol.upper(), "limit": 20},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch depth for {symbol}: {e}")
        return {}

    bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
    asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

    if not bids or not asks:
        return {}

    bid_vol_total = sum(q for _, q in bids)
    ask_vol_total = sum(q for _, q in asks)
    total_vol = bid_vol_total + ask_vol_total

    bid_vol_top5 = sum(q for _, q in bids[:5])
    ask_vol_top5 = sum(q for _, q in asks[:5])

    bid_vol_top3 = sum(q for _, q in bids[:3])
    ask_vol_top3 = sum(q for _, q in asks[:3])

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    spread = best_ask - best_bid

    features = {
        "symbol": symbol.upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "imbalance_pct": (bid_vol_total / total_vol * 100) if total_vol > 0 else 50.0,
        "spread": spread,
        "spread_pct": (spread / best_ask * 100) if best_ask > 0 else 0,
        "depth_pressure": (
            (bid_vol_top5 - ask_vol_top5) / (bid_vol_top5 + ask_vol_top5)
            if (bid_vol_top5 + ask_vol_top5) > 0
            else 0
        ),
        "bid_concentration": (bid_vol_top3 / bid_vol_total * 100) if bid_vol_total > 0 else 0,
        "ask_concentration": (ask_vol_top3 / ask_vol_total * 100) if ask_vol_total > 0 else 0,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_levels": len(bids),
        "ask_levels": len(asks),
    }

    return features


# ── Feature Matrix Builder (for ML) ───────────────────────────────


def build_feature_matrix(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """
    Build a complete feature matrix from raw market data.

    Combines OHLCV data with technical indicators and order book features.

    This is the primary input for ML model training (Phase 4).

    Args:
        symbol: Trading pair
        interval: Kline interval
        limit: Number of candles to include

    Returns:
        DataFrame with all features, NaN rows removed
    """
    df = ohlcv_to_dataframe(symbol, interval, limit)
    if df.empty:
        return df

    # Add technical indicators
    df = add_technical_indicators(df)

    # Add order book features (current snapshot only)
    ob_features = compute_order_book_features(symbol)
    if ob_features:
        df["ob_imbalance"] = ob_features.get("imbalance_pct", 50.0)
        df["ob_spread_pct"] = ob_features.get("spread_pct", 0)
        df["ob_depth_pressure"] = ob_features.get("depth_pressure", 0)

    # Drop NaN rows (warmup periods from indicators)
    df.dropna(inplace=True)

    return df
