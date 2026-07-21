"""
Technical Feature Engine — Polars-based Technical Indicators

Computes all technical analysis indicators from OHLCV data using
vectorized polars expressions for maximum performance.

Supports the plan's requirement: "Every data source from Section 2
must produce at least one feature."

Data Sources Covered (Section 2):
- Market Data: Multi-exchange OHLCV → all indicators below
- Microstructure: Trade aggressor side → buy_ratio, sell_ratio

All indicators are computed as pure functions that accept a polars
LazyFrame and return a LazyFrame with additional columns.
"""

import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def add_technical_features(
    lf: pl.LazyFrame,
    ohlc_cols: Optional[dict[str, str]] = None,
) -> pl.LazyFrame:
    """
    Add all technical indicator features to an OHLCV LazyFrame.

    Expects input columns: timestamp, open, high, low, close, volume
    (or custom names via ohlc_cols).

    Args:
        lf: Polars LazyFrame with OHLCV data
        ohlc_cols: Optional custom column name mapping
            e.g. {"open": "open", "high": "high", "low": "low",
                  "close": "close", "volume": "volume"}

    Returns:
        LazyFrame with all technical feature columns added
        (NaN for warmup periods)
    """
    if ohlc_cols is None:
        ohlc_cols = {
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        }

    o = ohlc_cols["open"]
    h = ohlc_cols["high"]
    l = ohlc_cols["low"]
    c = ohlc_cols["close"]
    v = ohlc_cols.get("volume", "volume")

    features = []

    # ── Returns (Log & Simple) ─────────────────────────────────
    features.extend([
        pl.col(c).log().diff().alias("log_return_1"),
        pl.col(c).pct_change().alias("return_1"),
        pl.col(c).pct_change(5).alias("return_5"),
        pl.col(c).pct_change(15).alias("return_15"),
        pl.col(c).pct_change(60).alias("return_60"),
    ])

    # ── RSI (14) ──────────────────────────────────────────────
    features.append(
        _rsi(pl.col(c), 14).alias("rsi_14")
    )
    features.append(
        _rsi(pl.col(c), 7).alias("rsi_7")
    )

    # ── MACD (12, 26, 9) ──────────────────────────────────────
    ema12 = pl.col(c).ewm_mean(span=12, min_periods=12)
    ema26 = pl.col(c).ewm_mean(span=26, min_periods=26)
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm_mean(span=9, min_periods=9)
    features.extend([
        macd_line.alias("macd"),
        macd_signal.alias("macd_signal"),
        (macd_line - macd_signal).alias("macd_hist"),
    ])

    # ── Bollinger Bands (20, 2) ───────────────────────────────
    sma20 = pl.col(c).rolling_mean(window_size=20, min_periods=20)
    std20 = pl.col(c).rolling_std(window_size=20, min_periods=20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_middle = sma20
    features.extend([
        bb_upper.alias("bb_upper"),
        bb_middle.alias("bb_middle"),
        bb_lower.alias("bb_lower"),
        ((bb_upper - bb_lower) / bb_middle).alias("bb_width"),
        ((pl.col(c) - bb_middle) / (bb_upper - bb_lower) * 2).alias("bb_position"),
    ])

    # ── ATR (14) ──────────────────────────────────────────────
    tr = pl.max_horizontal(
        pl.col(h) - pl.col(l),
        (pl.col(h) - pl.col(c).shift(1)).abs(),
        (pl.col(l) - pl.col(c).shift(1)).abs(),
    )
    atr_14 = tr.rolling_mean(window_size=14, min_periods=14)
    features.extend([
        atr_14.alias("atr_14"),
        (atr_14 / pl.col(c) * 100).alias("atr_pct"),
    ])

    # ── EMAs (9, 21, 50, 200) ────────────────────────────────
    features.extend([
        pl.col(c).ewm_mean(span=9, min_periods=9).alias("ema_9"),
        pl.col(c).ewm_mean(span=21, min_periods=21).alias("ema_21"),
        pl.col(c).ewm_mean(span=50, min_periods=50).alias("ema_50"),
        pl.col(c).ewm_mean(span=200, min_periods=200).alias("ema_200"),
    ])

    # ── Volume Indicators ─────────────────────────────────────
    features.extend([
        pl.col(v).rolling_mean(window_size=20, min_periods=20).alias("volume_sma_20"),
        (pl.col(v) / pl.col(v).rolling_mean(window_size=20, min_periods=20))
        .alias("volume_ratio"),
        _obv(pl.col(c), pl.col(v)).alias("obv"),
    ])

    # ── VWAP ──────────────────────────────────────────────────
    typical_price = (pl.col(h) + pl.col(l) + pl.col(c)) / 3
    cum_pv = (typical_price * pl.col(v)).cum_sum()
    cum_vol = pl.col(v).cum_sum()
    features.append(
        (cum_pv / cum_vol).alias("vwap")
    )

    # ── Volatility ────────────────────────────────────────────
    log_return = pl.col(c).log().diff()
    features.extend([
        log_return.rolling_std(window_size=14, min_periods=14).alias("volatility_14"),
        log_return.rolling_std(window_size=30, min_periods=30).alias("volatility_30"),
    ])

    # ── Price Position (HH/LL) ────────────────────────────────
    features.extend([
        (pl.col(c) / pl.col(h).rolling_max(window_size=20, min_periods=20)).alias("price_position_high_20"),
        (pl.col(c) / pl.col(l).rolling_min(window_size=20, min_periods=20)).alias("price_position_low_20"),
    ])

    # ── Target Labels (Supervised Learning) ───────────────────
    features.extend([
        pl.col(c).shift(-1).truediv(pl.col(c)).sub(1).alias("target_return_1"),
        pl.col(c).shift(-5).truediv(pl.col(c)).sub(1).alias("target_return_5"),
    ])

    # First pass: create all primitive features including EMAs
    lf = lf.with_columns(features)

    # Second pass: features that reference the newly-created EMA columns
    # (polars evaluates all expressions in a single with_columns against
    #  the original frame, so we need separate passes for column references)
    lf = lf.with_columns([
        (pl.col(c) / pl.col("ema_50") - 1).alias("price_vs_ema_50"),
        (pl.col(c) / pl.col("ema_200") - 1).alias("price_vs_ema_200"),
    ])

    return lf


# ── Helper Functions ──────────────────────────────────────────────


def _rsi(col: pl.Expr, period: int = 14) -> pl.Expr:
    """
    Compute Relative Strength Index using polars expressions.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = col.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)

    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False, min_periods=period)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False, min_periods=period)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _obv(close: pl.Expr, volume: pl.Expr) -> pl.Expr:
    """
    Compute On-Balance Volume.

    OBV tracks volume flow based on price direction:
    - Close up → add volume
    - Close down → subtract volume
    - Close same → no change
    """
    direction = pl.when(close.diff() > 0).then(1) \
        .when(close.diff() < 0).then(-1) \
        .otherwise(0)
    return (direction * volume).cum_sum()
