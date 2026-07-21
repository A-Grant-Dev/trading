"""
Technical Strategies — Momentum, Mean-Reversion, Breakout

Implements three core technical analysis strategies that detect
repeating price patterns.

Data Sources (Section 3):
- Multi-timeframe confluence of momentum / mean-reversion / breakout

All strategies are pure: accept a feature matrix, return signal series.
"""

import logging

import numpy as np
import polars as pl

from trading_bot.services.strategies.base import (
    BaseStrategy,
    get_feature_array,
    normalize_confidence,
)

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Trend momentum strategy.

    Goes long when short-term EMA is above medium-term EMA and
    RSI confirms bullish momentum. Goes short on the opposite.

    Signals:
        +1 (long):  ema_9 > ema_21 AND rsi_14 > 50 AND return_5 > 0
        -1 (short): ema_9 < ema_21 AND rsi_14 < 50 AND return_5 < 0
        0 (neutral): otherwise
    """

    name = "Momentum Strategy"
    description = "Trend following using EMA crossover + RSI confirmation"
    strategy_class = "trading_bot.services.strategies.technical.MomentumStrategy"
    min_history = 50

    default_params: dict = {
        "ema_fast": "ema_9",
        "ema_slow": "ema_21",
        "rsi_column": "rsi_14",
        "rsi_bull_threshold": 50,
        "rsi_bear_threshold": 50,
        "return_column": "return_5",
        "confidence_scalar": 0.8,
    }

    def generate_signals(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        confidence = np.zeros(n, dtype=float)

        ema_fast = get_feature_array(df, self.params["ema_fast"])
        ema_slow = get_feature_array(df, self.params["ema_slow"])
        rsi = get_feature_array(df, self.params["rsi_column"])
        ret = get_feature_array(df, self.params["return_column"])

        rsi_bull = self.params["rsi_bull_threshold"]
        rsi_bear = self.params["rsi_bear_threshold"]
        scalar = self.params["confidence_scalar"]

        # Long condition: fast EMA above slow EMA + RSI bullish + positive return
        long_mask = (ema_fast > ema_slow) & (rsi > rsi_bull) & (ret > 0)
        signals[long_mask] = 1

        # Short condition: fast EMA below slow EMA + RSI bearish + negative return
        short_mask = (ema_fast < ema_slow) & (rsi < rsi_bear) & (ret < 0)
        signals[short_mask] = -1

        # Confidence: strength of crossover + RSI extremity
        ema_diff = np.abs(ema_fast - ema_slow) / np.maximum(np.abs(ema_slow), 1e-10)
        rsi_extreme = np.abs(rsi - 50) / 50  # 0.0 to 1.0
        raw_confidence = scalar * (ema_diff + rsi_extreme) / 2
        confidence = normalize_confidence(raw_confidence)

        # Zero confidence for no-signal positions
        confidence[signals == 0] = 0.0

        return signals, confidence


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI extremes.

    Goes long when price touches lower BB and RSI is oversold.
    Goes short when price touches upper BB and RSI is overbought.

    Signals:
        +1 (long):  price < bb_lower AND rsi_14 < 30
        -1 (short): price > bb_upper AND rsi_14 > 70
        0 (neutral): otherwise
    """

    name = "Mean Reversion Strategy"
    description = "Anticipates price reversion from Bollinger Band extremes"
    strategy_class = "trading_bot.services.strategies.technical.MeanReversionStrategy"
    min_history = 50

    default_params: dict = {
        "bb_upper": "bb_upper",
        "bb_lower": "bb_lower",
        "close_column": "close",
        "rsi_column": "rsi_14",
        "oversold_threshold": 30,
        "overbought_threshold": 70,
        "confidence_scalar": 0.7,
    }

    def generate_signals(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        confidence = np.zeros(n, dtype=float)

        close = get_feature_array(df, self.params["close_column"])
        bb_upper = get_feature_array(df, self.params["bb_upper"])
        bb_lower = get_feature_array(df, self.params["bb_lower"])
        rsi = get_feature_array(df, self.params["rsi_column"])

        oversold = self.params["oversold_threshold"]
        overbought = self.params["overbought_threshold"]
        scalar = self.params["confidence_scalar"]

        # Long: price below lower BB + oversold RSI
        long_mask = (close < bb_lower) & (rsi < oversold)
        signals[long_mask] = 1

        # Short: price above upper BB + overbought RSI
        short_mask = (close > bb_upper) & (rsi > overbought)
        signals[short_mask] = -1

        # Confidence: how far price is from band + RSI extremity
        bb_range = np.maximum(bb_upper - bb_lower, 1e-10)

        long_distance = np.where(long_mask, (bb_lower - close) / bb_range, 0)
        short_distance = np.where(short_mask, (close - bb_upper) / bb_range, 0)
        rsi_distance_long = np.where(long_mask, (oversold - rsi) / oversold, 0)
        rsi_distance_short = np.where(short_mask, (rsi - overbought) / (100 - overbought), 0)

        raw_long_conf = scalar * (long_distance + rsi_distance_long) / 2
        raw_short_conf = scalar * (short_distance + rsi_distance_short) / 2
        confidence[long_mask] = normalize_confidence(raw_long_conf[long_mask])
        confidence[short_mask] = normalize_confidence(raw_short_conf[short_mask])

        return signals, confidence


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy using price position and volatility.

    Goes long when price breaks above the recent high with
    increasing volume and volatility expansion.
    Goes short on breakdown below recent low.

    Signals:
        +1 (long):  price_position_high > 0.98 AND volume_ratio > 1.2
        -1 (short): price_position_low < 1.02 AND volume_ratio > 1.2
        0 (neutral): otherwise
    """

    name = "Breakout Strategy"
    description = "Detects breakouts above resistance / below support with volume confirmation"
    strategy_class = "trading_bot.services.strategies.technical.BreakoutStrategy"
    min_history = 50

    default_params: dict = {
        "price_high_col": "price_position_high_20",
        "price_low_col": "price_position_low_20",
        "volume_col": "volume_ratio",
        "volatility_col": "volatility_14",
        "breakout_threshold_high": 0.98,
        "breakout_threshold_low": 1.02,
        "volume_threshold": 1.2,
        "confidence_scalar": 0.7,
    }

    def generate_signals(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        confidence = np.zeros(n, dtype=float)

        price_high = get_feature_array(df, self.params["price_high_col"])
        price_low = get_feature_array(df, self.params["price_low_col"])
        volume_ratio = get_feature_array(df, self.params["volume_col"])
        volatility = get_feature_array(df, self.params["volatility_col"])

        high_thresh = self.params["breakout_threshold_high"]
        low_thresh = self.params["breakout_threshold_low"]
        vol_thresh = self.params["volume_threshold"]
        scalar = self.params["confidence_scalar"]

        # Long breakout: near recent high + volume surge
        long_mask = (price_high > high_thresh) & (volume_ratio > vol_thresh)
        signals[long_mask] = 1

        # Short breakdown: near recent low + volume surge
        short_mask = (price_low < low_thresh) & (volume_ratio > vol_thresh)
        signals[short_mask] = -1

        # Confidence: breakout strength + volume spike
        # Use nanmean to avoid NaN propagation from warmup periods
        vol_mean = np.nanmean(volatility)
        vol_norm = np.where(
            (vol_mean > 0) & ~np.isnan(volatility),
            volatility / vol_mean,
            np.ones_like(volatility),
        )

        long_conf = np.where(long_mask, scalar * (
            (price_high - high_thresh) / (1 - high_thresh) +
            (volume_ratio - vol_thresh) / vol_thresh +
            vol_norm
        ) / 3, 0)

        short_conf = np.where(short_mask, scalar * (
            (low_thresh - price_low) / (low_thresh - 1) +
            (volume_ratio - vol_thresh) / vol_thresh +
            vol_norm
        ) / 3, 0)

        confidence[long_mask] = normalize_confidence(long_conf[long_mask])
        confidence[short_mask] = normalize_confidence(short_conf[short_mask])

        return signals, confidence
