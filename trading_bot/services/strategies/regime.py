"""
Market Regime Strategy

Classifies the current market regime (trend / range / high-vol / low-vol)
and generates signals based on the detected regime.

Uses volatility and price action features to determine regime.

Data Sources (Section 3):
- Market regime classification (trend / range / high-vol / low-vol)
- Volatility regime switching (GARCH + realized volatility)
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


class RegimeStrategy(BaseStrategy):
    """
    Market regime detection strategy.

    Classifies the current market into one of four regimes:
    - Strong Trend: High directional movement, normal volatility
    - Ranging: Low directional movement, normal volatility
    - High Volatility: Extreme price swings
    - Low Volatility: Low movement, low volatility

    Signal logic:
    - Strong Trend: Follow the trend (long if return_5 > 0, short if < 0)
    - Ranging: Mean-revert (reversal at extremes)
    - High Vol: Stand aside (neutral) — avoid whipsaws
    - Low Vol: Prepare for breakout (anticipate direction)
    """

    name = "Market Regime Strategy"
    description = "Adapts signal generation based on detected market regime"
    strategy_class = "trading_bot.services.strategies.regime.RegimeStrategy"
    min_history = 30

    default_params: dict = {
        "volatility_col": "volatility_14",
        "return_col": "return_5",
        "rsi_col": "rsi_14",
        "regime_lookback": 20,
        "high_vol_multiplier": 1.5,  # volatility > 1.5x avg = high vol
        "low_vol_multiplier": 0.5,   # volatility < 0.5x avg = low vol
        "trend_return_threshold": 0.02,  # 2% return = trending
        "confidence_scalar": 0.6,
    }

    def generate_signals(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        confidence = np.zeros(n, dtype=float)

        volatility = get_feature_array(df, self.params["volatility_col"])
        ret = get_feature_array(df, self.params["return_col"])
        rsi = get_feature_array(df, self.params["rsi_col"])

        lookback = self.params["regime_lookback"]
        high_vol_mult = self.params["high_vol_multiplier"]
        low_vol_mult = self.params["low_vol_multiplier"]
        trend_thresh = self.params["trend_return_threshold"]
        scalar = self.params["confidence_scalar"]

        # Compute rolling average volatility for regime detection
        vol_avg = np.full_like(volatility, np.nan)
        for i in range(len(volatility)):
            start = max(0, i - lookback)
            vol_avg[i] = np.nanmean(volatility[start:i + 1]) if i >= 5 else np.nan

        # Detect regime for each point
        for i in range(1, n):
            if np.isnan(vol_avg[i]) or np.isnan(volatility[i]):
                continue

            # Regime classification
            if volatility[i] > vol_avg[i] * high_vol_mult:
                # High Volatility → neutral (avoid whipsaws)
                signals[i] = 0
                confidence[i] = 0.0

            elif volatility[i] < vol_avg[i] * low_vol_mult:
                # Low Volatility → anticipate breakout in direction of momentum
                if ret[i] > trend_thresh * 0.5:
                    signals[i] = 1
                    confidence[i] = scalar * 0.4  # Low confidence — just anticipating
                elif ret[i] < -trend_thresh * 0.5:
                    signals[i] = -1
                    confidence[i] = scalar * 0.4
                else:
                    signals[i] = 0
                    confidence[i] = 0.0

            elif abs(ret[i]) > trend_thresh:
                # Strong Trend → follow it
                if ret[i] > 0 and rsi[i] > 50:
                    signals[i] = 1
                    confidence[i] = scalar * min(abs(ret[i]) / trend_thresh, 1.0)
                elif ret[i] < 0 and rsi[i] < 50:
                    signals[i] = -1
                    confidence[i] = scalar * min(abs(ret[i]) / trend_thresh, 1.0)
                else:
                    signals[i] = 0
                    confidence[i] = 0.0
            else:
                # Ranging → mean-revert at extremes
                if rsi[i] < 30:
                    signals[i] = 1
                    confidence[i] = scalar * 0.5
                elif rsi[i] > 70:
                    signals[i] = -1
                    confidence[i] = scalar * 0.5
                else:
                    signals[i] = 0
                    confidence[i] = 0.0

        return signals, confidence
