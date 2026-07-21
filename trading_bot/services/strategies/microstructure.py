"""
Microstructure Strategies — Order Book Imbalance, Absorption/Exhaustion

Analyses order book dynamics to detect:
- Order book imbalance shifts
- Absorption (price holding despite large sells)
- Exhaustion (reduced buying pressure after uptrend)

Data Sources (Section 3):
- Order-book imbalance + absorption / exhaustion detection
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


class OrderBookImbalanceStrategy(BaseStrategy):
    """
    Order book imbalance strategy.

    Goes long when buy-side volume dominates the order book (aggressive buyers).
    Goes short when sell-side volume dominates.

    Signals:
        +1 (long):  ob_imbalance_pct > imbalance_threshold AND depth_pressure > 0
        -1 (short): ob_imbalance_pct < (100 - imbalance_threshold) AND depth_pressure < 0
        0 (neutral): otherwise
    """

    name = "Order Book Imbalance"
    description = "Trades based on bid/ask volume imbalance in the order book"
    strategy_class = "trading_bot.services.strategies.microstructure.OrderBookImbalanceStrategy"
    min_history = 5

    default_params: dict = {
        "imbalance_col": "ob_imbalance_pct",
        "depth_pressure_col": "ob_depth_pressure",
        "imbalance_threshold": 55.0,  # >55% bids = bullish
        "confidence_scalar": 0.6,
    }

    def generate_signals(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        confidence = np.zeros(n, dtype=float)

        imbalance = get_feature_array(df, self.params["imbalance_col"])
        depth_pressure = get_feature_array(df, self.params["depth_pressure_col"])
        threshold = self.params["imbalance_threshold"]
        scalar = self.params["confidence_scalar"]

        # Long: strong buy-side dominance
        long_mask = (imbalance > threshold) & (depth_pressure > 0)
        signals[long_mask] = 1

        # Short: strong sell-side dominance
        short_mask = (imbalance < (100 - threshold)) & (depth_pressure < 0)
        signals[short_mask] = -1

        # Confidence: distance from neutral (50%)
        imbalance_extreme = np.abs(imbalance - 50) / 50  # 0.0 to 1.0
        raw_confidence = scalar * imbalance_extreme
        confidence = normalize_confidence(raw_confidence)
        confidence[signals == 0] = 0.0

        return signals, confidence


class AbsorptionStrategy(BaseStrategy):
    """
    Absorption / exhaustion detection strategy.

    Detects when large sell orders are being absorbed (price doesn't drop)
    signaling institutional buying. Also detects when buying pressure
    is exhausting (price doesn't rise despite large bids).

    Uses microstructure features from recent trades.

    Signals:
        +1 (long):  buy_volume_ratio > threshold (absorption of sells)
        -1 (short): buy_volume_ratio < (1 - threshold) (exhaustion of buys)
        0 (neutral): otherwise
    """

    name = "Absorption Detection"
    description = "Detects order absorption and exhaustion from trade flow"
    strategy_class = "trading_bot.services.strategies.microstructure.AbsorptionStrategy"
    min_history = 5

    default_params: dict = {
        "buy_ratio_col": "micro_buy_volume_ratio",
        "trade_count_col": "micro_trade_count",
        "volume_imbalance_col": "micro_volume_imbalance",
        "absorption_threshold": 0.65,
        "exhaustion_threshold": 0.35,
        "min_trades": 10,
        "confidence_scalar": 0.5,
    }

    def generate_signals(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        confidence = np.zeros(n, dtype=float)

        buy_ratio = get_feature_array(df, self.params["buy_ratio_col"])
        trade_count = get_feature_array(df, self.params["trade_count_col"])
        vol_imbalance = get_feature_array(df, self.params["volume_imbalance_col"])
        threshold = self.params["absorption_threshold"]
        min_trades = self.params["min_trades"]
        scalar = self.params["confidence_scalar"]

        # Only signal with sufficient trade data
        has_data = trade_count >= min_trades

        # Long: high buy volume ratio (absorption)
        long_mask = has_data & (buy_ratio > threshold)
        signals[long_mask] = 1

        # Short: low buy volume ratio (exhaustion)
        short_mask = has_data & (buy_ratio < self.params["exhaustion_threshold"])
        signals[short_mask] = -1

        # Confidence: distance from neutral + volume imbalance strength
        ratio_strength = np.abs(buy_ratio - 0.5) / 0.5
        imbalance_strength = np.abs(vol_imbalance)
        raw_confidence = scalar * (ratio_strength + imbalance_strength) / 2
        confidence = normalize_confidence(raw_confidence)
        confidence[signals == 0] = 0.0

        return signals, confidence
