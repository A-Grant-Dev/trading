"""
On-Chain Divergence Strategy

Detects divergences between on-chain metrics and price action.
For example: price dropping while exchange outflows increase = accumulation.

Data Sources (Section 3):
- On-chain divergence vs price (flows, SOPR, MVRV Z)
- Funding rate + OI extremes as reversal / continuation signals

Currently uses proxy features since on-chain data pipeline is stub.
Will be enhanced when real on-chain data is available.
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


class OnChainDivergenceStrategy(BaseStrategy):
    """
    On-chain divergence detection strategy.

    Detects when price action diverges from on-chain fundamentals.
    Currently uses available proxies; full on-chain features from
    Glassnode/CryptoQuant will be integrated in Phase 7.

    Signal logic:
    - Price drops + positive sentiment divergence → accumulation (long)
    - Price rises + negative sentiment divergence → distribution (short)
    - On-chain security (block height trend) + market context
    """

    name = "On-Chain Divergence"
    description = "Detects divergence between price and fundamental on-chain metrics"
    strategy_class = "trading_bot.services.strategies.onchain.OnChainDivergenceStrategy"
    min_history = 20

    default_params: dict = {
        "price_col": "close",
        "return_col": "return_15",
        "fear_greed_col": "sentiment_fear_greed",
        "volatility_col": "volatility_14",
        "fear_threshold": 25,  # Extreme fear
        "greed_threshold": 75,  # Extreme greed
        "confidence_scalar": 0.5,
    }

    def generate_signals(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        confidence = np.zeros(n, dtype=float)

        price = get_feature_array(df, self.params["price_col"])
        ret = get_feature_array(df, self.params["return_col"])
        fear_greed = get_feature_array(df, self.params["fear_greed_col"])
        volatility = get_feature_array(df, self.params["volatility_col"])

        fear_thresh = self.params["fear_threshold"]
        greed_thresh = self.params["greed_threshold"]
        scalar = self.params["confidence_scalar"]

        # On-chain divergence signals:
        for i in range(1, n):
            if np.isnan(price[i]) or np.isnan(fear_greed[i]):
                continue

            # Divergence: Price dropping but Fear & Greed is not extreme fear
            # (potential accumulation)
            if ret[i] < -0.02 and fear_greed[i] > fear_thresh:
                signals[i] = 1
                # Confidence increases with more divergence
                divergence = abs(ret[i]) * (fear_greed[i] - fear_thresh) / 50
                confidence[i] = normalize_confidence(scalar * divergence)

            # Divergence: Price rising but Fear & Greed is not extreme greed
            # (potential distribution)
            elif ret[i] > 0.02 and fear_greed[i] < greed_thresh:
                signals[i] = -1
                divergence = abs(ret[i]) * (greed_thresh - fear_greed[i]) / 50
                confidence[i] = normalize_confidence(scalar * divergence)

            # Extreme fear alone can be a buy signal (mean reversion of sentiment)
            elif fear_greed[i] < fear_thresh * 0.5:
                signals[i] = 1
                confidence[i] = scalar * 0.3

            # Extreme greed alone can be a sell signal
            elif fear_greed[i] > greed_thresh * 1.3:
                signals[i] = -1
                confidence[i] = scalar * 0.3

        return signals, confidence
