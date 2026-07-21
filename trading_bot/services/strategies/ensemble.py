"""
Ensemble Strategy — Weighted Combination of Multiple Strategies

Combines signals from multiple sub-strategies using dynamic weights.
Weights can be static (from ParamSet) or dynamic (based on recent
performance — updated by promoter in Phase 6).

All strategies must be pure functions accepting a feature matrix
and returning (+1/0/-1, confidence).

Data Sources (Section 3):
- Ensemble of strategies with dynamic weight allocation
"""

import logging
from typing import Optional

import numpy as np
import polars as pl

from trading_bot.services.strategies.base import BaseStrategy, normalize_confidence

logger = logging.getLogger(__name__)


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy that combines signals from multiple sub-strategies.

    Uses weighted voting where each sub-strategy contributes its
    signal × weight to the final decision.

    Weights can be:
    - Default (equal): All strategies weighted equally
    - Custom: Passed via params['weights']
    - Dynamic: Updated by the promoter (Phase 6) based on recent Sharpe
    """

    name = "Ensemble Strategy"
    description = "Weighted combination of all active sub-strategies"
    strategy_class = "trading_bot.services.strategies.ensemble.EnsembleStrategy"
    min_history = 50

    default_params: dict = {
        "weights": {},  # Empty = equal weight for all
        "min_strategies": 2,  # Minimum strategies needed for signal
        "agreement_threshold": 0.5,  # Fraction of weighted votes needed
        "confidence_scalar": 0.8,
    }

    def __init__(
        self,
        params: Optional[dict] = None,
        strategies: Optional[list[tuple[BaseStrategy, float]]] = None,
    ):
        """
        Initialize ensemble with list of (strategy, weight) tuples.

        Args:
            params: Override parameters
            strategies: List of (strategy_instance, weight) tuples.
                If None, strategies are loaded from active Strategy model records.
        """
        super().__init__(params)
        self.strategies: list[tuple[BaseStrategy, float]] = strategies or []

    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0) -> None:
        """Add a sub-strategy with its weight."""
        self.strategies.append((strategy, weight))

    def generate_signals(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        if not self.strategies:
            logger.warning("Ensemble has no sub-strategies")
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=float)

        if len(self.strategies) < self.params["min_strategies"]:
            logger.warning("Ensemble has fewer than min_strategies (%d < %d)",
                           len(self.strategies), self.params["min_strategies"])
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=float)

        # Collect signals from all sub-strategies
        weighted_signals = np.zeros(n, dtype=float)
        total_weight = 0.0
        strategy_confidences = []

        for strategy, weight in self.strategies:
            try:
                sub_signals, sub_confidence = strategy.generate_signals(df)
                weighted_signals += sub_signals.astype(float) * weight * sub_confidence
                total_weight += weight
                strategy_confidences.append(sub_confidence)
            except Exception as e:
                logger.error("Strategy '%s' failed: %s", strategy.name, e)
                continue

        if total_weight == 0:
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=float)

        # Normalize by total weight
        weighted_signals /= total_weight

        # Generate final signals
        signals = np.zeros(n, dtype=np.int8)
        threshold = self.params["agreement_threshold"]

        # Long: weighted vote exceeds positive threshold
        long_mask = weighted_signals > threshold
        signals[long_mask] = 1

        # Short: weighted vote exceeds negative threshold
        short_mask = weighted_signals < -threshold
        signals[short_mask] = -1

        # Ensemble confidence: average of individual confidences × signal strength
        if strategy_confidences:
            avg_confidence = np.mean(strategy_confidences, axis=0)
            signal_strength = np.abs(weighted_signals)
            raw_confidence = self.params["confidence_scalar"] * avg_confidence * signal_strength
        else:
            raw_confidence = np.abs(weighted_signals)

        confidence = normalize_confidence(raw_confidence)
        confidence[signals == 0] = 0.0

        n_long = np.sum(signals == 1)
        n_short = np.sum(signals == -1)
        logger.debug("Ensemble: %d long, %d short, %d neutral (from %d strategies)",
                     n_long, n_short, n - n_long - n_short, len(self.strategies))

        return signals, confidence
