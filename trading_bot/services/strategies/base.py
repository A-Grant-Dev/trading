"""
Base Strategy — Abstract Interface for All Trading Strategies

Defines the standardized interface that every strategy must implement.
All strategies are pure functions that accept a feature matrix and
return a signal series (+1 / 0 / -1) with confidence (0.0–1.0).

Inspired by Renaissance Technologies' systematic approach:
- Every strategy is a mathematical model, not a gut feeling
- Multiple independent signals feed into an ensemble
- Confidence scores allow weighted aggregation
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import polars as pl

from trading_bot.models import ParamSet, Strategy as StrategyModel

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Each strategy is a pure function: given a feature matrix,
    return a signal vector.

    Attributes:
        name: Human-readable strategy name
        description: Strategy description
        strategy_class: Python import path (auto-set by subclasses)
        default_params: Default parameter set for this strategy
        min_history: Minimum number of rows needed for valid signals
    """

    name: str = ""
    description: str = ""
    strategy_class: str = ""
    default_params: dict[str, Any] = {}
    min_history: int = 50  # Minimum data points needed

    def __init__(self, params: Optional[dict[str, Any]] = None):
        """
        Initialize strategy with optional parameter overrides.

        Args:
            params: Override specific parameters (e.g. {'rsi_period': 7})
        """
        self.params = {**self.default_params, **(params or {})}
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate that all required params are present and valid."""
        for key, value in self.default_params.items():
            if key not in self.params:
                self.params[key] = value

    @abstractmethod
    def generate_signals(
        self, df: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals from a feature matrix.

        Args:
            df: Polars DataFrame with feature columns.
                Must include at least the columns needed by this strategy.

        Returns:
            Tuple of (signals, confidence) where:
                signals: numpy array of int8 with values +1 (long), 0 (neutral), -1 (short)
                confidence: numpy array of float64 with values 0.0–1.0
                Both arrays have the same length as df.
        """
        ...

    def get_param_set(self, strategy_model: StrategyModel) -> ParamSet:
        """Get or create a ParamSet for this strategy with current params."""
        param_set, _ = ParamSet.objects.get_or_create(
            strategy=strategy_model,
            params=self.params,
            defaults={"is_candidate": True},
        )
        return param_set

    def __str__(self) -> str:
        return f"{self.name} ({self.strategy_class})"


# ── Signal Utilities ──────────────────────────────────────────────


def normalize_confidence(raw: np.ndarray) -> np.ndarray:
    """
    Normalize raw signal values to 0.0–1.0 confidence.

    Uses sigmoid-like scaling: confidence = abs(value) capped at 1.0
    """
    return np.clip(np.abs(raw), 0.0, 1.0)


def zscore_confidence(z_values: np.ndarray) -> np.ndarray:
    """
    Convert z-scores to confidence values.

    z = 0 → confidence 0.0 (no signal)
    z = 2 → confidence ~0.95
    z = 3 → confidence ~1.0
    """
    return np.clip(np.abs(z_values) / 3.0, 0.0, 1.0)


def get_feature_array(df: pl.DataFrame, feature_name: str) -> np.ndarray:
    """Safely extract a feature column as numpy array, returning zeros if missing."""
    if feature_name in df.columns:
        return df[feature_name].to_numpy().astype(float)
    logger.warning("Feature '%s' not found, returning zeros", feature_name)
    return np.zeros(len(df))
