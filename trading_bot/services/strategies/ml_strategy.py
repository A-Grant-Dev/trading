"""
ML-Based Strategy — Feature-Engineered ML Labels

Uses machine learning models to predict future returns and
generate trading signals. Requires trained models from Phase 6.

Data Sources (Section 3):
- Feature-engineered ML labels (future return quantiles 1h/4h/1d)

This is a placeholder for Phase 6 integration. The actual ML
models will be loaded and used for inference once the optimizer
pipeline trains them.
"""

import logging
from typing import Optional

import numpy as np
import polars as pl

from trading_bot.services.strategies.base import (
    BaseStrategy,
    get_feature_array,
    normalize_confidence,
)

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    Machine learning-based trading strategy.

    Uses trained ML models to predict future return direction
    and confidence. Requires one or more trained models from
    the optimizer pipeline (Phase 6).

    Falls back to target_return features when models are not available.
    """

    name = "ML Strategy"
    description = "Generates signals from trained ML model predictions"
    strategy_class = "trading_bot.services.strategies.ml_strategy.MLStrategy"
    min_history = 50

    default_params: dict = {
        "target_col": "target_return_1",  # Fallback target when no model
        "confidence_scalar": 0.5,
        "min_prediction_threshold": 0.001,  # 0.1% minimum predicted return
    }

    def __init__(self, params: Optional[dict] = None, model_path: Optional[str] = None):
        super().__init__(params)
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        """Load trained ML model (placeholder for Phase 6)."""
        if self._model is not None:
            return True
        if self.model_path:
            try:
                import joblib
                self._model = joblib.load(self.model_path)
                logger.info("Loaded ML model from %s", self.model_path)
                return True
            except Exception as e:
                logger.warning("Failed to load model %s: %s", self.model_path, e)
        return False

    def generate_signals(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        confidence = np.zeros(n, dtype=float)

        threshold = self.params["min_prediction_threshold"]
        scalar = self.params["confidence_scalar"]

        # Try ML model inference first
        if self._load_model() and self._model is not None:
            try:
                # Extract feature matrix (exclude non-feature columns)
                exclude = {"timestamp", "open", "high", "low", "close", "volume",
                           "target_return_1", "target_return_5"}
                feature_cols = [c for c in df.columns if c not in exclude]
                feature_matrix = df[feature_cols].to_numpy().astype(float)

                # Predict
                predictions = self._model.predict(feature_matrix)

                # Convert predictions to signals
                signals = np.where(predictions > threshold, 1,
                                   np.where(predictions < -threshold, -1, 0)).astype(np.int8)
                confidence = normalize_confidence(np.abs(predictions) / 0.01 * scalar)

                logger.debug("ML Strategy: %d long, %d short of %d rows",
                             np.sum(signals == 1), np.sum(signals == -1), n)
                return signals, confidence

            except Exception as e:
                logger.warning("ML inference failed: %s, falling back", e)

        # Fallback: use target_return features as signal
        target = get_feature_array(df, self.params["target_col"])

        long_mask = target > threshold
        short_mask = target < -threshold
        signals[long_mask] = 1
        signals[short_mask] = -1

        # Confidence proportional to expected return magnitude
        raw_confidence = scalar * np.abs(target) / 0.01  # Scale: 1% return = scalar confidence
        confidence = normalize_confidence(raw_confidence)
        confidence[signals == 0] = 0.0

        return signals, confidence
