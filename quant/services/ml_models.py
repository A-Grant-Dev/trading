"""
Machine Learning Models for Trading Signal Generation

Implements multiple ML models for price direction prediction, following
Renaissance's ensemble approach: many weak models > one strong model.

Models:
  - RandomForestModel: Baseline, interpretable, hard to overfit
  - XGBoostModel: Best for tabular data with many features
  - LSTMModel: Captures sequential dependencies (optional, requires torch)
  - EnsemblePredictor: Weighted voting with adaptive weights

Renaissance/Simons principle: Don't try to predict exact prices.
Predict probabilities. A 53% directional accuracy is enough when
combined with proper risk management and the Law of Large Numbers.
"""

import logging
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Model Storage ─────────────────────────────────────────────────

MODEL_DIR = Path(settings.BASE_DIR) / "quant" / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(symbol: str, model_name: str) -> Path:
    """Get the file path for a saved model."""
    return MODEL_DIR / f"{model_name}_{symbol.lower()}.pkl"


# ── Optional Model Imports ───────────────────────────────────────

_RANDOM_FOREST_AVAILABLE = False
_XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    _RANDOM_FOREST_AVAILABLE = True
except ImportError:
    RandomForestClassifier = None
    RandomForestRegressor = None
    StandardScaler = None

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None


# ── Base Class ────────────────────────────────────────────────────


class DirectionPredictor:
    """
    Base class for direction prediction models.

    Predicts P(up) or P(down) over the next N candles.
    Returns probability of upward movement (0.0 - 1.0).

    Subclasses must implement:
        - _train_model(X, y)
        - _predict_proba(X)
        - save(path) / load(path)
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._is_trained = False
        self._feature_columns: list[str] | None = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, X: np.ndarray, y: np.ndarray, feature_columns: list[str] | None = None) -> None:
        """Train the model."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> float:
        """Return probability of upward movement (0.0 - 1.0)."""
        raise NotImplementedError

    def predict_direction(self, X: np.ndarray) -> int:
        """Return 1 (up) or 0 (down/neutral)."""
        proba = self.predict_proba(X)
        return 1 if proba > 0.5 else 0

    def save(self, symbol: str) -> None:
        """Save model to disk."""
        path = _model_path(symbol, self.name)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved model {self.name} for {symbol}")

    @classmethod
    def load(cls, symbol: str, name: str) -> Optional["DirectionPredictor"]:
        """Load model from disk."""
        path = _model_path(symbol, name)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def get_feature_importance(self) -> dict | None:
        """Return feature importance if available."""
        return None


# ── Random Forest Model ──────────────────────────────────────────


class RandomForestModel(DirectionPredictor):
    """
    Random Forest classifier for direction prediction.

    Good baseline model — interpretable, robust to outliers, hard to overfit.
    Uses 100 trees with class weight balancing.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__(name="random_forest")
        if not _RANDOM_FOREST_AVAILABLE:
            raise ImportError("scikit-learn is not available. Install with: pip install scikit-learn")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def train(self, X: np.ndarray, y: np.ndarray, feature_columns: list[str] | None = None) -> None:
        """Train the Random Forest model."""
        if len(X) < 50:
            raise ValueError(f"Need at least 50 training samples, got {len(X)}")

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        # Convert regression target to binary classification
        y_binary = (y > 0).astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y_binary)

        self._is_trained = True
        self._feature_columns = feature_columns
        logger.info(f"RandomForest trained: {len(X)} samples, {X.shape[1]} features")

    def predict_proba(self, X: np.ndarray) -> float:
        """Predict probability of upward movement."""
        if not self._is_trained:
            return 0.5
        X = np.nan_to_num(np.array(X, dtype=float).reshape(1, -1), nan=0.0)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        # proba[0] = P(down), proba[1] = P(up) — return P(up)
        return float(proba[1]) if len(proba) > 1 else 0.5

    def get_feature_importance(self) -> dict | None:
        """Return feature importance scores."""
        if not self._is_trained or not hasattr(self.model, "feature_importances_"):
            return None
        importances = self.model.feature_importances_
        columns = self._feature_columns or [f"f{i}" for i in range(len(importances))]
        return dict(sorted(
            zip(columns, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:20])  # Top 20 features


# ── XGBoost Model ────────────────────────────────────────────────


class XGBoostModel(DirectionPredictor):
    """
    XGBoost classifier — typically the best for tabular feature data.

    Uses gradient boosting with early stopping and regularization.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6):
        super().__init__(name="xgboost")
        if not _XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not available. Install with: pip install xgboost")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

    def train(self, X: np.ndarray, y: np.ndarray, feature_columns: list[str] | None = None) -> None:
        """Train the XGBoost model."""
        if len(X) < 50:
            raise ValueError(f"Need at least 50 training samples, got {len(X)}")

        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        y_binary = (y > 0).astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y_binary)

        self._is_trained = True
        self._feature_columns = feature_columns
        logger.info(f"XGBoost trained: {len(X)} samples, {X.shape[1]} features")

    def predict_proba(self, X: np.ndarray) -> float:
        """Predict probability of upward movement."""
        if not self._is_trained:
            return 0.5
        X = np.nan_to_num(np.array(X, dtype=float).reshape(1, -1), nan=0.0)
        proba = self.model.predict_proba(X)[0]
        return float(proba[1]) if len(proba) > 1 else 0.5

    def get_feature_importance(self) -> dict | None:
        """Return feature importance scores."""
        if not self._is_trained:
            return None
        importances = self.model.feature_importances_
        columns = self._feature_columns or [f"f{i}" for i in range(len(importances))]
        return dict(sorted(
            zip(columns, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:20])


# ── Ensemble Predictor ────────────────────────────────────────────


class EnsemblePredictor:
    """
    Combines multiple models using weighted voting.

    Weights are determined by each model's recent performance
    (walk-forward validation accuracy or Sharpe ratio).

    Renaissance principle: Many weak models > one strong model.
    Use the ensemble's average probability as the final signal.
    """

    def __init__(self):
        self.models: list[DirectionPredictor] = []
        self.weights: list[float] = []
        self._recent_performance: dict[str, list[float]] = {}

    def add_model(self, model: DirectionPredictor, weight: float = 1.0) -> None:
        """Add a model to the ensemble with initial weight."""
        self.models.append(model)
        self.weights.append(weight)
        self._recent_performance[model.name] = []

    def train_all(self, X: np.ndarray, y: np.ndarray, feature_columns: list[str] | None = None) -> None:
        """Train all models in the ensemble."""
        for model in self.models:
            try:
                model.train(X, y, feature_columns)
            except Exception as e:
                logger.warning(f"Failed to train {model.name}: {e}")

    def predict(self, X: np.ndarray) -> float:
        """
        Weighted average of all model probabilities.

        Returns:
            Probability of upward movement (0.0 - 1.0)
        """
        if not self.models:
            return 0.5

        predictions = []
        for model in self.models:
            try:
                pred = model.predict_proba(X)
                predictions.append(pred)
            except Exception:
                predictions.append(0.5)

        predictions = np.array(predictions)
        weights = np.array(self.weights)

        # Normalize weights
        weights_sum = weights.sum()
        if weights_sum > 0:
            weights = weights / weights_sum

        return float(np.average(predictions, weights=weights))

    def predict_direction(self, X: np.ndarray) -> int:
        """Return 1 (up) or 0 (down/neutral)."""
        return 1 if self.predict(X) > 0.55 else 0  # 55% threshold for Law of Large Numbers

    def update_weights(self, recent_performance: list[float]) -> None:
        """
        Update model weights based on recent accuracy.

        Args:
            recent_performance: List of accuracy scores per model (same order as self.models)
        """
        if len(recent_performance) != len(self.models):
            logger.warning("Performance list length doesn't match model count")
            return

        # Store performance history
        for i, model in enumerate(self.models):
            self._recent_performance[model.name].append(recent_performance[i])

        # Calculate weights from recent accuracy (exponential moving)
        total = sum(max(0.01, p) for p in recent_performance)
        if total > 0:
            self.weights = [max(0.01, p) / total for p in recent_performance]
            logger.info(f"Updated ensemble weights: {dict(zip([m.name for m in self.models], [round(w, 3) for w in self.weights]))}")

    def save(self, symbol: str) -> None:
        """Save ensemble (saves each model individually + weights)."""
        for model in self.models:
            model.save(symbol)
        # Save weights
        import json
        weights_path = _model_path(symbol, "ensemble_weights")
        try:
            with open(weights_path, "w") as f:
                json.dump({
                    "model_names": [m.name for m in self.models],
                    "weights": self.weights,
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save ensemble weights: {e}")

    @classmethod
    def load(cls, symbol: str) -> Optional["EnsemblePredictor"]:
        """Load ensemble from disk (loads each available model + weights)."""
        ensemble = cls()
        for model_name in ["random_forest", "xgboost"]:
            model = DirectionPredictor.load(symbol, model_name)
            if model is not None:
                ensemble.add_model(model)
        if not ensemble.models:
            return None
        # Restore weights
        import json
        weights_path = _model_path(symbol, "ensemble_weights")
        try:
            if weights_path.exists():
                with open(weights_path) as f:
                    data = json.load(f)
                loaded_weights = data.get("weights", [])
                if len(loaded_weights) == len(ensemble.models):
                    ensemble.weights = loaded_weights
        except Exception as e:
            logger.warning(f"Failed to load ensemble weights: {e}")
        return ensemble

    def get_feature_importance(self) -> dict | None:
        """Aggregate feature importance across all models."""
        all_importances = {}
        for model in self.models:
            imp = model.get_feature_importance()
            if imp:
                for feature, score in imp.items():
                    all_importances[feature] = all_importances.get(feature, 0) + score * 0.5
        return dict(sorted(all_importances.items(), key=lambda x: x[1], reverse=True)[:20]) if all_importances else None
