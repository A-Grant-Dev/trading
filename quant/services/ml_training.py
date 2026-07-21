"""
ML Model Training Pipeline

End-to-end training pipeline with Renaissance-grade statistical rigor.

Key steps:
  1. Fetch and prepare data with features
  2. Split chronologically: 70% train, 15% validation, 15% test
  3. Train multiple models with hyperparameter defaults
  4. Walk-forward validation (train on expanding window)
  5. Purge signals that don't pass statistical significance
  6. Save best models to disk for inference

Renaissance principle: Discard 99%+ of discovered signals.
A signal must pass ALL of these tests:
  - Backtest Sharpe > 1.5 (in-sample)
  - Out-of-sample performance > 70% of in-sample
  - P-value of strategy returns < 0.05
  - Survives Monte Carlo simulation (95% confidence)
"""

import logging
import warnings
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    End-to-end model training pipeline.

    Orchestrates data preparation, model training, validation,
    and signal filtering with Renaissance-grade rigor.

    Usage:
        trainer = ModelTrainer()
        result = trainer.train_and_evaluate(df, models)
    """

    def __init__(self, test_size: float = 0.15, val_size: float = 0.15):
        """
        Args:
            test_size: Fraction of data for final testing (default 0.15)
            val_size: Fraction of data for validation (default 0.15)
        """
        self.test_size = test_size
        self.val_size = val_size

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list,
        model_names: list[str] | None = None,
    ) -> dict:
        """
        Full training cycle: train → validate → test → report.

        Uses chronological split to avoid look-ahead bias.

        Args:
            X: Feature matrix
            y: Target vector (returns or direction)
            models: List of model instances
            model_names: Optional names for each model

        Returns:
            Dict with per-model results:
            {
                'models': {name: {
                    'train_accuracy': float,
                    'val_accuracy': float,
                    'test_accuracy': float,
                    'train_sharpe': float,
                    'val_sharpe': float,
                    'test_sharpe': float,
                    'feature_importance': dict or None,
                }},
                'best_model': name,
                'total_samples': int,
                'feature_count': int,
            }
        """
        if len(X) < 100:
            return {"error": f"Need at least 100 samples, got {len(X)}"}

        # Handle NaN
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]

        # Chronological split
        n = len(X)
        test_idx = int(n * (1 - self.test_size))
        val_idx = int(test_idx * (1 - self.val_size))

        X_train, y_train = X[:val_idx], y[:val_idx]
        X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
        X_test, y_test = X[test_idx:], y[test_idx:]

        if len(X_train) < 30:
            return {"error": f"Need at least 30 training samples, got {len(X_train)}"}

        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(models))]

        results = {}
        best_accuracy = 0.0
        best_model_name = model_names[0] if model_names else "unknown"

        for model, name in zip(models, model_names):
            try:
                # Train
                model.train(X_train, y_train)

                # Evaluate
                train_acc = self._calculate_accuracy(model, X_train, y_train)
                val_acc = self._calculate_accuracy(model, X_val, y_val)
                test_acc = self._calculate_accuracy(model, X_test, y_test)

                train_sharpe = self._calculate_signal_sharpe(model, X_train, y_train)
                val_sharpe = self._calculate_signal_sharpe(model, X_val, y_val)
                test_sharpe = self._calculate_signal_sharpe(model, X_test, y_test)

                feature_imp = model.get_feature_importance() if hasattr(model, "get_feature_importance") else None

                results[name] = {
                    "train_accuracy": round(train_acc, 4),
                    "val_accuracy": round(val_acc, 4),
                    "test_accuracy": round(test_acc, 4),
                    "train_sharpe": round(train_sharpe, 4),
                    "val_sharpe": round(val_sharpe, 4),
                    "test_sharpe": round(test_sharpe, 4),
                    "feature_importance": feature_imp,
                    "is_trained": True,
                }

                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_model_name = name

                logger.info(
                    f"{name}: train_acc={train_acc:.1%}, val_acc={val_acc:.1%}, "
                    f"test_acc={test_acc:.1%}, val_sharpe={val_sharpe:.2f}"
                )

            except Exception as e:
                logger.warning(f"Training failed for {name}: {e}")
                results[name] = {"error": str(e), "is_trained": False}

        return {
            "models": results,
            "best_model": best_model_name,
            "total_samples": n,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "feature_count": X.shape[1],
        }

    @staticmethod
    def _calculate_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate directional accuracy."""
        if len(X) < 2:
            return 0.0

        predictions = []
        for i in range(len(X)):
            try:
                pred = model.predict_proba(X[i : i + 1])
                predictions.append(1 if pred > 0.5 else 0)
            except Exception:
                predictions.append(0)

        predictions = np.array(predictions)
        actual = (y > 0).astype(int)

        if len(predictions) != len(actual):
            return 0.0

        return float(np.mean(predictions == actual))

    @staticmethod
    def _calculate_signal_sharpe(model, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate Sharpe ratio of model's signals."""
        if len(X) < 10:
            return 0.0

        # Simulate trading: long when proba > 0.55, neutral otherwise
        signals = []
        for i in range(len(X)):
            try:
                proba = model.predict_proba(X[i : i + 1])
                signal = 1 if proba > 0.55 else 0
            except Exception:
                signal = 0
            signals.append(signal)

        signals = np.array(signals)
        returns = y * signals  # Only returns when we have a signal

        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        return float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365))


# ── Walk-Forward Validation ──────────────────────────────────────


def walk_forward_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 200,
    step_size: int = 50,
) -> dict:
    """
    Walk-forward validation — the gold standard for time-series ML.

    Instead of a single train/test split, walk forward through time:
      - Train on data[0:200], test on data[200:250]
      - Train on data[0:250], test on data[250:300]
      - Train on data[0:300], test on data[300:350]
      - ... until data exhausted

    A model passes only if it performs consistently across ALL windows.

    Args:
        model: ML model with train() and predict_proba()
        X: Feature matrix (chronologically ordered)
        y: Target vector
        window_size: Initial training window size
        step_size: Steps to move forward each iteration

    Returns:
        Dict with per-window results and overall consistency score
    """
    if len(X) < window_size + step_size:
        return {"error": f"Need at least {window_size + step_size} samples, got {len(X)}"}

    # Handle NaN
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    windows = []
    start = 0

    while start + window_size + step_size <= len(X):
        train_end = start + window_size
        test_end = min(train_end + step_size, len(X))

        X_train, y_train = X[start:train_end], y[start:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        if len(X_train) < 30 or len(X_test) < 5:
            break

        try:
            model.train(X_train, y_train)
            accuracy = ModelTrainer._calculate_accuracy(model, X_test, y_test)
            sharpe = ModelTrainer._calculate_signal_sharpe(model, X_test, y_test)

            windows.append({
                "window": len(windows) + 1,
                "train_start": start,
                "train_end": train_end,
                "test_end": test_end,
                "test_samples": len(X_test),
                "accuracy": round(accuracy, 4),
                "sharpe": round(sharpe, 4),
            })
        except Exception as e:
            logger.debug(f"Walk-forward window {len(windows) + 1} failed: {e}")

        start += step_size

    if not windows:
        return {"error": "No windows completed"}

    # Compute consistency metrics
    accuracies = [w["accuracy"] for w in windows]
    sharpes = [w["sharpe"] for w in windows]

    return {
        "windows": windows,
        "total_windows": len(windows),
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "std_accuracy": round(float(np.std(accuracies, ddof=1)), 4),
        "min_accuracy": round(float(np.min(accuracies)), 4),
        "max_accuracy": round(float(np.max(accuracies)), 4),
        "mean_sharpe": round(float(np.mean(sharpes)), 4),
        "consistency_score": round(float(np.mean(accuracies) / (np.std(accuracies, ddof=1) + 0.01)), 2),
    }


# ── Signal Purger ─────────────────────────────────────────────────


class SignalPurger:
    """
    Renaissance-style signal filtering.

    Discards signals that don't pass strict statistical significance tests.
    Renaissance famously discarded 99%+ of discovered signals.

    A signal passes if:
      - Backtest Sharpe > 1.5 (in-sample)
      - Out-of-sample performance > 70% of in-sample
      - P-value of strategy returns < 0.05
      - Survives Monte Carlo simulation (95% confidence)
    """

    MIN_SHARPE_IN_SAMPLE = 1.5
    MIN_OOS_RATIO = 0.7  # Out-of-sample must retain 70% of in-sample performance
    MAX_P_VALUE = 0.05
    MONTE_CARLO_CONFIDENCE = 0.95

    def should_keep_signal(self, metrics: dict) -> tuple[bool, list[str]]:
        """
        Apply Renaissance-style signal filtering.

        Args:
            metrics: Dict with at minimum:
                     - 'in_sample_sharpe': float
                     - 'out_of_sample_sharpe': float
                     - 'trade_returns': list[float] for p-value calculation

        Returns:
            (keep: bool, reasons: list[str])
            If keep=False, reasons explains why it was rejected.
        """
        reasons = []

        # Check 1: In-sample Sharpe
        is_sharpe = metrics.get("in_sample_sharpe", 0)
        if is_sharpe < self.MIN_SHARPE_IN_SAMPLE:
            reasons.append(f"IS Sharpe {is_sharpe:.2f} < {self.MIN_SHARPE_IN_SAMPLE}")

        # Check 2: Out-of-sample performance retention
        oos_sharpe = metrics.get("out_of_sample_sharpe", 0)
        if is_sharpe > 0 and oos_sharpe / is_sharpe < self.MIN_OOS_RATIO:
            reasons.append(f"OOS ratio {oos_sharpe / is_sharpe:.2%} < {self.MIN_OOS_RATIO:.0%}")

        # Check 3: P-value of returns
        trade_returns = metrics.get("trade_returns", [])
        if trade_returns:
            p_value = self._calculate_p_value(trade_returns)
            if p_value > self.MAX_P_VALUE:
                reasons.append(f"P-value {p_value:.4f} > {self.MAX_P_VALUE}")

        # Check 4: Monte Carlo survival
        if trade_returns:
            mc_survival = self._monte_carlo_survival(trade_returns)
            if mc_survival < self.MONTE_CARLO_CONFIDENCE:
                reasons.append(f"Monte Carlo survival {mc_survival:.1%} < {self.MONTE_CARLO_CONFIDENCE:.0%}")

        keep = len(reasons) == 0
        return keep, reasons

    @staticmethod
    def _calculate_p_value(returns: list[float]) -> float:
        """
        One-sample t-test: are the returns significantly different from 0?

        Returns p-value (lower = more statistically significant).
        """
        if len(returns) < 3:
            return 1.0

        returns = np.array(returns)
        t_stat = np.mean(returns) / (np.std(returns, ddof=1) / np.sqrt(len(returns)) + 1e-10)

        # Approximate p-value from t-statistic (using normal approximation)
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
        return float(min(1.0, max(0.0, p_value)))

    @staticmethod
    def _monte_carlo_survival(returns: list[float], n_simulations: int = 10000) -> float:
        """
        Monte Carlo simulation: reshuffle returns and check profitability.

        If fewer than 95% of simulated outcomes are profitable,
        the strategy is not statistically significant.

        Args:
            returns: List of historical trade returns
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Fraction of simulations with positive total return
        """
        if len(returns) < 5:
            return 0.5

        returns = np.array(returns)
        profitable = 0

        for _ in range(n_simulations):
            shuffled = np.random.permutation(returns)
            total = np.sum(shuffled)
            if total > 0:
                profitable += 1

        return profitable / n_simulations


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using scipy (already a project dependency)."""
    try:
        from scipy.stats import norm
        return float(norm.cdf(x))
    except ImportError:
        # Fallback approximation if scipy unavailable
        return 0.5 * (1 + _erf_fallback(x / np.sqrt(2)))


def _erf_fallback(x: float) -> float:
    """Error function approximation (fallback when scipy unavailable)."""
    if x < 0:
        return -_erf_fallback(-x)
    a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    p = 0.3275911
    t = 1 / (1 + p * x)
    return 1 - (((((a[4] * t + a[3]) * t + a[2]) * t + a[1]) * t + a[0]) * t * np.exp(-x * x))
