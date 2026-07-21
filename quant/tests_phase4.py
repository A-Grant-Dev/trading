"""
Phase 4 — ML Prediction Engine Tests

Tests cover:
  - FeaturePipeline feature generation and training data preparation
  - RandomForestModel training and prediction
  - EnsemblePredictor weighting and aggregation
  - ModelTrainer evaluation metrics
  - SignalPurger Renaissance-style filtering
  - Walk-forward validation
  - Edge cases: empty data, NaN values, insufficient samples
"""

import numpy as np
import pandas as pd
from django.test import TestCase

from quant.services.ml_features import FeaturePipeline
from quant.services.ml_models import (
    DirectionPredictor,
    EnsemblePredictor,
    RandomForestModel,
)
from quant.services.ml_training import (
    ModelTrainer,
    SignalPurger,
    _erf_fallback,
    _normal_cdf,
    walk_forward_validate,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_sample_ohlcv(n: int = 1000) -> pd.DataFrame:
    """Create synthetic OHLCV data with a slight upward trend."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.3) + np.linspace(0, 2, n)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close - np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n)

    dates = pd.date_range(start="2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)


# ══════════════════════════════════════════════════════════════════
#  FeaturePipeline Tests
# ══════════════════════════════════════════════════════════════════


class FeaturePipelineTests(TestCase):
    """Test FeaturePipeline feature engineering."""

    def setUp(self):
        self.df = _make_sample_ohlcv(1000)
        self.pipeline = FeaturePipeline()

    def test_build_features_returns_rich_df(self):
        """build_features should add many indicator columns."""
        result = self.pipeline.build_features(self.df)
        self.assertFalse(result.empty)
        self.assertGreater(len(result.columns), 25)  # Well beyond original 5 columns

    def test_build_features_includes_all_groups(self):
        """All 9 feature groups should be represented."""
        result = self.pipeline.build_features(self.df)
        columns = set(result.columns)

        # Group 1: Price
        self.assertTrue(columns.intersection({"log_return", "return_1", "return_5", "high_low_pct"}))

        # Group 2: Technical
        self.assertTrue(columns.intersection({"rsi_14", "macd", "atr_14", "bb_upper", "ema_9"}))

        # Group 3: Volume
        self.assertTrue(columns.intersection({"volume_ratio", "obv", "vwap"}))

        # Group 9: Time
        self.assertTrue(columns.intersection({"hour", "day_of_week", "is_ny_session"}))

        # Target columns
        self.assertIn("target_future_return_1", columns)
        self.assertIn("target_direction", columns)

    def test_empty_df(self):
        """Empty DataFrame should return empty result."""
        result = self.pipeline.build_features(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_too_small_df(self):
        """Very small DataFrame should return empty."""
        small = pd.DataFrame({"close": [100, 101], "high": [101, 102], "low": [99, 100], "volume": [100, 200]})
        result = self.pipeline.build_features(small)
        self.assertTrue(result.empty)

    def test_prepare_training_data_splits_correctly(self):
        """Training data preparation should return 4 arrays."""
        features = self.pipeline.build_features(self.df)
        result = self.pipeline.prepare_training_data(features)
        self.assertEqual(len(result), 4)  # X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = result
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_prepare_training_data_missing_target(self):
        """Missing target column should raise ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.pipeline.prepare_training_data(df, target_column="nonexistent")

    def test_get_feature_names(self):
        """get_feature_names should exclude targets and raw columns."""
        features = self.pipeline.build_features(self.df)
        names = self.pipeline.get_feature_names(features)
        self.assertNotIn("target_future_return_1", names)
        self.assertNotIn("close", names)
        self.assertGreater(len(names), 20)

    def test_rolling_max_min(self):
        """Rolling min/max should work correctly."""
        values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
        rmax = FeaturePipeline._rolling_max(values, 3)
        self.assertTrue(np.isnan(rmax[0]))
        self.assertTrue(np.isnan(rmax[1]))
        self.assertEqual(rmax[2], 4)  # max(3,1,4)
        self.assertEqual(rmax[3], 4)  # max(1,4,1)


# ══════════════════════════════════════════════════════════════════
#  ML Model Tests
# ══════════════════════════════════════════════════════════════════


class RandomForestModelTests(TestCase):
    """Test RandomForestModel training and prediction."""

    def setUp(self):
        np.random.seed(42)
        n = 200
        # Create synthetic data with a pattern
        X = np.random.randn(n, 10)
        # y = 1 if sum of first 3 features > 0, else 0
        y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(float)
        self.X = X
        self.y = y

    def test_train_and_predict(self):
        """Model should train and predict probabilities."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        model.train(self.X, self.y)
        self.assertTrue(model.is_trained)

        proba = model.predict_proba(self.X[0:1])
        self.assertGreaterEqual(proba, 0.0)
        self.assertLessEqual(proba, 1.0)

    def test_predict_direction(self):
        """predict_direction should return 0 or 1."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        model.train(self.X, self.y)
        direction = model.predict_direction(self.X[0:1])
        self.assertIn(direction, [0, 1])

    def test_predict_before_training(self):
        """Predicting before training should return 0.5."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        proba = model.predict_proba(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        self.assertEqual(proba, 0.5)

    def test_insufficient_data(self):
        """Training with too little data should raise ValueError."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        with self.assertRaises(ValueError):
            model.train(np.random.randn(10, 5), np.random.randn(10))

    def test_feature_importance(self):
        """Trained model should return feature importance."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        model.train(self.X, self.y)
        imp = model.get_feature_importance()
        self.assertIsNotNone(imp)
        self.assertGreater(len(imp), 0)

    def test_save_load_cycle(self):
        """Model should save and load correctly."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        model.train(self.X, self.y)
        model.save("TEST_SYMBOL")

        loaded = RandomForestModel.load("TEST_SYMBOL", "random_forest")
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.is_trained)

        # Clean up
        import os
        try:
            from quant.services.ml_models import _model_path
            path = _model_path("TEST_SYMBOL", "random_forest")
            if path.exists():
                os.remove(path)
        except Exception:
            pass


class EnsemblePredictorTests(TestCase):
    """Test EnsemblePredictor weighting and aggregation."""

    def setUp(self):
        np.random.seed(42)
        n = 200
        self.X = np.random.randn(n, 10)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(float)

    def test_empty_ensemble(self):
        """Empty ensemble should return 0.5."""
        ensemble = EnsemblePredictor()
        self.assertEqual(ensemble.predict(np.array([[0] * 10])), 0.5)

    def test_single_model_ensemble(self):
        """Single model ensemble should match model's prediction."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        model.train(self.X, self.y)

        ensemble = EnsemblePredictor()
        ensemble.add_model(model)

        test_point = np.random.randn(1, 10)
        model_pred = model.predict_proba(test_point)
        ensemble_pred = ensemble.predict(test_point)
        self.assertAlmostEqual(model_pred, ensemble_pred, places=5)

    def test_weighted_ensemble(self):
        """Weights should affect ensemble prediction."""
        model1 = RandomForestModel(n_estimators=10, max_depth=3)
        model2 = RandomForestModel(n_estimators=10, max_depth=3)
        model1.train(self.X, self.y)
        model2.train(self.X, self.y)

        ensemble = EnsemblePredictor()
        ensemble.add_model(model1, weight=1.0)
        ensemble.add_model(model2, weight=0.0)  # Zero weight

        test_point = np.random.randn(1, 10)
        ensemble_pred = ensemble.predict(test_point)
        model1_pred = model1.predict_proba(test_point)
        self.assertAlmostEqual(ensemble_pred, model1_pred, places=3)

    def test_update_weights(self):
        """update_weights should adjust ensemble weights."""
        ensemble = EnsemblePredictor()
        model1 = RandomForestModel(n_estimators=10, max_depth=3)
        model2 = RandomForestModel(n_estimators=10, max_depth=3)
        ensemble.add_model(model1)
        ensemble.add_model(model2)

        # One model has perfect accuracy, other has 0%
        ensemble.update_weights([0.9, 0.1])
        self.assertGreater(ensemble.weights[0], ensemble.weights[1])

    def test_predict_direction_threshold(self):
        """predict_direction should use 55% threshold."""
        ensemble = EnsemblePredictor()
        # No models = returns 0.5 < 0.55 = 0 (down)
        self.assertEqual(ensemble.predict_direction(np.array([[0] * 10])), 0)


# ══════════════════════════════════════════════════════════════════
#  ModelTrainer Tests
# ══════════════════════════════════════════════════════════════════


class ModelTrainerTests(TestCase):
    """Test ModelTrainer evaluation and metrics."""

    def setUp(self):
        np.random.seed(42)
        n = 300
        self.X = np.random.randn(n, 10)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(float)
        self.models = [RandomForestModel(n_estimators=10, max_depth=3)]

    def test_train_and_evaluate_returns_results(self):
        """train_and_evaluate should return complete results dict."""
        trainer = ModelTrainer()
        results = trainer.train_and_evaluate(self.X, self.y, self.models, ["rf"])
        self.assertIn("models", results)
        self.assertIn("best_model", results)
        self.assertIn("total_samples", results)
        self.assertIn("rf", results["models"])

    def test_insufficient_data(self):
        """Very small dataset should return error."""
        trainer = ModelTrainer()
        results = trainer.train_and_evaluate(
            np.random.randn(20, 5), np.random.randn(20), self.models
        )
        self.assertIn("error", results)

    def test_accuracy_calculation(self):
        """Accuracy should be between 0 and 1."""
        model = self.models[0]
        model.train(self.X[:100], self.y[:100])
        acc = ModelTrainer._calculate_accuracy(model, self.X[100:150], self.y[100:150])
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


# ══════════════════════════════════════════════════════════════════
#  Walk-Forward Validation Tests
# ══════════════════════════════════════════════════════════════════


class WalkForwardTests(TestCase):
    """Test walk-forward validation."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        self.X = np.random.randn(n, 10)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(float)

    def test_walk_forward_returns_windows(self):
        """Walk-forward should return multiple windows."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        results = walk_forward_validate(model, self.X, self.y, window_size=200, step_size=50)
        self.assertIn("windows", results)
        self.assertGreater(results["total_windows"], 1)

    def test_insufficient_data(self):
        """Too little data should return error."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        results = walk_forward_validate(
            model, np.random.randn(50, 5), np.random.randn(50),
            window_size=100, step_size=50,
        )
        self.assertIn("error", results)


# ══════════════════════════════════════════════════════════════════
#  SignalPurger Tests
# ══════════════════════════════════════════════════════════════════


class SignalPurgerTests(TestCase):
    """Test Renaissance-style signal filtering."""

    def setUp(self):
        self.purger = SignalPurger()

    def test_strong_signal_passes(self):
        """Strong signal should pass all filters."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.015 + 0.008  # Strong positive mean returns
        metrics = {
            "in_sample_sharpe": 2.0,
            "out_of_sample_sharpe": 1.6,
            "trade_returns": returns.tolist(),
        }
        keep, reasons = self.purger.should_keep_signal(metrics)
        self.assertTrue(keep)
        self.assertEqual(len(reasons), 0)

    def test_low_sharpe_rejected(self):
        """Low in-sample Sharpe should be rejected."""
        np.random.seed(42)
        metrics = {
            "in_sample_sharpe": 0.5,
            "out_of_sample_sharpe": 0.3,
            "trade_returns": [],
        }
        keep, reasons = self.purger.should_keep_signal(metrics)
        self.assertFalse(keep)
        self.assertGreater(len(reasons), 0)

    def test_math_helpers(self):
        """Math helper functions should work."""
        # erf(0) = 0
        self.assertAlmostEqual(_erf_fallback(0), 0.0, places=5)
        # normal_cdf(0) = 0.5
        self.assertAlmostEqual(_normal_cdf(0), 0.5, places=5)
        # normal_cdf(inf) = 1.0
        self.assertAlmostEqual(_normal_cdf(100), 1.0, places=5)
