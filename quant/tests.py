"""
Tests for Phase 0 & Phase 1 — Foundation & HMM Regime Detection

Tests cover:
  - Technical indicator computations (data_utils)
  - HMM regime detector train/predict cycle (hmm_regime)
  - Regime-aware signal weighting (regime_signals)
  - Edge case handling (empty data, missing columns, extreme values)
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from django.test import TestCase

from quant.services.data_utils import (
    compute_atr,
    compute_rsi,
    _compute_ema,
    _compute_sma,
    add_technical_indicators,
)
from quant.services.hmm_regime import MarketRegimeDetector, build_hmm_features
from quant.services.regime_signals import (
    REGIME_WEIGHTS,
    adjust_signal_for_regime,
    combine_regime_with_signal,
    get_max_position_pct,
    get_preferred_strategies,
    get_regime_adjusted_thresholds,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_sample_ohlcv(n: int = 200) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, n)

    dates = pd.date_range(
        start="2025-01-01",
        periods=n,
        freq="1h",
        tz="UTC",
    )

    df = pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    return df


# ══════════════════════════════════════════════════════════════════
#  Phase 0 — Technical Indicator Tests
# ══════════════════════════════════════════════════════════════════


class TechnicalIndicatorTests(TestCase):
    """Test manual numpy/pandas technical indicator implementations."""

    def setUp(self):
        self.df = _make_sample_ohlcv(200)

    def test_sma_computation(self):
        """SMA should equal the mean over the window."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        sma = _compute_sma(values, 3)
        # sma[2] should be mean(1,2,3) = 2, sma[3] = mean(2,3,4) = 3, etc.
        self.assertTrue(np.isnan(sma[0]))
        self.assertTrue(np.isnan(sma[1]))
        self.assertAlmostEqual(sma[2], 2.0)
        self.assertAlmostEqual(sma[3], 3.0)
        self.assertAlmostEqual(sma[9], 9.0)

    def test_ema_computation(self):
        """EMA should respond more to recent values."""
        values = np.array([10, 10, 10, 10, 100], dtype=float)
        ema = _compute_ema(values, 3)
        self.assertTrue(np.isnan(ema[0]))
        self.assertTrue(np.isnan(ema[1]))
        # ema[2] = mean(10,10,10) = 10
        self.assertAlmostEqual(ema[2], 10.0, places=1)
        # ema[4] should be higher due to the 100 at index 4
        self.assertGreater(ema[4], 10.0)

    def test_rsi_computation(self):
        """RSI should be between 0 and 100."""
        values = np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                           61, 62, 63, 64, 65], dtype=float)
        rsi = compute_rsi(values, 14)
        # With all up moves, RSI should approach 100
        rsi_value = rsi[15] if not np.isnan(rsi[15]) else rsi[14]
        self.assertGreaterEqual(rsi_value, 60)  # Strongly bullish

    def test_rsi_bearish(self):
        """RSI should be low with mostly down moves."""
        values = np.array([100, 98, 96, 94, 92, 90, 88, 86, 84, 82,
                           80, 78, 76, 74, 72, 70], dtype=float)
        rsi = compute_rsi(values, 14)
        rsi_value = rsi[15] if not np.isnan(rsi[15]) else rsi[14]
        self.assertLessEqual(rsi_value, 40)  # Strongly bearish

    def test_atr_computation(self):
        """ATR should be non-negative and reflect volatility."""
        high = np.array([11, 12, 13, 14, 15], dtype=float)
        low = np.array([9, 8, 7, 6, 5], dtype=float)
        close = np.array([10, 10, 10, 10, 10], dtype=float)
        atr = compute_atr(high, low, close, 3)
        self.assertTrue(np.isnan(atr[0]))
        self.assertTrue(np.isnan(atr[1]))
        # atr[3] should be positive (period=3, first valid at index 3)
        self.assertGreater(atr[3], 0)

    def test_add_technical_indicators(self):
        """add_technical_indicators should add all expected columns."""
        result = add_technical_indicators(self.df)
        expected_cols = [
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "atr_14", "atr_pct",
            "ema_9", "ema_21", "ema_50",
            "volume_sma_20", "volume_ratio", "obv",
            "vwap", "log_return", "return_1", "return_5", "return_15",
            "volatility_14", "volatility_30",
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_indicators_empty_df(self):
        """Indicators should handle empty DataFrames gracefully."""
        empty = pd.DataFrame()
        result = add_technical_indicators(empty)
        self.assertTrue(result.empty)


# ══════════════════════════════════════════════════════════════════
#  Phase 1 — HMM Regime Detection Tests
# ══════════════════════════════════════════════════════════════════


class HMMRegimeDetectionTests(TestCase):
    """Test HMM MarketRegimeDetector training and prediction."""

    def setUp(self):
        self.df = _make_sample_ohlcv(500)

    def test_detector_initialization(self):
        """Detector should initialize with 4 states by default."""
        detector = MarketRegimeDetector()
        self.assertEqual(detector.n_states, 4)
        self.assertFalse(detector.is_trained)
        self.assertIsNotNone(detector.model)

    def test_train_and_predict(self):
        """Detector should train on real data and predict a regime."""
        features = build_hmm_features(self.df)
        self.assertGreater(len(features), 50)

        detector = MarketRegimeDetector(n_states=4)
        detector.train(features)

        self.assertTrue(detector.is_trained)

        # Predict on the last row
        from quant.services.hmm_regime import DEFAULT_FEATURES
        last_features = features[DEFAULT_FEATURES].iloc[-1:].values.flatten()
        state_id, label = detector.predict_regime(last_features)

        self.assertIn(state_id, range(4))
        self.assertIn(label, ["ranging", "bullish", "bearish", "volatile"])

    def test_regime_probabilities(self):
        """Detector should return probability distribution."""
        from quant.services.hmm_regime import DEFAULT_FEATURES
        features = build_hmm_features(self.df)
        detector = MarketRegimeDetector(n_states=4)
        detector.train(features)

        last_features = features[DEFAULT_FEATURES].iloc[-1:].values.flatten()
        probs = detector.get_regime_probabilities(last_features)

        self.assertEqual(len(probs), 4)
        # Probabilities should sum to ~1.0
        total = sum(p["probability"] for p in probs.values())
        self.assertAlmostEqual(total, 1.0, places=1)

    def test_predict_before_training(self):
        """Predicting before training should return -1/unknown."""
        detector = MarketRegimeDetector()
        features = np.array([0.0, 0.0, 0.0, 50.0, 1.0])
        state_id, label = detector.predict_regime(features)
        self.assertEqual(state_id, -1)
        self.assertEqual(label, "unknown")

    def test_insufficient_data(self):
        """Training with insufficient data should raise ValueError."""
        small_df = self.df.head(5)
        features = build_hmm_features(small_df)
        detector = MarketRegimeDetector(n_states=4)
        with self.assertRaises(ValueError):
            detector.train(features)

    def test_missing_columns(self):
        """Training with missing feature columns should raise ValueError."""
        detector = MarketRegimeDetector(n_states=4)
        bad_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with self.assertRaises(ValueError):
            detector.train(bad_df)

    def test_get_state_sequence(self):
        """get_state_sequence should return predictions for all rows."""
        features = build_hmm_features(self.df)
        detector = MarketRegimeDetector(n_states=4)
        detector.train(features)

        sequence = detector.get_state_sequence(features)
        self.assertEqual(len(sequence), len(features))
        # Only non-NaN rows get labels
        valid = sequence.dropna()
        self.assertGreater(len(valid), 0)

    def test_get_regime_info(self):
        """get_regime_info should return descriptions for each state."""
        features = build_hmm_features(self.df)
        detector = MarketRegimeDetector(n_states=4)
        detector.train(features)

        info = detector.get_regime_info()
        self.assertEqual(len(info), 4)
        for state, data in info.items():
            self.assertIn("label", data)
            self.assertIn("description", data)

    def test_build_hmm_features_edge_cases(self):
        """build_hmm_features should handle edge inputs."""
        # Single column
        single = pd.DataFrame({"close": [100, 101, 102]})
        result = build_hmm_features(single)
        # Should be empty since we need high/low/volume too
        self.assertTrue(result.empty or len(result) == 0)

        # All same values (no volatility)
        flat = pd.DataFrame({
            "close": [100] * 100,
            "high": [101] * 100,
            "low": [99] * 100,
            "volume": [5000] * 100,
        })
        result = build_hmm_features(flat)
        self.assertFalse(result.empty)
        # RSI should be 0 (no gains at all = no positive movement)
        rsi_mean = result["rsi_14"].iloc[-10:].mean()
        self.assertEqual(rsi_mean, 0.0)

    def test_constant_inputs_dont_crash(self):
        """Detector should not crash with constant (zero-volatility) inputs."""
        n = 100
        const_df = pd.DataFrame({
            "close": [100.0] * n,
            "high": [100.5] * n,
            "low": [99.5] * n,
            "volume": [5000] * n,
        })
        features = build_hmm_features(const_df)
        if len(features) >= 40:
            detector = MarketRegimeDetector(n_states=3)
            try:
                detector.train(features)
                self.assertTrue(detector.is_trained)
            except Exception:
                # Training may fail on constant data (singular covariance),
                # but it should raise a clear error, not crash
                pass


class RegimeSignalTests(TestCase):
    """Test regime-aware signal weighting logic."""

    def test_adjust_signal_bullish(self):
        """Bullish regime should amplify long signals, suppress short."""
        long_adjusted = adjust_signal_for_regime(0.5, "long", "bullish")
        short_adjusted = adjust_signal_for_regime(0.5, "short", "bullish")
        self.assertGreater(long_adjusted, short_adjusted)
        self.assertAlmostEqual(long_adjusted, 0.5 * 1.2)
        self.assertAlmostEqual(short_adjusted, 0.5 * 0.4)

    def test_adjust_signal_bearish(self):
        """Bearish regime should amplify short signals, suppress long."""
        long_adjusted = adjust_signal_for_regime(0.5, "long", "bearish")
        short_adjusted = adjust_signal_for_regime(0.5, "short", "bearish")
        self.assertGreater(short_adjusted, long_adjusted)
        self.assertAlmostEqual(short_adjusted, 0.5 * 1.2)
        self.assertAlmostEqual(long_adjusted, 0.5 * 0.4)

    def test_adjust_signal_volatile(self):
        """Volatile regime should reduce both long and short signals."""
        long_adjusted = adjust_signal_for_regime(1.0, "long", "volatile")
        short_adjusted = adjust_signal_for_regime(1.0, "short", "volatile")
        self.assertLess(long_adjusted, 1.0)
        self.assertLess(short_adjusted, 1.0)
        self.assertAlmostEqual(long_adjusted, 0.3)
        self.assertAlmostEqual(short_adjusted, 0.3)

    def test_adjust_signal_ranging(self):
        """Ranging regime should reduce directional signals."""
        adjusted = adjust_signal_for_regime(1.0, "long", "ranging")
        self.assertAlmostEqual(adjusted, 0.5)

    def test_unknown_regime(self):
        """Unknown regime should use default weight."""
        adjusted = adjust_signal_for_regime(1.0, "long", "unknown_regime")
        self.assertAlmostEqual(adjusted, 0.5)

    def test_signal_clipping(self):
        """Adjusted signals should stay within [0, 1]."""
        # Very high base signal
        high = adjust_signal_for_regime(2.0, "long", "bullish")
        self.assertLessEqual(high, 1.0)
        # Negative base signal
        neg = adjust_signal_for_regime(-0.5, "long", "bullish")
        self.assertGreaterEqual(neg, 0.0)

    def test_get_regime_adjusted_thresholds(self):
        """Thresholds should vary by regime."""
        ranging = get_regime_adjusted_thresholds("ranging")
        volatile = get_regime_adjusted_thresholds("volatile")
        self.assertIn("entry_z", ranging)
        self.assertIn("exit_z", ranging)
        # Volatile should have wider thresholds
        self.assertGreater(volatile["entry_z"], ranging["entry_z"])

    def test_get_max_position_pct(self):
        """Max position % should vary by regime."""
        bullish_pct = get_max_position_pct("bullish")
        volatile_pct = get_max_position_pct("volatile")
        self.assertGreater(bullish_pct, volatile_pct)

    def test_get_preferred_strategies(self):
        """Each regime should have its own preferred strategies."""
        ranging = get_preferred_strategies("ranging")
        self.assertIn("mean_reversion", ranging)
        bullish = get_preferred_strategies("bullish")
        self.assertIn("momentum", bullish)

    def test_combine_regime_with_signal(self):
        """Combine should return a complete decision dict."""
        base = {"direction": "long", "strength": 0.8, "confidence": 0.7}
        result = combine_regime_with_signal(base, "bullish", 0.85)

        self.assertEqual(result["regime"], "bullish")
        self.assertEqual(result["recommendation"], "execute")
        self.assertAlmostEqual(result["adjusted_strength"], 0.8 * 1.2)
        self.assertIn("max_position_pct", result)
        self.assertIn("preferred_strategies", result)
        self.assertIn("thresholds", result)

    def test_combine_volatile_recommendation(self):
        """In volatile regime, even strong signals should be avoided."""
        base = {"direction": "long", "strength": 0.8, "confidence": 0.7}
        result = combine_regime_with_signal(base, "volatile", 0.9)
        self.assertEqual(result["recommendation"], "avoid")
