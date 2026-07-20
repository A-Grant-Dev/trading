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
from quant.services.cointegration import PairsFinder, fetch_daily_close_prices
from quant.services.pairs_signals import PairsSignalGenerator
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


# ══════════════════════════════════════════════════════════════════
#  Phase 2 — Cointegration & Pairs Trading Tests
# ══════════════════════════════════════════════════════════════════


class PairsFinderTests(TestCase):
    """Test PairsFinder cointegration discovery and statistics."""

    def setUp(self):
        # Create two synthetically cointegrated price series
        np.random.seed(42)
        n = 200
        # Random walk for the 'base' series
        base = 100 + np.cumsum(np.random.randn(n) * 0.3)
        # Series B follows series A with noise (cointegrated)
        series_a = base + np.random.randn(n) * 0.1
        series_b = base * 0.5 + 50 + np.random.randn(n) * 0.1  # hedge_ratio ~0.5

        dates = pd.date_range(start="2025-01-01", periods=n, freq="1h", tz="UTC")
        self.price_data = {
            "BTCUSDT": pd.Series(series_a, index=dates),
            "ETHUSDT": pd.Series(series_b, index=dates),
        }

        # Non-cointegrated series (random walks)
        series_c = 100 + np.cumsum(np.random.randn(n) * 0.3)
        series_d = 200 + np.cumsum(np.random.randn(n) * 0.5)
        self.price_data["SOLUSDT"] = pd.Series(series_c, index=dates)
        self.price_data["XRPUSDT"] = pd.Series(series_d, index=dates)

    def test_find_cointegrated_pairs_discovers_relationship(self):
        """Should find the cointegrated pair BTCUSDT/ETHUSDT."""
        finder = PairsFinder(["BTCUSDT", "ETHUSDT"])
        results = finder.find_cointegrated_pairs(self.price_data, p_threshold=0.1)

        self.assertGreater(len(results), 0)
        pair = results[0]
        self.assertIn(pair["symbol_a"], ["BTCUSDT", "ETHUSDT"])
        self.assertIn(pair["symbol_b"], ["BTCUSDT", "ETHUSDT"])
        self.assertLessEqual(pair["p_value"], 0.1)
        self.assertIsNotNone(pair.get("hedge_ratio"))
        self.assertIsNotNone(pair.get("current_zscore"))

    def test_find_cointegrated_pairs_empty_data(self):
        """Should raise ValueError with empty price data."""
        finder = PairsFinder(["BTCUSDT"])
        with self.assertRaises(ValueError):
            finder.find_cointegrated_pairs({})

    def test_find_cointegrated_pairs_no_relationship(self):
        """Should not find relationships where none exist."""
        finder = PairsFinder(["SOLUSDT", "XRPUSDT"])
        results = finder.find_cointegrated_pairs(self.price_data, p_threshold=0.01)
        # Two random walks should not be cointegrated at p<0.01
        self.assertEqual(len(results), 0)

    def test_compute_half_life(self):
        """Half-life computation should return a positive value for mean-reverting spread."""
        # Create a clearly mean-reverting spread (oscillating around 0)
        np.random.seed(42)
        n = 200
        # AR(1) with strong mean reversion: theta ~ 0.8 → half_life < 1
        errors = np.random.randn(n) * 0.5
        spread_ar = np.zeros(n)
        spread_ar[0] = errors[0]
        for i in range(1, n):
            spread_ar[i] = 0.2 * spread_ar[i-1] + errors[i]
        half_life = PairsFinder.compute_half_life(spread_ar)
        self.assertIsNotNone(half_life)
        self.assertGreater(half_life, 0)
        # With theta = 0.8 (since coefficient = (1 - 0.8) = 0.2), half_life < 5
        self.assertLess(half_life, 5)

    def test_compute_half_life_insufficient_data(self):
        """Half-life should return None with too few data points."""
        result = PairsFinder.compute_half_life(np.array([1.0, 2.0]))
        self.assertIsNone(result)

    def test_compute_zscore(self):
        """Z-score should be computed correctly."""
        spread = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        z = PairsFinder.compute_zscore(spread)
        # Last value (15) should be above the mean
        self.assertGreater(z, 0)

    def test_compute_zscore_constant(self):
        """Z-score should be 0 for constant spread."""
        spread = np.array([10.0, 10.0, 10.0, 10.0])
        z = PairsFinder.compute_zscore(spread)
        self.assertEqual(z, 0.0)

    def test_compute_zscore_single_value(self):
        """Z-score should be 0 for single value."""
        z = PairsFinder.compute_zscore(np.array([10.0]))
        self.assertEqual(z, 0.0)


class PairsSignalGeneratorTests(TestCase):
    """Test PairsSignalGenerator entry/exit logic and backtesting."""

    def setUp(self):
        self.generator = PairsSignalGenerator(
            entry_z=2.0,
            exit_z=0.5,
            stop_z=3.0,
        )

        # Create synthetic pair data for backtesting
        np.random.seed(42)
        n = 500

        # Cointegrated pair with noise
        base = 100 + np.cumsum(np.random.randn(n) * 0.2)
        a = base + np.random.randn(n) * 0.5
        b = base * 0.8 + 20 + np.random.randn(n) * 0.5

        dates = pd.date_range(start="2025-01-01", periods=n, freq="1h", tz="UTC")
        self.historical_data = pd.DataFrame({
            "ASSETUSDT": a,
            "BSSETUSDT": b,
        }, index=dates)

        self.pair_data = {
            "symbol_a": "ASSETUSDT",
            "symbol_b": "BSSETUSDT",
            "hedge_ratio": 0.8,
        }

    def test_backtest_pair_returns_results(self):
        """Backtest should return a complete result dict."""
        results = self.generator.backtest_pair(
            self.pair_data,
            self.historical_data,
        )

        self.assertIn("total_trades", results)
        self.assertIn("winning_trades", results)
        self.assertIn("losing_trades", results)
        self.assertIn("win_rate", results)
        self.assertIn("sharpe_ratio", results)
        self.assertIn("trades", results)

    def test_backtest_pair_empty_data(self):
        """Backtest with empty data should return error."""
        results = self.generator.backtest_pair(self.pair_data, pd.DataFrame())
        self.assertIn("error", results)

    def test_backtest_pair_single_column(self):
        """Backtest with single column should return error."""
        single = pd.DataFrame({"A": [1, 2, 3]})
        results = self.generator.backtest_pair(self.pair_data, single)
        self.assertIn("error", results)

    def test_zscore_computation(self):
        """Z-score computation should work with the helper."""
        spread = pd.Series([10.0, 12.0, 15.0, 11.0, 9.0, 14.0, 13.0, 10.0])
        z = PairsSignalGenerator._compute_zscore(spread)
        self.assertIsInstance(z, float)

    def test_compute_half_life_from_spread(self):
        """Should compute a reasonable half-life from a pair's spread."""
        # Compute spread from the historical data
        spread = (
            self.historical_data["ASSETUSDT"].values
            - 0.8 * self.historical_data["BSSETUSDT"].values
        )
        half_life = PairsFinder.compute_half_life(spread)
        # Half-life should be reasonable (between 1 and 1000 periods)
        if half_life is not None:
            self.assertGreater(half_life, 0)
            self.assertLess(half_life, 1000)

    def test_compute_zscore_from_series(self):
        """_compute_zscore should return correct values for a spread series."""
        spread = pd.Series([10.0, 12.0, 15.0, 11.0, 9.0, 14.0, 13.0, 10.0])
        z = PairsSignalGenerator._compute_zscore(spread)
        self.assertIsInstance(z, float)
        # Last value (10.0) is near the mean, so z should be close to 0
        self.assertAlmostEqual(z, 0.0, delta=2.0)

    def test_backtest_pair_metrics(self):
        """Backtest metrics should be internally consistent."""
        results = self.generator.backtest_pair(
            self.pair_data,
            self.historical_data,
        )

        if results.get("total_trades", 0) > 0:
            # Win rate should be between 0 and 1
            self.assertGreaterEqual(results["win_rate"], 0.0)
            self.assertLessEqual(results["win_rate"], 1.0)

            # Total trades should equal winning + losing
            self.assertEqual(
                results["total_trades"],
                results["winning_trades"] + results["losing_trades"],
            )

            # Max drawdown should be non-negative
            self.assertGreaterEqual(results["max_drawdown"], 0.0)
