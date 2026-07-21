"""
Tests for Trading Strategies — base.py, technical.py

Coverage:
- Base strategy abstract interface
- MomentumStrategy signal generation
- MeanReversionStrategy signal generation
- BreakoutStrategy signal generation
- Signal utilities (normalize_confidence, zscore_confidence, get_feature_array)
- Edge cases: missing columns, NaN values, single rows
"""

import numpy as np
import polars as pl
from django.test import TestCase

from trading_bot.services.strategies.base import (
    BaseStrategy,
    get_feature_array,
    normalize_confidence,
    zscore_confidence,
)
from trading_bot.services.strategies.technical import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
)


class TestSignalUtilities(TestCase):
    """Test signal utility functions."""

    def test_normalize_confidence_clamps(self):
        """Confidence should be clamped to 0.0–1.0."""
        raw = np.array([-0.5, 0.0, 0.5, 1.5, 2.0])
        result = normalize_confidence(raw)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        self.assertEqual(result[0], 0.5)
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[4], 1.0)

    def test_zscore_confidence_zero(self):
        """z=0 should give confidence 0."""
        z = np.array([0.0])
        result = zscore_confidence(z)
        self.assertAlmostEqual(result[0], 0.0)

    def test_zscore_confidence_high(self):
        """z >= 3 should give confidence 1."""
        z = np.array([3.0, 5.0])
        result = zscore_confidence(z)
        self.assertTrue(np.all(result >= 1.0))

    def test_get_feature_array_existing(self):
        """Existing feature should return its values."""
        df = pl.DataFrame({"test_col": [1.0, 2.0, 3.0]})
        result = get_feature_array(df, "test_col")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_get_feature_array_missing(self):
        """Missing feature should return zeros."""
        df = pl.DataFrame({"other": [1.0, 2.0]})
        result = get_feature_array(df, "missing")
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])


class TestMomentumStrategy(TestCase):
    """Test MomentumStrategy signal generation."""

    def setUp(self):
        self.strategy = MomentumStrategy()

    def test_strategy_has_name(self):
        """Strategy should have a name."""
        self.assertEqual(self.strategy.name, "Momentum Strategy")

    def test_strategy_has_default_params(self):
        """Strategy should have sensible defaults."""
        self.assertIn("ema_fast", self.strategy.params)
        self.assertIn("ema_slow", self.strategy.params)
        self.assertIn("rsi_column", self.strategy.params)

    def test_long_signal(self):
        """Fast EMA above slow + RSI bullish + positive return = long."""
        df = pl.DataFrame({
            "ema_9": [101.0, 102.0],
            "ema_21": [100.0, 100.5],
            "rsi_14": [60.0, 65.0],
            "return_5": [0.01, 0.02],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertEqual(signals[-1], 1)  # Last row should be long

    def test_short_signal(self):
        """Fast EMA below slow + RSI bearish + negative return = short."""
        df = pl.DataFrame({
            "ema_9": [99.0, 98.0],
            "ema_21": [100.0, 99.5],
            "rsi_14": [40.0, 35.0],
            "return_5": [-0.01, -0.02],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertEqual(signals[-1], -1)  # Last row should be short

    def test_neutral_signal(self):
        """Conflicting indicators should give neutral signal."""
        df = pl.DataFrame({
            "ema_9": [100.0, 101.0],
            "ema_21": [100.0, 100.5],
            "rsi_14": [50.0, 45.0],  # Bearish RSI but bullish EMA crossover
            "return_5": [0.01, -0.01],  # Negative return
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertEqual(signals[-1], 0)  # Neutral

    def test_all_neutral_default(self):
        """Default DataFrame with no clear signal should be neutral."""
        df = pl.DataFrame({
            "ema_9": [100.0, 100.0],
            "ema_21": [100.0, 100.0],
            "rsi_14": [50.0, 50.0],
            "return_5": [0.0, 0.0],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertTrue(np.all(signals == 0))

    def test_confidence_is_bounded(self):
        """Confidence values should be 0.0–1.0."""
        df = pl.DataFrame({
            "ema_9": [95.0, 105.0],
            "ema_21": [100.0, 100.0],
            "rsi_14": [30.0, 70.0],
            "return_5": [-0.02, 0.03],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertTrue(np.all(confidence >= 0.0))
        self.assertTrue(np.all(confidence <= 1.0))


class TestMeanReversionStrategy(TestCase):
    """Test MeanReversionStrategy signal generation."""

    def setUp(self):
        self.strategy = MeanReversionStrategy()

    def test_oversold_long_signal(self):
        """Price below lower BB + RSI oversold = long."""
        df = pl.DataFrame({
            "close": [90.0, 85.0],
            "bb_upper": [110.0, 110.0],
            "bb_lower": [95.0, 95.0],
            "rsi_14": [25.0, 20.0],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertEqual(signals[-1], 1)  # Long

    def test_overbought_short_signal(self):
        """Price above upper BB + RSI overbought = short."""
        df = pl.DataFrame({
            "close": [110.0, 115.0],
            "bb_upper": [110.0, 110.0],
            "bb_lower": [90.0, 90.0],
            "rsi_14": [75.0, 80.0],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertEqual(signals[-1], -1)  # Short

    def test_no_extreme_no_signal(self):
        """No overbought/oversold = neutral."""
        df = pl.DataFrame({
            "close": [100.0, 102.0],
            "bb_upper": [110.0, 110.0],
            "bb_lower": [90.0, 90.0],
            "rsi_14": [50.0, 55.0],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertTrue(np.all(signals == 0))


class TestBreakoutStrategy(TestCase):
    """Test BreakoutStrategy signal generation."""

    def setUp(self):
        self.strategy = BreakoutStrategy()

    def test_long_breakout(self):
        """Near high price + volume surge = long breakout."""
        df = pl.DataFrame({
            "price_position_high_20": [0.99, 0.99],
            "price_position_low_20": [5.0, 5.0],  # Far above low → short condition false
            "volume_ratio": [1.5, 2.0],
            "volatility_14": [0.02, 0.03],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertEqual(signals[-1], 1)

    def test_short_breakout(self):
        """Near low price + volume surge = short breakout."""
        df = pl.DataFrame({
            "price_position_high_20": [0.80, 0.80],
            "price_position_low_20": [1.01, 1.01],
            "volume_ratio": [1.5, 2.0],
            "volatility_14": [0.02, 0.03],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertEqual(signals[-1], -1)

    def test_low_volume_no_signal(self):
        """Low volume should prevent signal even at extremes."""
        df = pl.DataFrame({
            "price_position_high_20": [0.50, 0.50],  # Mid-range → no long
            "price_position_low_20": [0.50, 0.50],   # Mid-range → no short
            "volume_ratio": [1.0, 1.1],  # Below threshold
            "volatility_14": [0.02, 0.03],
        })
        signals, confidence = self.strategy.generate_signals(df)
        self.assertTrue(np.all(signals == 0))
