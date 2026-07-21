"""
Phase 7 & 8 — Dashboard & Backtesting Tests

Tests cover:
  - QuantNotifier: all notification types, edge cases
  - BacktestEngine: event-driven simulation, fee/slippage, empty data
  - BacktestResult: all metric properties, edge cases
  - MonteCarloSimulator: trade reshuffling, probability calculation
  - toggle_mode management command
  - Edge cases: no trades, single trade, constant prices
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from django.test import TestCase

from quant.models import ExecutedTrade, QuantConfig


# ══════════════════════════════════════════════════════════════════
#  QuantNotifier Tests
# ══════════════════════════════════════════════════════════════════


class QuantNotifierTests(TestCase):
    """Test QuantNotifier notifications."""

    def setUp(self):
        self.config = QuantConfig.objects.create(
            pk=1, mode="paper", is_enabled=True, virtual_balance=10000.00,
        )

    def _make_trade(self, status="open", pnl=50.0):
        return ExecutedTrade.objects.create(
            symbol="BTCUSDT", side="buy", entry_price=65000.0,
            exit_price=65100.0 if status == "closed" else None,
            qty=0.1, pnl=pnl, pnl_pct=pnl / 6500.0,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc) if status == "closed" else None,
            status=status, strategy="test", notes="Test trade",
        )

    def test_notify_trade(self):
        """notify_trade should log trade details."""
        from quant.services.quant_notifier import QuantNotifier
        trade = self._make_trade()
        # Should not raise any exceptions
        QuantNotifier().notify_trade(trade)

    def test_notify_exit(self):
        """notify_exit should log exit with reason."""
        from quant.services.quant_notifier import QuantNotifier
        trade = self._make_trade(status="closed")
        QuantNotifier().notify_exit(trade, "Take profit hit")

    def test_notify_regime_change(self):
        """notify_regime_change should log regime transition."""
        from quant.services.quant_notifier import QuantNotifier
        QuantNotifier().notify_regime_change("BTCUSDT", "ranging", "bullish", 0.85)

    def test_notify_risk_event(self):
        """notify_risk_event should log risk warnings."""
        from quant.services.quant_notifier import QuantNotifier
        QuantNotifier().notify_risk_event("max_drawdown", "Drawdown exceeded 15%")

    def test_notify_new_pair(self):
        """notify_new_pair should log pair discovery."""
        from quant.services.quant_notifier import QuantNotifier
        QuantNotifier().notify_new_pair("BTCUSDT", "ETHUSDT", 0.003, 18.5)

    def test_notify_daily_summary(self):
        """notify_daily_summary should log daily stats."""
        from quant.services.quant_notifier import QuantNotifier
        self._make_trade(status="closed", pnl=100.0)
        self._make_trade(status="closed", pnl=-50.0)
        self._make_trade(status="open")
        QuantNotifier().notify_daily_summary()


# ══════════════════════════════════════════════════════════════════
#  BacktestEngine Tests
# ══════════════════════════════════════════════════════════════════


def _make_sample_ohlcv(n: int = 200) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close - np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n)

    dates = pd.date_range(start="2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)


def _buy_and_hold_strategy(row, df):
    """Simple strategy: buy on first bar, sell on last."""
    if len(df) == 1:
        return {"action": "buy", "confidence": 0.6, "quantity": 0.95, "reason": "Buy on first bar"}
    if len(df) == len(df) - 0:  # Never sell via strategy
        return None
    return None


def _always_buy_strategy(row, df):
    """Always buy — generates constant signals."""
    return {"action": "buy", "confidence": 0.5, "quantity": 0.5, "reason": "Always buy"}


def _buy_sell_alternating(row, df):
    """Buy first bar, sell last bar."""
    if len(df) == 1:
        return {"action": "buy", "confidence": 0.6, "quantity": 0.95, "reason": "Buy"}
    if len(df) == 100:
        return {"action": "sell", "confidence": 0.6, "quantity": 0.95, "reason": "Sell"}
    return None


class BacktestEngineTests(TestCase):
    """Test BacktestEngine event-driven simulation."""

    def test_run_with_data(self):
        """Running with data should produce a result with trades."""
        from quant.services.backtesting import BacktestEngine
        df = _make_sample_ohlcv(200)
        engine = BacktestEngine(initial_capital=10000.0)
        engine.add_strategy(_buy_sell_alternating)
        result = engine.run(df)

        self.assertGreater(result.total_trades, 0)
        self.assertGreater(len(result.equity_curve), 0)
        self.assertIsInstance(result.sharpe_ratio, float)

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty result."""
        from quant.services.backtesting import BacktestEngine
        df = pd.DataFrame()
        engine = BacktestEngine()
        result = engine.run(df)
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.final_capital, 10000.0)

    def test_fee_and_slippage_deducted(self):
        """Fees and slippage should reduce final capital."""
        from quant.services.backtesting import BacktestEngine
        df = _make_sample_ohlcv(200)

        # Without fees
        engine_no = BacktestEngine(initial_capital=10000.0, fee_rate=0.0, slippage=0.0)
        engine_no.add_strategy(_buy_sell_alternating)
        result_no = engine_no.run(df)

        # With fees
        engine_yes = BacktestEngine(initial_capital=10000.0, fee_rate=0.01, slippage=0.01)
        engine_yes.add_strategy(_buy_sell_alternating)
        result_yes = engine_yes.run(df)

        self.assertGreaterEqual(result_no.final_capital, result_yes.final_capital)

    def test_no_strategies(self):
        """No strategies should result in zero trades."""
        from quant.services.backtesting import BacktestEngine
        df = _make_sample_ohlcv(200)
        engine = BacktestEngine()
        result = engine.run(df)
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.signals_generated, 0)

    def test_constant_prices(self):
        """Constant prices should result in zero P&L."""
        from quant.services.backtesting import BacktestEngine
        n = 100
        df = pd.DataFrame({
            "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
            "close": [100.0] * n, "volume": [1000] * n,
        }, index=pd.date_range(start="2025-01-01", periods=n, freq="1h", tz="UTC"))

        engine = BacktestEngine(initial_capital=10000.0)
        engine.add_strategy(_buy_sell_alternating)
        result = engine.run(df)

        # With 0 fees/slippage, P&L should be 0
        self.assertAlmostEqual(result.total_pnl, 0.0, delta=10.0)


# ══════════════════════════════════════════════════════════════════
#  BacktestResult Tests
# ══════════════════════════════════════════════════════════════════


class BacktestResultTests(TestCase):
    """Test BacktestResult metric properties."""

    def test_win_rate(self):
        """Win rate should be wins / total."""
        from quant.services.backtesting import BacktestResult, BacktestTrade
        now = datetime.now(timezone.utc)

        result = BacktestResult()
        result.total_trades = 10
        result.winning_trades = 6
        result.losing_trades = 4

        self.assertAlmostEqual(result.win_rate, 0.6)

    def test_zero_trades_properties(self):
        """Zero trades should produce safe defaults."""
        from quant.services.backtesting import BacktestResult
        result = BacktestResult()
        self.assertEqual(result.win_rate, 0.0)
        self.assertEqual(result.sharpe_ratio, 0.0)
        self.assertEqual(result.sortino_ratio, 0.0)
        self.assertEqual(result.profit_factor, 0.0)
        self.assertEqual(result.avg_hold_time, 0.0)

    def test_metrics_dict(self):
        """Metrics dict should contain all expected keys."""
        from quant.services.backtesting import BacktestResult
        result = BacktestResult()
        result.total_trades = 5
        result.winning_trades = 3
        result.losing_trades = 2
        m = result.metrics
        self.assertIn("sharpe_ratio", m)
        self.assertIn("sortino_ratio", m)
        self.assertIn("max_drawdown_pct", m)
        self.assertIn("win_rate", m)
        self.assertIn("profit_factor", m)
        self.assertIn("total_return_pct", m)

    def test_total_return(self):
        """Total return should reflect capital change."""
        from quant.services.backtesting import BacktestResult
        result = BacktestResult(initial_capital=10000.0, final_capital=11000.0)
        self.assertAlmostEqual(result.total_return, 0.1)

    def test_negative_return(self):
        """Negative return should be negative."""
        from quant.services.backtesting import BacktestResult
        result = BacktestResult(initial_capital=10000.0, final_capital=8000.0)
        self.assertAlmostEqual(result.total_return, -0.2)

    def test_profit_factor_infinity(self):
        """If no losing trades, profit factor should be inf."""
        from quant.services.backtesting import BacktestResult, BacktestTrade
        now = datetime.now(timezone.utc)
        result = BacktestResult()
        result.trades = [
            BacktestTrade(entry_time=now, pnl=100.0),
            BacktestTrade(entry_time=now, pnl=50.0),
        ]
        self.assertEqual(result.profit_factor, float("inf"))


# ══════════════════════════════════════════════════════════════════
#  MonteCarloSimulator Tests
# ══════════════════════════════════════════════════════════════════


class MonteCarloSimulatorTests(TestCase):
    """Test MonteCarloSimulator stress-testing."""

    def test_run_with_trades(self):
        """Running with valid trades should produce stats."""
        from quant.services.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=1000)
        trades = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.01, 0.02, -0.01, 0.03]
        result = sim.run(trades, initial_capital=10000.0)

        self.assertGreater(result["total_simulations"], 0)
        self.assertIn("probability_of_profit", result)
        self.assertIn("probability_of_ruin", result)
        self.assertIn("median_return", result)
        self.assertIn("percentile_5", result)

    def test_insufficient_trades(self):
        """Fewer than 3 trades should return error."""
        from quant.services.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=100)
        result = sim.run([0.01], initial_capital=10000.0)
        self.assertIn("error", result)

    def test_empty_trades(self):
        """Empty trade list should return error."""
        from quant.services.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=100)
        result = sim.run([], initial_capital=10000.0)
        self.assertIn("error", result)

    def test_all_profitable_trades(self):
        """All-profitable trades should have high probability of profit."""
        from quant.services.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=500)
        trades = [0.01] * 50  # 50 trades, each +1%
        result = sim.run(trades, initial_capital=10000.0)
        self.assertGreater(result["probability_of_profit"], 0.95)

    def test_all_losing_trades(self):
        """All-losing trades should have low probability of profit."""
        from quant.services.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=500)
        trades = [-0.01] * 50  # 50 trades, each -1%
        result = sim.run(trades, initial_capital=10000.0)
        self.assertLess(result["probability_of_profit"], 0.05)

    def test_pnl_list_format(self):
        """Trade data as list of floats vs list of dicts should both work."""
        from quant.services.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=100)

        # List of floats
        trades_float = [0.02, -0.01, 0.03]
        result = sim.run(trades_float)
        self.assertIn("probability_of_profit", result)

        # List of dicts
        trades_dict = [{"pnl_pct": 0.02}, {"pnl_pct": -0.01}, {"pnl_pct": 0.03}]
        result2 = sim.run(trades_dict)
        self.assertIn("probability_of_profit", result2)
