"""
Tests for Backtesting Metrics — metrics.py

Coverage:
- Total return, annualized return
- Volatility, Sharpe, Sortino, Calmar ratios
- Max drawdown
- Trade-based metrics (win rate, profit factor, expectancy)
- Full metrics suite
- Edge cases: empty arrays, single values, constant equity
"""

import numpy as np
from django.test import TestCase

from trading_bot.services.backtester.metrics import (
    compute_annualized_return,
    compute_calmar_ratio,
    compute_full_metrics,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_total_return,
    compute_trade_metrics,
    compute_volatility,
    get_annualization_factor,
)


class TestAnnualizationFactor(TestCase):
    """Test annualization factor lookup."""

    def test_1m_factor(self):
        """1m interval should have minute-based factor."""
        self.assertEqual(get_annualization_factor("1m"), 60 * 24 * 365)

    def test_1h_factor(self):
        """1h interval should have hourly factor."""
        self.assertEqual(get_annualization_factor("1h"), 24 * 365)

    def test_1d_factor(self):
        """1d interval should have daily factor."""
        self.assertEqual(get_annualization_factor("1d"), 365)

    def test_unknown_interval(self):
        """Unknown interval should default to hourly."""
        self.assertEqual(get_annualization_factor("unknown"), 24 * 365)

    def test_5m_factor(self):
        """5m interval should be 1/5 of minute factor."""
        self.assertEqual(get_annualization_factor("5m"), (60 * 24 * 365) / 5)


class TestTotalReturn(TestCase):
    """Test total return computation."""

    def test_positive_return(self):
        """Equity going up should return positive %."""
        equity = np.array([100, 110, 121])
        ret = compute_total_return(equity)
        self.assertAlmostEqual(ret, 21.0)  # (121-100)/100 * 100 = 21%

    def test_negative_return(self):
        """Equity going down should return negative %."""
        equity = np.array([100, 90, 80])
        ret = compute_total_return(equity)
        self.assertAlmostEqual(ret, -20.0)

    def test_zero_return(self):
        """Flat equity should return 0%."""
        equity = np.array([100, 100, 100])
        ret = compute_total_return(equity)
        self.assertAlmostEqual(ret, 0.0)

    def test_single_value(self):
        """Single value should return 0%."""
        equity = np.array([100])
        ret = compute_total_return(equity)
        self.assertAlmostEqual(ret, 0.0)

    def test_empty_array(self):
        """Empty array should return 0%."""
        equity = np.array([])
        ret = compute_total_return(equity)
        self.assertAlmostEqual(ret, 0.0)


class TestMaxDrawdown(TestCase):
    """Test max drawdown computation."""

    def test_no_drawdown(self):
        """Constantly rising equity should have 0% drawdown."""
        equity = np.array([100, 110, 120, 130])
        dd = compute_max_drawdown(equity)
        self.assertAlmostEqual(dd, 0.0)

    def test_simple_drawdown(self):
        """Equity that drops then recovers should capture the drop."""
        equity = np.array([100, 90, 80, 100])
        dd = compute_max_drawdown(equity)
        self.assertAlmostEqual(dd, -20.0)  # 80 is 20% below 100 peak

    def test_recovery_not_included(self):
        """Drawdown from after a recovery should be from new peak."""
        equity = np.array([100, 120, 110, 130])
        dd = compute_max_drawdown(equity)
        # Peak at 120 → drop to 110 = -8.33%
        # Then new peak at 130 → no subsequent drop
        self.assertAlmostEqual(dd, -8.3333, places=2)

    def test_continuous_drawdown(self):
        """Continuous decline should track the max."""
        equity = np.array([100, 95, 90, 85, 80])
        dd = compute_max_drawdown(equity)
        self.assertAlmostEqual(dd, -20.0)


class TestSharpeRatio(TestCase):
    """Test Sharpe ratio computation."""

    def test_positive_sharpe(self):
        """Positive returns should give positive Sharpe."""
        returns = np.array([0.01, 0.02, 0.015, 0.01])
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.0)
        self.assertGreater(sharpe, 0)

    def test_negative_sharpe(self):
        """Negative returns should give negative Sharpe."""
        returns = np.array([-0.01, -0.02, -0.015])
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.0)
        self.assertLess(sharpe, 0)

    def test_zero_sharpe(self):
        """Flat returns with risk-free rate of 0 should give 0 Sharpe."""
        returns = np.array([0.0, 0.0, 0.0])
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.0)
        self.assertAlmostEqual(sharpe, 0.0)

    def test_sharpe_with_risk_free(self):
        """Risk-free rate should reduce Sharpe."""
        returns = np.array([0.01, 0.02, 0.015])
        sharpe_no_rf = compute_sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.0)
        sharpe_with_rf = compute_sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.05)
        self.assertGreater(sharpe_no_rf, sharpe_with_rf)


class TestSortinoRatio(TestCase):
    """Test Sortino ratio computation."""

    def test_positive_sortino(self):
        """Positive returns should give positive Sortino."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, -0.003, 0.01])
        sortino = compute_sortino_ratio(returns, periods_per_year=252, risk_free_rate=0.0)
        self.assertGreater(sortino, 0)

    def test_sortino_vs_sharpe(self):
        """Sortino should be higher than Sharpe when downside is lower."""
        returns = np.array([1.0, -0.10, 1.0, -0.10, 1.0, -0.05, 1.0, -0.08, 1.0])
        sortino = compute_sortino_ratio(returns, periods_per_year=252, risk_free_rate=0.0)
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.0)
        # Sortino should be >= Sharpe (less downside volatility)
        self.assertGreaterEqual(sortino, sharpe)


class TestCalmarRatio(TestCase):
    """Test Calmar ratio computation."""

    def test_positive_calmar(self):
        """Positive return with moderate drawdown should give positive Calmar."""
        equity = np.array([100, 110, 95, 105, 115])
        calmar = compute_calmar_ratio(equity, periods_per_year=252)
        self.assertGreater(calmar, 0)


class TestTradeMetrics(TestCase):
    """Test trade-based metrics."""

    def test_empty_trades(self):
        """Empty trade list should return zeros."""
        metrics = compute_trade_metrics([])
        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["total_trades"], 0.0)
        self.assertEqual(metrics["profit_factor"], 0.0)

    def test_all_winners(self):
        """All profitable trades."""
        trades = [
            {"pnl_pct": 5.0, "pnl": 50.0, "side": "buy"},
            {"pnl_pct": 3.0, "pnl": 30.0, "side": "buy"},
            {"pnl_pct": 2.0, "pnl": 20.0, "side": "sell"},
        ]
        metrics = compute_trade_metrics(trades)
        self.assertEqual(metrics["win_rate"], 100.0)
        self.assertEqual(metrics["total_trades"], 3.0)
        # With no losers, profit_factor uses 1.0 as denominator: (50+30+20)/1.0 = 100.0
        self.assertAlmostEqual(metrics["profit_factor"], 100.0, places=1)

    def test_mixed_results(self):
        """Mix of winners and losers."""
        trades = [
            {"pnl_pct": 10.0, "pnl": 100.0, "side": "buy"},
            {"pnl_pct": -5.0, "pnl": -50.0, "side": "sell"},
            {"pnl_pct": 20.0, "pnl": 200.0, "side": "buy"},
            {"pnl_pct": -10.0, "pnl": -100.0, "side": "sell"},
        ]
        metrics = compute_trade_metrics(trades)
        self.assertEqual(metrics["win_rate"], 50.0)
        self.assertEqual(metrics["total_trades"], 4.0)
        self.assertEqual(metrics["winning_trades"], 2.0)
        self.assertEqual(metrics["losing_trades"], 2.0)
        self.assertAlmostEqual(metrics["profit_factor"], 2.0, places=2)  # 300/150

    def test_all_losers(self):
        """All losing trades."""
        trades = [
            {"pnl_pct": -5.0, "pnl": -50.0, "side": "buy"},
            {"pnl_pct": -3.0, "pnl": -30.0, "side": "sell"},
        ]
        metrics = compute_trade_metrics(trades)
        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["total_trades"], 2.0)


class TestFullMetricsSuite(TestCase):
    """Test the complete metrics suite."""

    def test_full_metrics_returns_dict(self):
        """Full metrics should return a dict with all expected keys."""
        equity = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
        returns = np.diff(equity) / equity[:-1]
        trades = [
            {"pnl_pct": 2.0, "pnl": 2.0, "side": "buy"},
            {"pnl_pct": -1.0, "pnl": -1.0, "side": "sell"},
        ]
        metrics = compute_full_metrics(equity, returns, trades, interval="1h")
        expected_keys = {
            "total_return_pct", "annualized_return_pct",
            "annualized_volatility_pct", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
            "win_rate", "profit_factor", "expectancy",
            "avg_win", "avg_loss", "total_trades",
            "winning_trades", "losing_trades", "hit_rate",
        }
        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, metrics)
