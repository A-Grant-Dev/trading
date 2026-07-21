"""
Integration Tests — End-to-End Workflows

Tests combined workflows across multiple modules:
- Strategy → Signal → Paper Trade execution
- Config → Risk checks → Trade execution
- Feature engine → Strategy → Backtest
- Circuit breaker → External call protection
- Full trade lifecycle (open → monitor → close)
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

import numpy as np
import polars as pl
from django.test import TestCase

from trading_bot.models import (
    AuditLog,
    BotConfig,
    ParamSet,
    Signal,
    Strategy,
    Trade,
)
from trading_bot.services.backtester.metrics import (
    compute_full_metrics,
    compute_sharpe_ratio,
    compute_trade_metrics,
)
from trading_bot.services.circuit_breaker import (
    circuit_breaker,
    get_breaker_state,
    reset_breaker,
)
from trading_bot.services.config import _get_defaults, load_config
from trading_bot.services.executor.position_sizer import calculate_position_size
from trading_bot.services.executor.risk import (
    check_can_open_trade,
    check_kill_switch,
    disarm_kill_switch,
    record_loss_for_circuit_breaker,
)
from trading_bot.services.strategies.technical import MomentumStrategy


class TestStrategyToTradeIntegration(TestCase):
    """Integration: Strategy signals → Trade execution."""

    def setUp(self):
        self.config = BotConfig.get_config()
        self.config.is_enabled = True
        self.config.mode = "paper"
        self.config.max_open_positions = 5
        self.config.virtual_balance = Decimal("10000.00")
        self.config.save()

        self.strategy_model = Strategy.objects.create(
            name="Integration Test Momentum",
            is_active=True,
            strategy_class="trading_bot.services.strategies.technical.MomentumStrategy",
            weight=1.0,
        )
        self.param_set = ParamSet.objects.create(
            strategy=self.strategy_model,
            params=MomentumStrategy.default_params,
            is_candidate=True,
        )

    def test_full_signal_to_trade_flow(self):
        """Complete flow: feature matrix → signal → risk check → trade."""
        # 1. Create a signal
        signal = Signal.objects.create(
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy=self.strategy_model,
            param_set=self.param_set,
            direction=1,
            confidence=0.85,
            status="pending",
            features_used={"ema_fast": "ema_9", "ema_slow": "ema_21"},
        )
        self.assertEqual(signal.status, "pending")

        # 2. Risk check
        entry_price = 50000.0
        qty, position_value, sizing_info = calculate_position_size(
            balance=10000.0,
            entry_price=entry_price,
            side="buy",
            method="percentage",
        )
        allowed, reason = check_can_open_trade(
            symbol=signal.symbol,
            side="buy",
            position_size_value=position_value,
            strategy_name=signal.strategy.name,
        )
        self.assertTrue(allowed, f"Risk check should pass: {reason}")

        # 3. Execute paper trade
        from trading_bot.services.executor.paper import execute_paper_trade
        trade = execute_paper_trade(
            signal=signal,
            current_price=entry_price,
            position_value=position_value,
        )
        self.assertIsNotNone(trade)
        self.assertEqual(trade.mode, "paper")
        self.assertEqual(trade.status, "open")
        self.assertEqual(trade.side, "buy")

        # 4. Verify signal status updated
        signal.refresh_from_db()
        self.assertEqual(signal.status, "filled")

        # 5. Close the trade
        from trading_bot.services.executor.paper import close_paper_trade
        exit_price = entry_price * 1.02  # 2% profit
        closed_trade = close_paper_trade(
            trade=trade,
            current_price=exit_price,
            exit_reason="take_profit",
        )
        self.assertIsNotNone(closed_trade)
        self.assertEqual(closed_trade.status, "closed")
        self.assertIsNotNone(closed_trade.pnl)
        self.assertGreater(float(closed_trade.pnl), 0)  # Should be profitable

        # 6. Verify audit trail
        audit_entries = AuditLog.objects.filter(action="trade_closed").order_by("-timestamp")
        self.assertGreaterEqual(audit_entries.count(), 1)
        self.assertIn("take_profit", audit_entries[0].message)


class TestConfigToRiskIntegration(TestCase):
    """Integration: Config changes → Risk engine behavior."""

    def test_config_disables_trading(self):
        """Disabling in config should block trades."""
        config = BotConfig.get_config()
        config.is_enabled = False
        config.mode = "paper"
        config.save()

        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertFalse(allowed)
        self.assertIn("disabled", reason.lower())

    def test_config_mode_blocks_backtest(self):
        """Backtest mode should block trade execution."""
        config = BotConfig.get_config()
        config.is_enabled = True
        config.mode = "backtest"
        config.save()

        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertFalse(allowed)

    def test_config_allow_paper_trading(self):
        """Setting mode to paper should enable trades."""
        config = BotConfig.get_config()
        config.is_enabled = True
        config.mode = "paper"
        config.save()

        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertTrue(allowed, f"Should be allowed: {reason}")


class TestFullMetricsIntegration(TestCase):
    """Integration: Trade list → Full metrics computation."""

    def test_metrics_from_trades(self):
        """Compute full metrics from a list of simulated trades."""
        equity_curve = np.array([10000, 10100, 10200, 10150, 10300, 10400, 10350, 10500])
        returns = np.diff(equity_curve) / equity_curve[:-1]
        trades = [
            {"pnl_pct": 2.0, "pnl": 200.0, "side": "buy"},
            {"pnl_pct": -1.5, "pnl": -150.0, "side": "sell"},
            {"pnl_pct": 3.0, "pnl": 300.0, "side": "buy"},
            {"pnl_pct": -0.5, "pnl": -50.0, "side": "sell"},
        ]
        metrics = compute_full_metrics(equity_curve, returns, trades, interval="1h")
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("total_return_pct", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        # Win rate: 2 wins / 4 trades = 50%
        self.assertAlmostEqual(metrics["win_rate"], 50.0, places=1)
        # Total return: 10500/10000 - 1 = 5%
        self.assertAlmostEqual(metrics["total_return_pct"], 5.0, places=1)


class TestCircuitBreakerIntegration(TestCase):
    """Integration: Circuit breaker protecting external calls."""

    def test_breaker_protects_failing_endpoint(self):
        """Circuit breaker should isolate failing external endpoints."""
        reset_breaker("integration_test")

        call_count = [0]

        @circuit_breaker(
            "integration_test",
            max_retries=1,  # Single retry only for faster test
            retry_delay=0.01,
            failure_threshold=1,  # Trip after 1 call
            cooldown_seconds=60,
        )
        def failing_api():
            call_count[0] += 1
            raise ConnectionError("API unavailable")

        from trading_bot.services.circuit_breaker import CircuitBreakerOpenError

        # Trip the breaker with first call
        with self.assertRaises(CircuitBreakerOpenError):
            failing_api()

        # Verify breaker is open
        state = get_breaker_state("integration_test")
        self.assertEqual(state["state"].value, "open")

        # Verify function is no longer called (blocked by breaker)
        call_count_before = call_count[0]
        with self.assertRaises(CircuitBreakerOpenError):
            failing_api()
        self.assertEqual(call_count[0], call_count_before)


class TestKillSwitchIntegration(TestCase):
    """Integration: Kill switch should block all trading."""

    def test_kill_switch_prevents_new_trades(self):
        """Activated kill switch should block trade opening."""
        config = BotConfig.get_config()
        config.is_enabled = True
        config.mode = "paper"
        config.save()

        # Activate kill switch
        AuditLog.objects.create(
            action="kill_switch",
            message="Kill switch: flattened all positions",
            details={"reason": "manual"},
            severity="critical",
        )

        # Trading should be blocked
        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertFalse(allowed)
        self.assertIn("kill switch", reason.lower())

        # Disarm
        disarm_kill_switch()

        # Trading should work again
        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertTrue(allowed, f"Should be allowed after disarm: {reason}")
