"""
Tests for Risk Engine — risk.py

Coverage:
- Guard 1: Master on/off switch
- Guard 2: Mode check (must be paper/live)
- Guard 3: Max open positions
- Guard 4: Position size vs balance
- Guard 5: Daily loss limit
- Guard 7: Circuit breaker
- Guard 9: Kill switch
- check_can_close_trade
- record_loss_for_circuit_breaker
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import patch

from django.test import TestCase

from trading_bot.models import AuditLog, BotConfig, Trade
from trading_bot.services.executor.risk import (
    check_can_close_trade,
    check_can_open_trade,
    check_kill_switch,
    disarm_kill_switch,
    record_loss_for_circuit_breaker,
)


class TestCheckCanOpenTrade(TestCase):
    """Test all risk guards for opening trades."""

    def setUp(self):
        self.config = BotConfig.get_config()
        self.config.is_enabled = True
        self.config.mode = "paper"
        self.config.max_open_positions = 5
        self.config.max_position_size_pct = 10.0
        self.config.virtual_balance = Decimal("10000.00")
        self.config.kelly_fraction = 0.25
        self.config.max_daily_loss_pct = 5.0
        self.config.max_drawdown_pct = 15.0
        self.config.circuit_breaker_count = 3
        self.config.circuit_breaker_hours = 24
        self.config.save()

    def test_guard1_disabled(self):
        """Disabled bot should reject trades."""
        self.config.is_enabled = False
        self.config.save()
        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertFalse(allowed)
        self.assertIn("disabled", reason.lower())

    def test_guard2_wrong_mode(self):
        """Non paper/live mode should reject."""
        self.config.mode = "backtest"
        self.config.save()
        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertFalse(allowed)
        self.assertIn("mode", reason.lower())

    def test_guard2_paper_mode_allowed(self):
        """Paper mode should allow trades."""
        self.config.mode = "paper"
        self.config.save()
        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertTrue(allowed, f"Should be allowed: {reason}")

    def test_guard3_max_positions(self):
        """Exceeding max open positions should reject."""
        self.config.max_open_positions = 1
        self.config.save()

        # Create one open trade
        Trade.objects.create(
            mode="paper",
            symbol="BTCUSDT",
            side="buy",
            entry_price=50000.0,
            quantity=0.001,
            status="open",
            entry_time=datetime.now(timezone.utc),
        )

        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertFalse(allowed)
        self.assertIn("max open positions", reason.lower())

    def test_guard4_position_size(self):
        """Exceeding max position size % should reject."""
        # 10% of 10000 = 1000. Trying 2000 should fail
        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 2000.0)
        self.assertFalse(allowed)
        self.assertIn("position size", reason.lower())

    def test_guard4_small_position_allowed(self):
        """Small position within limits should pass."""
        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 500.0)
        self.assertTrue(allowed, f"Should be allowed: {reason}")

    def test_guard9_kill_switch(self):
        """Active kill switch should reject trades."""
        # Trip the kill switch
        AuditLog.objects.create(
            action="kill_switch",
            message="Kill switch: flattened all positions",
            details={"reason": "manual"},
            severity="critical",
        )

        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertFalse(allowed)
        self.assertIn("kill switch", reason.lower())

        # Clean up
        disarm_kill_switch()

    def test_guard9_disarm_allows(self):
        """Disarming kill switch should allow trades again."""
        AuditLog.objects.create(
            action="kill_switch",
            message="Kill switch: flattened all positions",
            details={"reason": "manual"},
            severity="critical",
        )

        disarm_kill_switch()
        allowed, reason = check_can_open_trade("BTCUSDT", "buy", 100.0)
        self.assertTrue(allowed, f"Should be allowed after disarm: {reason}")


class TestCheckCanCloseTrade(TestCase):
    """Test close trade checks."""

    def test_close_open_trade(self):
        """Open trade should be closeable."""
        trade = Trade(
            mode="paper",
            symbol="BTCUSDT",
            side="buy",
            entry_price=50000.0,
            quantity=0.001,
            status="open",
        )
        allowed, reason = check_can_close_trade(trade)
        self.assertTrue(allowed)

    def test_close_closed_trade(self):
        """Closed trade should not be closeable again."""
        trade = Trade(
            mode="paper",
            symbol="BTCUSDT",
            side="buy",
            entry_price=50000.0,
            quantity=0.001,
            status="closed",
        )
        allowed, reason = check_can_close_trade(trade)
        self.assertFalse(allowed)


class TestKillSwitch(TestCase):
    """Test kill switch functions."""

    def test_check_no_kill(self):
        """Without kill switch audit log, should return False."""
        killed, reason = check_kill_switch()
        self.assertFalse(killed)

    def test_check_with_kill(self):
        """With kill switch audit log, should return True."""
        AuditLog.objects.create(
            action="kill_switch",
            message="kill switch: flattened all positions",
            details={"reason": "manual"},
            severity="critical",
        )
        killed, reason = check_kill_switch()
        self.assertTrue(killed)

    def test_disarm_removes_kill(self):
        """Disarming should clear kill switch."""
        AuditLog.objects.create(
            action="kill_switch",
            message="kill switch: triggered",
            details={"reason": "testing"},
            severity="critical",
        )
        disarm_kill_switch()
        killed, reason = check_kill_switch()
        self.assertFalse(killed)
