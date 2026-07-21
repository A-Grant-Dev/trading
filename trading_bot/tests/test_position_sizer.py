"""
Tests for Position Sizer — position_sizer.py

Coverage:
- Percentage-based sizing
- Kelly Criterion sizing (with sufficient trade history)
- Kelly fallback (insufficient trades)
- Auto method selection
- Edge cases: zero balance, zero entry price
"""

from decimal import Decimal
from unittest.mock import patch

from django.test import TestCase

from trading_bot.models import BotConfig, Trade
from trading_bot.services.executor.position_sizer import (
    _kelly_position_size,
    calculate_position_size,
)


class TestCalculatePositionSize(TestCase):
    """Test position size calculation."""

    def test_percentage_sizing(self):
        """Percentage sizing should use fixed % of balance."""
        qty, value, info = calculate_position_size(
            balance=10000.0,
            entry_price=50000.0,
            side="buy",
            method="percentage",
            fixed_pct=10.0,
        )
        # 10% of 10000 = 1000 USDT value
        self.assertAlmostEqual(value, 1000.0, places=2)
        # 1000 / 50000 = 0.02 BTC
        self.assertAlmostEqual(qty, 0.02, places=6)
        self.assertEqual(info["method"], "percentage")

    def test_zero_balance(self):
        """Zero balance should result in zero position."""
        qty, value, info = calculate_position_size(
            balance=0.0,
            entry_price=50000.0,
            method="percentage",
            fixed_pct=10.0,
        )
        self.assertEqual(qty, 0.0)
        self.assertEqual(value, 0.0)

    def test_zero_price(self):
        """Zero price should result in zero quantity."""
        qty, value, info = calculate_position_size(
            balance=10000.0,
            entry_price=0.0,
            method="percentage",
            fixed_pct=10.0,
        )
        self.assertEqual(qty, 0.0)
        self.assertEqual(value, 1000.0)

    def test_small_balance(self):
        """Small balance should work correctly."""
        qty, value, info = calculate_position_size(
            balance=50.0,
            entry_price=50000.0,
            method="percentage",
            fixed_pct=5.0,
        )
        # 5% of 50 = 2.5 USD
        self.assertAlmostEqual(value, 2.5, places=2)
        self.assertAlmostEqual(qty, 0.00005, places=6)

    def test_auto_method_with_few_trades(self):
        """Auto method with < 20 trades should use percentage."""
        qty, value, info = calculate_position_size(
            balance=10000.0,
            entry_price=50000.0,
            side="buy",
            method="auto",
        )
        self.assertEqual(info["method"], "percentage")


class TestKellyPositionSize(TestCase):
    """Test Kelly Criterion position sizing."""

    def test_insufficient_trades_fallback(self):
        """With < 20 trades, Kelly should fall back to percentage."""
        value, info = _kelly_position_size(
            balance=10000.0,
            kelly_fraction=0.25,
            min_trades=20,
        )
        self.assertIn("kelly_note", info)
        self.assertIsNone(info["kelly_pct"])

    def test_kelly_with_winners(self):
        """With winning trades, Kelly should give positive pct."""
        # Create some winning trades
        config = BotConfig.get_config()
        config.mode = "paper"

        from datetime import datetime, timezone
        for i in range(25):
            Trade.objects.create(
                mode="paper",
                symbol="TESTUSDT",
                side="buy" if i % 2 == 0 else "sell",
                entry_price=100.0,
                quantity=1.0,
                status="closed",
                pnl=Decimal("10.0") if i < 15 else Decimal("-5.0"),
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
            )

        value, info = _kelly_position_size(
            balance=10000.0,
            kelly_fraction=0.25,
            min_trades=5,
        )
        self.assertIsNotNone(info.get("kelly_pct"))
        self.assertGreater(value, 0)
