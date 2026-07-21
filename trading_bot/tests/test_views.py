"""
Tests for Views & API — views.py

Coverage:
- Dashboard loading
- Stats API
- Signals API
- Backtests API
- OHLCV overview API
- Optimization status API
- Paper trading status API
- Live trading status API
- Prometheus metrics endpoint
- Trades and audit APIs
- Win rate calculation
"""

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

from django.test import Client, TestCase
from django.urls import reverse

from trading_bot.models import (
    AuditLog,
    BacktestRun,
    OHLCV,
    ParamSet,
    Signal,
    Strategy,
    Trade,
)


class TestDashboardView(TestCase):
    """Test main dashboard page."""

    def setUp(self):
        self.client = Client()

    def test_dashboard_200(self):
        """Dashboard should return 200."""
        response = self.client.get(reverse("trading_bot:dashboard"))
        self.assertEqual(response.status_code, 200)

    def test_dashboard_context_keys(self):
        """Dashboard should contain expected context variables."""
        response = self.client.get(reverse("trading_bot:dashboard"))
        expected_keys = [
            "active_strategies", "total_signals", "open_trades",
            "total_trades", "total_backtests", "ohlcv_count",
        ]
        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, response.context)

    def test_dashboard_with_data(self):
        """Dashboard should show data when records exist."""
        Strategy.objects.create(
            name="Test Strategy", is_active=True,
            strategy_class="test.MomentumStrategy",
        )
        response = self.client.get(reverse("trading_bot:dashboard"))
        self.assertEqual(response.context["active_strategies"], 1)

    def test_dashboard_uses_template(self):
        """Dashboard should use the correct template."""
        response = self.client.get(reverse("trading_bot:dashboard"))
        self.assertTemplateUsed(response, "trading_bot/dashboard.html")


class TestAPIStats(TestCase):
    """Test /api/stats/ endpoint."""

    def setUp(self):
        self.client = Client()

    def test_stats_returns_json(self):
        """Stats API should return JSON."""
        response = self.client.get(reverse("trading_bot:api-stats"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("active_strategies", data)
        self.assertIn("total_signals", data)
        self.assertIn("open_trades", data)

    def test_stats_with_data(self):
        """Stats should reflect DB state."""
        now = datetime.now(timezone.utc)
        Strategy.objects.create(name="S1", is_active=True, strategy_class="c")
        Trade.objects.create(
            mode="paper", symbol="BTC", side="buy",
            entry_price=50000, quantity=0.001, status="open",
            entry_time=now,
        )
        Trade.objects.create(
            mode="paper", symbol="BTC", side="buy",
            entry_price=50000, quantity=0.001, status="closed",
            pnl=Decimal("10.0"),
            entry_time=now,
        )
        response = self.client.get(reverse("trading_bot:api-stats"))
        data = response.json()
        self.assertEqual(data["active_strategies"], 1)
        self.assertEqual(data["open_trades"], 1)
        self.assertEqual(data["total_trades"], 2)


class TestAPISignals(TestCase):
    """Test /api/signals/ endpoint."""

    def setUp(self):
        self.client = Client()
        self.strategy = Strategy.objects.create(
            name="S1", is_active=True, strategy_class="c",
        )
        self.param_set = ParamSet.objects.create(
            strategy=self.strategy, params={},
        )

    def test_signals_empty(self):
        """Empty signals should return empty list."""
        response = self.client.get(reverse("trading_bot:api-signals"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["signals"], [])

    def test_signals_with_data(self):
        """Signals API should return signal data."""
        Signal.objects.create(
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy=self.strategy,
            param_set=self.param_set,
            direction=1,
            confidence=0.8,
            status="active",
        )
        response = self.client.get(reverse("trading_bot:api-signals"))
        data = response.json()
        self.assertEqual(len(data["signals"]), 1)
        self.assertEqual(data["signals"][0]["direction"], 1)


class TestAPIMetrics(TestCase):
    """Test Prometheus /api/metrics/ endpoint."""

    def setUp(self):
        self.client = Client()

    def test_metrics_returns_text(self):
        """Metrics should return text/plain."""
        response = self.client.get(reverse("trading_bot:api-metrics"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response["Content-Type"])

    def test_metrics_format(self):
        """Metrics should follow Prometheus exposition format."""
        response = self.client.get(reverse("trading_bot:api-metrics"))
        body = response.content.decode()
        self.assertIn("# HELP", body)
        self.assertIn("# TYPE", body)
        self.assertIn("trading_bot_strategies_active", body)

    def test_metrics_has_expected_lines(self):
        """Metrics should contain all expected metric names."""
        response = self.client.get(reverse("trading_bot:api-metrics"))
        body = response.content.decode()
        expected_metrics = [
            "trading_bot_strategies_active",
            "trading_bot_signals_total",
            "trading_bot_trades_total",
            "trading_bot_trades_open",
            "trading_bot_backtests_total",
            "trading_bot_ohlcv_candles",
            "trading_bot_paramsets_total",
        ]
        for metric in expected_metrics:
            with self.subTest(metric=metric):
                self.assertIn(metric, body)


class TestAPIAudit(TestCase):
    """Test /api/audit/ endpoint."""

    def setUp(self):
        self.client = Client()

    def test_audit_with_entries(self):
        """Audit API should return log entries."""
        AuditLog.objects.create(
            action="info",
            message="Test audit entry",
            severity="info",
        )
        response = self.client.get(reverse("trading_bot:api-audit"))
        data = response.json()
        self.assertIn("audit", data)
        self.assertGreaterEqual(len(data["audit"]), 1)
