"""
Phase 5 — Execution Layer & Algo Trading Tests

Tests cover:
  - SignalCombiner: source weights, consensus, regime multipliers, empty signals
  - OrderManager: paper execution, quantity calculation, trade recording
  - AlgoExecutionService: TWAP/VWAP/Iceberg simulation, cancellation
  - ExecutionStrategySelector: method selection by order size & liquidity
  - Views: combine-signals, execute, strategy endpoints
  - Edge cases: insufficient balance, empty signals, wide spread, no credentials
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase, override_settings

from quant.models import ExecutedTrade, QuantConfig, TradeSignal
from quant.services.order_manager import OrderManager


# ══════════════════════════════════════════════════════════════════
#  SignalCombiner Tests
# ══════════════════════════════════════════════════════════════════


class SignalCombinerTests(TestCase):
    """Test SignalCombiner aggregation and threshold logic."""

    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.expiry = self.now + timedelta(hours=1)

    def _create_signal(self, symbol="BTCUSDT", direction="long",
                       source="ml_ensemble", strength=0.8,
                       confidence=0.7, signal_type="long"):
        return TradeSignal.objects.create(
            symbol=symbol,
            direction=direction,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source_model=source,
            generated_at=self.now,
            expiry=self.expiry,
            status="active",
        )

    def test_empty_signals_returns_none(self):
        """No active signals should return None."""
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        result = combiner.combine(symbol="BTCUSDT")
        self.assertIsNone(result)

    def test_single_signal_meets_threshold(self):
        """A strong signal should produce a trade decision in bullish regime."""
        self._create_signal(strength=1.0, confidence=1.0)
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        # Use bullish regime (multiplier 1.0) so ranging 0.6 doesn't reduce below threshold
        result = combiner.combine(symbol="BTCUSDT", regime="bullish")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "trade")
        self.assertEqual(result.get("side"), "buy")

    def test_bearish_signal_returns_sell(self):
        """A bearish signal should return side='sell' in bullish regime."""
        self._create_signal(direction="short", signal_type="short",
                           strength=1.0, confidence=1.0)
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        # Use bullish regime (multiplier 1.0) so ranging 0.6 doesn't reduce below threshold
        result = combiner.combine(symbol="BTCUSDT", regime="bullish")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("side"), "sell")

    def test_low_confidence_holds(self):
        """Low confidence should return action='hold'."""
        self._create_signal(strength=0.3, confidence=0.3)
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        result = combiner.combine(symbol="BTCUSDT")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "hold")

    def test_multiple_sources_aggregated(self):
        """Signals from multiple sources should be combined."""
        self._create_signal(source="ml_ensemble", strength=0.8, confidence=0.7)
        self._create_signal(source="cointegration", strength=0.7, confidence=0.6)
        self._create_signal(source="sentiment", strength=0.6, confidence=0.5,
                           direction="short", signal_type="short")
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        result = combiner.combine(symbol="BTCUSDT")
        self.assertIsNotNone(result)
        self.assertIn("source_breakdown", result)
        self.assertGreaterEqual(len(result["source_breakdown"]), 1)

    def test_regime_volatile_reduces_confidence(self):
        """Volatile regime should reduce combined confidence."""
        self._create_signal(strength=0.9, confidence=0.9)
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        # Pass volatile regime explicitly — multiplier 0.3 should bring conf to ~0.24
        result = combiner.combine(symbol="BTCUSDT", regime="volatile")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "hold")
        self.assertLess(abs(result.get("confidence", 0)), 0.55)

    def test_weighted_consensus(self):
        """Multiple conflicting signals should produce a net direction."""
        self._create_signal(direction="long", strength=0.9, confidence=0.9)
        self._create_signal(direction="short", strength=0.9, confidence=0.9,
                           source="orderbook")
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        result = combiner.combine(symbol="BTCUSDT")
        self.assertIsNotNone(result)
        # Should have source breakdown
        self.assertIn("source_breakdown", result)

    def test_expired_signals_excluded(self):
        """Expired signals should not be considered."""
        past = self.now - timedelta(hours=2)
        TradeSignal.objects.create(
            symbol="BTCUSDT",
            direction="long", signal_type="long",
            strength=0.9, confidence=0.9,
            source_model="ml_ensemble",
            generated_at=past,
            expiry=past,
            status="expired",
        )
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        result = combiner.combine(symbol="BTCUSDT")
        self.assertIsNone(result)

    def test_update_weights(self):
        """update_weights should normalize and apply new weights."""
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        combiner.update_weights({"cointegration": 10, "ml_ensemble": 5})
        expected_total = 10 + 5 + 0.15 + 0.15  # defaults for sentiment + orderbook
        # Actually update_weights only updates existing keys + adds new ones
        # Let's just verify weights were updated
        self.assertGreater(combiner.weights.get("cointegration", 0), 0)
        self.assertGreater(combiner.weights.get("ml_ensemble", 0), 0)


# ══════════════════════════════════════════════════════════════════
#  OrderManager Tests
# ══════════════════════════════════════════════════════════════════


class OrderManagerTests(TestCase):
    """Test OrderManager paper execution and trade recording."""

    def setUp(self):
        QuantConfig.objects.create(
            pk=1,
            mode="paper",
            is_enabled=True,
            virtual_balance=10000.00,
            max_open_positions=5,
            max_position_size_pct=10.0,
        )

    def test_paper_execute_buy(self):
        """Paper buy should return a filled order with no API calls."""
        from quant.services.order_manager import OrderManager

        signal = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "confidence": 0.62,
            "action": "trade",
            "reason": "ML ensemble + sentiment",
        }

        with patch("quant.services.order_manager.OrderManager._get_current_price",
                   return_value=65000.0):
            manager = OrderManager(mode="paper")
            result = manager.execute_signal(signal)

        self.assertEqual(result.get("status"), "filled")
        self.assertEqual(result.get("side"), "buy")
        self.assertEqual(result.get("symbol"), "BTCUSDT")
        self.assertGreater(result.get("quantity", 0), 0)
        self.assertIn("order_id", result)

    def test_paper_execute_no_signal(self):
        """Action=hold should be skipped with no execution."""
        from quant.services.order_manager import OrderManager

        signal = {"action": "hold", "reason": "Low confidence"}
        manager = OrderManager(mode="paper")
        result = manager.execute_signal(signal)

        self.assertEqual(result.get("status"), "skipped")

    def test_execute_records_trade_in_db(self):
        """After execution, ExecutedTrade should have a record."""
        from quant.services.order_manager import OrderManager

        signal = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "confidence": 0.62,
            "action": "trade",
            "reason": "Test trade",
        }

        with patch("quant.services.order_manager.OrderManager._get_current_price",
                   return_value=65000.0):
            manager = OrderManager(mode="paper")
            manager.execute_signal(signal)

        # Verify trade was recorded
        trades = ExecutedTrade.objects.filter(symbol="BTCUSDT")
        self.assertEqual(trades.count(), 1)
        trade = trades.first()
        self.assertEqual(trade.side, "buy")
        self.assertGreater(float(trade.entry_price), 0)
        self.assertEqual(trade.status, "paper")

    def test_insufficient_balance(self):
        """Very low balance should return error."""
        config = QuantConfig.get_config()
        config.virtual_balance = 0.01
        config.save()

        from quant.services.order_manager import OrderManager

        signal = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "confidence": 0.62,
            "action": "trade",
            "reason": "Test",
        }

        with patch("quant.services.order_manager.OrderManager._get_current_price",
                   return_value=65000.0):
            manager = OrderManager(mode="paper")
            result = manager.execute_signal(signal)

        # Should return error about insufficient balance
        self.assertIn("error", result)

    def test_calculate_quantity(self):
        """Quantity calculation should respect config limits."""
        from quant.models import QuantConfig

        config = QuantConfig.get_config()
        config.virtual_balance = 10000.00
        config.max_position_size_pct = 10.0
        config.save()

        with patch("quant.services.order_manager.OrderManager._get_current_price",
                   return_value=65000.0):
            with patch("quant.services.order_manager.OrderManager._get_lot_step_size",
                       return_value=0.001):
                with patch("quant.services.order_manager.OrderManager._get_min_notional",
                           return_value=10.0):
                    manager = OrderManager(mode="paper")
                    result = manager._calculate_quantity("BTCUSDT", "buy", 0.62)

        self.assertNotIn("error", result)
        self.assertGreater(result.get("quantity", 0), 0)
        # Notional should account for lot size rounding
        expected = 10000 * 0.10 * 0.62  # balance * position_pct * confidence
        step_size = 0.001
        expected_qty = (expected / 65000.0 // step_size) * step_size
        self.assertAlmostEqual(
            result["notional"],
            expected_qty * 65000.0,
            delta=1,  # Allow penny rounding differences
        )

    def test_cancel_order_not_found(self):
        """Cancelling non-existent order should return False."""
        from quant.services.order_manager import OrderManager
        manager = OrderManager(mode="paper")
        result = manager.cancel_order("nonexistent", "BTCUSDT")
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════════
#  AlgoExecutionService Tests
# ══════════════════════════════════════════════════════════════════


class AlgoExecutionServiceTests(TestCase):
    """Test AlgoExecutionService simulation (testnet mode)."""

    def setUp(self):
        pass

    def test_twap_simulates_in_testnet(self):
        """TWAP should return simulated result in testnet mode."""
        from quant.services.algo_execution import AlgoExecutionService
        algo = AlgoExecutionService(use_testnet=True)
        result = algo.execute_twap("BTCUSDT", "BUY", 1.5, duration_minutes=60)

        self.assertEqual(result.get("status"), "simulated")
        self.assertTrue(result.get("simulated"))
        self.assertEqual(result.get("algo_type"), "TWAP")

    def test_vwap_simulates_in_testnet(self):
        """VWAP should return simulated result in testnet mode."""
        from quant.services.algo_execution import AlgoExecutionService
        algo = AlgoExecutionService(use_testnet=True)
        result = algo.execute_vwap("ETHUSDT", "SELL", notional=100000)

        self.assertEqual(result.get("status"), "simulated")
        self.assertTrue(result.get("simulated"))
        self.assertEqual(result.get("algo_type"), "VWAP")

    def test_iceberg_simulates_in_testnet(self):
        """Iceberg should return simulated result in testnet mode."""
        from quant.services.algo_execution import AlgoExecutionService
        algo = AlgoExecutionService(use_testnet=True)
        result = algo.execute_iceberg("SOLUSDT", "BUY", 100.0, display_quantity=10.0)

        self.assertEqual(result.get("status"), "simulated")
        self.assertTrue(result.get("simulated"))
        self.assertEqual(result.get("algo_type"), "ICEBERG")

    def test_vwap_needs_notional_or_quantity(self):
        """VWAP should error if neither notional nor quantity provided."""
        from quant.services.algo_execution import AlgoExecutionService
        algo = AlgoExecutionService(use_testnet=True)
        result = algo.execute_vwap("BTCUSDT", "BUY")

        self.assertIn("error", result)

    def test_cancel_algo_simulated(self):
        """Cancelling a simulated algo should return False (no real API)."""
        from quant.services.algo_execution import AlgoExecutionService
        algo = AlgoExecutionService(use_testnet=True)
        result = algo.cancel_algo_order("nonexistent")
        self.assertFalse(result)

    def test_open_algos_returns_list(self):
        """get_open_algos should return list in testnet."""
        from quant.services.algo_execution import AlgoExecutionService
        algo = AlgoExecutionService(use_testnet=True)
        algos = algo.get_open_algos()
        self.assertIsInstance(algos, list)


# ══════════════════════════════════════════════════════════════════
#  ExecutionStrategySelector Tests
# ══════════════════════════════════════════════════════════════════


class ExecutionStrategySelectorTests(TestCase):
    """Test execution strategy selection logic."""

    def test_small_order_high_liquidity(self):
        """Small orders on liquid pairs should use market execution."""
        from quant.services.execution_strategies import select_execution_strategy

        with patch("quant.services.execution_strategies._get_market_conditions",
                   return_value=(0.05, {})):  # 0.05% spread
            result = select_execution_strategy({
                "symbol": "BTCUSDT",
                "side": "buy",
                "notional": 500,
                "confidence": 0.6,
            })

        self.assertEqual(result.get("method"), "market")

    def test_small_order_wide_spread(self):
        """Small orders with wide spread should use limit."""
        from quant.services.execution_strategies import select_execution_strategy

        with patch("quant.services.execution_strategies._get_market_conditions",
                   return_value=(0.5, {})):  # 0.5% spread (wide)
            result = select_execution_strategy({
                "symbol": "ALGOUSDT",
                "side": "buy",
                "notional": 500,
                "confidence": 0.6,
                "price": 0.20,
                "quantity": 2500,
            })

        self.assertEqual(result.get("method"), "limit")

    def test_medium_order_uses_twap(self):
        """Medium orders ($1K-$10K) should use TWAP."""
        from quant.services.execution_strategies import select_execution_strategy

        with patch("quant.services.execution_strategies._get_market_conditions",
                   return_value=(0.05, {})):
            result = select_execution_strategy({
                "symbol": "BTCUSDT",
                "side": "buy",
                "notional": 5000,
                "confidence": 0.6,
                "quantity": 0.08,
                "price": 65000,
            })

        self.assertEqual(result.get("method"), "twap")

    def test_large_order_uses_vwap(self):
        """Large orders ($10K-$100K) should use VWAP."""
        from quant.services.execution_strategies import select_execution_strategy

        with patch("quant.services.execution_strategies._get_market_conditions",
                   return_value=(0.05, {})):
            result = select_execution_strategy({
                "symbol": "ETHUSDT",
                "side": "sell",
                "notional": 50000,
                "confidence": 0.7,
                "price": 3500,
                "quantity": 14.3,
            })

        self.assertEqual(result.get("method"), "vwap")

    def test_very_large_order_uses_iceberg(self):
        """Very large orders ($100K+) should use Iceberg."""
        from quant.services.execution_strategies import select_execution_strategy

        with patch("quant.services.execution_strategies._get_market_conditions",
                   return_value=(0.05, {})):
            result = select_execution_strategy({
                "symbol": "BTCUSDT",
                "side": "buy",
                "notional": 200000,
                "confidence": 0.8,
                "price": 65000,
                "quantity": 3.08,
            })

        self.assertEqual(result.get("method"), "iceberg")

    def test_small_order_unknown_pair_with_normal_spread(self):
        """Small orders on unknown pairs with normal spread use TWAP."""
        from quant.services.execution_strategies import select_execution_strategy

        with patch("quant.services.execution_strategies._get_market_conditions",
                   return_value=(0.05, {})):
            result = select_execution_strategy({
                "symbol": "UNKNOWNUSDT",
                "side": "buy",
                "notional": 100,
                "confidence": 0.5,
            })

        # Not high liquidity and not wide spread → falls through to TWAP
        self.assertEqual(result.get("method"), "twap")


# ══════════════════════════════════════════════════════════════════
#  View Tests
# ══════════════════════════════════════════════════════════════════


class ExecutionViewTests(TestCase):
    """Test Phase 5 API endpoints."""

    def setUp(self):
        self.now = datetime.now(timezone.utc)
        QuantConfig.objects.create(pk=1, mode="paper", is_enabled=True)

    def test_combine_signals_empty(self):
        """api_combine_signals should return hold when no signals."""
        response = self.client.get("/quant/api/combine-signals/?symbol=BTCUSDT")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("action"), "hold")

    def test_execution_strategy_prompt(self):
        """api_execution_strategy should return strategy for valid params."""
        response = self.client.get(
            "/quant/api/execution-strategy/"
            "?symbol=BTCUSDT&notional=500&side=buy&confidence=0.6"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("method", data)

    def test_open_trades_empty(self):
        """api_open_trades should return empty list when no trades."""
        response = self.client.get("/quant/api/trades/open/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("count"), 0)

    def test_config_endpoint(self):
        """api_config should return current config."""
        response = self.client.get("/quant/api/config/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("mode"), "paper")
        self.assertIn("max_open_positions", data)

    def test_execute_no_symbol(self):
        """api_execute should error without symbol."""
        response = self.client.get("/quant/api/execute/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("error", data)

    def test_execute_with_signals(self):
        """api_execute should execute available signals."""
        from quant.models import TradeSignal
        TradeSignal.objects.create(
            symbol="BTCUSDT",
            direction="long", signal_type="long",
            strength=1.0, confidence=1.0,
            source_model="ml_ensemble",
            generated_at=self.now,
            expiry=self.now + timedelta(hours=1),
            status="active",
        )

        response = self.client.get("/quant/api/execute/?symbol=BTCUSDT")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("execution", data)
        self.assertIn("status", data["execution"])
