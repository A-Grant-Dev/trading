"""
Binance Algo Trading Integration — TWAP / VWAP / Iceberg Orders.

Leverages Binance's built-in algo trading products for sophisticated execution:

  1. TWAP — Time-weighted average price (accumulate/distribute evenly)
  2. VWAP — Volume-weighted average price (follow market volume)
  3. Iceberg — Hide true order size from the order book

API docs: Binance Algo Trading
  POST /sapi/v1/algo/spot/newOrderTwap
  POST /sapi/v1/algo/spot/newOrderVwap
  POST /sapi/v1/algo/spot/newOrderIceberg
  DELETE /sapi/v1/algo/spot/order
  GET  /sapi/v1/algo/spot/openOrders
  GET  /sapi/v1/algo/spot/order
  GET  /sapi/v1/algo/spot/historicalOrders

Renaissance principle: Large orders must be executed without moving the market.
TWAP/VWAP/Iceberg are the tools for this.
"""

import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone

import requests

from quant.services.order_manager import _signed_request

logger = logging.getLogger(__name__)

# ── Algo API Endpoints ─────────────────────────────────────────────

ALGO_SPOT_BASE = "/sapi/v1/algo/spot"

ALGO_ENDPOINTS = {
    "twap": f"{ALGO_SPOT_BASE}/newOrderTwap",
    "vwap": f"{ALGO_SPOT_BASE}/newOrderVwap",
    "iceberg": f"{ALGO_SPOT_BASE}/newOrderIceberg",
    "cancel": f"{ALGO_SPOT_BASE}/order",
    "open_orders": f"{ALGO_SPOT_BASE}/openOrders",
    "order_status": f"{ALGO_SPOT_BASE}/order",
    "historical": f"{ALGO_SPOT_BASE}/historicalOrders",
}

# ── Default Parameters ─────────────────────────────────────────────

DEFAULT_TWAP_DURATION_MINUTES = 60
DEFAULT_ICEBERG_DISPLAY_QTY = 0.1  # Show 10% at a time
MAX_ALGO_DURATION_HOURS = 24


class AlgoExecutionService:
    """
    Binance Algo Trading — sophisticated order execution.

    Usage:
        algo = AlgoExecutionService()
        result = algo.execute_twap("BTCUSDT", "BUY", 1.5, duration_minutes=120)
        if "error" not in result:
            print(f"TWAP order placed: {result['algo_id']}")
    """

    def __init__(self, use_testnet: bool = True):
        """
        Args:
            use_testnet: If True, algo orders are NOT actually sent
                         (Binance testnet doesn't support Algo API).
                         Algo calls in testnet mode are logged and simulated.
        """
        self.use_testnet = use_testnet

    def execute_twap(self, symbol: str, side: str, quantity: float,
                     duration_minutes: int = DEFAULT_TWAP_DURATION_MINUTES,
                     limit_price: float | None = None) -> dict:
        """
        Place a TWAP (Time-Weighted Average Price) order.

        Splits a large order into equal time slices over the duration.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            side: 'BUY' or 'SELL'
            quantity: Total quantity to execute
            duration_minutes: Total execution time in minutes
            limit_price: Optional max/min price cap

        Returns:
            Dict with algo_id, status, or error
        """
        if self.use_testnet:
            logger.info(
                "[TESTNET] TWAP %s %s %s over %d min — simulated",
                side, quantity, symbol, duration_minutes,
            )
            return self._simulate_algo("TWAP", symbol, side, quantity)

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": str(quantity),
            "duration": int(duration_minutes * 60_000),  # Convert to milliseconds
        }

        if limit_price is not None:
            params["limitPrice"] = str(limit_price)

        return self._post_algo("twap", params)

    def execute_vwap(self, symbol: str, side: str,
                     notional: float | None = None,
                     quantity: float | None = None) -> dict:
        """
        Place a VWAP (Volume-Weighted Average Price) order.

        Trades more during high-volume periods, less during low-volume,
        naturally blending into market activity.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            side: 'BUY' or 'SELL'
            notional: Total notional value (in quote asset)
            quantity: Total quantity (in base asset)
                     Provide EITHER notional or quantity, not both.

        Returns:
            Dict with algo_id, status, or error
        """
        if not notional and not quantity:
            return {"error": "Provide either notional or quantity"}

        if self.use_testnet:
            logger.info(
                "[TESTNET] VWAP %s %s %s — simulated",
                side, notional or quantity, symbol,
            )
            return self._simulate_algo("VWAP", symbol, side, notional or quantity)

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
        }

        if notional:
            params["totalValue"] = str(notional)
        if quantity:
            params["quantity"] = str(quantity)

        return self._post_algo("vwap", params)

    def execute_iceberg(self, symbol: str, side: str,
                        total_quantity: float,
                        display_quantity: float | None = None) -> dict:
        """
        Place an Iceberg order — hides true order size.

        Only a small portion ('display_quantity') shows on the order book.
        As each chunk fills, another replaces it.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            side: 'BUY' or 'SELL'
            total_quantity: Total quantity to execute
            display_quantity: Quantity to show at any one time
                              (default: 10% of total)

        Returns:
            Dict with algo_id, status, or error
        """
        display = display_quantity or (total_quantity * DEFAULT_ICEBERG_DISPLAY_QTY)

        if self.use_testnet:
            logger.info(
                "[TESTNET] Iceberg %s %s %s (show %s) — simulated",
                side, total_quantity, symbol, display,
            )
            return self._simulate_algo("ICEBERG", symbol, side, total_quantity)

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": str(total_quantity),
            "displayQuantity": str(display),
        }

        return self._post_algo("iceberg", params)

    def cancel_algo_order(self, algo_id: str) -> bool:
        """Cancel an active algo order."""
        params = {"algoId": algo_id}
        result = _signed_request("DELETE", ALGO_ENDPOINTS["cancel"], params)
        if "error" not in result:
            logger.info(f"Algo order {algo_id} cancelled")
            return True
        logger.error(f"Failed to cancel algo {algo_id}: {result}")
        return False

    def get_open_algos(self) -> list:
        """List all active algo orders."""
        result = _signed_request("GET", ALGO_ENDPOINTS["open_orders"])
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return []

    def get_algo_status(self, algo_id: str) -> dict:
        """Query a specific algo order status."""
        params = {"algoId": algo_id}
        return _signed_request("GET", ALGO_ENDPOINTS["order_status"], params)

    def get_historical_algos(self, limit: int = 20) -> list:
        """View past algo orders."""
        params = {"limit": limit}
        result = _signed_request("GET", ALGO_ENDPOINTS["historical"], params)
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return []

    # ── Internal Methods ───────────────────────────────────────────

    def _post_algo(self, algo_type: str, params: dict) -> dict:
        """Send an algo order to Binance API."""
        endpoint = ALGO_ENDPOINTS.get(algo_type)
        if not endpoint:
            return {"error": f"Unknown algo type: {algo_type}"}

        result = _signed_request("POST", endpoint, params)
        if "error" in result:
            logger.error(
                "Algo %s failed for %s: %s",
                algo_type.upper(), params.get("symbol", "?"), result["error"],
            )
        else:
            logger.info(
                "Algo %s placed: %s %s — algoId=%s",
                algo_type.upper(), params.get("side", "?"),
                params.get("symbol", "?"), result.get("algoId", "?"),
            )
        return result

    def _simulate_algo(self, algo_type: str, symbol: str,
                       side: str, amount: float) -> dict:
        """Simulate an algo order (testnet mode)."""
        from datetime import datetime, timezone

        algo_id = f"sim_{algo_type.lower()}_{int(time.time())}_{symbol}"
        return {
            "status": "simulated",
            "algo_id": algo_id,
            "algo_type": algo_type,
            "symbol": symbol.upper(),
            "side": side.upper(),
            "amount": amount,
            "simulated": True,
            "note": f"Binance testnet doesn't support Algo API. "
                    f"Order {algo_id} simulated locally.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
