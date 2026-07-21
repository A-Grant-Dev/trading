"""
Order Manager — Lifecycle management of trading orders.

Responsible for:
  - Creating orders via Binance REST API (or paper mode)
  - Monitoring fill status
  - Handling cancellations
  - Recording trades in ExecutedTrade model
  - Error handling (partial fills, rejections, network issues)

Renaissance principle: The computer decides, the computer executes.
No human in the loop. Orders are created, filled, and tracked automatically.
"""

import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

# ── Binance API Config ─────────────────────────────────────────────

BINANCE_API = "https://api.binance.com"
BINANCE_API_TESTNET = "https://testnet.binance.vision"

# Loaded from settings or env
API_KEY = ""
SECRET_KEY = ""
USE_TESTNET = True  # Default to testnet for safety


def _load_credentials() -> tuple[str, str, bool]:
    """Load Binance API credentials from Django settings."""
    global API_KEY, SECRET_KEY, USE_TESTNET
    if API_KEY:
        return API_KEY, SECRET_KEY, USE_TESTNET

    from django.conf import settings
    API_KEY = getattr(settings, "BINANCE_API_KEY", "")
    SECRET_KEY = getattr(settings, "BINANCE_SECRET_KEY", "")
    USE_TESTNET = getattr(settings, "BINANCE_USE_TESTNET", True)

    if not API_KEY or not SECRET_KEY:
        logger.warning(
            "Binance API credentials not configured. "
            "Set BINANCE_API_KEY and BINANCE_SECRET_KEY in settings. "
            "Falling back to paper trading."
        )

    return API_KEY, SECRET_KEY, USE_TESTNET


def _base_url() -> str:
    """Get the base URL for Binance API (testnet or live)."""
    _, _, use_testnet = _load_credentials()
    return BINANCE_API_TESTNET if use_testnet else BINANCE_API


def _sign_request(params: dict) -> dict:
    """Sign a Binance API request with HMAC-SHA256."""
    api_key, secret_key, _ = _load_credentials()
    if not api_key or not secret_key:
        return params

    query_string = "&".join(
        f"{k}={v}" for k, v in sorted(params.items())
    )
    signature = hmac.new(
        secret_key.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    params["signature"] = signature
    return params


def _signed_request(method: str, path: str,
                     params: dict | None = None) -> dict:
    """Make a signed request to Binance REST API."""
    api_key, secret_key, _ = _load_credentials()
    url = f"{_base_url()}{path}"
    headers = {"X-MBX-APIKEY": api_key} if api_key else {}

    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    params = _sign_request(params)

    try:
        if method.upper() == "GET":
            resp = requests.get(url, params=params, headers=headers, timeout=15)
        else:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            resp = requests.post(url, data=params, headers=headers, timeout=15)

        if resp.status_code == 200:
            return resp.json()
        else:
            logger.error(
                "Binance API error %s: %s — %s",
                resp.status_code, path, resp.text[:500],
            )
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text[:500]}
    except requests.exceptions.Timeout:
        logger.error(f"Binance API timeout: {path}")
        return {"error": "timeout"}
    except Exception as e:
        logger.error(f"Binance API request failed: {e}")
        return {"error": str(e)}


# ── Order Manager ──────────────────────────────────────────────────


class OrderManager:
    """
    Manages the lifecycle of orders from creation to execution.

    Supports three modes:
      - paper: Simulated execution (no real API calls)
      - testnet: Executes on Binance testnet (default)
      - live: Executes on live Binance (use with caution!)

    Usage:
        manager = OrderManager()
        trade = manager.execute_signal({
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'confidence': 0.62,
            'reason': 'ML ensemble + sentiment',
        })
    """

    def __init__(self, mode: str | None = None):
        """
        Args:
            mode: 'paper', 'testnet', or 'live'. Defaults to QuantConfig.
        """
        if mode:
            self.mode = mode
        else:
            try:
                config = QuantConfig.objects.get(pk=1)
                self.mode = config.mode
            except Exception:
                self.mode = "paper"  # Safest default

    def execute_signal(self, signal_order: dict) -> dict:
        """
        Execute a combined trading signal.

        Args:
            signal_order: Dict from SignalCombiner.combine() with keys:
                symbol, side, confidence, reason, action, etc.

        Returns:
            Result dict with:
                trade_id, order_id, symbol, side, qty, price,
                status, error (if any)
        """
        if signal_order.get("action") != "trade":
            return {
                "status": "skipped",
                "reason": signal_order.get("reason", "No trade action"),
            }

        symbol = signal_order.get("symbol", "").upper()
        side = signal_order.get("side", "").lower()
        confidence = signal_order.get("confidence", 0.5)

        if not symbol or side not in ("buy", "sell"):
            return {"status": "error", "error": f"Invalid symbol/side: {symbol}/{side}"}

        # Calculate quantity based on available balance and confidence
        qty_result = self._calculate_quantity(symbol, side, confidence)
        if "error" in qty_result:
            return qty_result

        quantity = qty_result["quantity"]
        reason = signal_order.get("reason", "Combined signal")

        # Execute based on mode
        if self.mode == "paper":
            result = self._paper_execute(symbol, side, quantity, confidence, reason)
        elif self.mode == "live":
            result = self._live_execute(symbol, side, quantity, confidence, reason)
        else:  # testnet (default)
            result = self._live_execute(symbol, side, quantity, confidence, reason)

        # Record trade in ExecutedTrade model
        if "error" not in result:
            self._record_trade(result, signal_order)
            logger.info(
                "TRADE EXECUTED (%s): %s %s %s @ %s — conf=%.1f%% — %s",
                self.mode.upper(), side.upper(), quantity, symbol,
                result.get("price", "N/A"), confidence * 100, reason,
            )

        return result

    def _calculate_quantity(self, symbol: str, side: str,
                            confidence: float) -> dict:
        """Calculate order quantity based on available balance."""
        try:
            from quant.models import QuantConfig
            config = QuantConfig.objects.get(pk=1)

            if self.mode == "paper":
                available = float(config.virtual_balance)
            else:
                balance = self._get_asset_balance(
                    "USDT" if side == "buy" else symbol.replace("USDT", "")
                )
                available = float(balance.get("free", 0)) if balance else 0

            if available <= 0:
                return {"error": f"Insufficient {symbol} balance: {available}"}

            # Position size = balance * max_position_size_pct * confidence
            position_pct = config.max_position_size_pct / 100.0
            allocation = available * position_pct * confidence

            # Get current price
            price = self._get_current_price(symbol)
            if not price:
                return {"error": f"Cannot get price for {symbol}"}

            # Convert to base asset quantity
            if side == "buy":
                quantity = allocation / price
            else:
                quantity = allocation  # Already in base asset

            # Apply lot size filter
            step_size = self._get_lot_step_size(symbol)
            if step_size > 0:
                quantity = (quantity // step_size) * step_size

            # Ensure min notional
            min_notional = self._get_min_notional(symbol)
            if quantity * price < min_notional:
                return {
                    "error": f"Order too small: {quantity * price:.2f} USDT "
                             f"< min notional {min_notional} USDT"
                }

            return {
                "quantity": round(quantity, 8),
                "price": price,
                "notional": round(quantity * price, 2),
            }

        except Exception as e:
            logger.exception(f"Quantity calculation failed for {symbol}")
            return {"error": str(e)}

    def _paper_execute(self, symbol: str, side: str, quantity: float,
                       confidence: float, reason: str) -> dict:
        """Simulate trade execution (virtual P&L tracking)."""
        price = self._get_current_price(symbol) or 0
        notional = quantity * price

        # Simulate 0.1% slippage for realism
        slippage = notional * 0.001
        executed_price = price * (1.001 if side == "buy" else 0.999)

        return {
            "status": "filled",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": round(executed_price, 8),
            "notional": round(notional, 2),
            "slippage": round(slippage, 2),
            "order_id": f"paper_{int(time.time())}_{symbol}",
            "reason": reason,
        }

    def _live_execute(self, symbol: str, side: str, quantity: float,
                      confidence: float, reason: str) -> dict:
        """Execute on Binance (testnet or live)."""
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quoteOrderQty": f"{quantity:.2f}" if side == "buy" else None,
            "quantity": f"{quantity:.8f}" if side == "sell" else None,
            "newOrderRespType": "FULL",
        }

        if params["quoteOrderQty"] is None:
            del params["quoteOrderQty"]
        if params["quantity"] is None:
            del params["quantity"]

        result = _signed_request("POST", "/api/v3/order", params)

        if "error" in result:
            return result

        # Parse fill info
        fills = result.get("fills", [])
        avg_price = 0.0
        total_qty = 0.0
        total_cost = 0.0

        for fill in fills:
            f_price = float(fill.get("price", 0))
            f_qty = float(fill.get("qty", 0))
            f_comm = float(fill.get("commission", 0))
            avg_price = (avg_price * total_qty + f_price * f_qty) / (total_qty + f_qty) if (total_qty + f_qty) > 0 else f_price
            total_qty += f_qty
            total_cost += f_price * f_qty

        return {
            "status": "filled" if result.get("status") == "FILLED" else result.get("status", "unknown"),
            "symbol": symbol,
            "side": side,
            "quantity": total_qty or quantity,
            "price": round(avg_price, 8) if avg_price else 0,
            "notional": round(total_cost, 2),
            "order_id": result.get("orderId", f"live_{int(time.time())}"),
            "binance_result": result,
        }

    def _record_trade(self, result: dict, signal_order: dict) -> None:
        """Record an executed trade in the ExecutedTrade model."""
        from quant.models import ExecutedTrade, TradeSignal

        # Find the original signal
        signal = None
        try:
            symbol = result.get("symbol", "")
            potential = TradeSignal.objects.filter(
                symbol=symbol, status="active"
            ).order_by("-generated_at").first()
            if potential:
                signal = potential
                signal.status = "executed"
                signal.save(update_fields=["status"])
        except Exception:
            pass

        now = datetime.now(timezone.utc)
        ExecutedTrade.objects.create(
            signal=signal,
            symbol=result.get("symbol", ""),
            side=result.get("side", ""),
            entry_price=result.get("price", 0),
            qty=result.get("quantity", 0),
            entry_time=now,
            status="open" if self.mode != "paper" else "paper",
            order_id=result.get("order_id", ""),
            strategy=signal_order.get("reason", "combined")[:50],
            notes=f"Confidence: {signal_order.get('confidence', 0):.1%} | "
                  f"Mode: {self.mode} | {signal_order.get('reason', '')}",
        )

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order on Binance."""
        params = {"symbol": symbol.upper(), "orderId": order_id}
        result = _signed_request("DELETE", "/api/v3/order", params)
        if "error" not in result:
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return True

        logger.error(f"Failed to cancel order {order_id}: {result.get('error')}")
        return False

    def get_open_orders(self, symbol: str | None = None) -> list:
        """Get all currently open orders from Binance."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        result = _signed_request("GET", "/api/v3/openOrders", params)
        if isinstance(result, list):
            return result
        return []

    def get_order_status(self, order_id: str, symbol: str) -> dict:
        """Check fill status of a specific order."""
        params = {"symbol": symbol.upper(), "orderId": order_id}
        result = _signed_request("GET", "/api/v3/order", params)
        return result

    # ── Helper Methods ────────────────────────────────────────────

    @staticmethod
    def _get_current_price(symbol: str) -> float | None:
        """Get latest price for a symbol from Binance REST."""
        try:
            resp = requests.get(
                f"{BINANCE_API}/api/v3/ticker/price",
                params={"symbol": symbol.upper()},
                timeout=10,
            )
            if resp.status_code == 200:
                return float(resp.json()["price"])
        except Exception as e:
            logger.debug(f"Failed to get price for {symbol}: {e}")
        return None

    @staticmethod
    def _get_asset_balance(asset: str) -> dict | None:
        """Get balance for a specific asset from Binance account."""
        result = _signed_request("GET", "/api/v3/account")
        if "error" in result:
            return None

        for balance in result.get("balances", []):
            if balance["asset"] == asset.upper():
                return {
                    "asset": balance["asset"],
                    "free": float(balance["free"]),
                    "locked": float(balance["locked"]),
                }
        return {"asset": asset.upper(), "free": 0.0, "locked": 0.0}

    @staticmethod
    def _get_lot_step_size(symbol: str) -> float:
        """Get LOT_SIZE step size from exchange info."""
        try:
            resp = requests.get(
                f"{BINANCE_API}/api/v3/exchangeInfo",
                params={"symbol": symbol.upper()},
                timeout=10,
            )
            if resp.status_code != 200:
                return 0.0

            data = resp.json()
            for sym in data.get("symbols", []):
                if sym["symbol"] == symbol.upper():
                    for filt in sym.get("filters", []):
                        if filt["filterType"] == "LOT_SIZE":
                            return float(filt["stepSize"])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _get_min_notional(symbol: str) -> float:
        """Get MIN_NOTIONAL from exchange info."""
        try:
            resp = requests.get(
                f"{BINANCE_API}/api/v3/exchangeInfo",
                params={"symbol": symbol.upper()},
                timeout=10,
            )
            if resp.status_code != 200:
                return 10.0  # Default minimum

            data = resp.json()
            for sym in data.get("symbols", []):
                if sym["symbol"] == symbol.upper():
                    for filt in sym.get("filters", []):
                        if filt["filterType"] == "MIN_NOTIONAL":
                            return float(filt.get("minNotional", "10"))
                        if filt["filterType"] == "NOTIONAL":
                            return float(filt.get("minNotional", "10"))
        except Exception:
            pass
        return 10.0  # Safe default
