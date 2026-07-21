"""
Execution Strategy Selector — Chooses the optimal execution method.

Decides whether to use:
  - Regular MARKET/LIMIT order (small orders, liquid pairs)
  - Algo TWAP (medium orders, moderate liquidity)
  - Algo VWAP (best average price over time)
  - Algo Iceberg (large orders, low liquidity)

Renaissance principle: Every trade must be executed with minimal market impact.
The execution method is as important as the signal itself.
"""

import logging

import requests

logger = logging.getLogger(__name__)

BINANCE_API = "https://api.binance.com"

# ── Thresholds ─────────────────────────────────────────────────────

SMALL_ORDER_THRESHOLD_USDT = 1_000      # < $1K → market/limit order
MEDIUM_ORDER_THRESHOLD_USDT = 10_000    # $1K-$10K → TWAP
LARGE_ORDER_THRESHOLD_USDT = 100_000    # $10K-$100K → VWAP
# > $100K → Iceberg

HIGH_LIQUIDITY_PAIRS = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
                        "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
                        "DOTUSDT", "LINKUSDT", "MATICUSDT", "UNIUSDT"}

WIDE_SPREAD_PCT = 0.1  # 0.1% spread is considered wide


def select_execution_strategy(order_info: dict) -> dict:
    """
    Select the optimal execution strategy for a given order.

    Args:
        order_info: Dict with keys:
            symbol: Trading pair (e.g., BTCUSDT)
            side: 'buy' or 'sell'
            notional: Total order value in USDT (or estimated)
            quantity: Quantity in base asset
            price: Estimated entry price
            confidence: Signal confidence (0-1)

    Returns:
        Dict with:
            method: 'market', 'limit', 'twap', 'vwap', 'iceberg'
            params: Parameters for the chosen method
            reason: Why this method was chosen
    """
    symbol = order_info.get("symbol", "").upper()
    side = order_info.get("side", "buy").upper()
    notional = order_info.get("notional", 0)
    quantity = order_info.get("quantity", 0)
    price = order_info.get("price", 0)
    confidence = order_info.get("confidence", 0.5)

    if not notional and quantity and price:
        notional = quantity * price

    # Get market conditions
    spread_pct, depth = _get_market_conditions(symbol)
    is_high_liquidity = symbol in HIGH_LIQUIDITY_PAIRS
    spread_is_wide = spread_pct > WIDE_SPREAD_PCT if spread_pct is not None else False

    # Strategy selection logic
    if notional < SMALL_ORDER_THRESHOLD_USDT and is_high_liquidity:
        # Small order, high liquidity → fast market order
        return {
            "method": "market",
            "params": {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
            "reason": f"Small order (${notional:.0f}) on high-liquidity pair — direct market execution",
        }

    if notional < SMALL_ORDER_THRESHOLD_USDT and spread_is_wide:
        # Small order but wide spread → use limit order to avoid slippage
        limit_price = price * (0.999 if side == "BUY" else 1.001)
        return {
            "method": "limit",
            "params": {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": round(limit_price, 8),
            },
            "reason": f"Small order but spread is {spread_pct:.3f}% — limit order to avoid wide spread",
        }

    if notional < MEDIUM_ORDER_THRESHOLD_USDT:
        # Medium order → TWAP over 30-60 minutes
        duration = max(30, min(60, int(notional / 100)))
        return {
            "method": "twap",
            "params": {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "duration_minutes": duration,
            },
            "reason": f"Medium order (${notional:.0f}) — TWAP over {duration}min to minimize slippage",
        }

    if notional < LARGE_ORDER_THRESHOLD_USDT:
        # Large order → VWAP (volume-aware)
        return {
            "method": "vwap",
            "params": {
                "symbol": symbol,
                "side": side,
                "notional": notional,
            },
            "reason": f"Large order (${notional:.0f}) — VWAP for volume-aware execution",
        }

    # Very large order → Iceberg (hide position size)
    display_qty = quantity * 0.1
    return {
        "method": "iceberg",
        "params": {
            "symbol": symbol,
            "side": side,
            "total_quantity": quantity,
            "display_quantity": display_qty,
        },
        "reason": f"Very large order (${notional:.0f}) — Iceberg to hide position size",
    }


def _get_market_conditions(symbol: str) -> tuple[float | None, dict]:
    """
    Fetch current market conditions for a symbol.

    Returns:
        (spread_pct: float | None, depth_info: dict)
    """
    try:
        # Get order book for spread
        resp = requests.get(
            f"{BINANCE_API}/api/v3/depth",
            params={"symbol": symbol.upper(), "limit": 5},
            timeout=10,
        )
        if resp.status_code != 200:
            return None, {}

        data = resp.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if not bids or not asks:
            return None, {}

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread_pct = ((best_ask - best_bid) / best_bid) * 100

        # Compute shallow depth (sum of top 5 bids/asks)
        bid_depth = sum(float(b[1]) * float(b[0]) for b in bids)
        ask_depth = sum(float(a[1]) * float(a[0]) for a in asks)

        depth_info = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_pct": round(spread_pct, 4),
            "bid_depth_5": round(bid_depth, 2),
            "ask_depth_5": round(ask_depth, 2),
        }

        return spread_pct, depth_info

    except Exception as e:
        logger.debug(f"Failed to get market conditions for {symbol}: {e}")
        return None, {}
