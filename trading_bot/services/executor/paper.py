"""
Paper Trading Engine — Realistic Fill Simulation

Simulates trade execution using configurable pricing models:
- Current market price (from live data or OHLCV)
- Slippage model (fixed % or order-book-based)
- Fee model (maker/taker or flat %)
- Partial fill simulation for large orders

The paper executor mirrors the real execution pipeline so that
promoting from paper → live is seamless.

All paper trades are stored in the Trade model with mode='paper'.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from django.db import transaction

from trading_bot.models import AuditLog, BotConfig, ParamSet, Signal, Strategy, Trade

logger = logging.getLogger(__name__)


def execute_paper_trade(
    signal: Signal,
    current_price: float,
    quantity: Optional[float] = None,
    position_value: Optional[float] = None,
    slippage_pct: Optional[float] = None,
    fee_pct: Optional[float] = None,
) -> Optional[Trade]:
    """
    Execute a paper trade based on a signal and current market price.

    Handles both long (buy) and short (sell) signals.

    Args:
        signal: The Signal model instance to execute
        current_price: Current market price in quote currency
        quantity: Exact quantity to trade (overrides position sizing)
        position_value: Trade value in quote currency for sizing
        slippage_pct: Slippage % override (default: 0.05%)
        fee_pct: Fee % override (default: 0.1%)

    Returns:
        Trade model instance or None if rejected by risk checks
    """
    from trading_bot.services.executor.risk import check_can_open_trade
    from trading_bot.services.executor.position_sizer import calculate_position_size

    config = BotConfig.get_config()

    if slippage_pct is None:
        slippage_pct = 0.0005  # 0.05%
    if fee_pct is None:
        fee_pct = 0.001  # 0.1%

    # Determine side from signal direction
    if signal.direction == 1:
        side = "buy"
    elif signal.direction == -1:
        side = "sell"
    else:
        logger.info("Signal %d is neutral, no trade to execute", signal.id)
        return None

    # Calculate position size
    balance = float(config.virtual_balance)

    if quantity is not None:
        # Use exact quantity
        qty = quantity
        pos_value = qty * current_price
        sizing_info = {"method": "exact", "quantity": qty}
    elif position_value is not None:
        # Use exact position value
        qty = position_value / current_price if current_price > 0 else 0
        pos_value = position_value
        sizing_info = {"method": "fixed_value", "position_value": pos_value}
    else:
        # Auto-calculate
        qty, pos_value, sizing_info = calculate_position_size(
            balance=balance,
            entry_price=current_price,
            side=side,
        )

    if qty <= 0 or pos_value <= 0:
        logger.warning("Signal %d: invalid position size (qty=%s, value=%s)", signal.id, qty, pos_value)
        return None

    # Risk check
    allowed, reason = check_can_open_trade(
        symbol=signal.symbol,
        side=side,
        position_size_value=pos_value,
        strategy_name=signal.strategy.name if signal.strategy else None,
    )

    if not allowed:
        logger.info("Signal %d rejected by risk: %s", signal.id, reason)
        AuditLog.objects.create(
            action="error",
            message=f"Paper trade rejected for {signal.symbol}: {reason}",
            details={"signal_id": signal.id, "side": side, "reason": reason},
            severity="warning",
        )
        return None

    # Apply slippage to entry price
    if side == "buy":
        entry_price = current_price * (1 + slippage_pct)
    else:
        entry_price = current_price * (1 - slippage_pct)

    # Calculate fees
    entry_fee = pos_value * fee_pct

    # Create trade record
    try:
        with transaction.atomic():
            trade = Trade.objects.create(
                mode="paper",
                signal=signal,
                symbol=signal.symbol,
                side=side,
                entry_price=round(Decimal(str(entry_price)), 8),
                quantity=round(Decimal(str(qty)), 8),
                entry_time=datetime.now(timezone.utc),
                status="open",
                entry_fee=round(Decimal(str(entry_fee)), 8),
                slippage=round(Decimal(str(slippage_pct * 100)), 4),
                strategy=signal.strategy,
                param_set=signal.param_set,
                notes=f"Paper trade from signal #{signal.id} ({signal.strategy.name if signal.strategy else '?'})",
            )

            # Update signal status
            signal.status = "filled"
            signal.save(update_fields=["status"])

            AuditLog.objects.create(
                action="trade_opened",
                message=(
                    f"Paper {side.upper()} {signal.symbol}: "
                    f"{qty:.4f} @ ${entry_price:.2f} (${pos_value:.2f})"
                ),
                details={
                    "trade_id": trade.id,
                    "signal_id": signal.id,
                    "symbol": signal.symbol,
                    "side": side,
                    "quantity": qty,
                    "entry_price": entry_price,
                    "position_value": pos_value,
                    "slippage_pct": slippage_pct,
                    "fee": entry_fee,
                    "sizing": sizing_info,
                },
                severity="info",
            )

            logger.info(
                "Paper %s %s: %.4f @ %.2f (value=%.2f)",
                side.upper(), signal.symbol, qty, entry_price, pos_value,
            )

            return trade

    except Exception as e:
        logger.exception("Failed to create paper trade for signal %d: %s", signal.id, e)
        return None


def close_paper_trade(
    trade: Trade,
    current_price: float,
    slippage_pct: Optional[float] = None,
    fee_pct: Optional[float] = None,
    exit_reason: str = "signal",
) -> Optional[Trade]:
    """
    Close an open paper trade at the current market price.

    Args:
        trade: The open Trade model instance
        current_price: Current market price
        slippage_pct: Slippage override
        fee_pct: Fee override
        exit_reason: Reason for closing ('signal', 'stop_loss', 'take_profit', 'manual')

    Returns:
        Updated Trade instance or None if close rejected
    """
    from trading_bot.services.executor.risk import check_can_close_trade

    if slippage_pct is None:
        slippage_pct = 0.0005
    if fee_pct is None:
        fee_pct = 0.001

    allowed, reason = check_can_close_trade(trade)
    if not allowed:
        logger.warning("Cannot close trade %d: %s", trade.id, reason)
        return None

    # Apply slippage to exit price
    if trade.side == "buy":
        exit_price = current_price * (1 - slippage_pct)
    else:
        exit_price = current_price * (1 + slippage_pct)

    # Calculate PnL
    entry = float(trade.entry_price)
    exit_f = float(exit_price)
    qty = float(trade.quantity)

    if trade.side == "buy":
        pnl = qty * (exit_f - entry)
    else:
        pnl = qty * (entry - exit_f)

    pnl_pct = (exit_f - entry) / entry * 100 if entry > 0 else 0
    if trade.side == "sell":
        pnl_pct = -pnl_pct

    exit_fee = abs(qty * exit_f) * fee_pct
    net_pnl = pnl - float(trade.entry_fee or 0) - exit_fee

    # Update trade
    try:
        with transaction.atomic():
            trade.exit_price = round(Decimal(str(exit_f)), 8)
            trade.exit_time = datetime.now(timezone.utc)
            trade.status = "closed"
            trade.pnl = round(Decimal(str(net_pnl)), 8)
            trade.pnl_pct = round(pnl_pct, 4)
            trade.exit_fee = round(Decimal(str(exit_fee)), 8)
            trade.save(
                update_fields=[
                    "exit_price", "exit_time", "status",
                    "pnl", "pnl_pct", "exit_fee",
                ]
            )

            AuditLog.objects.create(
                action="trade_closed",
                message=(
                    f"Paper trade #{trade.id} CLOSED {trade.symbol}: "
                    f"${net_pnl:.2f} ({pnl_pct:.2f}%) — {exit_reason}"
                ),
                details={
                    "trade_id": trade.id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry_price": float(trade.entry_price),
                    "exit_price": exit_f,
                    "pnl": net_pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "entry_fee": float(trade.entry_fee or 0),
                    "exit_fee": exit_fee,
                },
                severity="info",
            )

            logger.info(
                "Paper trade #%d closed: %s %.2f → %.2f (PnL=%.2f, %.2f%%)",
                trade.id, trade.symbol, entry, exit_f, net_pnl, pnl_pct,
            )

            return trade

    except Exception as e:
        logger.exception("Failed to close paper trade %d: %s", trade.id, e)
        return None


def get_open_paper_trades() -> list[Trade]:
    """Get all open paper trades."""
    return list(Trade.objects.filter(mode="paper", status="open").select_related("signal", "strategy"))


def get_paper_portfolio_summary() -> dict[str, Any]:
    """Get a summary of the paper trading portfolio."""
    config = BotConfig.get_config()
    initial_balance = float(config.virtual_balance)

    closed_trades = Trade.objects.filter(mode="paper", status="closed")
    open_trades = Trade.objects.filter(mode="paper", status="open")
    total_trades = Trade.objects.filter(mode="paper").count()

    total_pnl = sum(float(t.pnl or 0) for t in closed_trades)
    current_balance = initial_balance + total_pnl

    # Open trade exposure
    open_exposure = sum(
        float(t.quantity) * float(t.entry_price)
        for t in open_trades
    )

    # Returns
    total_return_pct = (current_balance - initial_balance) / initial_balance * 100 if initial_balance > 0 else 0

    # Win rate
    winners = closed_trades.filter(pnl__gt=0).count()
    losers = closed_trades.filter(pnl__lt=0).count()
    win_rate = (winners / (winners + losers) * 100) if (winners + losers) > 0 else 0

    return {
        "initial_balance": initial_balance,
        "current_balance": round(current_balance, 2),
        "open_exposure": round(open_exposure, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_trades": total_trades,
        "open_trades": open_trades.count(),
        "closed_trades": closed_trades.count(),
        "winning_trades": winners,
        "losing_trades": losers,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
    }
