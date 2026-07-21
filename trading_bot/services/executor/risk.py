"""
Risk Engine — Position Limits, Circuit Breakers, and Safety Checks

Enforces risk constraints on all trading decisions:
- Max open positions
- Max position size as % of balance
- Max daily loss % (halts trading if exceeded)
- Max drawdown % (halts trading if exceeded)
- Circuit breaker (consecutive losses trigger cooldown)
- Correlation limits (prevent over-concentration in correlated assets)

Every risk check is logged to AuditLog for full traceability.
"""

import logging
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from typing import Optional

from django.db.models import Sum, Q

from trading_bot.models import AuditLog, BotConfig, Trade

logger = logging.getLogger(__name__)


def check_can_open_trade(
    symbol: str,
    side: str,
    position_size_value: float,
    strategy_name: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Check if a new trade is allowed under current risk constraints.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        side: 'buy' or 'sell'
        position_size_value: Trade value in quote currency (e.g., 100.0 USD)
        strategy_name: Strategy name for audit logging

    Returns:
        Tuple of (allowed: bool, reason: str)
    """
    config = BotConfig.get_config()

    # ── Guard 1: Master on/off switch ──────────────────────────
    if not config.is_enabled:
        return False, "Trading is disabled (BotConfig.is_enabled = False)"

    # ── Guard 2: Mode must be paper or live ────────────────────
    if config.mode not in ("paper", "live"):
        return False, f"Mode is '{config.mode}', not paper or live"

    # ── Guard 3: Max open positions ────────────────────────────
    open_positions = Trade.objects.filter(status="open", mode=config.mode).count()
    if open_positions >= config.max_open_positions:
        return (
            False,
            f"Max open positions reached ({open_positions}/{config.max_open_positions})",
        )

    # ── Guard 4: Position size vs balance ──────────────────────
    balance = float(config.virtual_balance) if config.mode == "paper" else float(config.real_balance_limit)
    max_position_value = balance * (config.max_position_size_pct / 100)

    if position_size_value > max_position_value:
        return (
            False,
            f"Position size ${position_size_value:.2f} exceeds max "
            f"${max_position_value:.2f} ({config.max_position_size_pct}% of ${balance:.2f})",
        )

    # ── Guard 5: Daily loss limit ──────────────────────────────
    today = date.today()
    daily_losses = Trade.objects.filter(
        mode=config.mode,
        status="closed",
        pnl__lt=0,
        exit_time__date=today,
    ).aggregate(total_loss=Sum("pnl"))["total_loss"] or Decimal("0")

    daily_loss_pct = abs(float(daily_losses)) / balance * 100 if balance > 0 else 0
    if daily_loss_pct >= config.max_daily_loss_pct:
        return (
            False,
            f"Daily loss limit reached: {daily_loss_pct:.1f}% ≥ "
            f"{config.max_daily_loss_pct}% max",
        )

    # ── Guard 6: Drawdown limit ────────────────────────────────
    peak_balance = _get_peak_balance(config.mode)
    current_balance = _get_current_balance(config.mode)
    if peak_balance > 0:
        drawdown_pct = (peak_balance - current_balance) / peak_balance * 100
        if drawdown_pct >= config.max_drawdown_pct:
            return (
                False,
                f"Drawdown limit reached: {drawdown_pct:.1f}% ≥ "
                f"{config.max_drawdown_pct}% max",
            )

    # ── Guard 7: Circuit breaker ───────────────────────────────
    breaker_active, breaker_reason = _check_circuit_breaker(config)
    if breaker_active:
        return False, breaker_reason

    # ── Guard 8: Correlation limits ────────────────────────────
    correlation_allowed, correlation_reason = _check_correlation_limits(symbol, config)
    if not correlation_allowed:
        return False, correlation_reason

    # ── Guard 9: Kill switch ───────────────────────────────────
    killed, kill_reason = check_kill_switch()
    if killed:
        return False, kill_reason

    return True, "ok"


def check_can_close_trade(trade: Trade) -> tuple[bool, str]:
    """
    Check if an open trade can be closed.

    Always allows closing — risk checks only apply to opening new positions.
    """
    if trade.status != "open":
        return False, f"Trade {trade.id} is not open (status={trade.status})"
    return True, "ok"


# ── Kill Switch ──────────────────────────────────────────────────


def check_kill_switch() -> tuple[bool, str]:
    """
    Check if the kill switch has been activated.

    The kill switch is triggered by AuditLog entries with action='kill_switch'.
    Once active, it blocks all new trades until disarmed.

    Returns:
        Tuple of (is_killed: bool, reason: str)
    """
    last_kill = (
        AuditLog.objects.filter(action="kill_switch")
        .order_by("-timestamp")
        .first()
    )
    if last_kill is None:
        return False, ""
    return True, f"Kill switch is active (triggered at {last_kill.timestamp.strftime('%H:%M UTC')})"


def disarm_kill_switch() -> bool:
    """
    Disarm the kill switch by removing all active kill switch audit entries.

    Returns:
        True if any entries were removed
    """
    deleted, _ = AuditLog.objects.filter(action="kill_switch").delete()
    if deleted:
        logger.info("Kill switch disarmed (%d entries removed)", deleted)
    return deleted > 0


# ── Correlation Limits ─────────────────────────────────────────────


def _check_correlation_limits(symbol: str, config: BotConfig) -> tuple[bool, str]:
    """
    Check correlation limits — prevents multiple open positions in the same symbol.

    Args:
        symbol: Trading pair to check
        config: BotConfig singleton

    Returns:
        Tuple of (allowed: bool, reason: str)
    """
    existing = Trade.objects.filter(
        mode=config.mode,
        status="open",
        symbol=symbol.upper(),
    ).count()
    if existing > 0:
        return False, f"Already have an open position in {symbol}"
    return True, ""


def record_loss_for_circuit_breaker(config: BotConfig, trade: Trade) -> bool:
    """
    Track a losing trade for circuit breaker logic.

    If the number of consecutive losses exceeds the circuit_breaker_count,
    trips the breaker and prevents new trades for circuit_breaker_hours.

    Args:
        config: BotConfig singleton
        trade: The losing trade

    Returns:
        True if circuit breaker was tripped
    """
    if trade.pnl is not None and trade.pnl >= 0:
        return False  # Not a loss

    # Count consecutive losses
    recent_closed = Trade.objects.filter(
        mode=config.mode,
        status="closed",
    ).order_by("-exit_time")[:config.circuit_breaker_count]

    consecutive_losses = 0
    for t in recent_closed:
        if t.pnl is not None and t.pnl < 0:
            consecutive_losses += 1
        else:
            break

    if consecutive_losses >= config.circuit_breaker_count:
        # Trip breaker — cooldown is calculated from timestamp in _check_circuit_breaker
        cooldown_until = datetime.now(timezone.utc) + timedelta(
            hours=config.circuit_breaker_hours
        )

        AuditLog.objects.create(
            action="error",
            message=(
                f"Circuit breaker TRIPPED after {consecutive_losses} consecutive losses. "
                f"Trading halted until {cooldown_until.strftime('%Y-%m-%d %H:%M UTC')}"
            ),
            details={
                "consecutive_losses": consecutive_losses,
                "cooldown_until": cooldown_until.isoformat(),
                "circuit_breaker_count": config.circuit_breaker_count,
                "circuit_breaker_hours": config.circuit_breaker_hours,
            },
            severity="critical",
        )
        logger.warning(
            "Circuit breaker tripped after %d consecutive losses. "
            "Cooldown until %s",
            consecutive_losses, cooldown_until,
        )
        return True

    return False


# ── Internal Helpers ────────────────────────────────────────────────


def _get_peak_balance(mode: str) -> float:
    """Get the peak balance from BotConfig history."""
    config = BotConfig.get_config()
    balance = float(config.virtual_balance) if mode == "paper" else float(config.real_balance_limit)

    # Start from initial config balance
    peak = balance

    # Look at closed trades with positive PnL to estimate peak
    closed_trades = Trade.objects.filter(mode=mode, status="closed").order_by("exit_time")
    running_balance = balance
    for t in closed_trades:
        if t.pnl is not None:
            running_balance += float(t.pnl)
            if running_balance > peak:
                peak = running_balance

    return peak


def _get_current_balance(mode: str) -> float:
    """Get the current estimated balance."""
    config = BotConfig.get_config()
    balance = float(config.virtual_balance) if mode == "paper" else float(config.real_balance_limit)

    # Add realized PnL from closed trades
    total_pnl = (
        Trade.objects.filter(mode=mode, status="closed")
        .aggregate(total=Sum("pnl"))["total"]
        or Decimal("0")
    )
    return balance + float(total_pnl)


def _check_circuit_breaker(config: BotConfig) -> tuple[bool, str]:
    """
    Check if the circuit breaker is active.

    Looks at AuditLog for the most recent circuit breaker trip.
    If the cooldown period hasn't elapsed, blocks trading.

    Returns:
        Tuple of (is_breaked: bool, reason: str)
    """
    last_breaker = (
        AuditLog.objects.filter(
            action="error",
            severity="critical",
            message__icontains="circuit breaker",
        )
        .order_by("-timestamp")
        .first()
    )

    if last_breaker is None:
        return False, ""

    # Check if cooldown has elapsed
    cooldown_end = last_breaker.timestamp + timedelta(
        hours=config.circuit_breaker_hours
    )

    if datetime.now(timezone.utc) < cooldown_end:
        remaining = (cooldown_end - datetime.now(timezone.utc)).total_seconds()
        return (
            True,
            f"Circuit breaker active for {remaining:.0f}s more "
            f"(tripped at {last_breaker.timestamp.strftime('%H:%M UTC')})",
        )

    return False, ""
