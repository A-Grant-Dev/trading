"""
Position Sizer — Kelly Criterion and Percentage-Based Position Sizing

Determines trade quantity based on account balance and risk parameters.

Methods:
1. Percentage sizing: Fixed % of available balance per trade
2. Kelly Criterion: Optimal bet size based on historical win rate and
   average win/loss ratio (fractional Kelly for safety)
"""

import logging
from decimal import Decimal
from typing import Optional

from trading_bot.models import BotConfig, Trade

logger = logging.getLogger(__name__)


def calculate_position_size(
    balance: float,
    entry_price: float,
    side: str = "buy",
    method: str = "auto",
    kelly_fraction: Optional[float] = None,
    fixed_pct: Optional[float] = None,
) -> tuple[float, float, dict]:
    """
    Calculate position size and trade value.

    Args:
        balance: Available balance in quote currency
        entry_price: Expected entry price per unit
        side: 'buy' or 'sell'
        method: 'kelly', 'percentage', or 'auto' (uses config)
        kelly_fraction: Fractional Kelly multiplier (default: config.kelly_fraction)
        fixed_pct: Fixed percentage of balance (default: config.max_position_size_pct)

    Returns:
        Tuple of (quantity, position_value, info_dict)
        where info_dict contains details about how the size was calculated
    """
    config = BotConfig.get_config()

    if kelly_fraction is None:
        kelly_fraction = config.kelly_fraction
    if fixed_pct is None:
        fixed_pct = config.max_position_size_pct

    if method == "auto":
        # Use Kelly if we have enough trade history, otherwise use percentage
        closed_trades = Trade.objects.filter(status="closed").count()
        if closed_trades >= 20:
            method = "kelly"
        else:
            method = "percentage"

    info = {"method": method, "balance": balance, "entry_price": entry_price}

    if method == "kelly":
        position_value, kelly_info = _kelly_position_size(balance, kelly_fraction)
        info.update(kelly_info)
    else:
        position_value = balance * (fixed_pct / 100)
        info["position_size_pct"] = fixed_pct
        info["position_value"] = position_value

    # Calculate quantity
    if entry_price > 0:
        quantity = position_value / entry_price
    else:
        quantity = 0.0

    return quantity, position_value, info


def _kelly_position_size(
    balance: float,
    kelly_fraction: float,
    min_trades: int = 20,
) -> tuple[float, dict]:
    """
    Calculate position size using the Kelly Criterion.

    Kelly % = W - (1-W) / R
    where:
        W = Win rate (decimal)
        R = Average win / average loss ratio

    Returns:
        Tuple of (position_value, info_dict)
    """
    closed_trades = Trade.objects.filter(status="closed")

    if closed_trades.count() < min_trades:
        # Fall back to percentage
        pct = BotConfig.get_config().max_position_size_pct
        position_value = balance * (pct / 100)
        return position_value, {
            "kelly_note": f"Insufficient trades ({closed_trades.count()} < {min_trades}), using {pct}%",
            "kelly_pct": None,
        }

    # Calculate win rate and avg win/loss
    winners = closed_trades.filter(pnl__gt=0)
    losers = closed_trades.filter(pnl__lt=0)

    n_winners = winners.count()
    n_losers = losers.count()
    total = n_winners + n_losers

    if total == 0:
        pct = BotConfig.get_config().max_position_size_pct
        position_value = balance * (pct / 100)
        return position_value, {
            "kelly_note": "No closed trades, using percentage fallback",
            "kelly_pct": None,
        }

    win_rate = n_winners / total

    avg_win = float(
        abs(sum(w.pnl or 0 for w in winners) / n_winners)
        if n_winners > 0 else 0.001
    )
    avg_loss = float(
        abs(sum(l.pnl or 0 for l in losers) / n_losers)
        if n_losers > 0 else 0.001
    )

    # Kelly formula
    r_ratio = None
    if avg_loss > 0:
        r_ratio = avg_win / avg_loss
        kelly_pct = win_rate - ((1 - win_rate) / r_ratio) if r_ratio > 0 else 0
    else:
        kelly_pct = win_rate

    # Apply fractional Kelly for safety
    kelly_pct = max(0.0, min(kelly_pct, 0.25)) * kelly_fraction

    # Fallback if Kelly is too small
    if kelly_pct < 0.01:
        kelly_pct = float(BotConfig.get_config().max_position_size_pct) / 100 * 0.5

    position_value = balance * kelly_pct

    return position_value, {
        "kelly_pct": round(kelly_pct * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "r_ratio": round(r_ratio, 2) if r_ratio is not None else None,
    }
