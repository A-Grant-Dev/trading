"""
Backtesting Metrics — Performance Evaluation Suite

Computes a comprehensive set of trading performance metrics from
equity curves and trade lists.

Metrics computed:
- Total Return %, Annualized Return %
- Volatility (annualized)
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown %
- Win Rate %, Profit Factor, Expectancy
- Average Win / Average Loss
- Total Trades, Hit Rate
- R-squared (consistency of returns)

All functions accept numpy arrays for vectorized computation.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 365
HOURS_PER_YEAR = 24 * 365
MINUTES_PER_YEAR = 60 * 24 * 365


def get_annualization_factor(interval: str) -> float:
    """Get the annualization factor for a given interval."""
    factors = {
        "1m": MINUTES_PER_YEAR,
        "5m": MINUTES_PER_YEAR / 5,
        "15m": MINUTES_PER_YEAR / 15,
        "30m": MINUTES_PER_YEAR / 30,
        "1h": HOURS_PER_YEAR,
        "2h": HOURS_PER_YEAR / 2,
        "4h": HOURS_PER_YEAR / 4,
        "6h": HOURS_PER_YEAR / 6,
        "8h": HOURS_PER_YEAR / 8,
        "12h": HOURS_PER_YEAR / 12,
        "1d": TRADING_DAYS_PER_YEAR,
        "3d": TRADING_DAYS_PER_YEAR / 3,
        "1w": 52,
    }
    return factors.get(interval, HOURS_PER_YEAR)


# ── Core Metrics ───────────────────────────────────────────────────


def compute_total_return(equity_curve: np.ndarray) -> float:
    """Compute total return % from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    start = equity_curve[0]
    end = equity_curve[-1]
    if start == 0:
        return 0.0
    return float((end - start) / start * 100)


def compute_annualized_return(
    equity_curve: np.ndarray,
    periods_per_year: float,
) -> float:
    """Compute annualized return %."""
    total_ret = compute_total_return(equity_curve) / 100
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return 0.0
    return float(((1 + total_ret) ** (periods_per_year / n_periods) - 1) * 100)


def compute_volatility(
    returns: np.ndarray,
    periods_per_year: float,
) -> float:
    """Compute annualized volatility from periodic returns."""
    if len(returns) < 2:
        return 0.0
    return float(np.nanstd(returns) * np.sqrt(periods_per_year) * 100)


def compute_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: float,
    risk_free_rate: float = 0.05,
) -> float:
    """
    Compute annualized Sharpe Ratio.

    Sharpe = (mean(returns) - risk_free_rate/periods) / std(returns) * sqrt(periods)
    """
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = np.nanmean(excess_returns)
    std_returns = np.nanstd(excess_returns)
    if std_returns == 0 or np.isnan(std_returns):
        return 0.0
    return float(mean_excess / std_returns * np.sqrt(periods_per_year))


def compute_sortino_ratio(
    returns: np.ndarray,
    periods_per_year: float,
    risk_free_rate: float = 0.05,
) -> float:
    """
    Compute annualized Sortino Ratio.

    Uses downside deviation (negative returns only) instead of total std.
    """
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = np.nanmean(excess_returns)

    # Downside deviation: std of negative returns only
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        downside = np.nanstd(returns)  # Fall back to regular std
    else:
        downside = np.nanstd(negative_returns)

    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(mean_excess / downside * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Compute maximum drawdown percentage.

    Drawdown = (current - peak) / peak * 100
    Max DD = minimum drawdown (largest peak-to-trough decline)
    """
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak * 100
    return float(np.min(drawdowns))


def compute_calmar_ratio(
    equity_curve: np.ndarray,
    periods_per_year: float,
) -> float:
    """
    Compute Calmar Ratio.

    Calmar = Annualized Return / |Max Drawdown|
    """
    ann_return = compute_annualized_return(equity_curve, periods_per_year)
    max_dd = abs(compute_max_drawdown(equity_curve))
    if max_dd == 0 or np.isnan(max_dd):
        return 0.0
    return float(ann_return / max_dd)


# ── Trade-Based Metrics ────────────────────────────────────────────


def compute_trade_metrics(trades: list[dict]) -> dict[str, float]:
    """
    Compute metrics from a list of closed trades.

    Each trade dict should have:
        - 'pnl_pct': PnL as percentage
        - 'pnl': PnL in quote currency
        - 'side': 'buy' or 'sell'

    Returns dict with:
        win_rate, profit_factor, expectancy, avg_win, avg_loss,
        total_trades, winning_trades, losing_trades, hit_rate
    """
    if not trades:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_trades": 0.0,
            "winning_trades": 0.0,
            "losing_trades": 0.0,
            "hit_rate": 0.0,
            "avg_hold_bars": 0.0,
        }

    pnl_pcts = np.array([t.get("pnl_pct", 0) or 0 for t in trades])
    pnls = np.array([t.get("pnl", 0) or 0 for t in trades])
    total = len(pnl_pcts)

    winners = pnl_pcts[pnl_pcts > 0]
    losers = pnl_pcts[pnl_pcts < 0]
    n_winners = len(winners)
    n_losers = len(losers)

    win_rate = (n_winners / total * 100) if total > 0 else 0.0
    total_wins = np.sum(pnls[pnls > 0]) if np.any(pnls > 0) else 0
    total_losses = abs(np.sum(pnls[pnls < 0])) if np.any(pnls < 0) else 1.0

    avg_win = float(np.mean(winners)) if n_winners > 0 else 0.0
    avg_loss = float(np.mean(losers)) if n_losers > 0 else 0.0

    return {
        "win_rate": round(win_rate, 2),
        "profit_factor": round(float(total_wins / total_losses), 4) if total_losses > 0 else 0.0,
        "expectancy": round(float(np.mean(pnl_pcts)), 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "total_trades": float(total),
        "winning_trades": float(n_winners),
        "losing_trades": float(n_losers),
        "hit_rate": round(win_rate / 100, 4),
    }


# ── Full Metrics Suite ─────────────────────────────────────────────


def compute_full_metrics(
    equity_curve: np.ndarray,
    returns: np.ndarray,
    trades: list[dict],
    interval: str = "1h",
    risk_free_rate: float = 0.05,
) -> dict[str, float]:
    """
    Compute the complete metrics suite for a backtest.

    Args:
        equity_curve: Array of account equity over time
        returns: Array of periodic returns (same length as equity_curve - 1)
        trades: List of closed trade dicts with pnl_pct and pnl
        interval: Candle interval for annualization
        risk_free_rate: Annual risk-free rate (default 5%)

    Returns:
        Dict of all computed metrics
    """
    periods_per_year = get_annualization_factor(interval)

    total_return = compute_total_return(equity_curve)
    ann_return = compute_annualized_return(equity_curve, periods_per_year)
    volatility = compute_volatility(returns, periods_per_year)
    sharpe = compute_sharpe_ratio(returns, periods_per_year, risk_free_rate)
    sortino = compute_sortino_ratio(returns, periods_per_year, risk_free_rate)
    max_dd = compute_max_drawdown(equity_curve)
    calmar = compute_calmar_ratio(equity_curve, periods_per_year)
    trade_metrics = compute_trade_metrics(trades)

    metrics = {
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(ann_return, 2),
        "annualized_volatility_pct": round(volatility, 2),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown_pct": round(max_dd, 2),
    }
    metrics.update(trade_metrics)

    return metrics
