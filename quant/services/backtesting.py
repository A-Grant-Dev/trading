"""
Backtesting Engine — Event-driven simulation of trading strategies.

Features:
  - Uses MarketData table for historical data
  - Walks through time candle by candle
  - Supports multiple strategy types simultaneously
  - Tracks every trade: entry, exit, P&L, fees, slippage
  - Reports comprehensive metrics: Sharpe, Sortino, Calmar, drawdown

Renaissance principle: Every strategy must be rigorously backtested.
99%+ of discovered signals will be discarded after backtesting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A single trade recorded during backtesting."""
    entry_time: datetime
    exit_time: datetime | None = None
    side: str = "buy"
    entry_price: float = 0.0
    exit_price: float | None = None
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fee: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    """
    Comprehensive backtest report.

    Provides all key metrics for evaluating a strategy's performance.
    """

    initial_capital: float = 10000.0
    final_capital: float = 10000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    signals_generated: int = 0
    signals_discarded: int = 0

    @property
    def total_return(self) -> float:
        """Total return as a fraction."""
        return (self.final_capital - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0.0

    @property
    def win_rate(self) -> float:
        """Fraction of trades that were profitable."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def avg_win(self) -> float:
        """Average winning trade P&L."""
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        """Average losing trade P&L."""
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def avg_hold_time(self) -> float:
        """Average trade holding time in hours."""
        durations = []
        for t in self.trades:
            if t.exit_time and t.entry_time:
                delta = (t.exit_time - t.entry_time).total_seconds()
                durations.append(delta / 3600)
        return float(np.mean(durations)) if durations else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio of strategy returns."""
        if len(self.equity_curve) < 10:
            return 0.0
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) < 5 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(365))

    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio — only penalizes downside volatility."""
        if len(self.equity_curve) < 10:
            return 0.0
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        downside = returns[returns < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
        if downside_std == 0:
            return 0.0
        return float(np.mean(returns) / downside_std * np.sqrt(365))

    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio — return / max drawdown."""
        if self.max_drawdown_pct == 0:
            return 0.0
        annualized_return = self.total_return * 12  # Approximate
        return annualized_return / self.max_drawdown_pct if self.max_drawdown_pct > 0 else 0.0

    @property
    def metrics(self) -> dict:
        """All metrics as a flat dict."""
        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return_pct": round(self.total_return * 100, 4),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 4) if self.profit_factor != float("inf") else "inf",
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "avg_hold_time_hours": round(self.avg_hold_time, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "signals_generated": self.signals_generated,
            "signals_discarded": self.signals_discarded,
            "total_fees": round(self.total_fees, 2),
            "total_pnl": round(self.total_pnl, 2),
        }


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Walks through historical data candle by candle, calling the
    strategy function on each candle to generate signals, and
    tracking P&L from executed trades.

    Usage:
        engine = BacktestEngine(initial_capital=10000.0)
        engine.add_strategy(my_strategy_fn)
        result = engine.run(df)
        print(result.metrics)
    """

    def __init__(self, initial_capital: float = 10000.0,
                 fee_rate: float = 0.001,
                 slippage: float = 0.001):
        """
        Args:
            initial_capital: Starting capital in quote currency
            fee_rate: Trading fee as fraction (default 0.1%)
            slippage: Slippage as fraction (default 0.1%)
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.strategies: list[Callable] = []

    def add_strategy(self, strategy_fn: Callable) -> None:
        """
        Add a signal-generating function to the backtest.

        The strategy function receives the current row/index and the
        full DataFrame, and should return a dict with:
            action: 'buy', 'sell', or None
            confidence: 0.0–1.0
            quantity: fraction of capital to use (0.0–1.0)
            reason: str (optional)
        """
        self.strategies.append(strategy_fn)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest over historical data.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
                and any additional feature columns. Index must be datetime.

        Returns:
            BacktestResult with all computed metrics
        """
        result = BacktestResult(initial_capital=self.initial_capital)
        capital = self.initial_capital
        position = 0.0  # Current position size (in base asset)
        entry_price = 0.0
        entry_time: datetime | None = None

        equity_curve = [capital]

        if df.empty or len(df) < 20:
            result.final_capital = capital
            result.equity_curve = equity_curve
            return result

        for i in range(len(df)):
            row = df.iloc[i]
            current_price = float(row.get("close", 0))
            current_time = df.index[i]
            if hasattr(current_time, "to_pydatetime"):
                current_time = current_time.to_pydatetime()

            if current_price <= 0:
                equity_curve.append(capital)
                continue

            # ── Generate signals ──────────────────────────────────
            combined_signal = None
            for strategy_fn in self.strategies:
                try:
                    signal = strategy_fn(row, df.iloc[:i + 1])
                    if signal and signal.get("action") in ("buy", "sell"):
                        combined_signal = signal
                        result.signals_generated += 1
                except Exception as e:
                    logger.debug(f"Strategy error at index {i}: {e}")

            # ── Execute signals ────────────────────────────────────
            if combined_signal and position == 0:
                action = combined_signal["action"]
                confidence = combined_signal.get("confidence", 0.5)
                qty_pct = combined_signal.get("quantity", 0.95)

                if action == "buy":
                    fee = capital * qty_pct * self.fee_rate
                    slippage_cost = capital * qty_pct * self.slippage
                    position = (capital * qty_pct - fee - slippage_cost) / current_price
                    entry_price = current_price
                    entry_time = current_time
                    result.total_fees += fee
                    result.trades.append(BacktestTrade(
                        entry_time=current_time,
                        side="buy",
                        entry_price=current_price,
                        quantity=position,
                        reason=combined_signal.get("reason", "Signal"),
                    ))
                elif action == "sell" and position > 0:
                    # Close position
                    fee = position * current_price * self.fee_rate
                    slippage_cost = position * current_price * self.slippage
                    gross = position * current_price
                    pnl = gross - (position * entry_price) - fee - slippage_cost
                    pnl_pct = pnl / (position * entry_price) if entry_price > 0 else 0
                    result.total_fees += fee
                    capital = capital + pnl
                    result.total_pnl += pnl

                    if len(result.trades) > 0:
                        last_trade = result.trades[-1]
                        last_trade.exit_time = current_time
                        last_trade.exit_price = current_price
                        last_trade.pnl = pnl
                        last_trade.pnl_pct = pnl_pct
                        if pnl > 0:
                            result.winning_trades += 1
                        else:
                            result.losing_trades += 1
                        result.total_trades += 1

                    position = 0.0
                    entry_price = 0.0
                    entry_time = None

            # ── Track equity ──────────────────────────────────────
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)

            # ── Track drawdown ────────────────────────────────────
            peak = max(equity_curve)
            dd = peak - current_equity
            dd_pct = dd / peak if peak > 0 else 0
            if dd > result.max_drawdown:
                result.max_drawdown = dd
                result.max_drawdown_pct = dd_pct

        # Close any remaining position at final price
        if position > 0 and len(df) > 0:
            final_price = float(df.iloc[-1].get("close", 0))
            if final_price > 0:
                fee = position * final_price * self.fee_rate
                pnl = position * (final_price - entry_price) - fee
                pnl_pct = pnl / (position * entry_price) if entry_price > 0 else 0
                capital += pnl
                result.total_pnl += pnl
                result.total_fees += fee

                if len(result.trades) > 0:
                    last_trade = result.trades[-1]
                    last_trade.exit_time = df.index[-1]
                    last_trade.exit_price = final_price
                    last_trade.pnl = pnl
                    last_trade.pnl_pct = pnl_pct
                    if pnl > 0:
                        result.winning_trades += 1
                    else:
                        result.losing_trades += 1
                    result.total_trades += 1

        result.final_capital = capital
        result.equity_curve = equity_curve

        # Mark remaining signals as discarded (Renaissance rule)
        result.signals_discarded = max(0, result.signals_generated - result.total_trades)

        logger.info(
            "Backtest complete: %.0f→%.2f, %d trades, win_rate=%.1f%%, "
            "Sharpe=%.2f, max_dd=%.1f%%",
            self.initial_capital, capital, result.total_trades,
            result.win_rate * 100, result.sharpe_ratio,
            result.max_drawdown_pct * 100,
        )

        return result
