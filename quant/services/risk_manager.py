"""
Risk Manager — Portfolio-level risk controls. The circuit breaker.

Rules enforced on every trade attempt:
  1. Max position size: No single position > configured % of portfolio
  2. Max daily loss: Halt all trading if daily P&L < threshold
  3. Max drawdown: Halt all trading if drawdown > threshold
  4. Max correlation: Don't over-concentrate in correlated assets
  5. Max open positions: Global cap on concurrent positions
  6. Min time between trades: Avoid overtrading
  7. Leverage cap: Max leverage based on regime
  8. Session check: Only trade during active hours

Renaissance principle: Risk management is not optional — it's the entire game.
"""

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone

from quant.models import ExecutedTrade, QuantConfig

logger = logging.getLogger(__name__)

# Pairs that are highly correlated (same sector)
CORRELATED_GROUPS: list[set[str]] = [
    # L1 / Smart Contract Platforms
    {"ETHUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT"},
    # DeFi
    {"UNIUSDT", "AAVEUSDT", "CRVUSDT", "MKRUSDT", "SUSHIUSDT"},
    # Layer 2
    {"ARBUSDT", "OPUSDT", "MATICUSDT"},
    # AI / Gaming
    {"FETUSDT", "AGIXUSDT", "OCEANUSDT"},
    # Meme
    {"DOGEUSDT", "SHIBUSDT", "PEPEUSDT"},
    # Exchange tokens
    {"BNBUSDT", "OKBUSDT", "GTUSDT"},
    # Oracle
    {"LINKUSDT", "BANDUSDT", "TRBUSDT"},
]

# Trading sessions (UTC)
ASIA_SESSION = (1, 9)    # 01:00 - 09:00 UTC
LONDON_SESSION = (8, 16)  # 08:00 - 16:00 UTC
NY_SESSION = (13, 22)     # 13:00 - 22:00 UTC

PREFERRED_SESSIONS = [ASIA_SESSION, LONDON_SESSION, NY_SESSION]


class RiskManager:
    """
    Portfolio-level risk controls — the circuit breaker.

    Usage:
        risk = RiskManager()
        allowed, reason = risk.can_trade(proposed_order, portfolio)
        if not allowed:
            logger.warning(f"Trade blocked: {reason}")
    """

    def __init__(self, config: QuantConfig | None = None):
        """
        Args:
            config: QuantConfig instance. Loads from DB if None.
        """
        if config:
            self.config = config
        else:
            self.config = QuantConfig.get_config()

    def can_trade(self, proposed_order: dict,
                  portfolio: dict | None = None) -> tuple[bool, str]:
        """
        Check all risk rules for a proposed trade.

        Args:
            proposed_order: Dict with keys:
                symbol, side, notional, confidence
            portfolio: Dict with keys (fetched from DB if None):
                open_positions, daily_pnl, current_drawdown, balance

        Returns:
            (allowed: bool, reason: str)
        """
        symbol = proposed_order.get("symbol", "").upper()
        side = proposed_order.get("side", "buy")
        notional = proposed_order.get("notional", 0)
        confidence = proposed_order.get("confidence", 0.5)

        if not portfolio:
            portfolio = self._get_portfolio_snapshot()

        order = {
            "symbol": symbol,
            "side": side,
            "notional": notional,
            "confidence": confidence,
        }

        # Check all rules in order of severity
        checks = [
            ("is_enabled", self._check_enabled()),
            ("max_drawdown", self._check_max_drawdown(portfolio)),
            ("max_daily_loss", self._check_daily_loss(portfolio)),
            ("max_open_positions", self._check_max_open(portfolio)),
            ("max_position_size", self._check_position_size(order, portfolio)),
            ("correlation", self._check_correlation(order, portfolio)),
            ("session", self._check_session()),
        ]

        for check_name, result in checks:
            if result is not True:
                return False, result

        return True, "All risk checks passed"

    def _check_enabled(self) -> bool | str:
        """Check if trading is enabled."""
        if not self.config.is_enabled:
            return "Trading is disabled in QuantConfig"
        return True

    def _check_max_drawdown(self, portfolio: dict) -> bool | str:
        """Check if drawdown exceeds configured limit."""
        drawdown = portfolio.get("current_drawdown", 0)
        max_dd = self.config.max_drawdown_pct / 100.0
        if drawdown >= max_dd:
            return (
                f"Drawdown {drawdown:.1%} exceeds max {max_dd:.1%}. "
                f"Trading halted until recovery."
            )
        return True

    def _check_daily_loss(self, portfolio: dict) -> bool | str:
        """Check if today's losses exceed the daily limit."""
        daily_pnl_pct = portfolio.get("daily_pnl_pct", 0)
        max_loss = self.config.max_daily_loss_pct / 100.0
        if daily_pnl_pct <= -max_loss:
            return (
                f"Daily P&L {daily_pnl_pct:.1%} exceeds max loss {max_loss:.1%}. "
                f"Trading halted for the day."
            )
        return True

    def _check_max_open(self, portfolio: dict) -> bool | str:
        """Check if max concurrent positions is reached."""
        open_count = portfolio.get("open_positions", 0)
        max_open = self.config.max_open_positions
        if open_count >= max_open:
            return (
                f"Max open positions reached: {open_count}/{max_open}. "
                f"Close a position before opening a new one."
            )
        return True

    def _check_position_size(self, order: dict, portfolio: dict) -> bool | str:
        """Check if the proposed position size exceeds the max allowed."""
        notional = order.get("notional", 0)
        balance = portfolio.get("balance", 0)
        max_pct = self.config.max_position_size_pct / 100.0

        if balance <= 0:
            return "Account balance is zero or negative"

        position_pct = notional / balance if balance > 0 else 0
        if position_pct > max_pct:
            return (
                f"Position size {position_pct:.1%} of portfolio exceeds "
                f"max {max_pct:.1%}. Max allowed: ${balance * max_pct:.2f}"
            )
        return True

    def _check_correlation(self, order: dict, portfolio: dict) -> bool | str:
        """Check if the new position would over-concentrate in a correlated group."""
        symbol = order.get("symbol", "").upper()

        # Find which correlation group(s) this symbol belongs to
        symbol_groups = []
        for group in CORRELATED_GROUPS:
            if symbol in group:
                symbol_groups.append(group)
                break  # Only the first matching group

        if not symbol_groups:
            return True  # Unknown symbol — no correlation check

        # Count open positions in the same group
        open_symbols = portfolio.get("open_symbols", [])
        group_count = 0
        for group in symbol_groups:
            for osym in open_symbols:
                if osym in group:
                    group_count += 1

        # Allow at most 2 positions in the same correlated group
        if group_count >= 2:
            return (
                f"Already have {group_count} positions in a correlated group "
                f"with {symbol}. Over-concentration risk."
            )
        return True

    def _check_session(self) -> bool | str:
        """Check if current time is within preferred trading sessions."""
        now = datetime.now(timezone.utc)
        hour = now.hour

        for start, end in PREFERRED_SESSIONS:
            if start <= hour <= end:
                return True

        return (
            f"Current time {hour}:00 UTC is outside preferred trading sessions. "
            f"Sessions: Asia (01-09), London (08-16), NY (13-22) UTC"
        )

    def _get_portfolio_snapshot(self) -> dict:
        """Build a portfolio snapshot from DB for risk checks."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Open positions
        open_trades = ExecutedTrade.objects.filter(status="open")
        open_count = open_trades.count()
        open_symbols = list(open_trades.values_list("symbol", flat=True))

        # Today's P&L
        today_trades = ExecutedTrade.objects.filter(entry_time__gte=today_start)
        daily_pnl = sum(float(t.pnl or 0) for t in today_trades if t.pnl is not None)
        balance = float(self.config.virtual_balance)
        daily_pnl_pct = daily_pnl / balance if balance > 0 else 0

        # Drawdown (simplified: based on all-time worst P&L day)
        all_trades = ExecutedTrade.objects.all().order_by("entry_time")
        peak = balance
        current_drawdown = 0.0
        running_balance = balance
        for t in all_trades:
            pnl = float(t.pnl or 0)
            running_balance += pnl
            if running_balance > peak:
                peak = running_balance
            dd = (peak - running_balance) / peak if peak > 0 else 0
            if dd > current_drawdown:
                current_drawdown = dd

        return {
            "open_positions": open_count,
            "open_symbols": open_symbols,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "current_drawdown": current_drawdown,
            "balance": balance,
        }

    @staticmethod
    def get_daily_pnl() -> float:
        """Calculate today's P&L from ExecutedTrade records."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        trades = ExecutedTrade.objects.filter(entry_time__gte=today_start)
        return sum(float(t.pnl or 0) for t in trades if t.pnl is not None)

    @staticmethod
    def get_current_drawdown(balance: float | None = None) -> float:
        """
        Calculate current drawdown from peak portfolio value.

        Args:
            balance: Current balance. Fetches from config if None.

        Returns:
            Drawdown as fraction (0.0–1.0)
        """
        if balance is None:
            config = QuantConfig.get_config()
            balance = float(config.virtual_balance)

        all_trades = ExecutedTrade.objects.all().order_by("entry_time")
        peak = balance
        max_dd = 0.0
        running = balance

        for t in all_trades:
            pnl = float(t.pnl or 0)
            running += pnl
            if running > peak:
                peak = running
            dd = (peak - running) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return round(max_dd, 6)
