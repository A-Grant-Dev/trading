"""
Quant Notifier — Sends notifications for important trading events.

Events that trigger notification:
  - New trade executed (with details)
  - Stop loss hit
  - Take profit hit
  - Regime change detected
  - New cointegrated pair discovered
  - Drawdown exceeds threshold
  - Risk manager blocks a trade (with reason)
  - Daily P&L summary

Renaissance principle: The machine runs autonomously, but the operator
needs to know when something significant happens.
"""

import logging
from datetime import datetime, timezone

from quant.models import ExecutedTrade, QuantConfig

logger = logging.getLogger(__name__)


class QuantNotifier:
    """
    Sends notifications for quant trading events.

    Since this is a local/personal app, notifications are logged to
    the console and stored in the ExecutedTrade notes field.
    For future enhancement: add email, Telegram, or Discord integration.

    Usage:
        notifier = QuantNotifier()
        notifier.notify_trade(trade)
        notifier.notify_risk_event('max_drawdown', 'Drawdown exceeded 15%')
    """

    def __init__(self, config: QuantConfig | None = None):
        self.config = config or QuantConfig.get_config()

    def notify_trade(self, trade: ExecutedTrade) -> None:
        """Send notification when a trade is executed."""
        pnl_str = f"${float(trade.pnl or 0):.2f}" if trade.pnl else "N/A"
        msg = (
            f"[TRADE] {trade.side.upper()} {trade.symbol} "
            f"qty={float(trade.qty):.4f} @ ${float(trade.entry_price):.2f} "
            f"| P&L: {pnl_str} | Mode: {self.config.mode} | "
            f"Strategy: {trade.strategy or 'N/A'}"
        )
        logger.info(msg)

    def notify_exit(self, trade: ExecutedTrade, reason: str) -> None:
        """Send notification when a trade exits (stop loss or take profit)."""
        pnl_str = f"${float(trade.pnl or 0):.2f}" if trade.pnl else "N/A"
        pnl_pct = f"{float(trade.pnl_pct or 0) * 100:.2f}%" if trade.pnl_pct else "N/A"
        msg = (
            f"[EXIT] {trade.side.upper()} {trade.symbol} "
            f"| Exit: ${float(trade.exit_price or 0):.2f} "
            f"| P&L: {pnl_str} ({pnl_pct}) "
            f"| Reason: {reason}"
        )
        logger.info(msg)

    def notify_regime_change(self, symbol: str, old_regime: str,
                             new_regime: str, confidence: float) -> None:
        """Send notification when the market regime changes."""
        msg = (
            f"[REGIME] {symbol}: {old_regime} → {new_regime} "
            f"(confidence: {confidence:.1%})"
        )
        logger.info(msg)

    def notify_risk_event(self, event_type: str, details: str) -> None:
        """Send notification when a risk rule is triggered."""
        msg = f"[RISK] {event_type}: {details}"
        logger.warning(msg)

    def notify_new_pair(self, symbol_a: str, symbol_b: str,
                        p_value: float, half_life: float) -> None:
        """Send notification when a new cointegrated pair is discovered."""
        msg = (
            f"[PAIR] New cointegrated pair: {symbol_a}/{symbol_b} "
            f"(p={p_value:.4f}, half-life={half_life:.1f}h)"
        )
        logger.info(msg)

    def notify_daily_summary(self) -> None:
        """Send end-of-day P&L summary."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        today_trades = ExecutedTrade.objects.filter(entry_time__gte=today_start)
        total = today_trades.count()
        wins = today_trades.filter(pnl__gt=0).count()
        losses = today_trades.filter(pnl__lt=0).count()
        total_pnl = sum(float(t.pnl or 0) for t in today_trades if t.pnl is not None)
        open_count = ExecutedTrade.objects.filter(status="open").count()
        balance = float(self.config.virtual_balance)

        msg = (
            f"[DAILY] {now.strftime('%Y-%m-%d')}: "
            f"{total} trades ({wins}W/{losses}L) "
            f"| P&L: ${total_pnl:.2f} ({total_pnl / balance * 100:.2f}%) "
            f"| Open: {open_count} | Balance: ${balance:.2f}"
        )
        logger.info(msg)
