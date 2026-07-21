"""
Management command to run the paper trading engine.

Simulates trade execution on live signals using the paper executor:
- Fetches current price from OHLCV data
- Executes pending signals as paper trades
- Monitors and closes open positions based on signals
- Records full audit trail

Usage:
    # Run paper trading cycle once
    python manage.py run_paper_trading

    # Run with continuous monitoring (every N seconds)
    python manage.py run_paper_trading --continuous --interval 60

    # Show paper portfolio status
    python manage.py run_paper_trading --status

    # Close all open paper positions
    python manage.py run_paper_trading --flatten
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from django.core.management.base import BaseCommand

from trading_bot.models import AuditLog, BotConfig, Signal, Strategy as StrategyModel

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run paper trading engine — execute signals and manage positions"

    def add_arguments(self, parser):
        parser.add_argument(
            "--status",
            action="store_true",
            default=False,
            help="Show paper portfolio status and exit",
        )
        parser.add_argument(
            "--flatten",
            action="store_true",
            default=False,
            help="Close all open paper positions and exit",
        )
        parser.add_argument(
            "--continuous",
            action="store_true",
            default=False,
            help="Run in continuous monitoring mode",
        )
        parser.add_argument(
            "--interval",
            type=int,
            default=60,
            help="Polling interval in seconds (default: 60)",
        )
        parser.add_argument(
            "--symbol",
            type=str,
            default="BTCUSDT",
            help="Trading pair (default: BTCUSDT)",
        )

    def handle(self, *args, **options):
        if options["status"]:
            self._show_status()
            return

        if options["flatten"]:
            self._flatten_all()
            return

        symbol = options["symbol"].upper()
        interval = options["interval"]
        continuous = options["continuous"]

        self.stdout.write(self.style.SUCCESS(
            f"\n📋 Paper Trading Engine\n"
            f"{'='*60}\n"
            f"  Symbol:    {symbol}\n"
            f"  Mode:      {'Continuous (every {}s)'.format(interval) if continuous else 'Single pass'}\n"
            f"{'='*60}\n"
        ))

        if continuous:
            self._run_continuous(symbol, interval)
        else:
            self._run_single_pass(symbol)

    def _run_single_pass(self, symbol: str):
        """Run a single paper trading cycle."""
        t0 = time.time()

        # Step 1: Get current price
        current_price = self._get_current_price(symbol)
        if current_price is None:
            self.stdout.write(self.style.ERROR("Could not get current price"))
            return

        self.stdout.write(f"  Current price: ${current_price:.2f}")

        # Step 2: Process pending signals
        n_entered = self._process_new_signals(symbol, current_price)

        # Step 3: Check for signal-based closes
        n_closed = self._process_signal_closes(symbol, current_price)

        duration = time.time() - t0
        self.stdout.write(self.style.SUCCESS(
            f"\n  ✅ Cycle complete: {n_entered} entered, {n_closed} closed ({duration:.1f}s)"
        ))

        # Show portfolio
        self._show_portfolio_snapshot()

    def _run_continuous(self, symbol: str, interval: int):
        """Run paper trading continuously."""
        self.stdout.write(f"  Press Ctrl+C to stop\n")

        try:
            while True:
                self._run_single_pass(symbol)
                self.stdout.write(f"  Sleeping {interval}s...\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            self.stdout.write(self.style.SUCCESS("\n  Paper trading stopped"))

    def _process_new_signals(self, symbol: str, current_price: float) -> int:
        """Execute pending signals as paper trades."""
        from trading_bot.services.executor.paper import execute_paper_trade

        config = BotConfig.get_config()
        if not config.is_enabled:
            self.stdout.write(self.style.WARNING("  Trading is disabled (is_enabled=False)"))
            return 0

        pending_signals = Signal.objects.filter(
            symbol=symbol,
            status="pending",
        ).select_related("strategy", "param_set").order_by("timestamp")[:10]

        if not pending_signals:
            return 0

        n_executed = 0
        for signal in pending_signals:
            trade = execute_paper_trade(
                signal=signal,
                current_price=current_price,
            )
            if trade:
                n_executed += 1
                self.stdout.write(
                    f"  ✅ Entered {signal.get_direction_display()} {symbol}: "
                    f"{float(trade.quantity):.4f} @ ${float(trade.entry_price):.2f}"
                )
            else:
                self.stdout.write(
                    f"  ⏸️  Skipped signal #{signal.id} ({signal.get_direction_display()}): risk check failed"
                )

        return n_executed

    def _process_signal_closes(self, symbol: str, current_price: float) -> int:
        """Close positions based on new signals (exit = 0 / neutral)."""
        from trading_bot.services.executor.paper import close_paper_trade

        # Get new neutral signals for this symbol
        close_signals = Signal.objects.filter(
            symbol=symbol,
            direction=0,
            status="active",
        ).select_related("strategy").order_by("-timestamp")[:10]

        if not close_signals:
            return 0

        # Get open paper trades for this symbol
        from trading_bot.models import Trade
        open_trades = Trade.objects.filter(
            mode="paper",
            status="open",
            symbol=symbol,
        ).order_by("entry_time")

        n_closed = 0
        for signal in close_signals:
            # Find matching trade by strategy
            matching_trades = [t for t in open_trades if t.strategy_id == signal.strategy_id]
            if matching_trades:
                trade = matching_trades[0]
                closed = close_paper_trade(
                    trade=trade,
                    current_price=current_price,
                    exit_reason="signal",
                )
                if closed:
                    n_closed += 1
                    self.stdout.write(
                        f"  🔒 Closed {trade.side.upper()} {symbol}: "
                        f"${float(closed.pnl or 0):.2f} PnL"
                    )

            # Mark signal as filled
            signal.status = "filled"
            signal.save(update_fields=["status"])

        return n_closed

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current market price from OHLCV data."""
        from trading_bot.models import OHLCV

        latest = OHLCV.objects.filter(symbol=symbol).order_by("-timestamp").first()
        if latest:
            return float(latest.close)

        # Fallback: try to fetch live price
        try:
            import ccxt
            exchange = ccxt.binance({"timeout": 5000})
            ticker = exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            logger.warning("Failed to fetch live price: %s", e)
            return None

    def _show_status(self):
        """Display paper trading portfolio status."""
        from trading_bot.services.executor.paper import get_paper_portfolio_summary

        summary = get_paper_portfolio_summary()
        config = BotConfig.get_config()

        self.stdout.write(self.style.SUCCESS(
            f"\n📊 Paper Trading Portfolio\n"
            f"{'='*60}"
        ))
        self.stdout.write(
            f"  Balance:     ${summary['current_balance']:>10,.2f}  "
            f"({summary['total_return_pct']:+.2f}%)"
        )
        self.stdout.write(
            f"  Open Exposure: ${summary['open_exposure']:>10,.2f}"
        )
        self.stdout.write(
            f"  Open Trades: {summary['open_trades']:>3}  "
            f"|  Closed: {summary['closed_trades']}  "
            f"|  Total: {summary['total_trades']}"
        )
        self.stdout.write(
            f"  Win Rate:    {summary['win_rate']:>5.1f}%  "
            f"({summary['winning_trades']}W / {summary['losing_trades']}L)"
        )
        self.stdout.write(
            f"  Total PnL:   ${summary['total_pnl']:>+10,.2f}"
        )
        self.stdout.write(
            f"  Mode:        {config.mode.upper()}  "
            f"|  Enabled: {'YES' if config.is_enabled else 'NO'}\n"
        )

    def _show_portfolio_snapshot(self):
        """Display a quick portfolio snapshot after trading cycle."""
        from trading_bot.services.executor.paper import get_paper_portfolio_summary

        summary = get_paper_portfolio_summary()
        self.stdout.write(
            f"  Portfolio: ${summary['current_balance']:,.2f} "
            f"({summary['total_return_pct']:+.2f}%) "
            f"| {summary['open_trades']} open, {summary['total_trades']} total"
        )

    def _flatten_all(self):
        """Close all open paper positions."""
        from trading_bot.services.executor.paper import close_paper_trade, get_open_paper_trades

        open_trades = get_open_paper_trades()
        if not open_trades:
            self.stdout.write(self.style.WARNING("No open paper positions to flatten"))
            return

        self.stdout.write(f"Closing {len(open_trades)} open position(s)...")

        for trade in open_trades:
            # Get current price
            current_price = self._get_current_price(trade.symbol)
            if current_price is None:
                self.stdout.write(self.style.ERROR(f"  ❌ Cannot get price for {trade.symbol}, skipping"))
                continue

            closed = close_paper_trade(
                trade=trade,
                current_price=current_price,
                exit_reason="manual_flatten",
            )
            if closed:
                self.stdout.write(
                    f"  ✅ Closed {trade.side.upper()} {trade.symbol}: "
                    f"${float(closed.pnl or 0):+.2f}"
                )

        AuditLog.objects.create(
            action="info",
            message=f"Flattened all {len(open_trades)} paper positions",
            details={"n_positions": len(open_trades), "action": "flatten"},
            severity="info",
        )
        self.stdout.write(self.style.SUCCESS("  All positions closed"))
