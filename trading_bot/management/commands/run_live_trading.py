"""
Management command for live trading execution via CCXT Binance.

Usage:
    # Show live trading status
    python manage.py run_live_trading --status

    # Execute a single signal as a live trade (dry run by default)
    python manage.py run_live_trading --execute --signal-id 1 --dry-run

    # Execute a live trade for real (requires BINANCE_API_KEY)
    python manage.py run_live_trading --execute --signal-id 1 --no-dry-run

    # Emergency flatten all positions
    python manage.py run_live_trading --flatten --dry-run

    # ACTUALLY flatten (real orders!)
    python manage.py run_live_trading --flatten --no-dry-run

    # Check exchange connection
    python manage.py run_live_trading --check-connection

    # Promote paper ParamSet to live
    python manage.py run_live_trading --promote --param-set-id 1
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from django.core.management.base import BaseCommand

from trading_bot.models import AuditLog, BotConfig, ParamSet, Signal, Trade

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run live trading engine — execute signals on Binance"

    def add_arguments(self, parser):
        parser.add_argument(
            "--status",
            action="store_true",
            default=False,
            help="Show live trading portfolio status",
        )
        parser.add_argument(
            "--execute",
            action="store_true",
            default=False,
            help="Execute a signal as a live trade",
        )
        parser.add_argument(
            "--signal-id",
            type=int,
            default=None,
            help="Signal ID to execute (used with --execute)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=True,
            help="Simulate without sending to exchange (default: True)",
        )
        parser.add_argument(
            "--no-dry-run",
            action="store_false",
            dest="dry_run",
            help="Actually send orders to exchange",
        )
        parser.add_argument(
            "--flatten",
            action="store_true",
            default=False,
            help="Emergency close ALL open live positions",
        )
        parser.add_argument(
            "--check-connection",
            action="store_true",
            default=False,
            help="Check Binance exchange connection",
        )
        parser.add_argument(
            "--promote",
            action="store_true",
            default=False,
            help="Promote a ParamSet from paper to live",
        )
        parser.add_argument(
            "--param-set-id",
            type=int,
            default=None,
            help="ParamSet ID to promote (used with --promote)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Skip confirmation for flatten (non-interactive mode)",
        )

    def handle(self, *args, **options):
        if options["status"]:
            self._show_status()
        elif options["check_connection"]:
            self._check_connection()
        elif options["flatten"]:
            self._flatten_all(options["dry_run"], options.get("force", False))
        elif options["execute"]:
            self._execute_signal(options["signal_id"], options["dry_run"])
        elif options["promote"]:
            self._promote_param_set(options["param_set_id"])
        else:
            self._show_status()

    def _show_status(self):
        """Display live trading status."""
        from trading_bot.services.executor.live import get_live_portfolio_summary

        summary = get_live_portfolio_summary()
        config = BotConfig.get_config()

        self.stdout.write(self.style.SUCCESS(
            f"\n🔴 Live Trading Portfolio\n"
            f"{'='*60}"
        ))
        self.stdout.write(
            f"  Balance Limit: ${summary['balance_limit']:>10,.2f}\n"
            f"  Current Value: ${summary['current_value']:>10,.2f}\n"
            f"  Open Exposure: ${summary['open_exposure']:>10,.2f}\n"
            f"  Total Trades:  {summary['total_trades']:>3}\n"
            f"  Open/Closed:   {summary['open_trades']}/{summary['closed_trades']}\n"
            f"  Win Rate:      {summary['win_rate']:>5.1f}%\n"
            f"  Total PnL:     ${summary['total_pnl']:>+10,.2f}\n"
            f"  API Keys:      {'✅ Configured' if summary['has_api_keys'] else '❌ Not configured'}\n"
            f"  Mode:          {config.mode.upper()}\n"
            f"  Enabled:       {'YES' if config.is_enabled else 'NO'}"
        )

        # Show open trades
        open_trades = Trade.objects.filter(mode="live", status="open").select_related("strategy")
        if open_trades:
            self.stdout.write(f"\n  📋 Open Positions:")
            for t in open_trades:
                self.stdout.write(
                    f"    #{t.id} {t.side.upper()} {t.symbol}: "
                    f"{float(t.quantity):.6f} @ ${float(t.entry_price):.2f}"
                )

        self.stdout.write("")

    def _check_connection(self):
        """Check Binance exchange connection."""
        from trading_bot.services.executor.live import get_exchange

        config = BotConfig.get_config()
        exchange = get_exchange(config)

        if exchange is None:
            self.stdout.write(self.style.ERROR(
                "❌ Cannot connect: Binance API keys not configured\n\n"
                "Set these in your .env file:\n"
                "  BINANCE_API_KEY=your_api_key\n"
                "  BINANCE_API_SECRET=your_api_secret\n"
                "  BINANCE_USE_TESTNET=true  (optional, for testing)"
            ))
            return

        try:
            # Fetch account info to verify connectivity
            account = exchange.fetch_balance()
            total_usdt = account.get("USDT", {}).get("total", 0)
            self.stdout.write(self.style.SUCCESS(
                f"✅ Connected to Binance {'testnet' if exchange.urls.get('api', '').__contains__('testnet') else 'mainnet'}\n"
                f"  Balance: ${float(total_usdt):.2f} USDT\n"
                f"  Account: {account.get('info', {}).get('accountType', 'unknown')}"
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Connection failed: {e}"))

    def _execute_signal(self, signal_id: Optional[int], dry_run: bool):
        """Execute a signal as a live trade."""
        if signal_id is None:
            self.stdout.write(self.style.ERROR("Usage: --execute --signal-id <id> [--dry-run|--no-dry-run]"))
            return

        from trading_bot.services.executor.live import execute_live_trade

        try:
            signal = Signal.objects.select_related("strategy", "param_set").get(id=signal_id)
        except Signal.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Signal #{signal_id} not found"))
            return

        if signal.status != "pending":
            self.stdout.write(self.style.WARNING(f"Signal #{signal_id} status is '{signal.status}', not 'pending'"))
            return

        self.stdout.write(f"Executing signal #{signal_id}: {signal.get_direction_display()} {signal.symbol}")

        trade = execute_live_trade(
            signal=signal,
            dry_run=dry_run,
        )

        if trade:
            mode = "DRY RUN" if dry_run else "LIVE"
            self.stdout.write(self.style.SUCCESS(
                f"✅ [{mode}] Trade #{trade.id} opened: "
                f"{trade.side.upper()} {trade.symbol} {float(trade.quantity):.6f} @ ${float(trade.entry_price):.2f}"
            ))
        else:
            self.stdout.write(self.style.ERROR("❌ Trade execution failed (check risk limits or connection)"))

    def _flatten_all(self, dry_run: bool, force: bool = False):
        """Emergency flatten all live positions."""
        from trading_bot.services.executor.live import flatten_all_live

        if dry_run:
            self.stdout.write(self.style.WARNING(
                "⚠️  DRY RUN — no real orders will be sent\n"
                "  Use --no-dry-run to actually close positions\n"
            ))

        open_trades = Trade.objects.filter(mode="live", status="open").count()
        self.stdout.write(f"Open live positions: {open_trades}")

        if open_trades == 0:
            self.stdout.write(self.style.WARNING("No open positions to flatten"))
            return

        # Confirm for non-dry-run (only if interactive)
        if not dry_run:
            import sys
            if sys.stdin.isatty():
                self.stdout.write(self.style.WARNING(
                    "\n⚠️  REALLY flatten all positions with REAL orders?"
                ))
                self.stdout.write("  Type 'FLATTEN' to confirm: ")
                confirm = input("  Type 'FLATTEN' to confirm: ")
                if confirm != "FLATTEN":
                    self.stdout.write(self.style.WARNING("Cancelled"))
                    return
            elif not force:
                self.stdout.write(self.style.ERROR(
                    "Non-interactive mode: use --force to flatten without confirmation"
                ))
                return
            else:
                self.stdout.write(self.style.WARNING("  --force: flattening without confirmation"))

        n_closed = flatten_all_live(dry_run=dry_run)
        mode = "DRY RUN" if dry_run else "LIVE"
        self.stdout.write(self.style.SUCCESS(f"✅ [{mode}] Flattened {n_closed} positions"))

    def _promote_param_set(self, param_set_id: Optional[int]):
        """Promote a paper ParamSet to live."""
        if param_set_id is None:
            self.stdout.write(self.style.ERROR("Usage: --promote --param-set-id <id>"))
            # Show available candidates
            candidates = ParamSet.objects.filter(is_candidate=True, is_live=False).select_related("strategy")
            if candidates:
                self.stdout.write("\nAvailable candidates:")
                for p in candidates:
                    sharpe = p.metrics.get("sharpe_ratio", "?")
                    self.stdout.write(f"  #{p.id}: {p.strategy.name} (sharpe={sharpe})")
            return

        try:
            param_set = ParamSet.objects.select_related("strategy").get(id=param_set_id)
        except ParamSet.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"ParamSet #{param_set_id} not found"))
            return

        # Demote any existing live ParamSets for this strategy
        ParamSet.objects.filter(
            strategy=param_set.strategy, is_live=True
        ).exclude(id=param_set.id).update(is_live=False)

        param_set.is_live = True
        param_set.is_candidate = False
        param_set.save(update_fields=["is_live", "is_candidate"])

        AuditLog.objects.create(
            action="param_promoted",
            message=f"Manual promotion: ParamSet #{param_set.id} for {param_set.strategy.name} → LIVE",
            details={"param_set_id": param_set.id, "strategy": param_set.strategy.name},
            severity="info",
        )

        self.stdout.write(self.style.SUCCESS(
            f"✅ Promoted #{param_set.id}: {param_set.strategy.name} → LIVE"
        ))
