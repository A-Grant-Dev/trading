"""
Management command to run full backtests on strategies.

Uses the vectorized backtesting engine with:
- Strategy signal generation from feature matrix
- Walk-forward analysis with configurable splits
- Full metrics suite
- Persistence to BacktestRun model

Usage:
    # Backtest Momentum strategy on BTCUSDT
    python manage.py run_backtest --strategy Momentum --symbol BTCUSDT --interval 1h --limit 240

    # Backtest with walk-forward analysis
    python manage.py run_backtest --strategy Momentum --symbol BTCUSDT --walk-forward --n-train 100 --n-test 50

    # Backtest all active strategies
    python manage.py run_backtest --all

    # Show stored backtest results
    python manage.py run_backtest --status
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np
from django.core.management.base import BaseCommand

from trading_bot.models import BacktestRun, ParamSet, Strategy as StrategyModel
from trading_bot.services.backtester.vectorbt_engine import backtest_strategy
from trading_bot.services.backtester.walk_forward import (
    combinatorial_purged_cv_splits,
    run_walk_forward,
    walk_forward_splits,
)
from trading_bot.services.features.engine import build_feature_matrix

logger = logging.getLogger(__name__)

# ── Strategy Registry ──────────────────────────────────────────────

STRATEGY_CLASSES = {}

try:
    from trading_bot.services.strategies.technical import MomentumStrategy, MeanReversionStrategy, BreakoutStrategy
    from trading_bot.services.strategies.microstructure import OrderBookImbalanceStrategy, AbsorptionStrategy
    from trading_bot.services.strategies.regime import RegimeStrategy
    from trading_bot.services.strategies.onchain import OnChainDivergenceStrategy
    from trading_bot.services.strategies.ml_strategy import MLStrategy
    from trading_bot.services.strategies.ensemble import EnsembleStrategy

    STRATEGY_CLASSES = {
        "Momentum": MomentumStrategy,
        "Mean Reversion": MeanReversionStrategy,
        "Breakout": BreakoutStrategy,
        "Order Book Imbalance": OrderBookImbalanceStrategy,
        "Absorption": AbsorptionStrategy,
        "Market Regime": RegimeStrategy,
        "On-Chain Divergence": OnChainDivergenceStrategy,
        "ML Strategy": MLStrategy,
    }
except Exception as e:
    logger.warning("Failed to import strategies: %s", e)


class Command(BaseCommand):
    help = "Run full backtests on trading strategies"

    def add_arguments(self, parser):
        parser.add_argument(
            "--strategy",
            type=str,
            default=None,
            help="Strategy name to backtest (default: first active strategy)",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            default=False,
            help="Backtest all active strategies",
        )
        parser.add_argument(
            "--symbol",
            type=str,
            default="BTCUSDT",
            help="Trading pair (default: BTCUSDT)",
        )
        parser.add_argument(
            "--interval",
            type=str,
            default="1h",
            choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            help="Candle interval (default: 1h)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=240,
            help="Number of candles to use (default: 240)",
        )
        parser.add_argument(
            "--walk-forward",
            action="store_true",
            default=False,
            help="Use walk-forward analysis instead of single backtest",
        )
        parser.add_argument(
            "--n-train",
            type=int,
            default=100,
            help="Training window size for walk-forward (default: 100)",
        )
        parser.add_argument(
            "--n-test",
            type=int,
            default=50,
            help="Test window size for walk-forward (default: 50)",
        )
        parser.add_argument(
            "--initial-capital",
            type=float,
            default=10000.0,
            help="Starting capital (default: 10000)",
        )
        parser.add_argument(
            "--position-size",
            type=float,
            default=10.0,
            help="Position size as %% of capital (default: 10%%)",
        )
        parser.add_argument(
            "--store",
            action="store_true",
            default=True,
            help="Store results in BacktestRun model (default: True)",
        )
        parser.add_argument(
            "--no-store",
            action="store_false",
            dest="store",
            help="Dry run — print results without storing",
        )
        parser.add_argument(
            "--status",
            action="store_true",
            default=False,
            help="Show stored backtest results and exit",
        )

    def handle(self, *args, **options):
        if options["status"]:
            self._show_status()
            return

        symbol = options["symbol"].upper()
        interval = options["interval"]
        limit = options["limit"]
        walk_forward = options["walk_forward"]
        store = options["store"]
        initial_capital = options["initial_capital"]
        position_size = options["position_size"]

        # Determine which strategies to backtest
        strategies_to_test = []
        if options["all"]:
            strategies_to_test = [
                (s.name, s.strategy_class)
                for s in StrategyModel.objects.filter(is_active=True)
            ]
        elif options["strategy"]:
            name = options["strategy"]
            strategies_to_test = [(name, name)]
        else:
            # First active strategy
            first = StrategyModel.objects.filter(is_active=True).first()
            if first:
                strategies_to_test = [(first.name, first.strategy_class)]

        if not strategies_to_test:
            self.stdout.write(self.style.WARNING("No strategies found to backtest"))
            return

        self.stdout.write(self.style.SUCCESS(
            f"\n🧪 Backtest Engine\n"
            f"{'='*60}\n"
            f"  Symbol:   {symbol}\n"
            f"  Interval: {interval}\n"
            f"  Candles:  {limit}\n"
            f"  Capital:  ${initial_capital:,.2f}\n"
            f"  Position: {position_size}%\n"
            f"  Walk-Fwd: {walk_forward}\n"
            f"  Store:    {store}\n"
            f"{'='*60}\n"
        ))

        # Build feature matrix once
        self.stdout.write("Building feature matrix...")
        self.stdout.flush()
        lf = build_feature_matrix(symbol, interval, limit=limit)
        df = lf.collect()

        if df.is_empty():
            self.stdout.write(self.style.ERROR("No data available"))
            return

        prices = df["close"].to_numpy().astype(float)
        self.stdout.write(f"Feature matrix: {len(df)} rows x {len(df.columns)} cols\n")

        for strategy_name, strategy_class_path in strategies_to_test:
            self._run_single_backtest(
                strategy_name=strategy_name,
                strategy_class_path=strategy_class_path,
                df=df,
                prices=prices,
                symbol=symbol,
                interval=interval,
                initial_capital=initial_capital,
                position_size=position_size,
                walk_forward=walk_forward,
                n_train=options["n_train"],
                n_test=options["n_test"],
                store=store,
            )

    def _run_single_backtest(
        self,
        strategy_name: str,
        strategy_class_path: str,
        df,
        prices: np.ndarray,
        symbol: str,
        interval: str,
        initial_capital: float,
        position_size: float,
        walk_forward: bool,
        n_train: int,
        n_test: int,
        store: bool,
    ):
        """Run a single backtest for a strategy."""
        self.stdout.write(f"\n{'─'*60}")
        self.stdout.write(f"  Strategy: {strategy_name}")
        self.stdout.write(f"{'─'*60}")

        # Create strategy instance
        strategy_cls = STRATEGY_CLASSES.get(strategy_name)
        if strategy_cls is None:
            self.stdout.write(self.style.WARNING(f"  ⚠️  No class registered for '{strategy_name}', skipping"))
            return

        try:
            strategy = strategy_cls()
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  ❌ Failed to create strategy: {e}"))
            return

        # Generate signals
        try:
            signals, confidences = strategy.generate_signals(df)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  ❌ Signal generation failed: {e}"))
            return

        n_long = int(np.sum(signals == 1))
        n_short = int(np.sum(signals == -1))
        self.stdout.write(f"  Signals: {n_long} long, {n_short} short, "
                          f"{len(signals) - n_long - n_short} neutral")

        # Run backtest
        t0 = time.time()

        if walk_forward:
            splits = walk_forward_splits(
                n_samples=len(prices),
                n_train=n_train,
                n_test=n_test,
            )
            self.stdout.write(f"  Walk-forward folds: {len(splits)}")

            wf_result = run_walk_forward(
                prices=prices,
                signals=signals,
                confidences=confidences,
                splits=splits,
                initial_capital=initial_capital,
                position_size_pct=position_size,
            )
            result = wf_result["overall_metrics"]
            bt_result = {"n_trades": wf_result["total_trades"], "equity_curve": []}
            n_trades = wf_result["total_trades"]

            self.stdout.write(f"  Folds completed: {wf_result['n_folds']}/{wf_result['n_splits']}")
            for fold in wf_result.get("fold_results", []):
                fm = fold.get("metrics", {})
                self.stdout.write(
                    f"    Fold {fold['fold']}: {fold['n_trades']} trades, "
                    f"sharpe={fm.get('sharpe_ratio', 0):.2f}, "
                    f"return={fm.get('total_return_pct', 0):.1f}%"
                )
        else:
            bt_result = backtest_strategy(
                prices=prices,
                signals=signals,
                confidences=confidences,
                initial_capital=initial_capital,
                position_size_pct=position_size,
            )
            result = bt_result["metrics"]
            n_trades = bt_result["n_trades"]

        duration = time.time() - t0

        # Print results
        self.stdout.write("")
        self._print_metrics(result)

        self.stdout.write(f"\n  ⏱️  Duration: {duration:.2f}s")

        # Store to DB
        if store and result:
            try:
                strategy_model = StrategyModel.objects.filter(name=strategy_name).first()
                if strategy_model:
                    param_set, _ = ParamSet.objects.get_or_create(
                        strategy=strategy_model,
                        params=strategy.default_params,
                        defaults={"is_candidate": True},
                    )
                    BacktestRun.objects.create(
                        param_set=param_set,
                        symbol=symbol,
                        interval=interval,
                        start_date=datetime.now(timezone.utc),
                        end_date=datetime.now(timezone.utc),
                        status="completed",
                        metrics=result,
                        equity_curve=bt_result.get("equity_curve", []),
                        duration_seconds=duration,
                    )
                    self.stdout.write(self.style.SUCCESS("  ✅ Stored to database"))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"  ⚠️  Failed to store: {e}"))

    def _print_metrics(self, metrics: dict):
        """Pretty-print backtest metrics."""
        if not metrics:
            self.stdout.write(self.style.WARNING("  No metrics computed"))
            return

        lines = [
            ("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%"),
            ("Ann. Return", f"{metrics.get('annualized_return_pct', 0):.2f}%"),
            ("Ann. Volatility", f"{metrics.get('annualized_volatility_pct', 0):.2f}%"),
            ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.4f}"),
            ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.4f}"),
            ("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.4f}"),
            ("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
            ("Win Rate", f"{metrics.get('win_rate', 0):.1f}%"),
            ("Profit Factor", f"{metrics.get('profit_factor', 0):.4f}"),
            ("Expectancy", f"{metrics.get('expectancy', 0):.4f}"),
            ("Total Trades", f"{metrics.get('total_trades', 0):.0f}"),
        ]

        for label, value in lines:
            self.stdout.write(f"    {label:<20} {value}")

    def _show_status(self):
        """Display stored backtest results."""
        backtests = BacktestRun.objects.select_related(
            "param_set__strategy"
        ).order_by("-created_at")[:20]

        total = BacktestRun.objects.count()
        self.stdout.write(self.style.SUCCESS(
            f"\n📊 Backtest Results ({total} total)\n"
            f"{'='*60}"
        ))

        if not backtests:
            self.stdout.write(self.style.WARNING("  No backtests stored yet"))
            self.stdout.write("  Run: python manage.py run_backtest --strategy Momentum")
            self.stdout.write("")
            return

        for bt in backtests:
            strat_name = bt.param_set.strategy.name if bt.param_set else "?"
            sharpe = bt.metrics.get("sharpe_ratio", "?")
            win_rate = bt.metrics.get("win_rate", "?")
            ret = bt.metrics.get("total_return_pct", "?")
            trades = bt.metrics.get("total_trades", "?")

            self.stdout.write(
                f"  {strat_name:<20} {bt.symbol:<10} "
                f"sharpe={sharpe}  win={win_rate}%  "
                f"ret={ret}%  trades={trades}  "
                f"({bt.created_at.strftime('%m/%d %H:%M')})"
            )
        self.stdout.write("")
