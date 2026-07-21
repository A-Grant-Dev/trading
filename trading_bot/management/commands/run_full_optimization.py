"""
Management command to run the full optimization cycle.

Orchestrates the self-improvement loop:
1. Builds feature matrix
2. Runs Optuna hyperparameter search
3. Backtests top candidates
4. Promotes best ParamSet if it beats current live set

Usage:
    # Full optimization cycle for Momentum strategy
    python manage.py run_full_optimization --strategy Momentum --symbol BTCUSDT --n-trials 50

    # Optimize all active strategies
    python manage.py run_full_optimization --all --n-trials 30

    # Show optimization status and leaderboard
    python manage.py run_full_optimization --status

    # Quick test with minimal trials
    python manage.py run_full_optimization --strategy Momentum --symbol BTCUSDT --n-trials 5 --no-store
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from django.core.management.base import BaseCommand

from trading_bot.models import AuditLog, BacktestRun, BotConfig, ParamSet, Strategy as StrategyModel

logger = logging.getLogger(__name__)

# ── Strategy Registry ──────────────────────────────────────────────

STRATEGY_CLASSES: dict[str, Any] = {}

try:
    from trading_bot.services.strategies.technical import (
        BreakoutStrategy,
        MeanReversionStrategy,
        MomentumStrategy,
    )
    from trading_bot.services.strategies.microstructure import (
        AbsorptionStrategy,
        OrderBookImbalanceStrategy,
    )
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
    help = "Run full optimization cycle — hyperparameter search, backtest, and promotion"

    def add_arguments(self, parser):
        parser.add_argument(
            "--strategy",
            type=str,
            default=None,
            help="Strategy name to optimize (default: first active strategy)",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            default=False,
            help="Optimize all strategies (one at a time)",
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
            help="Number of candles (default: 240)",
        )
        parser.add_argument(
            "--n-trials",
            type=int,
            default=50,
            help="Number of Optuna trials per strategy (default: 50)",
        )
        parser.add_argument(
            "--n-jobs",
            type=int,
            default=4,
            help="Number of parallel jobs (default: 4)",
        )
        parser.add_argument(
            "--maximize",
            type=str,
            default="sharpe_ratio",
            choices=["sharpe_ratio", "sortino_ratio", "total_return_pct", "profit_factor"],
            help="Metric to maximize (default: sharpe_ratio)",
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
            help="Store results in DB (default: True)",
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
            help="Show optimization status and exit",
        )

    def handle(self, *args, **options):
        if options["status"]:
            self._show_status()
            return

        symbol = options["symbol"].upper()
        interval = options["interval"]
        limit = options["limit"]
        n_trials = options["n_trials"]
        n_jobs = options["n_jobs"]
        maximize = options["maximize"]
        store = options["store"]
        initial_capital = options["initial_capital"]
        position_size = options["position_size"]

        # Determine which strategies to optimize
        strategies_to_run: list[tuple[StrategyModel, Any]] = []
        if options["all"]:
            for s in StrategyModel.objects.filter(is_active=True):
                cls = STRATEGY_CLASSES.get(s.name)
                if cls:
                    strategies_to_run.append((s, cls))
        elif options["strategy"]:
            name = options["strategy"]
            cls = STRATEGY_CLASSES.get(name)
            strategy_model = StrategyModel.objects.filter(name=name).first()
            if cls and strategy_model:
                strategies_to_run.append((strategy_model, cls))
        else:
            first = StrategyModel.objects.filter(is_active=True).first()
            if first:
                cls = STRATEGY_CLASSES.get(first.name)
                if cls:
                    strategies_to_run.append((first, cls))

        if not strategies_to_run:
            self.stdout.write(self.style.WARNING("No strategies found to optimize"))
            self.stdout.write("  Strategy options: " + ", ".join(STRATEGY_CLASSES.keys()))
            self.stdout.write("  Or run: python manage.py run_full_optimization --status")
            return

        self.stdout.write(self.style.SUCCESS(
            f"\n🔬 Full Optimization Cycle\n"
            f"{'='*60}\n"
            f"  Symbol:    {symbol}\n"
            f"  Interval:  {interval}\n"
            f"  Candles:   {limit}\n"
            f"  Trials:    {n_trials} per strategy\n"
            f"  Parallel:  {n_jobs} jobs\n"
            f"  Maximize:  {maximize}\n"
            f"  Capital:   ${initial_capital:,.2f}\n"
            f"  Position:  {position_size}%\n"
            f"  Store:     {store}\n"
            f"{'='*60}\n"
        ))

        # Build feature matrix once
        self.stdout.write("\nBuilding feature matrix...")
        self.stdout.flush()
        from trading_bot.services.features.engine import build_feature_matrix

        lf = build_feature_matrix(symbol, interval, limit=limit)
        df = lf.collect()

        if df.is_empty() or len(df) < 50:
            self.stdout.write(self.style.ERROR("Insufficient data for optimization"))
            return

        prices = df["close"].to_numpy().astype(float)
        self.stdout.write(f"Feature matrix: {len(df)} rows x {len(df.columns)} cols\n")

        # ── Run Optimization Loop ───────────────────────────────
        t0_total = time.time()
        all_results = []

        for strategy_model, strategy_cls in strategies_to_run:
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write(f"  Optimizing: {strategy_model.name}")
            self.stdout.write(f"{'─'*60}")

            try:
                result = self._optimize_strategy(
                    strategy_model=strategy_model,
                    strategy_cls=strategy_cls,
                    df=df,
                    prices=prices,
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    maximize=maximize,
                    initial_capital=initial_capital,
                    position_size=position_size,
                    symbol=symbol,
                    interval=interval,
                    store=store,
                )
                all_results.append(result)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  ❌ Optimization failed: {e}"))
                logger.exception("Optimization failed for %s", strategy_model.name)

        duration_total = time.time() - t0_total

        # ── Summary ─────────────────────────────────────────────
        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"  📊 Optimization Summary\n"
            f"{'='*60}"
        ))
        for r in all_results:
            strategy = r.get("strategy", "?")
            trials = r.get("n_trials", 0)
            best = r.get("best_value", None)
            promoted = r.get("promoted", False)
            bt_sharpe = r.get("backtest_sharpe", None)

            self.stdout.write(
                f"  {strategy:<25} trials={trials:<4} "
                f"best={str(best or '?'):<10} "
                f"bt_sharpe={bt_sharpe or '?'} "
                f"{'✅ PROMOTED' if promoted else '⏸️  not promoted'}"
            )

        self.stdout.write(f"\n  Total duration: {duration_total:.1f}s")
        self.stdout.write(f"{'='*60}\n")

    def _optimize_strategy(
        self,
        strategy_model: StrategyModel,
        strategy_cls: Any,
        df: Any,
        prices: np.ndarray,
        n_trials: int,
        n_jobs: int,
        maximize: str,
        initial_capital: float,
        position_size: float,
        symbol: str,
        interval: str,
        store: bool,
    ) -> dict[str, Any]:
        """Run full optimization for a single strategy."""
        from trading_bot.services.optimizer.ray_parallel import run_parallel_trials
        from trading_bot.services.backtester.vectorbt_engine import backtest_strategy

        t0 = time.time()

        # ── Step 1: Hyperparameter Search ──────────────────────
        self.stdout.write(f"  Step 1/3: Running {n_trials} Optuna trials...")
        self.stdout.flush()

        result = run_parallel_trials(
            strategy_name=strategy_model.name,
            strategy_class=strategy_cls,
            df=df,
            prices=prices,
            n_trials=n_trials,
            n_jobs=n_jobs,
            initial_capital=initial_capital,
            position_size_pct=position_size,
            maximize_metric=maximize,
            study_name=f"{strategy_model.name.lower().replace(' ', '_')}_optimization",
        )

        if result.get("n_trials", 0) == 0:
            return {
                "strategy": strategy_model.name,
                "n_trials": 0,
                "error": "No trials completed",
            }

        best_params = result.get("best_params", {})
        best_value = result.get("best_value")
        trial_duration = result.get("duration_seconds", 0)

        self.stdout.write(
            f"  ✓ {result['n_trials']} trials in {trial_duration:.1f}s, "
            f"best {maximize}={best_value:.4f}"
        )

        # ── Step 2: Backtest Best Params ───────────────────────
        self.stdout.write("  Step 2/3: Backtesting best parameter set...")
        self.stdout.flush()

        # Create strategy with best params
        best_strategy = strategy_cls(params=best_params)
        signals, confidences = best_strategy.generate_signals(df)
        n_signals = int(np.sum(signals != 0))

        bt_result = backtest_strategy(
            prices=prices,
            signals=signals,
            confidences=confidences,
            initial_capital=initial_capital,
            position_size_pct=position_size,
        )

        backtest_metrics = bt_result["metrics"]
        bt_sharpe = backtest_metrics.get("sharpe_ratio", 0)

        self.stdout.write(
            f"  ✓ Backtest: sharpe={bt_sharpe:.4f}, "
            f"return={backtest_metrics.get('total_return_pct', 0):.1f}%, "
            f"trades={bt_result['n_trades']}"
        )

        # ── Step 3: Promote (if beats current) ─────────────────
        promoted = False
        if store and bt_result["n_trades"] > 0:
            self.stdout.write("  Step 3/3: Evaluating promotion...")
            self.stdout.flush()

            # Create or update ParamSet
            param_set, created = ParamSet.objects.get_or_create(
                strategy=strategy_model,
                params=best_params,
                defaults={"is_candidate": True},
            )

            if not created:
                param_set.is_candidate = True
                param_set.metrics = backtest_metrics
                param_set.save(update_fields=["is_candidate", "metrics"])

            # Log optimization run
            AuditLog.objects.create(
                action="optimization_run",
                message=(
                    f"Optimization for {strategy_model.name}: "
                    f"{result['n_trials']} trials, best {maximize}={best_value:.4f}"
                ),
                details={
                    "strategy": strategy_model.name,
                    "n_trials": result["n_trials"],
                    "maximize_metric": maximize,
                    "best_value": best_value,
                    "best_params": best_params,
                    "backtest_sharpe": bt_sharpe,
                },
                severity="info",
            )

            # Attempt promotion
            from trading_bot.services.optimizer.promoter import promote_param_set

            promoted = promote_param_set(
                param_set=param_set,
                strategy_model=strategy_model,
                metrics=backtest_metrics,
            )

            if promoted:
                self.stdout.write(self.style.SUCCESS("  ✅ PROMOTED to live!"))

                # Store backtest result
                BacktestRun.objects.create(
                    param_set=param_set,
                    symbol=symbol,
                    interval=interval,
                    start_date=datetime.now(timezone.utc),
                    end_date=datetime.now(timezone.utc),
                    status="completed",
                    metrics=backtest_metrics,
                    equity_curve=bt_result.get("equity_curve", []),
                    duration_seconds=time.time() - t0,
                )
            else:
                self.stdout.write("  ⏸️  Met promotion criteria (sharpe not above live threshold)")
        else:
            self.stdout.write("  ⏸️  Skipping promotion (no-store or 0 trades)")

        duration = time.time() - t0
        self.stdout.write(f"  ⏱️  Duration: {duration:.1f}s\n")

        return {
            "strategy": strategy_model.name,
            "n_trials": result.get("n_trials", 0),
            "best_value": best_value,
            "best_params": best_params,
            "backtest_sharpe": bt_sharpe,
            "backtest_trades": bt_result["n_trades"],
            "promoted": promoted,
            "duration_seconds": round(duration, 2),
        }

    def _show_status(self):
        """Display optimization status and leaderboard."""
        from trading_bot.models import ParamSet

        total_studies = ParamSet.objects.values("strategy__name").distinct().count()
        total_params = ParamSet.objects.count()
        live_params = ParamSet.objects.filter(is_live=True).select_related("strategy")
        candidate_params = ParamSet.objects.filter(
            is_candidate=True, is_live=False
        ).select_related("strategy").order_by("-metrics__sharpe_ratio")[:20]

        self.stdout.write(self.style.SUCCESS(
            f"\n📊 Optimization Status\n"
            f"{'='*60}\n"
            f"  Total ParamSets:  {total_params}\n"
            f"  Strategies w/ data: {total_studies}\n"
            f"  Live ParamSets:   {live_params.count()}\n"
            f"  Candidates:       {candidate_params.count()}\n"
            f"{'='*60}\n"
        ))

        if live_params:
            self.stdout.write(self.style.SUCCESS("  🏆 LIVE ParamSets:"))
            for p in live_params:
                sharpe = p.metrics.get("sharpe_ratio", "?")
                win_rate = p.metrics.get("win_rate", "?")
                trades = p.metrics.get("total_trades", "?")
                self.stdout.write(
                    f"    {p.strategy.name:<25} sharpe={sharpe}  "
                    f"win={win_rate}%  trades={trades}  "
                    f"({p.created_at.strftime('%m/%d')})"
                )
            self.stdout.write("")

        if candidate_params:
            self.stdout.write(self.style.WARNING("  📋 CANDIDATE ParamSets (by Sharpe):"))
            for p in candidate_params:
                sharpe = p.metrics.get("sharpe_ratio", "?")
                self.stdout.write(
                    f"    {p.strategy.name:<25} sharpe={sharpe}  "
                    f"({p.created_at.strftime('%m/%d')})"
                )
            self.stdout.write("")
        else:
            self.stdout.write(self.style.WARNING("  No candidate ParamSets yet"))
            self.stdout.write("  Run: python manage.py run_full_optimization --strategy Momentum --n-trials 20")
            self.stdout.write("")

        # Recent optimization audits
        recent_optimizations = AuditLog.objects.filter(
            action="optimization_run"
        ).order_by("-timestamp")[:5]

        if recent_optimizations:
            self.stdout.write("  Recent optimizations:")
            for a in recent_optimizations:
                self.stdout.write(f"    {a.timestamp.strftime('%m/%d %H:%M')} — {a.message[:80]}")
            self.stdout.write("")
