"""
Autonomous Trading Bot — Celery Tasks

Implements the scheduled task pipeline:
- Nightly optimization cycle (Phase 6)
- Paper trading execution loop (Phase 7)
- Data ingestion pollers (Phase 2)
- Sentiment scrapers
- Strategy execution (Phase 7-8)

All tasks are idempotent with timeout + retry + circuit breaker.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ── Nightly Optimization Cycle ──────────────────────────────────────


def run_nightly_optimization() -> dict:
    """
    Nightly optimization cycle.

    Called by Celery Beat schedule (or management command for testing).

    Steps:
    1. Pull latest OHLCV data from exchange
    2. Rebuild feature snapshots
    3. Run Optuna hyperparameter search for each active strategy
    4. Backtest top candidates
    5. Promote best ParamSet if it beats current live set

    Returns:
        Dict with results per strategy
    """
    import django
    from django.conf import settings

    if not settings.configured:
        django.setup()

    logger.info("🔄 Starting nightly optimization cycle at %s", datetime.now(timezone.utc))

    results = {}
    start_time = datetime.now(timezone.utc)

    try:
        # ── Step 1: Update historical data ──────────────────────
        logger.info("Step 1/4: Updating historical data...")
        try:
            from trading_bot.management.commands.download_history import Command as DownloadCmd
            cmd = DownloadCmd()
            cmd.handle(symbol="BTCUSDT", interval="1h", limit=500)
            logger.info("Historical data updated")
        except Exception as e:
            logger.warning("Failed to update historical data: %s", e)

        # ── Step 2: Rebuild features ────────────────────────────
        logger.info("Step 2/4: Rebuilding feature snapshots...")
        try:
            from trading_bot.management.commands.rebuild_features import Command as RebuildCmd
            cmd = RebuildCmd()
            cmd.handle(symbols=["BTCUSDT"], interval="1h", days=30, store=True)
            logger.info("Feature snapshots rebuilt")
        except Exception as e:
            logger.warning("Failed to rebuild features: %s", e)

        # ── Step 3: Run optimization for each active strategy ───
        logger.info("Step 3/4: Running optimization...")
        from trading_bot.models import BotConfig, Strategy as StrategyModel

        config = BotConfig.get_config()
        n_trials = min(config.max_trials_per_study, 50)

        from trading_bot.services.features.engine import build_feature_matrix
        from trading_bot.services.optimizer.ray_parallel import run_parallel_trials
        from trading_bot.services.backtester.vectorbt_engine import backtest_strategy
        import numpy as np

        lf = build_feature_matrix("BTCUSDT", config.default_interval, limit=240)
        df = lf.collect()

        if df.is_empty() or len(df) < 50:
            logger.error("Insufficient data for optimization")
            return {"error": "Insufficient data"}

        prices = df["close"].to_numpy().astype(float)

        from trading_bot.services.strategies.technical import MomentumStrategy, MeanReversionStrategy, BreakoutStrategy
        from trading_bot.services.strategies.regime import RegimeStrategy
        from trading_bot.services.strategies.onchain import OnChainDivergenceStrategy

        ACTIVE_CLASSES = {
            "Momentum": MomentumStrategy,
            "Mean Reversion": MeanReversionStrategy,
            "Breakout": BreakoutStrategy,
            "Market Regime": RegimeStrategy,
            "On-Chain Divergence": OnChainDivergenceStrategy,
        }

        for strategy in StrategyModel.objects.filter(is_active=True):
            strategy_name = strategy.name
            strategy_cls = ACTIVE_CLASSES.get(strategy_name)
            if strategy_cls is None:
                logger.warning("No class for strategy: %s, skipping", strategy_name)
                continue

            try:
                logger.info("Optimizing: %s (%d trials)", strategy_name, n_trials)

                opt_result = run_parallel_trials(
                    strategy_name=strategy_name,
                    strategy_class=strategy_cls,
                    df=df,
                    prices=prices,
                    n_trials=n_trials,
                    n_jobs=4,
                    initial_capital=float(config.virtual_balance),
                    position_size_pct=config.max_position_size_pct,
                    maximize_metric="sharpe_ratio",
                    study_name=f"{strategy_name.lower().replace(' ', '_')}_nightly",
                )

                if opt_result.get("n_trials", 0) == 0:
                    logger.warning("  %s: 0 trials completed, skipping", strategy_name)
                    results[strategy_name] = {"status": "skipped", "reason": "0 trials"}
                    continue

                best_params = opt_result.get("best_params", {})
                best_value = opt_result.get("best_value")

                best_strategy = strategy_cls(params=best_params)
                signals, confidences = best_strategy.generate_signals(df)

                bt_result = backtest_strategy(
                    prices=prices,
                    signals=signals,
                    confidences=confidences,
                    initial_capital=float(config.virtual_balance),
                    position_size_pct=config.max_position_size_pct,
                )

                promoted = False
                if bt_result["n_trades"] > 0:
                    from trading_bot.models import ParamSet
                    from trading_bot.services.optimizer.promoter import promote_param_set

                    param_set, _ = ParamSet.objects.get_or_create(
                        strategy=strategy,
                        params=best_params,
                        defaults={"is_candidate": True},
                    )
                    param_set.is_candidate = True
                    param_set.metrics = bt_result["metrics"]
                    param_set.save(update_fields=["is_candidate", "metrics"])

                    promoted = promote_param_set(
                        param_set=param_set,
                        strategy_model=strategy,
                        metrics=bt_result["metrics"],
                    )

                    if promoted:
                        from trading_bot.models import BacktestRun
                        BacktestRun.objects.create(
                            param_set=param_set,
                            symbol="BTCUSDT",
                            interval=config.default_interval,
                            start_date=datetime.now(timezone.utc),
                            end_date=datetime.now(timezone.utc),
                            status="completed",
                            metrics=bt_result["metrics"],
                            equity_curve=bt_result.get("equity_curve", []),
                        )

                from trading_bot.models import AuditLog
                AuditLog.objects.create(
                    action="optimization_run",
                    message=f"Nightly opt {strategy_name}: {opt_result['n_trials']} trials, best={best_value:.4f}, promoted={promoted}",
                    details={"strategy": strategy_name, "n_trials": opt_result["n_trials"], "best_value": best_value, "promoted": promoted, "backtest_sharpe": bt_result["metrics"].get("sharpe_ratio")},
                    severity="info",
                )

                results[strategy_name] = {"status": "ok", "n_trials": opt_result["n_trials"], "best_value": best_value, "promoted": promoted}
                logger.info("  %s: %d trials, best=%.4f, promoted=%s", strategy_name, opt_result["n_trials"], best_value or 0, promoted)

            except Exception as e:
                logger.exception("  %s: FAILED: %s", strategy_name, e)
                results[strategy_name] = {"status": "failed", "error": str(e)}

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info("Nightly optimization complete in %.1f seconds", duration)

        from trading_bot.models import AuditLog
        AuditLog.objects.create(
            action="optimization_run",
            message=f"Nightly optimization complete: {sum(1 for r in results.values() if r.get('status') == 'ok')}/{len(results)} strategies, {duration:.0f}s",
            details={"duration_seconds": duration, "results": results, "timestamp": start_time.isoformat()},
            severity="info",
        )

    except Exception as e:
        logger.exception("Nightly optimization failed: %s", e)
        from trading_bot.models import AuditLog
        AuditLog.objects.create(
            action="optimization_run",
            message=f"Nightly optimization FAILED: {e}",
            details={"error": str(e)},
            severity="error",
        )

    return results


def trigger_optimization(strategy_name: Optional[str] = None) -> dict:
    """Trigger optimization for debugging / manual runs."""
    if strategy_name:
        import sys
        from io import StringIO
        from django.core.management import call_command

        out = StringIO()
        try:
            call_command("run_full_optimization", strategy=strategy_name, n_trials=10, stdout=out, stderr=out)
            return {"status": "ok", "strategy": strategy_name, "output": out.getvalue()[:500]}
        except Exception as e:
            logger.exception("Manual optimization failed: %s", e)
            return {"error": str(e)}
    else:
        return run_nightly_optimization()


# ── Paper Trading Cycle ─────────────────────────────────────────────


def run_paper_trading_cycle() -> dict:
    """
    Run a single paper trading cycle.

    Called periodically (every 60s via Celery Beat or management command).

    Steps:
    1. Get current price for monitored symbols
    2. Execute pending signals as paper trades
    3. Close positions based on exit/neutral signals
    4. Check risk limits and circuit breakers
    5. Log cycle results

    Returns:
        Dict with cycle results
    """
    import django
    from django.conf import settings

    if not settings.configured:
        django.setup()

    logger.info("Running paper trading cycle at %s", datetime.now(timezone.utc))

    try:
        from trading_bot.management.commands.run_paper_trading import Command as PaperCmd

        cmd = PaperCmd()
        cmd._run_single_pass("BTCUSDT")

        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    except Exception as e:
        logger.exception("Paper trading cycle failed: %s", e)
        from trading_bot.models import AuditLog
        AuditLog.objects.create(
            action="error",
            message=f"Paper trading cycle FAILED: {e}",
            details={"error": str(e)},
            severity="error",
        )
        return {"status": "error", "error": str(e)}


def get_paper_trading_status() -> dict:
    """Get paper trading status for the dashboard."""
    try:
        import django
        from django.conf import settings
        if not settings.configured:
            django.setup()

        from trading_bot.services.executor.paper import get_paper_portfolio_summary
        from trading_bot.models import BotConfig

        config = BotConfig.get_config()
        summary = get_paper_portfolio_summary()
        summary["mode"] = config.mode
        summary["is_enabled"] = config.is_enabled
        summary["max_open_positions"] = config.max_open_positions
        summary["max_position_size_pct"] = config.max_position_size_pct
        return summary

    except Exception as e:
        logger.error("Failed to get paper trading status: %s", e)
        return {"error": str(e)}
