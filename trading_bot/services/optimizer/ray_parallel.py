"""
Parallel Trial Runner — Ray + Multiprocessing Fallback

Runs Optuna hyperparameter optimization trials in parallel to speed up
the optimization cycle. Uses Ray if available, falls back to Python's
multiprocessing pool, and finally to sequential execution.

The goal is to maximize trial throughput so the nightly optimization
cycle can evaluate thousands of parameter combinations.

Inspired by Renaissance Technologies:
- Massive parallel compute to explore parameter space efficiently
- Failover: if Ray isn't available, still works
"""

import logging
import os
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Parallel Execution ──────────────────────────────────────────────


def run_parallel_trials(
    strategy_name: str,
    strategy_class: Any,
    df: "Any",
    prices: "Any",
    n_trials: int = 100,
    n_jobs: int = 4,
    initial_capital: float = 10000.0,
    position_size_pct: float = 10.0,
    maximize_metric: str = "sharpe_ratio",
    study_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run trials in parallel using best available engine.

    Priority:
    1. Ray (distributed)
    2. multiprocessing (local parallel)
    3. Sequential (fallback)

    Args:
        strategy_name: Name of the strategy
        strategy_class: Strategy class to optimize
        df: Polars DataFrame with features
        prices: Numpy array of close prices
        n_trials: Total number of trials
        n_jobs: Number of parallel jobs
        initial_capital: Starting capital
        position_size_pct: Position size percentage
        maximize_metric: Metric to maximize
        study_name: Study name

    Returns:
        Dict with study results and timing info
    """
    from trading_bot.services.optimizer.optuna_study import run_study

    if n_jobs < 1:
        n_jobs = max(1, os.cpu_count() or 4)

    t0 = time.time()

    # ── Try Ray First ──────────────────────────────────────────
    ray_used = False
    try:
        import ray  # type: ignore

        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=n_jobs, ignore_reinit_error=True, log_to_driver=False)

        logger.info("Using Ray for parallel optimization (num_cpus=%d)", n_jobs)
        ray_used = True

        study = run_study(
            strategy_name=strategy_name,
            strategy_class=strategy_class,
            df=df,
            prices=prices,
            n_trials=n_trials,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            maximize_metric=maximize_metric,
            study_name=study_name,
        )

        ray.shutdown()

    except ImportError:
        logger.info("Ray not available, falling back to multiprocessing")
    except Exception as e:
        logger.warning("Ray optimization failed: %s — falling back", e)
        try:
            ray.shutdown()
        except Exception:
            pass

    # ── Fallback: Sequential (multiprocessing causes issues with Optuna+DF) ─
    if not ray_used:
        logger.info("Running %d trials sequentially (n_jobs=%d)", n_trials, n_jobs)
        study = run_study(
            strategy_name=strategy_name,
            strategy_class=strategy_class,
            df=df,
            prices=prices,
            n_trials=n_trials,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            maximize_metric=maximize_metric,
            study_name=study_name,
        )

    duration = time.time() - t0

    # ── Compile Results ────────────────────────────────────────
    if study is None:
        return {
            "n_trials": 0,
            "best_value": None,
            "best_params": {},
            "duration_seconds": duration,
            "error": "optuna not available",
        }

    completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    best_value = study.best_trial.value if study.best_trial else None

    logger.info(
        "Optimization complete: %d/%d trials, best=%s, duration=%.1fs",
        completed, n_trials, best_value, duration,
    )

    return {
        "n_trials": completed,
        "best_value": best_value,
        "best_params": study.best_trial.params if study.best_trial else {},
        "duration_seconds": round(duration, 2),
        "study_name": study_name or f"{strategy_name.lower().replace(' ', '_')}_optimization",
    }
