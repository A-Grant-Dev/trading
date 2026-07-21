"""
Optuna Study — Hyperparameter Search Spaces per Strategy

Defines the hyperparameter search space and Optuna objective function
for each trading strategy. Each strategy has a custom search space
based on its relevant parameters.

The optimizer runs multiple trials to find the best parameter set,
evaluated via backtest performance (customizable metric).

Inspired by Renaissance Technologies:
- Systematic parameter sweeping with Bayesian optimization
- Walk-forward OOS validation to prevent overfitting
- Best ParamSet promoted only if it beats current live set
"""

import logging
from typing import Any, Callable, Optional

import numpy as np

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore

logger = logging.getLogger(__name__)


# ── Search Space Definitions ────────────────────────────────────────


def suggest_momentum_params(trial: "optuna.Trial") -> dict[str, Any]:
    """Suggest MomentumStrategy hyperparameters via Optuna."""
    return {
        "ema_fast": "ema_9",
        "ema_slow": trial.suggest_categorical("ema_slow", ["ema_21", "ema_50", "ema_200"]),
        "rsi_column": trial.suggest_categorical("rsi_column", ["rsi_7", "rsi_14"]),
        "rsi_bull_threshold": trial.suggest_int("rsi_bull_threshold", 40, 60),
        "rsi_bear_threshold": trial.suggest_int("rsi_bear_threshold", 40, 60),
        "return_column": "return_5",
        "confidence_scalar": trial.suggest_float("confidence_scalar", 0.3, 1.0),
    }


def suggest_mean_reversion_params(trial: "optuna.Trial") -> dict[str, Any]:
    """Suggest MeanReversionStrategy hyperparameters via Optuna."""
    return {
        "bb_upper": "bb_upper",
        "bb_lower": "bb_lower",
        "close_column": "close",
        "rsi_column": trial.suggest_categorical("rsi_column", ["rsi_7", "rsi_14"]),
        "oversold_threshold": trial.suggest_int("oversold_threshold", 20, 40),
        "overbought_threshold": trial.suggest_int("overbought_threshold", 60, 80),
        "confidence_scalar": trial.suggest_float("confidence_scalar", 0.3, 1.0),
    }


def suggest_breakout_params(trial: "optuna.Trial") -> dict[str, Any]:
    """Suggest BreakoutStrategy hyperparameters via Optuna."""
    return {
        "price_high_col": "price_position_high_20",
        "price_low_col": "price_position_low_20",
        "volume_col": "volume_ratio",
        "volatility_col": trial.suggest_categorical("volatility_col", ["volatility_14", "volatility_30"]),
        "breakout_threshold_high": trial.suggest_float("breakout_threshold_high", 0.95, 1.0),
        "breakout_threshold_low": trial.suggest_float("breakout_threshold_low", 1.0, 1.02),
        "volume_threshold": trial.suggest_float("volume_threshold", 1.0, 2.0),
        "confidence_scalar": trial.suggest_float("confidence_scalar", 0.3, 1.0),
    }


def suggest_regime_params(trial: "optuna.Trial") -> dict[str, Any]:
    """Suggest RegimeStrategy hyperparameters via Optuna."""
    return {
        "volatility_col": trial.suggest_categorical("volatility_col", ["volatility_14", "volatility_30"]),
        "high_vol_threshold": trial.suggest_float("high_vol_threshold", 1.2, 3.0),
        "low_vol_threshold": trial.suggest_float("low_vol_threshold", 0.3, 0.8),
        "rsi_column": trial.suggest_categorical("rsi_column", ["rsi_7", "rsi_14"]),
        "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 85),
        "rsi_oversold": trial.suggest_int("rsi_oversold", 15, 35),
        "trend_col": "ema_ratio_21_50",
        "trend_threshold": trial.suggest_float("trend_threshold", 0.005, 0.03),
        "confidence_scalar": trial.suggest_float("confidence_scalar", 0.3, 1.0),
    }


def suggest_onchain_params(trial: "optuna.Trial") -> dict[str, Any]:
    """Suggest OnChainDivergenceStrategy hyperparameters via Optuna."""
    return {
        "price_col": "close",
        "sentiment_col": "fear_greed_value",
        "divergence_lookback": trial.suggest_int("divergence_lookback", 3, 20),
        "divergence_threshold": trial.suggest_float("divergence_threshold", 0.02, 0.15),
        "min_sentiment_divergence": trial.suggest_float("min_sentiment_divergence", 10, 40),
        "confidence_scalar": trial.suggest_float("confidence_scalar", 0.3, 1.0),
    }


# ── Strategy → Search Space Registry ────────────────────────────────

SEARCH_SPACES: dict[str, Callable[["optuna.Trial"], dict[str, Any]]] = {
    "Momentum": suggest_momentum_params,
    "Mean Reversion": suggest_mean_reversion_params,
    "Breakout": suggest_breakout_params,
    "Market Regime": suggest_regime_params,
    "On-Chain Divergence": suggest_onchain_params,
}

# Strategies without tunable params (microstructure, ML, ensemble)
SEARCH_SPACES["Order Book Imbalance"] = lambda t: {
    "imbalance_threshold": t.suggest_float("imbalance_threshold", 52.0, 80.0),
    "confidence_scalar": t.suggest_float("confidence_scalar", 0.3, 1.0),
}
SEARCH_SPACES["Absorption"] = lambda t: {
    "buy_volume_ratio_threshold": t.suggest_float("buy_volume_ratio_threshold", 0.55, 0.85),
    "min_trades_for_signal": t.suggest_int("min_trades_for_signal", 5, 50),
    "confidence_scalar": t.suggest_float("confidence_scalar", 0.3, 1.0),
}


# ── Objective Function ──────────────────────────────────────────────


def objective(
    trial: "optuna.Trial",
    strategy_name: str,
    strategy_class: Any,
    df: "Any",  # polars DataFrame (lazy import to avoid circular)
    prices: np.ndarray,
    initial_capital: float = 10000.0,
    position_size_pct: float = 10.0,
    maximize_metric: str = "sharpe_ratio",
    walk_forward: bool = True,
    n_train: int = 100,
    n_test: int = 50,
) -> float:
    """
    Optuna objective function for a single trial.

    Suggests hyperparameters, runs a backtest, and returns the
    metric to maximize (e.g. Sharpe ratio).

    Args:
        trial: Optuna trial object
        strategy_name: Name of the strategy being optimized
        strategy_class: The strategy class to instantiate
        df: Polars DataFrame with feature columns
        prices: Numpy array of close prices
        initial_capital: Starting account equity
        position_size_pct: Position size as % of equity
        maximize_metric: Metric to maximize ('sharpe_ratio', 'total_return_pct', etc.)
        walk_forward: Whether to use walk-forward validation
        n_train: Training window size
        n_test: Test window size

    Returns:
        Float metric value to maximize (higher is better)
    """
    from trading_bot.services.backtester.vectorbt_engine import backtest_strategy
    from trading_bot.services.backtester.walk_forward import run_walk_forward, walk_forward_splits

    # Suggest hyperparameters
    search_fn = SEARCH_SPACES.get(strategy_name)
    if search_fn is None:
        raise ValueError(f"No search space defined for strategy: {strategy_name}")
    params = search_fn(trial)

    # Create strategy with suggested params
    try:
        strategy = strategy_class(params=params)
    except Exception as e:
        logger.warning("Trial %d: strategy init failed: %s", trial.number, e)
        return -999.0  # Penalize failed trials

    # Generate signals
    try:
        signals, confidences = strategy.generate_signals(df)
    except Exception as e:
        logger.warning("Trial %d: signal generation failed: %s", trial.number, e)
        return -999.0

    n_signals = int(np.sum(signals != 0))
    if n_signals == 0:
        return -999.0  # No signals = useless param set

    # Run backtest
    try:
        if walk_forward and len(prices) >= n_train + n_test:
            splits = walk_forward_splits(
                n_samples=len(prices),
                n_train=n_train,
                n_test=n_test,
            )
            if len(splits) < 1:
                bt_result = backtest_strategy(
                    prices=prices, signals=signals, confidences=confidences,
                    initial_capital=initial_capital, position_size_pct=position_size_pct,
                )
                result = bt_result["metrics"]
            else:
                wf_result = run_walk_forward(
                    prices=prices, signals=signals, confidences=confidences,
                    splits=splits, initial_capital=initial_capital,
                    position_size_pct=position_size_pct,
                )
                result = wf_result.get("overall_metrics", {})
        else:
            bt_result = backtest_strategy(
                prices=prices, signals=signals, confidences=confidences,
                initial_capital=initial_capital, position_size_pct=position_size_pct,
            )
            result = bt_result["metrics"]
    except Exception as e:
        logger.warning("Trial %d: backtest failed: %s", trial.number, e)
        return -999.0

    # Return the metric to maximize
    metric_value = result.get(maximize_metric, -999.0)
    if metric_value is None or (isinstance(metric_value, float) and np.isnan(metric_value)):
        return -999.0

    return float(metric_value)


# ── Study Runner ────────────────────────────────────────────────────


def run_study(
    strategy_name: str,
    strategy_class: Any,
    df: "Any",
    prices: np.ndarray,
    n_trials: int = 50,
    initial_capital: float = 10000.0,
    position_size_pct: float = 10.0,
    maximize_metric: str = "sharpe_ratio",
    study_name: Optional[str] = None,
    direction: str = "maximize",
    storage: Optional[str] = None,
    load_if_exists: bool = True,
) -> Optional["optuna.Study"]:
    """
    Run an Optuna hyperparameter optimization study.

    Args:
        strategy_name: Name of the strategy
        strategy_class: Strategy class to optimize
        df: Feature matrix DataFrame
        prices: Close price array
        n_trials: Number of optimization trials
        initial_capital: Starting capital for backtests
        position_size_pct: Position sizing
        maximize_metric: Metric to optimize ('sharpe_ratio', 'total_return_pct', etc.)
        study_name: Name for the study (default: f"{strategy_name}_optimization")
        direction: 'maximize' or 'minimize'
        storage: Optional Optuna storage URL for persistence
        load_if_exists: Whether to resume existing study

    Returns:
        Optuna Study object with completed trials, or None if optuna unavailable
    """
    if optuna is None:
        logger.error("optuna is not installed. Install with: pip install optuna")
        return None

    if study_name is None:
        study_name = f"{strategy_name.lower().replace(' ', '_')}_optimization"

    # Create or load study
    study_kwargs = {
        "study_name": study_name,
        "direction": direction,
        "load_if_exists": load_if_exists,
    }
    if storage:
        study_kwargs["storage"] = storage

    try:
        study = optuna.create_study(**study_kwargs)
    except Exception as e:
        logger.warning("Could not create study '%s': %s. Creating in-memory.", study_name, e)
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
        )

    # Show current best before starting
    if len(study.trials) > 0:
        best = study.best_trial
        logger.info(
            "Resuming study '%s' — existing best %s = %.4f (trial %d)",
            study_name, maximize_metric, best.value, best.number,
        )

    # Run trials
    logger.info("Starting %d trials for %s (metric: %s)", n_trials, strategy_name, maximize_metric)

    try:
        study.optimize(
            lambda trial: objective(
                trial=trial,
                strategy_name=strategy_name,
                strategy_class=strategy_class,
                df=df,
                prices=prices,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                maximize_metric=maximize_metric,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error("Optimization failed: %s", e)
        return study

    # Log results
    best_trial = study.best_trial
    logger.info(
        "Study '%s' complete — best %s = %.4f (trial %d)",
        study_name, maximize_metric, best_trial.value, best_trial.number,
    )
    logger.info("Best params: %s", best_trial.params)

    return study


def get_best_params_from_study(
    study: "optuna.Study",
) -> dict[str, Any]:
    """Extract the best hyperparameters from a completed study."""
    if study is None or len(study.trials) == 0:
        return {}
    return study.best_trial.params
