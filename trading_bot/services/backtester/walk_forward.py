"""
Walk-Forward Analysis & Combinatorial Purged Cross-Validation

Implements robust out-of-sample testing methodologies to prevent
overfitting and data leakage.

Methods:
1. Walk-Forward Analysis: Train on expanding/rolling window, test on next period
2. Combinatorial Purged CV: Multiple train/test splits with purge gaps
   to prevent leakage from overlapping data

Inspired by Renaissance Technologies' rigorous validation:
- Never test on data that leaked into the training set
- Walk-forward mimics live trading conditions
- Purged CV provides robust confidence intervals
"""

import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Split Generators ───────────────────────────────────────────────


def walk_forward_splits(
    n_samples: int,
    n_train: int = 100,
    n_test: int = 50,
    step: Optional[int] = None,
) -> list[tuple[int, int, int, int]]:
    """
    Generate expanding walk-forward train/test splits.

    Each split: train on [train_start:train_end], test on [test_start:test_end]
    The training window expands by `step` after each fold.

    Args:
        n_samples: Total number of samples
        n_train: Initial number of training samples
        n_test: Number of test samples per fold
        step: How many steps to advance each fold (default: n_test)

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    if step is None:
        step = n_test

    splits = []
    split_start = n_train

    while split_start + n_test <= n_samples:
        train_end = split_start
        test_start = split_start
        test_end = min(test_start + n_test, n_samples)
        splits.append((0, train_end, test_start, test_end))
        split_start += step if step else n_test

    return splits


def rolling_window_splits(
    n_samples: int,
    window_size: int = 150,
    n_test: int = 50,
    step: Optional[int] = None,
) -> list[tuple[int, int, int, int]]:
    """
    Generate rolling window train/test splits.

    Unlike walk-forward, the training window is fixed-size and slides forward.

    Args:
        n_samples: Total number of samples
        window_size: Size of the training window (fixed)
        n_test: Number of test samples per fold
        step: How many steps to advance (default: n_test)

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    if step is None:
        step = n_test

    splits = []
    train_start = 0

    while train_start + window_size + n_test <= n_samples:
        train_end = train_start + window_size
        test_start = train_end
        test_end = min(test_start + step, n_samples)

        splits.append((train_start, train_end, test_start, test_end))
        train_start += step

    return splits


def combinatorial_purged_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    purge_pct: float = 0.1,
) -> list[tuple[int, int, int, int]]:
    """
    Generate Combinatorial Purged Cross-Validation splits.

    Divides data into `n_splits` groups. For each fold, one group is held out
    for testing and the rest for training, with a purge gap between train and
    test to prevent leakage from overlapping labels.

    Args:
        n_samples: Total number of samples
        n_splits: Number of CV folds
        purge_pct: Fraction of data to purge between train/test (0.0-0.5)

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    fold_size = n_samples // n_splits
    purge_size = int(n_samples * purge_pct)

    splits = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n_samples)

        # Training: everything outside the test fold
        # With purge gap before the test window
        train_end = max(0, test_start - purge_size)
        train_start = 0

        if train_end - train_start < fold_size * 0.5:
            # Skip folds with insufficient training data
            continue

        splits.append((train_start, train_end, test_start, test_end))

    return splits


# ── Walk-Forward Runner ────────────────────────────────────────────


def run_walk_forward(
    prices: np.ndarray,
    signals: np.ndarray,
    confidences: np.ndarray,
    splits: list[tuple[int, int, int, int]],
    initial_capital: float = 10000.0,
    position_size_pct: float = 10.0,
    fee_pct: float = 0.001,
) -> dict[str, Any]:
    """
    Run a walk-forward backtest across multiple train/test splits.

    For each split:
    - Trains on the training portion (in future, this will train ML models)
    - Tests on the testing portion using the strategy signals
    - Computes metrics for each fold

    Args:
        prices: Array of close prices
        signals: Array of signal values (+1, 0, -1)
        confidences: Array of confidence values (0.0-1.0)
        splits: List of (train_start, train_end, test_start, test_end)
        initial_capital: Starting account equity
        position_size_pct: Position size as % of equity per trade
        fee_pct: Trading fee as % of trade value

    Returns:
        Dict with fold_results (list), overall_metrics, equity_curves
    """
    from trading_bot.services.backtester.vectorbt_engine import backtest_strategy
    from trading_bot.services.backtester.metrics import compute_full_metrics

    fold_results = []
    all_trades = []
    all_equity_curves = []

    for fold_idx, (tr_s, tr_e, ts_s, ts_e) in enumerate(splits):
        # Extract test portion signals/prices
        test_prices = prices[ts_s:ts_e]
        test_signals = signals[ts_s:ts_e]
        test_conf = confidences[ts_s:ts_e]

        if len(test_prices) < 10:
            continue

        # Run backtest on the test portion
        bt_result = backtest_strategy(
            prices=test_prices,
            signals=test_signals,
            confidences=test_conf,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            fee_pct=fee_pct,
        )

        # Compute per-fold metrics
        fold_metrics = compute_full_metrics(
            equity_curve=np.array(bt_result["equity_curve"]),
            returns=np.array(bt_result["returns"]),
            trades=bt_result["trades"],
            interval="1h",
        )

        fold_results.append({
            "fold": fold_idx + 1,
            "train_range": (int(tr_s), int(tr_e)),
            "test_range": (int(ts_s), int(ts_e)),
            "metrics": fold_metrics,
            "n_trades": len(bt_result["trades"]),
        })
        all_trades.extend(bt_result["trades"])
        all_equity_curves.append(bt_result["equity_curve"])

    # Compute overall metrics across all folds
    if all_trades:
        # Combine all equity curves
        combined_equity = [initial_capital]
        combined_returns = []

        for ec in all_equity_curves:
            if len(ec) > 1:
                chunk_returns = np.diff(ec) / ec[:-1]
                combined_returns.extend(chunk_returns.tolist())
                combined_equity.extend(ec[1:])

        overall_metrics = compute_full_metrics(
            equity_curve=np.array(combined_equity),
            returns=np.array(combined_returns),
            trades=all_trades,
            interval="1h",
        )
    else:
        overall_metrics = {}

    return {
        "fold_results": fold_results,
        "overall_metrics": overall_metrics,
        "n_folds": len(fold_results),
        "total_trades": len(all_trades),
        "n_splits": len(splits),
    }
