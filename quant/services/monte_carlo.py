"""
Monte Carlo Simulation — Stress-test strategies by randomizing trade outcomes.

Takes actual trade results and reshuffles them to create thousands of
alternate histories, showing the range of possible outcomes.

If less than 95% of simulated outcomes are profitable, the strategy
is not statistically significant and should be discarded.

Renaissance principle: A single backtest result means nothing.
The distribution of possible outcomes is what matters.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for stress-testing trading strategies.

    Usage:
        sim = MonteCarloSimulator(n_simulations=10000)
        result = sim.run(trades)
        logger.info(f"Probability of profit: {result['probability_of_profit']:.1%}")
    """

    def __init__(self, n_simulations: int = 10000, random_seed: int | None = 42):
        """
        Args:
            n_simulations: Number of alternate histories to generate
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        if random_seed is not None:
            np.random.seed(random_seed)

    def run(self, trades: list[dict] | list[float],
            initial_capital: float = 10000.0) -> dict[str, Any]:
        """
        Run Monte Carlo simulation on a list of trades.

        Args:
            trades: List of trade dicts (with 'pnl_pct' key) or
                    list of P&L percentages as floats
            initial_capital: Starting capital for simulation

        Returns:
            Dict with:
                median_return, mean_return, std_return,
                percentile_5, percentile_25, percentile_75, percentile_95,
                probability_of_profit, probability_of_ruin,
                max_drawdown_median, max_drawdown_95th,
                total_simulations
        """
        pnl_values = self._extract_pnl(trades)

        if len(pnl_values) < 3:
            return {
                "error": f"Need at least 3 trades for MC simulation, got {len(pnl_values)}",
                "total_simulations": 0,
            }

        n_trades = len(pnl_values)
        final_equities = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            # Randomly reshuffle trade order
            shuffled = np.random.permutation(pnl_values)

            # Simulate equity curve
            equity = initial_capital
            peak = initial_capital
            sim_dd = 0.0

            for pnl_pct in shuffled:
                equity *= (1 + pnl_pct)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > sim_dd:
                    sim_dd = dd

            final_equities.append(equity)
            max_drawdowns.append(sim_dd)

        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)

        profitable = np.sum(final_equities > initial_capital)
        ruined = np.sum(final_equities <= 0)

        result = {
            "total_trades": n_trades,
            "total_simulations": self.n_simulations,
            "initial_capital": initial_capital,
            "median_return": round(float(np.median(final_equities)), 2),
            "mean_return": round(float(np.mean(final_equities)), 2),
            "std_return": round(float(np.std(final_equities, ddof=1)), 2),
            "min_return": round(float(np.min(final_equities)), 2),
            "max_return": round(float(np.max(final_equities)), 2),
            "percentile_5": round(float(np.percentile(final_equities, 5)), 2),
            "percentile_25": round(float(np.percentile(final_equities, 25)), 2),
            "percentile_75": round(float(np.percentile(final_equities, 75)), 2),
            "percentile_95": round(float(np.percentile(final_equities, 95)), 2),
            "probability_of_profit": round(float(profitable / self.n_simulations), 4),
            "probability_of_ruin": round(float(ruined / self.n_simulations), 4),
            "median_return_pct": round(float((np.median(final_equities) - initial_capital) / initial_capital * 100), 2),
            "max_drawdown_median": round(float(np.median(max_drawdowns)), 6),
            "max_drawdown_95th": round(float(np.percentile(max_drawdowns, 95)), 6),
        }

        # Renaissance rule: discard if < 95% probability of profit
        if result["probability_of_profit"] < 0.95:
            result["verdict"] = "DISCARD — insufficient statistical significance"
            result["discard"] = True
        else:
            result["verdict"] = f"KEEP — {result['probability_of_profit']:.1%} probability of profit"
            result["discard"] = False

        logger.info(
            "MC simulation: %d trades × %d sims → profit_prob=%.1f%%, "
            "ruin_prob=%.1f%%, median_return=%+.2f%%",
            n_trades, self.n_simulations,
            result["probability_of_profit"] * 100,
            result["probability_of_ruin"] * 100,
            result["median_return_pct"],
        )

        return result

    def run_from_backtest(self, backtest_result) -> dict[str, Any]:
        """
        Run Monte Carlo from a BacktestResult object.

        Args:
            backtest_result: BacktestResult instance with .trades list

        Returns:
            Same as run() with additional backtest context
        """
        if not backtest_result.trades:
            return {"error": "No trades in backtest result", "total_simulations": 0}

        pnl_values = [t.pnl_pct for t in backtest_result.trades]

        result = self.run(pnl_values, initial_capital=backtest_result.initial_capital)
        result["backtest_return_pct"] = backtest_result.total_return * 100
        result["backtest_sharpe"] = backtest_result.sharpe_ratio

        return result

    @staticmethod
    def _extract_pnl(trades: list[dict] | list[float]) -> np.ndarray:
        """Extract P&L percentages from various trade formats."""
        if not trades:
            return np.array([])

        # If already a list of floats, use directly
        if isinstance(trades[0], (int, float)):
            return np.array(trades, dtype=float)

        # If list of dicts with pnl_pct key
        if isinstance(trades[0], dict):
            values = [t.get("pnl_pct", 0) for t in trades]
            return np.array(values, dtype=float)

        return np.array([])
