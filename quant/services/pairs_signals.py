"""
Pairs Trading Signal Generator

Generates long/short trading signals for cointegrated asset pairs.

Core strategy (pairs trading / statistical arbitrage):
    1. Find two cointegrated assets (done by PairsFinder)
    2. When the spread between them widens (z > entry_threshold):
       → Short the outperformer, long the underperformer
    3. When the spread reverts (z < exit_threshold):
       → Close both positions
    4. If the spread continues widening (z > stop_threshold):
       → Close at a loss — the relationship may have broken

Renaissance/Simons principle: This is the strategy that launched Medallion.
Find pairs where prices move together. When they temporarily diverge,
bet on convergence. Win 51% of the time, but the wins > losses.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from quant.models import Pair, TradeSignal
from quant.services.cointegration import PairsFinder
from quant.services.regime_signals import adjust_signal_for_regime

logger = logging.getLogger(__name__)

# Default thresholds (in standard deviations)
ENTRY_Z = 2.0      # Enter when |z-score| > 2.0
EXIT_Z = 0.5       # Exit when |z-score| < 0.5
STOP_Z = 3.0       # Stop loss when |z-score| > 3.0

# Signal expiry (hours)
SIGNAL_EXPIRY_HOURS = 4


class PairsSignalGenerator:
    """
    Generates trading signals for cointegrated pairs based on z-score thresholds.

    Usage:
        generator = PairsSignalGenerator()
        signal = generator.evaluate_pair(pair_data)
        backtest_results = generator.backtest_pair(pair_data, historical_df)
    """

    ENTRY_Z = ENTRY_Z
    EXIT_Z = EXIT_Z
    STOP_Z = STOP_Z

    def __init__(
        self,
        entry_z: float = ENTRY_Z,
        exit_z: float = EXIT_Z,
        stop_z: float = STOP_Z,
    ):
        """
        Args:
            entry_z: Z-score threshold to enter a trade (default: 2.0)
            exit_z: Z-score threshold to exit a trade (default: 0.5)
            stop_z: Z-score threshold for stop loss (default: 3.0)
        """
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

    def evaluate_pair(
        self,
        pair: Pair,
        spread_data: pd.Series | None = None,
        current_zscore: float | None = None,
        regime_label: str = "ranging",
    ) -> TradeSignal | None:
        """
        Evaluate a cointegrated pair and generate a signal if thresholds are breached.

        Args:
            pair: Pair model instance
            spread_data: Optional Series of spread values for z-score recalculation
            current_zscore: Optional pre-computed z-score (avoids recalculation)
            regime_label: Current market regime for signal adjustment

        Returns:
            TradeSignal instance if signal generated, None otherwise
        """
        # Determine z-score
        if current_zscore is not None:
            z = current_zscore
        elif spread_data is not None:
            z = self._compute_zscore(spread_data)
        elif pair.current_zscore is not None:
            z = pair.current_zscore
        else:
            # Not enough data to evaluate
            return None

        abs_z = abs(z)
        now = datetime.now(timezone.utc)

        # Generate signal based on z-score
        if abs_z >= self.entry_z:
            # Entry signal
            if z > 0:
                # Spread widened positively — price_a is overpriced vs price_b
                # → Short symbol_a, long symbol_b
                direction = "short"  # Short the spread (bet on convergence)
                base_strength = min(1.0, (abs_z - self.entry_z) / (self.stop_z - self.entry_z) + 0.5)
            else:
                # Spread widened negatively — price_a is underpriced vs price_b
                # → Long symbol_a, short symbol_b
                direction = "long"  # Long the spread (bet on convergence)
                base_strength = min(1.0, (abs_z - self.entry_z) / (self.stop_z - self.entry_z) + 0.5)

            # Apply regime adjustment
            adjusted_strength = adjust_signal_for_regime(base_strength, direction, regime_label)

            # Only generate signal if adjusted strength is meaningful
            if adjusted_strength < 0.3:
                return None

            confidence = min(1.0, max(0.3, adjusted_strength))

            # Create the signal (store for both symbols)
            signal = TradeSignal.objects.create(
                pair=pair,
                signal_type="long" if direction == "long" else "short",
                direction=direction,
                strength=round(adjusted_strength, 4),
                confidence=round(confidence, 4),
                source_model="cointegration",
                expiry=now + timedelta(hours=SIGNAL_EXPIRY_HOURS),
                status="active",
                metadata={
                    "z_score": round(z, 4),
                    "entry_threshold": self.entry_z,
                    "stop_threshold": self.stop_z,
                    "hedge_ratio": float(pair.hedge_ratio) if pair.hedge_ratio else None,
                    "half_life_hours": pair.half_life,
                    "regime": regime_label,
                    "signal_type": "entry",
                },
            )

            pair.total_signals += 1
            pair.current_zscore = z
            pair.save(update_fields=["total_signals", "current_zscore"])

            logger.info(
                f"Pairs signal: {pair.symbol_a}/{pair.symbol_b} "
                f"{direction.upper()} (z={z:.2f}, conf={confidence:.0%})"
            )

            return signal

        elif abs_z <= self.exit_z:
            # Z-score has reverted — close any open positions for this pair
            self._close_active_signals(pair)
            return None

        else:
            # No signal — spread within normal range
            return None

    def _close_active_signals(self, pair: Pair) -> int:
        """
        Close all active signals for a pair (spread has reverted).

        Args:
            pair: The Pair model instance

        Returns:
            Number of signals closed
        """
        now = datetime.now(timezone.utc)
        active = TradeSignal.objects.filter(
            pair=pair,
            status="active",
            source_model="cointegration",
        )

        count = active.update(
            status="cancelled",
            metadata={"cancelled_at": now.isoformat(), "reason": "spread_reverted"},
        )

        if count > 0:
            logger.info(f"Closed {count} active signal(s) for {pair} (spread reverted)")

        return count

    @staticmethod
    def _compute_zscore(spread_data: pd.Series) -> float:
        """
        Compute z-score of the spread.

        Args:
            spread_data: Series of spread values

        Returns:
            Current z-score
        """
        return PairsFinder.compute_zscore(spread_data)

    # ── Backtesting ──────────────────────────────────────────────

    def backtest_pair(
        self,
        pair_data: dict,
        historical_data: pd.DataFrame,
    ) -> dict:
        """
        Run a backtest of the pairs trading strategy on historical data.

        Simulates trading the spread: enters when |z| > entry threshold,
        exits when |z| < exit threshold or |z| > stop threshold.

        Args:
            pair_data: Dict with 'symbol_a', 'symbol_b', 'hedge_ratio'
            historical_data: DataFrame with columns for both symbols' prices
                            Format: index=datetime, columns=[symbol_a, symbol_b]

        Returns:
            Dict with backtest results:
            {
                'total_trades': int,
                'winning_trades': int,
                'losing_trades': int,
                'win_rate': float,
                'total_pnl': float,
                'sharpe_ratio': float,
                'max_drawdown': float,
                'avg_hold_periods': float,
                'avg_win': float,
                'avg_loss': float,
                'profit_factor': float,
                'trades': list[dict],
            }
        """
        if historical_data.empty or len(historical_data.columns) < 2:
            return {"error": "Insufficient data for backtest"}

        symbol_a = pair_data.get("symbol_a", historical_data.columns[0])
        symbol_b = pair_data.get("symbol_b", historical_data.columns[1])

        # Get hedge ratio
        if "hedge_ratio" in pair_data and pair_data["hedge_ratio"]:
            hedge_ratio = pair_data["hedge_ratio"]
        else:
            # Compute hedge ratio from data
            prices_a = historical_data[symbol_a].values.astype(float)
            prices_b = historical_data[symbol_b].values.astype(float)
            import statsmodels.api as sm
            X = sm.add_constant(prices_b)
            model = sm.OLS(prices_a, X).fit()
            hedge_ratio = float(model.params[1])

        # Compute spread and z-score
        prices_a = historical_data[symbol_a].values.astype(float)
        prices_b = historical_data[symbol_b].values.astype(float)
        spread = prices_a - hedge_ratio * prices_b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread, ddof=1)
        z_scores = (spread - spread_mean) / spread_std if spread_std > 0 else np.zeros_like(spread)

        # Simulate trading
        position = 0  # -1 = short spread, 0 = flat, 1 = long spread
        entry_z = 0.0
        entry_idx = 0
        trades = []
        trade_pnl_values = []

        for i in range(1, len(z_scores)):
            z = z_scores[i]

            if position == 0:
                # Look for entry
                if z > self.entry_z:
                    position = -1  # Short the spread
                    entry_z = z
                    entry_idx = i
                elif z < -self.entry_z:
                    position = 1  # Long the spread
                    entry_z = z
                    entry_idx = i

            elif position == -1:
                # Short spread position — exit on reversion or stop
                if z < self.exit_z:
                    # Profitable exit — spread reverted
                    exit_z = z
                    pnl = entry_z - exit_z  # Simplified: short profited from z decrease
                    pnl_pct = abs(entry_z - exit_z) / (abs(entry_z) + 1e-10)
                    is_win = pnl > 0
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "short",
                        "entry_z": round(float(entry_z), 4),
                        "exit_z": round(float(exit_z), 4),
                        "pnl": round(float(pnl), 4),
                        "pnl_pct": round(float(pnl_pct), 4),
                        "is_win": is_win,
                    })
                    trade_pnl_values.append(pnl)
                    position = 0

                elif z > self.stop_z:
                    # Stop loss — spread continued widening
                    pnl = entry_z - z  # Loss: z went further positive
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "short",
                        "entry_z": round(float(entry_z), 4),
                        "exit_z": round(float(z), 4),
                        "pnl": round(float(pnl), 4),
                        "is_win": False,
                    })
                    trade_pnl_values.append(pnl)
                    position = 0

            elif position == 1:
                # Long spread position
                if z > -self.exit_z:
                    # Profitable exit
                    pnl = -(entry_z - z)  # Long profited from z increase (becoming less negative)
                    pnl_pct = abs(entry_z - z) / (abs(entry_z) + 1e-10)
                    is_win = pnl > 0
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "long",
                        "entry_z": round(float(entry_z), 4),
                        "exit_z": round(float(z), 4),
                        "pnl": round(float(pnl), 4),
                        "pnl_pct": round(float(pnl_pct), 4),
                        "is_win": is_win,
                    })
                    trade_pnl_values.append(pnl)
                    position = 0

                elif z < -self.stop_z:
                    # Stop loss
                    pnl = -(entry_z - z)
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "long",
                        "entry_z": round(float(entry_z), 4),
                        "exit_z": round(float(z), 4),
                        "pnl": round(float(pnl), 4),
                        "is_win": False,
                    })
                    trade_pnl_values.append(pnl)
                    position = 0

        # Compute summary metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t["is_win"])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(t["pnl"] for t in trades)

        avg_win = (
            sum(t["pnl"] for t in trades if t["is_win"]) / winning_trades
            if winning_trades > 0
            else 0.0
        )
        avg_loss = (
            sum(t["pnl"] for t in trades if not t["is_win"]) / losing_trades
            if losing_trades > 0
            else 0.0
        )

        profit_factor = (
            abs(sum(t["pnl"] for t in trades if t["pnl"] > 0)) /
            abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
            if any(t["pnl"] < 0 for t in trades)
            else float("inf")
        )

        # Sharpe ratio (annualized)
        if len(trade_pnl_values) > 1 and np.std(trade_pnl_values) > 0:
            sharpe = round(
                float(np.mean(trade_pnl_values) / np.std(trade_pnl_values) * np.sqrt(365)),
                2,
            )
        else:
            sharpe = 0.0

        # Max drawdown of cumulative PnL
        cumulative = np.cumsum(trade_pnl_values) if trade_pnl_values else [0]
        if len(cumulative) > 0:
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / (running_max + 1e-10)
            max_dd = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0
        else:
            max_dd = 0.0

        # Average hold periods
        hold_periods = [t["exit_idx"] - t["entry_idx"] for t in trades]
        avg_hold = float(np.mean(hold_periods)) if hold_periods else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(float(total_pnl), 4),
            "sharpe_ratio": sharpe,
            "max_drawdown": round(max_dd, 4),
            "avg_hold_periods": round(avg_hold, 1),
            "avg_win": round(float(avg_win), 4),
            "avg_loss": round(float(avg_loss), 4),
            "profit_factor": round(float(profit_factor), 2) if profit_factor != float("inf") else None,
            "trades": trades,
        }
