"""
Kelly Criterion Position Sizer — Optimal position sizing for max geometric growth.

Kelly formula (general):
    f* = (b*p - q) / b
where:
    b = net odds received (win_amount / loss_amount)
    p = probability of winning
    q = 1 - p = probability of losing

Practical adjustment:
    - Full Kelly: f* — maximises growth, high drawdown risk
    - Half Kelly: f*/2 — moderate growth, manageable drawdown
    - Quarter Kelly: f*/4 — conservative, low drawdown (recommended for retail)

Renaissance principle: "It's not about how often you win — it's about how much
you win when you're right vs. how much you lose when you're wrong."
"""

import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class KellyPositionSizer:
    """
    Position sizing using the Kelly Criterion.

    Usage:
        sizer = KellyPositionSizer(fraction='quarter')
        size = sizer.calculate_position_size(
            capital=10000.0,
            win_probability=0.55,
            avg_win=0.02,    # 2% average win
            avg_loss=0.01,   # 1% average loss
        )
        # Returns position size in quote currency
    """

    FRACTION_MAP = {
        "full": 1.0,
        "half": 0.5,
        "quarter": 0.25,
    }

    def __init__(self, fraction: str = "quarter"):
        """
        Args:
            fraction: 'full', 'half', or 'quarter' (default: quarter)
        """
        self.kelly_fraction = self.FRACTION_MAP.get(fraction, 0.25)
        if fraction not in self.FRACTION_MAP:
            logger.warning(
                "Unknown Kelly fraction '%s', defaulting to quarter. "
                "Use 'full', 'half', or 'quarter'.",
                fraction,
            )

    def calculate_position_size(
        self,
        capital: float,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        max_position_pct: float = 100.0,
    ) -> dict:
        """
        Calculate optimal position size using the Kelly Criterion.

        Args:
            capital: Available capital for this trade (in quote currency)
            win_probability: P(profit) — from model/signal history (0.0–1.0)
            avg_win: Average fractional win (e.g., 0.02 = 2% gain)
            avg_loss: Average fractional loss (e.g., 0.01 = 1% loss)
            max_position_pct: Hard cap as % of capital (default 100% = no cap)

        Returns:
            Dict with:
                kelly_percent: Full Kelly % of capital
                applied_percent: Kelly % after fraction multiplier + cap
                position_size: USD/quote amount
                is_capped: Whether the position was capped
                parameters: Input parameters for audit trail
                warnings: List of any warnings
        """
        warnings: list[str] = []

        # ── Input validation ─────────────────────────────────------
        if capital <= 0:
            return {
                "kelly_percent": 0.0,
                "applied_percent": 0.0,
                "position_size": 0.0,
                "is_capped": False,
                "parameters": self._params_dict(locals()),
                "warnings": ["Capital must be positive"],
            }

        if win_probability <= 0 or win_probability >= 1:
            return {
                "kelly_percent": 0.0,
                "applied_percent": 0.0,
                "position_size": 0.0,
                "is_capped": False,
                "parameters": self._params_dict(locals()),
                "warnings": [f"win_probability={win_probability} out of range (0, 1)"],
            }

        if avg_win <= 0:
            return {
                "kelly_percent": 0.0,
                "applied_percent": 0.0,
                "position_size": 0.0,
                "is_capped": False,
                "parameters": self._params_dict(locals()),
                "warnings": ["avg_win must be positive"],
            }

        if avg_loss <= 0:
            return {
                "kelly_percent": 0.0,
                "applied_percent": 0.0,
                "position_size": 0.0,
                "is_capped": False,
                "parameters": self._params_dict(locals()),
                "warnings": ["avg_loss must be positive"],
            }

        # ── Win/loss ratio ────────────────────────────────────────
        loss_prob = 1.0 - win_probability

        # General Kelly formula: f* = (b*p - q) / b
        # where b = avg_win / avg_loss (net odds)
        b = avg_win / avg_loss

        kelly_percent = (b * win_probability - loss_prob) / b

        # Clamp: never bet more than 100% or less than 0%
        if kelly_percent < 0:
            warnings.append(
                f"Negative Kelly ({kelly_percent:.2%}) — no position recommended"
            )
            kelly_percent = 0.0
        elif kelly_percent > 1.0:
            warnings.append(
                f"Kelly > 100% ({kelly_percent:.1%}) — clamping to 100%. "
                "Consider using a smaller fraction."
            )
            kelly_percent = 1.0

        # ── Apply fraction (quarter/half/full) ─────────────────────
        applied_percent = kelly_percent * self.kelly_fraction

        # ── Apply hard cap ─────────────────────────────────────────
        max_pct = max_position_pct / 100.0
        is_capped = applied_percent > max_pct
        if is_capped:
            applied_percent = max_pct
            warnings.append(f"Capped at {max_position_pct:.0f}% of capital")

        position_size = capital * applied_percent

        logger.info(
            "Kelly: k=%.1f%%, applied=%.1f%%, size=%.2f, "
            "p=%.1f%%, avg_win=%.2f%%, avg_loss=%.2f%%, fraction=%s",
            kelly_percent * 100, applied_percent * 100, position_size,
            win_probability * 100, avg_win * 100, avg_loss * 100,
            self._fraction_name(),
        )

        return {
            "kelly_percent": round(kelly_percent, 6),
            "applied_percent": round(applied_percent, 6),
            "position_size": round(position_size, 2),
            "capital_used": round(position_size / capital, 6) if capital > 0 else 0.0,
            "is_capped": is_capped,
            "parameters": self._params_dict(locals()),
            "warnings": warnings,
        }

    def calculate_from_trade_history(self, capital: float, trades: list[dict],
                                      max_position_pct: float = 100.0) -> dict:
        """
        Calculate Kelly position size from a list of historical trades.

        Each trade dict must have:
            - 'pnl_pct': Fractional P&L (e.g., 0.02 = +2%, -0.01 = -1%)

        Automatically computes win_probability, avg_win, and avg_loss.

        Args:
            capital: Available capital
            trades: List of historical trade dicts
            max_position_pct: Hard cap

        Returns:
            Same as calculate_position_size, plus derived trade stats
        """
        if not trades:
            return {
                "kelly_percent": 0.0,
                "applied_percent": 0.0,
                "position_size": 0.0,
                "is_capped": False,
                "parameters": self._params_dict(locals()),
                "warnings": ["No trade history available"],
                "trade_stats": {"total_trades": 0},
            }

        pnl_values = [t.get("pnl_pct", 0) for t in trades if isinstance(t.get("pnl_pct"), (int, float))]
        if not pnl_values:
            return self.calculate_position_size(capital, 0.51, 0.01, 0.01, max_position_pct)

        wins = [v for v in pnl_values if v > 0]
        losses = [v for v in pnl_values if v < 0]

        total = len(pnl_values)
        win_probability = len(wins) / total if total > 0 else 0.5
        avg_win = sum(wins) / len(wins) if wins else 0.01
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.01

        result = self.calculate_position_size(
            capital, win_probability, avg_win, avg_loss, max_position_pct,
        )

        result["trade_stats"] = {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_probability, 4),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
        }

        return result

    @staticmethod
    def _params_dict(local_vars: dict) -> dict:
        """Extract parameter dict from local variables (excluding self)."""
        return {
            "capital": local_vars.get("capital", 0),
            "win_probability": local_vars.get("win_probability", 0),
            "avg_win": local_vars.get("avg_win", 0),
            "avg_loss": local_vars.get("avg_loss", 0),
            "max_position_pct": local_vars.get("max_position_pct", 100),
        }

    def _fraction_name(self) -> str:
        """Return human-readable fraction name."""
        for name, val in self.FRACTION_MAP.items():
            if val == self.kelly_fraction:
                return name
        return f"{self.kelly_fraction:.2f}x"
