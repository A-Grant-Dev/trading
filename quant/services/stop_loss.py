"""
Stop Loss & Take Profit Manager — Dynamic exit strategies.

Strategies:
  1. Fixed percentage: -2% stop, +4% take profit
  2. ATR-based: 2x ATR stop, 4x ATR take profit
  3. Volatility-adjusted: Wider stops in high vol, tighter in low
  4. Trailing stop: Lock in profits as trade moves favorably
  5. Time-based: Exit if trade hasn't hit target within N hours

Renaissance principle: A trade without a stop loss is not a trade —
it's a gamble. Define your exit before you enter.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

logger = logging.getLogger(__name__)

BINANCE_API = "https://api.binance.com"

# ── Default Parameters ─────────────────────────────────────────────

DEFAULT_STOP_LOSS_PCT = 0.02      # 2% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.04    # 4% take profit
DEFAULT_ATR_MULTIPLIER_STOP = 2.0
DEFAULT_ATR_MULTIPLIER_TP = 4.0
DEFAULT_TRAILING_ACTIVATION_PCT = 0.01  # Activate trailing at 1% profit
DEFAULT_TRAILING_DISTANCE_PCT = 0.005   # 0.5% trailing distance
DEFAULT_TIME_EXIT_HOURS = 24

# Regime volatility multipliers
REGIME_VOL_MULTIPLIERS: dict[str, float] = {
    "ranging": 1.0,    # Normal stops
    "bullish": 0.8,    # Tighter stops — trend is strong
    "bearish": 0.8,    # Tighter stops — trend is strong
    "volatile": 2.0,   # Wider stops — noise is higher
}


class StopLossManager:
    """
    Dynamic stop-loss and take-profit management.

    Usage:
        sl = StopLossManager()
        stops = sl.calculate_stops(
            entry_price=65000.0,
            side='buy',
            symbol='BTCUSDT',
            regime='ranging',
        )
        # Use stops['stop_loss'], stops['take_profit'], etc.
    """

    def __init__(self, strategy: str = "atr"):
        """
        Args:
            strategy: 'fixed', 'atr', 'volatility', 'trailing', 'time'
        """
        self.strategy = strategy

    def calculate_stops(self, entry_price: float, side: str,
                        symbol: str | None = None,
                        regime: str = "ranging") -> dict:
        """
        Calculate stop loss and take profit levels.

        Args:
            entry_price: Entry price of the position
            side: 'buy' or 'sell'
            symbol: Trading pair (for fetching ATR if needed)
            regime: Current market regime label

        Returns:
            Dict with:
                stop_loss: Price level to stop out
                take_profit: Price level to take profit
                trailing_activation: Price level to activate trailing stop
                trailing_distance: Trailing distance in price units
                time_exit_hours: Max hold time before forced exit
                strategy: Strategy used
                stop_pct: Percentage distance to stop
                tp_pct: Percentage distance to take profit
        """
        # Get ATR for dynamic strategies
        atr = self._fetch_atr(symbol) if symbol else None

        # Regime adjustment
        regime_mult = REGIME_VOL_MULTIPLIERS.get(regime, 1.0)

        if self.strategy == "atr" and atr and atr > 0:
            return self._atr_based(entry_price, side, atr, regime_mult)
        elif self.strategy == "fixed":
            return self._fixed_pct(entry_price, side, regime_mult)
        else:
            # Default to fixed if ATR unavailable
            return self._fixed_pct(entry_price, side, regime_mult)

    def _fixed_pct(self, entry_price: float, side: str,
                   regime_mult: float) -> dict:
        """Fixed percentage stop loss and take profit."""
        stop_pct = DEFAULT_STOP_LOSS_PCT * regime_mult
        tp_pct = DEFAULT_TAKE_PROFIT_PCT * regime_mult

        if side == "buy":
            stop_loss = entry_price * (1 - stop_pct)
            take_profit = entry_price * (1 + tp_pct)
            trailing_activation = entry_price * (1 + DEFAULT_TRAILING_ACTIVATION_PCT)
        else:
            stop_loss = entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 - tp_pct)
            trailing_activation = entry_price * (1 - DEFAULT_TRAILING_ACTIVATION_PCT)

        trailing_distance = entry_price * DEFAULT_TRAILING_DISTANCE_PCT

        return self._build_result(entry_price, side, stop_loss, take_profit,
                                  trailing_activation, trailing_distance,
                                  stop_pct, tp_pct, "fixed")

    def _atr_based(self, entry_price: float, side: str,
                   atr: float, regime_mult: float) -> dict:
        """ATR-based stop loss and take profit."""
        stop_distance = atr * DEFAULT_ATR_MULTIPLIER_STOP * regime_mult
        tp_distance = atr * DEFAULT_ATR_MULTIPLIER_TP * regime_mult

        stop_pct = stop_distance / entry_price if entry_price > 0 else DEFAULT_STOP_LOSS_PCT
        tp_pct = tp_distance / entry_price if entry_price > 0 else DEFAULT_TAKE_PROFIT_PCT

        if side == "buy":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
            trailing_activation = entry_price + (atr * regime_mult)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
            trailing_activation = entry_price - (atr * regime_mult)

        trailing_distance = atr * 0.5 * regime_mult

        return self._build_result(entry_price, side, stop_loss, take_profit,
                                  trailing_activation, trailing_distance,
                                  stop_pct, tp_pct, "atr")

    def _build_result(self, entry_price: float, side: str,
                      stop_loss: float, take_profit: float,
                      trailing_activation: float, trailing_distance: float,
                      stop_pct: float, tp_pct: float,
                      strategy: str) -> dict:
        """Build and return the stops dict."""
        time_exit_hours = DEFAULT_TIME_EXIT_HOURS

        # Validate stop direction
        if side == "buy":
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.98  # Fallback: 2% below
            if take_profit <= entry_price:
                take_profit = entry_price * 1.04  # Fallback: 4% above
        else:
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.02  # Fallback: 2% above
            if take_profit >= entry_price:
                take_profit = entry_price * 0.96  # Fallback: 4% below

        return {
            "entry_price": entry_price,
            "side": side,
            "stop_loss": round(stop_loss, 8),
            "take_profit": round(take_profit, 8),
            "trailing_activation": round(trailing_activation, 8),
            "trailing_distance": round(trailing_distance, 8),
            "time_exit_hours": time_exit_hours,
            "time_exit_at": (
                datetime.now(timezone.utc) + timedelta(hours=time_exit_hours)
            ).isoformat(),
            "stop_pct": round(stop_pct, 6),
            "tp_pct": round(tp_pct, 6),
            "strategy": strategy,
        }

    def should_exit(self, position: dict, current_price: float,
                    highest_price: float | None = None,
                    lowest_price: float | None = None) -> tuple[bool, str]:
        """
        Check if any exit condition is triggered.

        Args:
            position: Dict with keys: entry_price, side, stop_loss,
                      take_profit, trailing_activation, trailing_distance,
                      entry_time, time_exit_hours
            current_price: Current market price
            highest_price: Highest price since entry (for trailing stops)
            lowest_price: Lowest price since entry

        Returns:
            (should_exit: bool, reason: str)
        """
        entry_price = position.get("entry_price", 0)
        side = position.get("side", "buy").lower()
        stop_loss = position.get("stop_loss", 0)
        take_profit = position.get("take_profit", 0)
        trailing_activation = position.get("trailing_activation", 0)
        trailing_distance = position.get("trailing_distance", 0)
        entry_time_str = position.get("entry_time")
        time_exit_hours = position.get("time_exit_hours", DEFAULT_TIME_EXIT_HOURS)

        # ── Stop loss check ───────────────────────────────────────
        if side == "buy" and current_price <= stop_loss:
            return True, f"Stop loss hit: {current_price:.2f} <= {stop_loss:.2f}"
        if side == "sell" and current_price >= stop_loss:
            return True, f"Stop loss hit: {current_price:.2f} >= {stop_loss:.2f}"

        # ── Take profit check ─────────────────────────────────────
        if side == "buy" and current_price >= take_profit:
            return True, f"Take profit hit: {current_price:.2f} >= {take_profit:.2f}"
        if side == "sell" and current_price <= take_profit:
            return True, f"Take profit hit: {current_price:.2f} <= {take_profit:.2f}"

        # ── Trailing stop check ───────────────────────────────────
        if highest_price and trailing_activation and trailing_distance:
            if highest_price >= trailing_activation:
                # Activate trailing stop
                if side == "buy":
                    trailing_stop = highest_price - trailing_distance
                    if current_price <= trailing_stop:
                        return True, (
                            f"Trailing stop hit: {current_price:.2f} "
                            f"<= {trailing_stop:.2f} (trailed from {highest_price:.2f})"
                        )
                else:
                    trailing_stop = highest_price + trailing_distance
                    if current_price >= trailing_stop:
                        return True, (
                            f"Trailing stop hit: {current_price:.2f} "
                            f">= {trailing_stop:.2f} (trailed from {highest_price:.2f})"
                        )

        # ── Time-based exit ───────────────────────────────────────
        if entry_time_str and time_exit_hours:
            try:
                if isinstance(entry_time_str, str):
                    entry_time = datetime.fromisoformat(entry_time_str)
                else:
                    entry_time = entry_time_str

                if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)

                elapsed = datetime.now(timezone.utc) - entry_time
                if elapsed.total_seconds() >= time_exit_hours * 3600:
                    return True, f"Time exit: {elapsed.total_seconds() / 3600:.1f}h >= {time_exit_hours}h"
            except (ValueError, TypeError):
                pass

        return False, "No exit condition triggered"

    @staticmethod
    def _fetch_atr(symbol: str, period: int = 14) -> float | None:
        """Fetch current ATR for a symbol from Binance klines."""
        try:
            resp = requests.get(
                f"{BINANCE_API}/api/v3/klines",
                params={"symbol": symbol.upper(), "interval": "1h", "limit": period + 1},
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            if len(data) < period + 1:
                return None

            # Would use compute_atr from data_utils, but compute here for
            # independence from DB dependencies
            highs = np.array([float(k[2]) for k in data])
            lows = np.array([float(k[3]) for k in data])
            closes = np.array([float(k[4]) for k in data])

            tr_values = []
            for i in range(1, len(data)):
                hl = highs[i] - lows[i]
                hc = abs(highs[i] - closes[i - 1])
                lc = abs(lows[i] - closes[i - 1])
                tr_values.append(max(hl, hc, lc))

            atr = float(np.mean(tr_values[-period:]))
            return round(atr, 8)

        except Exception as e:
            logger.debug(f"Failed to fetch ATR for {symbol}: {e}")
            return None

    @staticmethod
    def calculate_atr_from_klines(klines: list[list]) -> float:
        """
        Calculate ATR from raw Binance kline data.

        Args:
            klines: List of Binance klines (at least 15 needed)

        Returns:
            ATR value
        """
        if len(klines) < 15:
            return 0.0

        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]

        tr_values = []
        for i in range(1, len(klines)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr_values.append(max(hl, hc, lc))

        period = min(14, len(tr_values))
        return round(float(np.mean(tr_values[-period:])), 8)
