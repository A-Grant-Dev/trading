"""
Cointegration & Pairs Discovery Service

Core engine for statistical arbitrage — the strategy that launched
Renaissance Technologies' Medallion Fund.

Implements the Engle-Granger two-step cointegration test to discover
pairs of assets whose prices share a long-term statistical relationship.
When the spread between them temporarily widens, we trade the reversion.

Renaissance/Simons principle: Don't predict direction. Find relationships.
When a statistical relationship breaks temporarily, bet on it to return.

Key formulas:
    spread = price_a - hedge_ratio * price_b
    z_score = (spread - mean(spread)) / std(spread)
    half_life = ln(2) / theta  (from OLS on lagged spread)
"""

import logging
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from quant.models import Pair
from quant.services.data_feeds import get_cache, set_cache

logger = logging.getLogger(__name__)

BINANCE_API = "https://api.binance.com"

# Default thresholds
DEFAULT_P_THRESHOLD = 0.05
DEFAULT_LOOKBACK_DAYS = 90
MIN_DATA_POINTS = 30  # Minimum price points for meaningful cointegration test


class PairsFinder:
    """
    Scans a universe of symbols to discover cointegrated trading pairs.

    Runs the Engle-Granger cointegration test on every pair combination
    and returns those that pass the p-value threshold, sorted by
    mean-reversion half-life.

    Usage:
        finder = PairsFinder(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        pairs = finder.find_cointegrated_pairs(price_data)
    """

    def __init__(self, symbols: list[str]):
        """
        Args:
            symbols: List of trading pair symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        """
        self.universe = [s.upper() for s in symbols]

    def find_cointegrated_pairs(
        self,
        price_data: dict[str, pd.Series],
        p_threshold: float = DEFAULT_P_THRESHOLD,
    ) -> list[dict]:
        """
        Run Engle-Granger cointegration test on all symbol pairs.

        Tests every unique pair combination in the universe. For each pair,
        computes the cointegration test statistic, p-value, hedge ratio,
        half-life, and current z-score.

        Args:
            price_data: Dict mapping symbol -> pd.Series of close prices
                        (aligned on same datetime index)
            p_threshold: Maximum p-value for significance (default 0.05)

        Returns:
            List of dicts sorted by half-life (fastest mean-reversion first):
            [{
                'symbol_a': str, 'symbol_b': str,
                'p_value': float, 'coint_statistic': float,
                'hedge_ratio': float, 'half_life': float,
                'current_zscore': float,
                'spread_mean': float, 'spread_std': float,
                'n_data_points': int,
                'adf_statistic': float, 'adf_pvalue': float,
            }, ...]

        Raises:
            ValueError: If no valid price data provided
        """
        if not price_data:
            raise ValueError("No price data provided")

        results = []
        n = len(self.universe)

        for i in range(n):
            for j in range(i + 1, n):
                sym_a = self.universe[i]
                sym_b = self.universe[j]

                # Get aligned price series
                prices_a = price_data.get(sym_a)
                prices_b = price_data.get(sym_b)

                if prices_a is None or prices_b is None:
                    continue

                # Align and drop NaN
                combined = pd.concat(
                    [prices_a.rename("a"), prices_b.rename("b")],
                    axis=1,
                ).dropna()

                if len(combined) < MIN_DATA_POINTS:
                    continue

                series_a = combined["a"].values.astype(float)
                series_b = combined["b"].values.astype(float)

                try:
                    result = self._test_pair(series_a, series_b, sym_a, sym_b, p_threshold)
                    if result and result["p_value"] <= p_threshold:
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Cointegration test failed for {sym_a}/{sym_b}: {e}")
                    continue

        # Sort by half-life (fastest mean-reversion first)
        results.sort(key=lambda r: r.get("half_life", float("inf")))

        logger.info(
            f"Found {len(results)} cointegrated pairs "
            f"from {n * (n - 1) // 2} combinations "
            f"(p<{p_threshold})"
        )
        return results

    def _test_pair(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        sym_a: str,
        sym_b: str,
        p_threshold: float = DEFAULT_P_THRESHOLD,
    ) -> Optional[dict]:
        """
        Run Engle-Granger cointegration test on a single pair.

        Step 1: Regress price_a = alpha + beta * price_b + epsilon
        Step 2: Test epsilon for stationarity using ADF test
        Step 3: If cointegrated, compute half-life and z-score

        Args:
            p_threshold: Maximum p-value for significance (passed from caller)
        """
        # Step 1: OLS regression to find hedge ratio
        X = sm.add_constant(series_b)
        ols_model = sm.OLS(series_a, X).fit()
        hedge_ratio = float(ols_model.params[1])  # beta coefficient
        alpha = float(ols_model.params[0])  # intercept

        # Compute the spread
        spread = series_a - hedge_ratio * series_b

        # Step 2: Engle-Granger cointegration test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coint_stat, p_value, _ = coint(series_a, series_b)

        if np.isnan(p_value) or p_value > p_threshold:
            return None

        # Step 3: Compute half-life of mean reversion
        half_life = self.compute_half_life(spread)

        # Step 4: Compute current z-score
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread, ddof=1))
        current_zscore = (
            float((spread[-1] - spread_mean) / spread_std)
            if spread_std > 0
            else 0.0
        )

        # Step 5: ADF test on spread for stationarity verification
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(spread, maxlag=1)

        return {
            "symbol_a": sym_a,
            "symbol_b": sym_b,
            "p_value": round(float(p_value), 6),
            "coint_statistic": round(float(coint_stat), 4),
            "hedge_ratio": round(float(hedge_ratio), 6),
            "alpha": round(float(alpha), 6),
            "half_life": round(float(half_life), 2) if half_life is not None else None,
            "current_zscore": round(float(current_zscore), 4),
            "spread_mean": round(spread_mean, 8),
            "spread_std": round(spread_std, 8),
            "n_data_points": len(series_a),
            "adf_statistic": round(float(adf_stat), 4),
            "adf_pvalue": round(float(adf_pvalue), 6),
        }

    @staticmethod
    def compute_half_life(spread: np.ndarray) -> Optional[float]:
        """
        Compute mean reversion half-life using OLS on lagged spread.

        Formula:
            Δspread_t = theta * (spread_{t-1} - mean) + epsilon
            half_life = ln(2) / theta

        The half-life represents the expected time (in periods) for the
        spread to revert halfway back to its mean after a deviation.

        A shorter half-life = faster mean reversion = more trading opportunities.

        Args:
            spread: Array of spread values (price_a - hedge_ratio * price_b)

        Returns:
            Half-life in periods (same units as input data), or None if
            the spread doesn't show mean-reverting behavior
        """
        spread = np.array(spread, dtype=float)
        if len(spread) < 20:
            return None

        # Create lagged spread: X = spread_{t-1}, y = Δspread_t = spread_t - spread_{t-1}
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)

        # Add constant for OLS
        X = sm.add_constant(spread_lag)
        try:
            model = sm.OLS(spread_diff, X).fit()
            theta = float(model.params[1])  # Coefficient on lagged spread
        except Exception:
            return None

        # Half-life = ln(2) / theta
        # The OLS coefficient on spread_lag is -theta for mean-reverting series
        # (since Δspread = -θ * spread_{t-1} + ...)
        # Negative OLS coefficient = mean-reverting
        if theta < 0:
            return float(np.log(2) / (-theta))
        else:
            # No mean reversion detected (positive = trending)
            return None

    @staticmethod
    def compute_zscore(spread: pd.Series | np.ndarray) -> float:
        """
        Compute the current z-score of the spread.

        z = (current_spread - mean_spread) / std_spread

        Rules of thumb:
            |z| < 1.0:  Spread is within normal range — no action
            1.0 < |z| < 2.0:  Spread is wide — monitor closely
            2.0 < |z| < 3.0:  Spread is very wide — ENTER trade
            |z| > 3.0:  Extremely wide — ENTER with wider stop or investigate

        Args:
            spread: Array-like of spread values

        Returns:
            Current z-score as a float
        """
        spread = np.array(spread, dtype=float)
        if len(spread) < 2:
            return 0.0

        mean = np.mean(spread)
        std = np.std(spread, ddof=1)

        if std == 0:
            return 0.0

        return float((spread[-1] - mean) / std)

    @staticmethod
    def update_pair_model(pair_result: dict) -> Pair:
        """
        Update or create a Pair model instance from cointegration results.

        Args:
            pair_result: Dict from find_cointegrated_pairs()

        Returns:
            Pair model instance
        """
        pair, _ = Pair.objects.update_or_create(
            symbol_a=pair_result["symbol_a"],
            symbol_b=pair_result["symbol_b"],
            defaults={
                "is_active": True,
                "coint_p_value": pair_result["p_value"],
                "coint_statistic": pair_result["coint_statistic"],
                "half_life": pair_result.get("half_life"),
                "hedge_ratio": pair_result["hedge_ratio"],
                "current_zscore": pair_result["current_zscore"],
                "last_tested": datetime.now(timezone.utc),
            },
        )
        return pair


# ── Data Fetching Helpers ──────────────────────────────────────────


def fetch_daily_close_prices(
    symbols: list[str],
    days: int = DEFAULT_LOOKBACK_DAYS,
    interval: str = "1d",
) -> dict[str, pd.Series]:
    """
    Fetch daily close prices for multiple symbols from Binance REST API.

    Uses cached results when available (cache TTL: 5 minutes).

    Args:
        symbols: List of trading pair symbols
        days: Number of days of historical data to fetch
        interval: Kline interval (default: 1d for daily closes)

    Returns:
        Dict mapping symbol -> pd.Series of close prices with datetime index
    """
    cache_key = f"close_prices_{'_'.join(sorted(symbols))}_{days}d"
    cached = get_cache(cache_key)
    if cached:
        return cached

    price_data: dict[str, pd.Series] = {}
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    for symbol in symbols:
        try:
            resp = requests.get(
                f"{BINANCE_API}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 1000,
                },
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()

            if not raw:
                continue

            dates = [datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc) for k in raw]
            closes = [float(k[4]) for k in raw]

            price_data[symbol] = pd.Series(closes, index=pd.DatetimeIndex(dates), name=symbol)

        except Exception as e:
            logger.warning(f"Failed to fetch daily data for {symbol}: {e}")
            continue

    set_cache(cache_key, price_data)
    logger.info(f"Fetched daily close prices for {len(price_data)} symbols")
    return price_data


def fetch_ohlcv_for_pair(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single symbol from Binance REST API.

    Used for detailed pair analysis and backtesting.

    Args:
        symbol: Trading pair symbol
        interval: Kline interval
        limit: Number of candles

    Returns:
        DataFrame with OHLCV columns and datetime index, or empty DataFrame on failure
    """
    from quant.services.data_utils import ohlcv_to_dataframe

    return ohlcv_to_dataframe(symbol, interval, limit)
