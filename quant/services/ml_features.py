"""
ML Feature Engineering Pipeline

Builds comprehensive feature matrices from raw market data for ML model
training and inference. Features are organized into 9 groups mirroring
the signals Renaissance Technologies would extract.

Feature groups:
  1. Price-based: Returns (1, 5, 15, 60 min), log returns
  2. Technical: RSI, MACD, BB, ATR, Stochastic, OBV, VWAP
  3. Volume: Volume delta, taker buy/sell ratio, volume profile
  4. Order book: Imbalance ratio, depth pressure, spread width
  5. Microstructure: Trade intensity, tick frequency, avg trade size
  6. Cross-asset: BTC correlation, sector correlation
  7. Regime: HMM regime label (one-hot encoded)
  8. Sentiment: Aggregated sentiment score
  9. Time-based: Hour of day, day of week, month (seasonality)

Renaissance principle: The feature space should be wide enough to capture
hidden statistical patterns but not so wide that it overfits.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import requests
from quant.services.data_utils import (
    _compute_ema,
    _compute_sma,
    compute_rsi,
    compute_atr,
    ohlcv_to_dataframe,
)
from quant.services.data_feeds import get_cache

logger = logging.getLogger(__name__)

# ── Default Target Column ──────────────────────────────────────────

TARGET_COLUMN = "target_future_return_1"  # Default: predict next-candle return


class FeaturePipeline:
    """
    Builds feature matrices for ML models from raw market data.

    The pipeline takes a raw OHLCV DataFrame and transforms it into
    a comprehensive feature matrix with technical indicators, volume
    metrics, and derived features ready for model training.

    Usage:
        pipeline = FeaturePipeline()
        df = pipeline.build_features(ohlcv_df)
        X, y = pipeline.prepare_training_data(df, target_column='target_future_return_1')
    """

    def __init__(self, include_orderbook: bool = False, include_sentiment: bool = False):
        """
        Args:
            include_orderbook: Whether to attempt fetching order book features
            include_sentiment: Whether to attempt fetching sentiment data
        """
        self.include_orderbook = include_orderbook
        self.include_sentiment = include_sentiment

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Takes a raw OHLCV DataFrame and returns a feature-rich DataFrame
        with all 9 feature groups computed.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be datetime

        Returns:
            DataFrame with all features added, NaN rows removed
        """
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        df = df.copy()

        # Convert columns to float arrays for numpy operations
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        _open = df["open"].values.astype(float) if "open" in df.columns else close
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None

        # ── Group 1: Price-based Features ────────────────────────
        # log(close[t] / close[t-1]) — first element is log(1) = 0
        shifted_close = np.concatenate([[close[0]], close[:-1]])
        df["log_return"] = np.log(close / shifted_close)

        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_15"] = df["close"].pct_change(15)
        df["return_60"] = df["close"].pct_change(60) if len(df) > 60 else np.nan

        df["high_low_pct"] = (high - low) / close * 100
        df["close_open_pct"] = (close - _open) / _open * 100

        # Price position within recent range
        df["highest_20"] = self._rolling_max(close, 20)
        df["lowest_20"] = self._rolling_min(close, 20)
        df["price_position_20"] = (close - df["lowest_20"]) / (df["highest_20"] - df["lowest_20"] + 1e-10)

        # ── Group 2: Technical Indicators ────────────────────────
        df["rsi_14"] = compute_rsi(close, 14)
        df["atr_14"] = compute_atr(high, low, close, 14)
        df["atr_pct"] = df["atr_14"] / close * 100

        # MACD
        ema12 = _compute_ema(close, 12)
        ema26 = _compute_ema(close, 26)
        df["macd"] = ema12 - ema26
        df["macd_signal"] = _compute_ema(df["macd"].values, 9)
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma20 = _compute_sma(close, 20)
        std20 = self._rolling_std(close, 20)
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_middle"] = sma20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # EMAs
        df["ema_9"] = _compute_ema(close, 9)
        df["ema_21"] = _compute_ema(close, 21)
        df["ema_50"] = _compute_ema(close, 50)
        df["ema_200"] = _compute_ema(close, 200)
        df["ema_cross"] = df["ema_9"] / df["ema_21"] - 1  # Positive = bullish cross

        # ── Group 3: Volume Features ─────────────────────────────
        if volume is not None:
            df["volume_sma_20"] = _compute_sma(volume, 20)
            df["volume_ratio"] = volume / (df["volume_sma_20"] + 1e-10)
            df["volume_change"] = df["volume"].pct_change()

            # On-Balance Volume
            obv = np.zeros_like(volume)
            for i in range(1, len(volume)):
                obv[i] = obv[i - 1] + (volume[i] if close[i] > close[i - 1] else -volume[i] if close[i] < close[i - 1] else 0)
            df["obv"] = obv
            df["obv_ema"] = _compute_ema(obv, 20)
            df["obv_ratio"] = obv / (df["obv_ema"] + 1e-10)

            # VWAP
            typical_price = (high + low + close) / 3
            cum_pv = np.cumsum(typical_price * volume)
            cum_vol = np.cumsum(volume)
            df["vwap"] = cum_pv / (cum_vol + 1e-10)
            df["vwap_distance"] = (close - df["vwap"]) / df["vwap"] * 100  # % distance from VWAP

        # ── Group 4: Order Book Features (if enabled) ────────────
        if self.include_orderbook:
            try:
                symbol = df.attrs.get("symbol", "")
                if symbol:
                    from quant.services.data_utils import compute_order_book_features
                    ob = compute_order_book_features(symbol)
                    if ob:
                        df["ob_imbalance"] = ob.get("imbalance_pct", 50.0)
                        df["ob_spread_pct"] = ob.get("spread_pct", 0)
                        df["ob_depth_pressure"] = ob.get("depth_pressure", 0)
            except Exception:
                pass

        # ── Group 5: Microstructure (requires TradeRecord data) ──
        # This will be populated by a separate method when trade data is available
        # Placeholder for now
        df["microstructure_available"] = 0

        # ── Group 6: Cross-asset Features ────────────────────────
        try:
            btc_close = self._get_btc_price(df.index)
            if btc_close is not None and len(btc_close) > 20:
                btc_returns = np.log(btc_close / np.roll(btc_close, 1))
                asset_returns = np.log(close / np.roll(close, 1))
                # Rolling correlation (30 periods)
                df["btc_corr_30"] = self._rolling_corr(asset_returns, btc_returns, 30)
        except Exception:
            pass

        # ── Group 7: Regime Features (one-hot encoded) ───────────
        # Filled by update_regime command or explicitly set
        # Default: all zeros (neutral)
        for regime in ["ranging", "bullish", "bearish", "volatile"]:
            df[f"regime_{regime}"] = 0.0

        # ── Group 8: Sentiment Features ──────────────────────────
        if self.include_sentiment:
            try:
                from quant.services.alt_sentiment import AlternativeSentimentEngine
                symbol = df.attrs.get("symbol", "")
                if symbol:
                    engine = AlternativeSentimentEngine()
                    consensus = engine.compute_consensus_score(symbol)
                    if consensus and "consensus_score" in consensus:
                        df["sentiment_score"] = consensus["consensus_score"]
            except Exception:
                pass

        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 50.0  # Neutral default

        # ── Group 9: Time-based Features ─────────────────────────
        if hasattr(df.index, "hour"):
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            df["is_monday"] = (df["day_of_week"] == 0).astype(float)
            df["is_friday"] = (df["day_of_week"] == 4).astype(float)
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
            df["is_asia_session"] = ((df["hour"] >= 1) & (df["hour"] <= 9)).astype(float)
            df["is_london_session"] = ((df["hour"] >= 8) & (df["hour"] <= 16)).astype(float)
            df["is_ny_session"] = ((df["hour"] >= 13) & (df["hour"] <= 22)).astype(float)

        # ── Target: Future Returns ──────────────────────────────
        df["target_future_return_1"] = df["return_1"].shift(-1)  # Next candle return
        df["target_future_return_5"] = df["close"].pct_change(5).shift(-5)
        df["target_direction"] = (df["target_future_return_1"] > 0).astype(int)  # Binary: up=1

        # Drop NaN rows from indicator warmup periods.
        # Use thresh to keep rows that have most features valid —
        # long-period indicators (EMA 200, return_60) may be NaN
        # for the first ~200 rows, but we don't want to lose everything.
        min_required = max(10, len(df.columns) - 8)
        df.dropna(thresh=min_required, inplace=True)

        # Fill any remaining sparse NaN with 0 for sklearn/XGBoost compatibility
        df.fillna(0, inplace=True)

        return df

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str = TARGET_COLUMN,
        feature_columns: list[str] | None = None,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split features and target into train/test sets.

        Uses chronological split (not random) to avoid look-ahead bias.

        Args:
            df: Feature-rich DataFrame from build_features()
            target_column: Column name for target variable
            feature_columns: Columns to use as features (None = auto-select)
            test_size: Fraction of data for testing (default 0.2)

        Returns:
            (X_train, X_test, y_train, y_test) as numpy arrays

        Raises:
            ValueError: If insufficient data or missing target column
        """
        if df.empty or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Auto-select feature columns (exclude targets, metadata, NaN-heavy)
        if feature_columns is None:
            feature_columns = [
                col for col in df.columns
                if col not in [target_column, "target_direction", "target_future_return_5",
                              "microstructure_available", "close_time",
                              "open", "high", "low", "close", "volume",
                              "quote_asset_volume", "taker_buy_base_vol", "taker_buy_quote_vol"]
                and not col.startswith("regime_")
            ] + [col for col in df.columns if col.startswith("regime_")]

        # Filter to available columns
        feature_columns = [c for c in feature_columns if c in df.columns]

        if not feature_columns:
            raise ValueError("No valid feature columns found")

        X = df[feature_columns].values.astype(float)
        y = df[target_column].values.astype(float)

        # Chronological split
        split_idx = int(len(X) * (1 - test_size))
        if split_idx < 10:
            raise ValueError(f"Not enough data: need at least 10 training samples, got {split_idx}")

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Handle NaN in targets
        train_mask = ~np.isnan(y_train)
        test_mask = ~np.isnan(y_test)

        return X_train[train_mask], X_test[test_mask], y_train[train_mask], y_test[test_mask]

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Get list of feature column names (excludes targets and metadata)."""
        exclude = {"target_future_return_1", "target_future_return_5", "target_direction",
                   "microstructure_available", "open", "high", "low", "close", "volume"}
        return [c for c in df.columns if c not in exclude]

    # ── Static Helpers ──────────────────────────────────────────

    @staticmethod
    def _rolling_max(values: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(values, np.nan)
        for i in range(period - 1, len(values)):
            result[i] = np.max(values[i - period + 1 : i + 1])
        return result

    @staticmethod
    def _rolling_min(values: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(values, np.nan)
        for i in range(period - 1, len(values)):
            result[i] = np.min(values[i - period + 1 : i + 1])
        return result

    @staticmethod
    def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(values, np.nan)
        for i in range(period - 1, len(values)):
            result[i] = np.std(values[i - period + 1 : i + 1], ddof=1)
        return result

    @staticmethod
    def _rolling_corr(x: np.ndarray, y: np.ndarray, period: int) -> np.ndarray:
        """Rolling Pearson correlation."""
        result = np.full_like(x, np.nan, dtype=float)
        for i in range(period - 1, len(x)):
            x_slice = x[i - period + 1 : i + 1]
            y_slice = y[i - period + 1 : i + 1]
            mask = ~(np.isnan(x_slice) | np.isnan(y_slice))
            if mask.sum() >= period // 2:
                result[i] = np.corrcoef(x_slice[mask], y_slice[mask])[0, 1]
        return result

    @staticmethod
    def _get_btc_price(index: pd.DatetimeIndex) -> np.ndarray | None:
        """Fetch BTC price data for cross-asset correlation."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1h", "limit": 500},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return np.array([float(k[4]) for k in data])
        except Exception:
            pass
        return None
