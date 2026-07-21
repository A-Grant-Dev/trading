"""
Hidden Markov Model — Market Regime Detection

Identifies the current "market regime" (quiet, trending up, trending down,
highly volatile) using a Gaussian HMM. This regime label feeds into all
downstream models — different strategies activate in different regimes.

Renaissance/Simons principle: Don't predict price direction directly.
First identify which ENVIRONMENT the market is in, then apply the
appropriate statistical model. An HMM is the ideal tool for this.

States (default, 4 regimes):
    0: Low volatility / ranging (sideways consolidation)
    1: Bullish trend / low volatility (steady uptrend)
    2: Bearish trend / low volatility (steady downtrend)
    3: High volatility / chaotic (large swings, uncertain direction)
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from hmmlearn import hmm

logger = logging.getLogger(__name__)

# Default feature columns expected by the HMM
DEFAULT_FEATURES = ["log_return", "volatility_14", "volume_change", "rsi_14", "atr_pct"]

# Pre-defined regime labels — mapped after training
REGIME_LABELS = {
    0: "ranging",
    1: "bullish",
    2: "bearish",
    3: "volatile",
}

# Regime descriptions for dashboard display
REGIME_DESCRIPTIONS = {
    "ranging": "Low volatility, sideways consolidation. Best for mean-reversion and range-bound strategies.",
    "bullish": "Steady upward trend with normal volatility. Long bias with standard position sizing.",
    "bearish": "Steady downward trend with normal volatility. Short bias or defensive positioning.",
    "volatile": "High uncertainty, large price swings. Severe position size reduction, wider stops.",
}


class MarketRegimeDetector:
    """
    Hidden Markov Model that identifies latent market regimes.

    Trains on historical features to learn the statistical properties
    of each regime state, then predicts the current regime in real-time.

    Usage:
        detector = MarketRegimeDetector(n_states=4)
        detector.train(df)
        regime_id, regime_label = detector.predict_regime(feature_vector)
        confidence = detector.get_regime_probabilities(feature_vector)
    """

    def __init__(self, n_states: int = 4):
        """
        Initialize the HMM detector.

        Args:
            n_states: Number of hidden states (default 4)
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            tol=0.01,
            random_state=42,
            init_params="stmc",  # Initialize all params
        )
        self._is_trained = False
        self._regime_map: dict[int, str] = {}
        self._feature_means: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained

    def train(self, df: pd.DataFrame, feature_columns: list[str] = None) -> None:
        """
        Train the HMM on historical data.

        Learns the transition probabilities and emission distributions
        for each hidden state. After training, states are automatically
        labeled based on their statistical properties.

        Args:
            df: DataFrame with feature columns and datetime index
            feature_columns: List of column names to use as features.
                            Defaults to [log_return, volatility_14, volume_change, rsi_14, atr_pct]

        Raises:
            ValueError: If data is insufficient or features are missing
        """
        if feature_columns is None:
            feature_columns = DEFAULT_FEATURES

        # Validate data
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        df_clean = df[feature_columns].dropna()
        if len(df_clean) < self.n_states * 10:
            raise ValueError(
                f"Not enough data: need at least {self.n_states * 10} samples, "
                f"got {len(df_clean)}"
            )

        # Store normalization parameters
        for col in feature_columns:
            self._feature_means[col] = float(df_clean[col].mean())
            self._feature_stds[col] = float(df_clean[col].std())

        # Normalize features
        X = (df_clean.values - np.array([self._feature_means[c] for c in feature_columns])) / \
            np.array([self._feature_stds[c] for c in feature_columns])

        # Handle NaN/Inf in normalized values
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)

        # Train HMM
        try:
            self.model.fit(X)
            self._is_trained = True
        except Exception as e:
            logger.error(f"HMM training failed: {e}")
            raise

        # Auto-label states
        self._label_states(df_clean, feature_columns)

        logger.info(
            f"HMM trained: {self.n_states} states, "
            f"{len(df_clean)} samples, "
            f"{len(feature_columns)} features"
        )

    def _label_states(self, df: pd.DataFrame, feature_columns: list[str]) -> None:
        """
        Automatically label hidden states based on their characteristics.

        Uses the training data decoded states to compute:
        - Mean return for each state → bullish/bearish/ranging
        - Mean volatility for each state → volatile vs calm

        The labeling is deterministic given the same training data.
        """
        X = (df[feature_columns].values - np.array([self._feature_means[c] for c in feature_columns])) / \
            np.array([self._feature_stds[c] for c in feature_columns])
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)

        state_sequence = self.model.predict(X)

        return_idx = feature_columns.index("log_return")
        vol_idx = feature_columns.index("volatility_14") if "volatility_14" in feature_columns else None

        state_stats = {}
        for state in range(self.n_states):
            mask = state_sequence == state
            if mask.sum() == 0:
                state_stats[state] = {"mean_return": 0.0, "mean_vol": 1.0}
                continue

            mean_return = float(np.mean(df[feature_columns].values[mask][:, return_idx]))
            mean_vol = float(np.mean(df[feature_columns].values[mask][:, vol_idx])) if vol_idx else 0.5
            state_stats[state] = {"mean_return": mean_return, "mean_vol": mean_vol}

        # Find the state with highest volatility
        vol_state = max(state_stats, key=lambda s: state_stats[s]["mean_vol"])

        # Among remaining states, find bullish (highest return) and bearish (lowest return)
        calm_states = [s for s in range(self.n_states) if s != vol_state]
        if calm_states:
            bullish_state = max(calm_states, key=lambda s: state_stats[s]["mean_return"])
            bearish_state = min(calm_states, key=lambda s: state_stats[s]["mean_return"])
        else:
            bullish_state = 0
            bearish_state = 0

        # Assign labels
        self._regime_map = {}
        for s in range(self.n_states):
            if s == vol_state:
                self._regime_map[s] = "volatile"
            elif s == bullish_state and bullish_state != bearish_state:
                self._regime_map[s] = "bullish"
            elif s == bearish_state and bullish_state != bearish_state:
                self._regime_map[s] = "bearish"
            else:
                self._regime_map[s] = "ranging"

        logger.debug(f"State labeling complete: {self._regime_map}")

    def predict_regime(self, feature_vector: np.ndarray) -> tuple[int, str]:
        """
        Predict the current market regime.

        Args:
            feature_vector: 1D numpy array of features (same order as training)

        Returns:
            Tuple of (state_number, label) where label is one of:
            'ranging', 'bullish', 'bearish', 'volatile'
        """
        if not self._is_trained:
            return -1, "unknown"

        # Normalize
        feature_vector = np.array(feature_vector, dtype=float).reshape(1, -1)
        normalized = self._normalize(feature_vector)

        state = int(self.model.predict(normalized)[0])
        label = self._regime_map.get(state, f"state_{state}")
        return state, label

    def get_regime_probabilities(self, feature_vector: np.ndarray) -> dict:
        """
        Get probability distribution across all regimes.

        Returns the posterior probability of each regime given the
        current observation. Higher confidence = more certain.

        Args:
            feature_vector: 1D numpy array of features

        Returns:
            Dict mapping state_number -> {
                'label': str, 'probability': float
            }
            Sorted by probability descending.
        """
        if not self._is_trained:
            return {}

        feature_vector = np.array(feature_vector, dtype=float).reshape(1, -1)
        normalized = self._normalize(feature_vector)

        probabilities = self.model.predict_proba(normalized)[0]

        results = {}
        for state in range(self.n_states):
            results[state] = {
                "label": self._regime_map.get(state, f"state_{state}"),
                "probability": round(float(probabilities[state]), 4),
            }

        # Sort by probability descending
        return dict(
            sorted(results.items(), key=lambda x: x[1]["probability"], reverse=True)
        )

    def get_state_sequence(self, df: pd.DataFrame, feature_columns: list[str] = None) -> pd.Series:
        """
        Predict regime for the entire DataFrame.

        Returns a Series with the same index as df, containing
        regime labels for each row.

        Args:
            df: DataFrame with feature columns
            feature_columns: List of feature column names

        Returns:
            Series with regime labels indexed by df's index
        """
        if not self._is_trained:
            return pd.Series(index=df.index, dtype=str)

        if feature_columns is None:
            feature_columns = DEFAULT_FEATURES

        df_clean = df[feature_columns].dropna()
        X = self._normalize(df_clean.values)
        states = self.model.predict(X)

        result = pd.Series(index=df_clean.index, dtype=str)
        for i, state in enumerate(states):
            result.iloc[i] = self._regime_map.get(int(state), f"state_{int(state)}")

        return result.reindex(df.index)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize feature vector using stored means/stds."""
        if not self._feature_means:
            return X
        means = np.array([self._feature_means.get(c, 0) for c in DEFAULT_FEATURES[:X.shape[1]]])
        stds = np.array([self._feature_stds.get(c, 1) for c in DEFAULT_FEATURES[:X.shape[1]]])
        stds = np.where(stds == 0, 1, stds)
        result = (X - means) / stds
        return np.nan_to_num(result, nan=0.0, posinf=3.0, neginf=-3.0)

    def get_regime_info(self) -> dict:
        """
        Get descriptive info about each regime state.

        Returns:
            Dict mapping state_number -> {
                'label': str,
                'description': str,
            }
        """
        info = {}
        for state in range(self.n_states):
            label = self._regime_map.get(state, f"state_{state}")
            info[state] = {
                "label": label,
                "description": REGIME_DESCRIPTIONS.get(label, "Unknown regime"),
            }
        return info


# ── Feature Builder ────────────────────────────────────────────────

# Reuse technical indicator functions from data_utils to avoid duplication
from quant.services.data_utils import compute_rsi, compute_atr


def build_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform OHLCV data into HMM-ready features.

    Computes the feature columns needed by MarketRegimeDetector:
        - log_return: Log returns of close price
        - volatility_14: 14-period rolling std of log returns
        - volume_change: % change in volume
        - rsi_14: 14-period RSI
        - atr_pct: ATR as % of close price

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with additional feature columns, NaN rows removed
    """
    # Validate required columns
    required = ["close", "high", "low"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Missing required columns for HMM features: {missing}")
        return pd.DataFrame()

    df = df.copy()

    # Ensure we have numpy float arrays
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float) if "volume" in df.columns else None

    # Log returns
    log_return = np.full_like(close, 0.0, dtype=float)
    log_return[1:] = np.log(close[1:] / close[:-1])
    df["log_return"] = log_return

    # Volatility (14-period rolling std of log returns)
    df["volatility_14"] = df["log_return"].rolling(14).std()

    # Volume change %
    if volume is not None:
        df["volume_change"] = df["volume"].pct_change()
    else:
        df["volume_change"] = 0.0

    # RSI (14) — reuse from data_utils
    df["rsi_14"] = compute_rsi(close, 14)

    # ATR % (14) — reuse from data_utils
    df["atr_pct"] = compute_atr(high, low, close, 14) / close * 100

    # Drop NaN rows from indicator warmup
    df.dropna(inplace=True)

    return df
