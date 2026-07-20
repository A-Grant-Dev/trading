"""
Regime-Aware Signal Weighting

Adjusts trading signal confidence and strength based on the current
market regime detected by the HMM. This is the bridge between regime
detection (Phase 1) and signal generation (Phases 2-4).

Renaissance/Simons principle: Different market environments require
different mathematical models. A strategy that works in a trending
market will fail in a ranging market, and vice versa.

Adjustment rules:
    Ranging (0):     Reduce signal confidence, prefer mean-reversion
    Bullish (1):     Amplify long signals, reduce short signals
    Bearish (2):     Amplify short signals, reduce long signals
    Volatile (3):    Reduce all signals significantly, widen thresholds
"""

import logging

logger = logging.getLogger(__name__)

# ── Regime Signal Weights ──────────────────────────────────────────

REGIME_WEIGHTS = {
    "ranging": {
        "long": 0.5,
        "short": 0.5,
        "description": "Ranging — low conviction on directional signals",
        "max_position_pct": 5.0,
        "preferred_strategies": ["mean_reversion", "cointegration"],
    },
    "bullish": {
        "long": 1.2,
        "short": 0.4,
        "description": "Bullish trend — amplify longs, suppress shorts",
        "max_position_pct": 10.0,
        "preferred_strategies": ["momentum", "breakout", "ml_ensemble"],
    },
    "bearish": {
        "long": 0.4,
        "short": 1.2,
        "description": "Bearish trend — amplify shorts, suppress longs",
        "max_position_pct": 8.0,
        "preferred_strategies": ["momentum", "hedging"],
    },
    "volatile": {
        "long": 0.3,
        "short": 0.3,
        "description": "High volatility — severely reduce all positions",
        "max_position_pct": 2.0,
        "preferred_strategies": ["volatility_arbitrage", "cash"],
    },
}

# Default weight for unknown regimes
DEFAULT_WEIGHT = 0.5

# Threshold adjustments per regime
REGIME_THRESHOLDS = {
    "ranging": {"entry_z": 2.5, "exit_z": 0.8},
    "bullish": {"entry_z": 2.0, "exit_z": 0.5},
    "bearish": {"entry_z": 2.0, "exit_z": 0.5},
    "volatile": {"entry_z": 3.0, "exit_z": 1.0},
}


def adjust_signal_for_regime(
    base_signal: float,
    direction: str,
    regime_label: str,
) -> float:
    """
    Adjust signal confidence based on market regime.

    Applies a multiplier that amplifies or suppresses the base signal
    based on how well the signal's direction aligns with the current regime.

    Args:
        base_signal: Original signal confidence (0.0 to 1.0)
        direction: Signal direction ('long' or 'short')
        regime_label: Current regime label (ranging, bullish, bearish, volatile)

    Returns:
        Adjusted signal confidence (0.0 to 1.0, clipped)
    """
    regime = REGIME_WEIGHTS.get(regime_label)
    if not regime:
        return base_signal * DEFAULT_WEIGHT

    weight = regime.get(direction, DEFAULT_WEIGHT)
    adjusted = base_signal * weight

    # Clip to valid range
    return max(0.0, min(1.0, adjusted))


def get_regime_adjusted_thresholds(regime_label: str) -> dict:
    """
    Get entry/exit thresholds adjusted for the current regime.

    In volatile regimes, wider thresholds avoid false signals.
    In ranging regimes, tighter thresholds capture small movements.

    Args:
        regime_label: Current regime label

    Returns:
        Dict with 'entry_z' and 'exit_z' thresholds
    """
    return REGIME_THRESHOLDS.get(regime_label, REGIME_THRESHOLDS["ranging"])


def get_max_position_pct(regime_label: str) -> float:
    """
    Get maximum position size as percentage of portfolio for this regime.

    Args:
        regime_label: Current regime label

    Returns:
        Max position % (e.g., 10.0 = 10% of portfolio)
    """
    regime = REGIME_WEIGHTS.get(regime_label)
    if regime:
        return regime["max_position_pct"]
    return 5.0


def get_preferred_strategies(regime_label: str) -> list[str]:
    """
    Get list of preferred strategy types for the current regime.

    Different strategies perform best in different market environments.
    This helps the SignalCombiner (Phase 5) weight signals appropriately.

    Args:
        regime_label: Current regime label

    Returns:
        List of strategy type strings
    """
    regime = REGIME_WEIGHTS.get(regime_label)
    if regime:
        return regime["preferred_strategies"]
    return ["mean_reversion"]


def combine_regime_with_signal(
    base_signal: dict,
    regime_label: str,
    regime_confidence: float,
) -> dict:
    """
    Combine a base signal with regime information for a final decision.

    This is the main integration point between HMM regime detection
    and all downstream signal generators.

    Args:
        base_signal: Dict with 'direction', 'strength', 'confidence'
        regime_label: Current regime label
        regime_confidence: HMM confidence in this regime (0.0-1.0)

    Returns:
        Dict with adjusted signal info:
            - original_strength
            - adjusted_strength
            - regime_weight_applied
            - max_position_pct
            - preferred_strategies
            - recommendation: 'execute', 'reduce', 'avoid'
    """
    direction = base_signal.get("direction", "long")
    strength = base_signal.get("strength", 0.5)
    confidence = base_signal.get("confidence", 0.5)

    adjusted = adjust_signal_for_regime(strength, direction, regime_label)

    # Determine recommendation based on adjusted strength
    if adjusted > 0.6:
        recommendation = "execute"
    elif adjusted > 0.3:
        recommendation = "reduce"
    else:
        recommendation = "avoid"

    return {
        "original_strength": strength,
        "adjusted_strength": round(adjusted, 4),
        "regime": regime_label,
        "regime_confidence": regime_confidence,
        "regime_weight_applied": REGIME_WEIGHTS.get(regime_label, {}).get(direction, DEFAULT_WEIGHT),
        "max_position_pct": get_max_position_pct(regime_label),
        "preferred_strategies": get_preferred_strategies(regime_label),
        "thresholds": get_regime_adjusted_thresholds(regime_label),
        "recommendation": recommendation,
    }
