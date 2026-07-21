"""
Signal Combiner — Aggregates signals from all quant sources into trade orders.

Sources:
  - Phase 2: Pairs trading signals (cointegration)
  - Phase 3: Sentiment-based signals (alt data, fear & greed)
  - Phase 4: ML model predictions (ensemble)
  - Phase 1: Regime override (HMM)

Renaissance principle: The computer decides, the computer executes.
No human in the loop. Combine many weak signals into one strong decision.
"""

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from quant.models import TradeSignal
from quant.services.data_feeds import get_cache

logger = logging.getLogger(__name__)

# ── Default Weights ────────────────────────────────────────────────

# Map from source_model codes to weight keys
SOURCE_KEY_MAP: dict[str, str] = {
    "cointegration": "cointegration",
    "ml_ensemble": "ml_ensemble",
    "sentiment": "sentiment",
    "alt_data": "sentiment",      # Alternative data maps to sentiment weight
    "futures": "sentiment",       # Futures data maps to sentiment weight
    "orderbook": "orderbook",
    "hmm_regime": "",            # Regime handled separately, not weighted
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "cointegration": 0.40,   # Highest conviction — stat arb
    "ml_ensemble": 0.30,     # Medium conviction — ML predictions
    "sentiment": 0.15,       # Lower conviction — contrarian signals
    "orderbook": 0.15,       # Short-term micro-signal
}

CONFIDENCE_THRESHOLD = 0.55  # > 55% = trade (law of large numbers)

# Regime multipliers — adjust combined confidence
REGIME_MULTIPLIERS: dict[str, float] = {
    "ranging": 0.6,    # Low conviction — chop
    "bullish": 1.0,    # Normal conviction — trend is friend
    "bearish": 1.0,    # Normal conviction — trend is friend
    "volatile": 0.3,   # Very low conviction — random noise
}


class SignalCombiner:
    """
    Aggregates signals from all sources into actionable trade orders.

    Usage:
        combiner = SignalCombiner()
        order = combiner.combine(symbol="BTCUSDT")
        if order:
            order_manager.execute(order)
    """

    def __init__(self, weights: dict[str, float] | None = None):
        """
        Args:
            weights: Source weights. Defaults to pre-tuned values.
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def combine(self, symbol: str | None = None,
                regime: str | None = None) -> dict | None:
        """
        Collect all active signals for a symbol and combine into an order.

        Args:
            symbol: Trading symbol (e.g., BTCUSDT). If None, scans all.
            regime: Current market regime label. Auto-detected if None.

        Returns:
            Order dict with keys:
                symbol, side, confidence, reason, source_breakdown
            or None if confidence below threshold or no signals.
        """
        # Collect active signals across all sources
        now = datetime.now(timezone.utc)
        all_signals = TradeSignal.objects.filter(
            status="active",
            expiry__gt=now,
        )

        if symbol:
            all_signals = all_signals.filter(symbol=symbol.upper())

        if not all_signals.exists():
            return None

        # Group signals by source and direction
        source_groups: dict[str, list[TradeSignal]] = {}
        for sig in all_signals:
            src = sig.source_model
            if src not in source_groups:
                source_groups[src] = []
            source_groups[src].append(sig)

        # Compute weighted consensus per source
        source_votes: dict[str, dict] = {}
        for src, signals in source_groups.items():
            source_votes[src] = self._compute_source_consensus(src, signals)

        # Apply regime multiplier
        regime = regime or self._detect_regime(symbol)
        regime_mult = REGIME_MULTIPLIERS.get(regime, 1.0)

        # Combine into final decision
        return self._aggregate(source_votes, regime, regime_mult, symbol)

    def _compute_source_consensus(self, source: str,
                                  signals: list[TradeSignal]) -> dict:
        """
        Compute a single consensus from one source's signals.

        Returns dict with:
            long_confidence, short_confidence, neutral_confidence,
            weighted_direction, top_reason
        """
        long_conf = 0.0
        short_conf = 0.0
        neutral_conf = 0.0
        total_weight = 0.0

        reasons: list[str] = []

        for sig in signals:
            w = sig.strength * sig.confidence
            total_weight += w

            if sig.direction == "long" or sig.signal_type == "long":
                long_conf += w
                if sig.metadata and "reason" in sig.metadata:
                    reasons.append(sig.metadata["reason"])
            elif sig.direction == "short" or sig.signal_type == "short":
                short_conf += w
                if sig.metadata and "reason" in sig.metadata:
                    reasons.append(sig.metadata["reason"])
            else:
                neutral_conf += w

        # Normalize to 0-1 range
        if total_weight > 0:
            long_conf /= total_weight
            short_conf /= total_weight
            neutral_conf /= total_weight

        # Net direction (-1 to 1)
        net = long_conf - short_conf

        # Average signal quality for this source (strength * confidence)
        avg_strength = total_weight / len(signals) if signals else 0.0

        top_reason = reasons[0] if reasons else f"{source} signal"

        return {
            "long_confidence": round(long_conf, 4),
            "short_confidence": round(short_conf, 4),
            "net_signal": round(net, 4),
            "avg_strength": round(avg_strength, 4),
            "signal_count": len(signals),
            "total_weight": round(total_weight, 4),
            "top_reason": top_reason,
        }

    def _aggregate(self, source_votes: dict[str, dict],
                   regime: str, regime_mult: float,
                   symbol: str | None) -> dict | None:
        """
        Combine weighted source votes into a single order decision.

        Implements the core aggregation logic:
          1. Map each source to its weight key, get configured weight
          2. Multiply net_signal by avg_strength to respect signal quality
          3. Apply regime multiplier
          4. If absolute net confidence > threshold → generate order
        """
        total_confidence = 0.0
        source_breakdown: dict[str, Any] = {}
        reasons: list[str] = []
        total_weight_applied = 0.0

        for src, vote in source_votes.items():
            # Map source_model to weight key BEFORE checking weight
            src_key = SOURCE_KEY_MAP.get(src, src)
            if not src_key:  # hmm_regime → skip (handled separately)
                continue

            weight = self.weights.get(src_key, 0.05)
            if weight <= 0:
                continue

            # Scale net_signal by avg_strength so weak signals don't get full weight
            avg_strength = vote.get("avg_strength", 1.0)
            contribution = vote["net_signal"] * avg_strength * weight

            total_confidence += contribution
            total_weight_applied += weight

            source_breakdown[src] = {
                "net_signal": vote["net_signal"],
                "avg_strength": avg_strength,
                "weight": weight,
                "contribution": round(contribution, 4),
                "signal_count": vote["signal_count"],
                "top_reason": vote["top_reason"],
            }
            reasons.append(f"{src}({vote['top_reason']})")

        # Apply regime multiplier to final confidence
        if total_weight_applied > 0:
            total_confidence /= total_weight_applied
        total_confidence *= regime_mult

        # Clamp to [-1, 1]
        total_confidence = max(-1.0, min(1.0, total_confidence))

        logger.info(
            "SignalCombiner %s: conf=%.3f (threshold=%.2f), "
            "regime=%s, regime_mult=%.1f, reasons=%s",
            symbol or "ALL", abs(total_confidence),
            CONFIDENCE_THRESHOLD, regime, regime_mult, reasons,
        )

        # Decision: abs confidence > threshold → trade
        if abs(total_confidence) < CONFIDENCE_THRESHOLD:
            return {
                "symbol": symbol,
                "side": None,
                "confidence": round(total_confidence, 4),
                "action": "hold",
                "reason": f"Confidence {abs(total_confidence):.1%} below {CONFIDENCE_THRESHOLD:.0%} threshold",
                "regime": regime,
                "regime_multiplier": regime_mult,
                "source_breakdown": source_breakdown,
            }

        side = "buy" if total_confidence > 0 else "sell"

        return {
            "symbol": symbol,
            "side": side,
            "confidence": round(abs(total_confidence), 4),
            "action": "trade",
            "reason": "; ".join(reasons[:3]),
            "regime": regime,
            "regime_multiplier": regime_mult,
            "source_breakdown": source_breakdown,
        }

    @staticmethod
    def _detect_regime(symbol: str | None = None) -> str:
        """Detect current regime from cache or default to ranging."""
        try:
            if symbol:
                data = get_cache(f"regime:{symbol}")
                if data and "regime_label" in data:
                    return data["regime_label"].lower()

            all_data = get_cache("regime:all", {})
            if all_data:
                # Return most common regime across symbols
                regimes = [
                    d.get("regime_label", "ranging").lower()
                    for d in all_data.values()
                ]
                if regimes:
                    return Counter(regimes).most_common(1)[0][0]
        except Exception as e:
            logger.debug(f"Failed to detect regime: {e}")

        return "ranging"

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update source weights dynamically based on recent performance.

        Args:
            new_weights: Dict mapping source key → weight
                         (e.g., {'cointegration': 0.5, 'ml_ensemble': 0.3})
        """
        total = sum(new_weights.values())
        if total <= 0:
            logger.warning("Invalid weights (total <= 0), keeping current")
            return

        # Normalize to sum to 1.0
        normalized = {k: v / total for k, v in new_weights.items()}
        self.weights.update(normalized)
        logger.info("Updated signal weights: %s", self.weights)
