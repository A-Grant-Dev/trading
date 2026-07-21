"""
Sentiment → Quant Signal Converter

Transforms multi-source sentiment data into actionable trading signals.
Implements Renaissance's contrarian approach: extreme sentiment is often
a counter-indicator — when everyone is bullish, the smart money is selling.

Architecture:
  1. Aggregate sentiment scores from all sources (news, trends, on-chain)
  2. Apply regime-aware adjustments (sentiment is unreliable in high vol)
  3. Convert to -1.0 to 1.0 signal with dead zone in the middle
  4. Return signal with confidence and metadata

Renaissance/Simons principle: Markets are driven by emotion in the short
term and fundamentals in the long term. Extreme sentiment = opportunity
to fade the crowd.
"""

import logging
from datetime import datetime, timedelta, timezone

from quant.models import TradeSignal

logger = logging.getLogger(__name__)


def sentiment_to_signal(
    sentiment_data: dict,
    regime_label: str = "ranging",
) -> dict:
    """
    Convert multi-source sentiment into a -1.0 to 1.0 trading signal.

    The conversion applies a Renaissance-style contrarian approach:
      - Extreme bullish sentiment → sell signal (fade the crowd)
      - Extreme bearish sentiment → buy signal (buy the fear)
      - Neutral sentiment → no signal (wait for opportunity)

    Regime adjustment:
      - High volatility (volatile): Sentiment is unreliable → 0.0 signal
      - Trending regimes: Amplify contrarian signals slightly

    Args:
        sentiment_data: Dict with sentiment scores (from sentiment app or
                       AlternativeSentimentEngine). Expected keys:
                       - 'overall_sentiment': {'score': 0-100}
                       - 'fear_greed': {'value': 0-100}
                       - 'breakdown': {'bullish': int, 'bearish': int, 'total': int}
        regime_label: Current HMM regime label ('ranging', 'bullish', 'bearish', 'volatile')

    Returns:
        Dict with:
          - signal: Float from -1.0 (extreme buy) to 1.0 (extreme sell)
            (contrarian: -1.0 means sentiment is extremely bearish → BUY signal)
          - strength: Absolute signal strength (0.0-1.0)
          - direction: Underlying direction ('long' or 'short' or None)
          - sources_used: How many sources contributed
          - confidence: How confident we are in this signal
          - details: Per-source breakdown
    """
    now = datetime.now(timezone.utc)

    # Extract scores from various sources
    scores = _extract_scores(sentiment_data)

    if not scores:
        return {
            "signal": 0.0,
            "strength": 0.0,
            "direction": None,
            "sources_used": 0,
            "confidence": 0.0,
            "regime": regime_label,
            "details": {"note": "No sentiment data available"},
        }

    # Compute average sentiment across all sources
    avg_sentiment = sum(s["score"] for s in scores.values()) / len(scores)

    # Regime-based volatility adjustment
    regime_multiplier = _get_regime_multiplier(regime_label)

    # Apply contrarian conversion: 0-100 → -1.0 to 1.0
    # With dead zone in middle (35-65 → no signal)
    raw_signal = _contrarian_convert(avg_sentiment)

    # Apply regime multiplier
    adjusted_signal = raw_signal * regime_multiplier

    # Compute confidence based on:
    # - Number of sources available
    # - Extremity of sentiment (more extreme = more confidence in contrarian bet)
    source_confidence = min(1.0, len(scores) / 5.0)
    extremity = abs(avg_sentiment - 50) / 50  # 0.0 to 1.0
    confidence = source_confidence * max(0.3, extremity)

    # Determine direction (contrarian)
    if adjusted_signal > 0.3:
        direction = "long"  # Bearish sentiment → contrarian long
        strength = min(1.0, adjusted_signal)
    elif adjusted_signal < -0.3:
        direction = "short"  # Bullish sentiment → contrarian short
        strength = min(1.0, abs(adjusted_signal))
    else:
        direction = None
        strength = 0.0

    return {
        "signal": round(adjusted_signal, 4),
        "strength": round(strength, 4),
        "direction": direction,
        "sources_used": len(scores),
        "confidence": round(confidence, 4),
        "regime": regime_label,
        "regime_multiplier": regime_multiplier,
        "avg_sentiment": round(avg_sentiment, 1),
        "details": scores,
    }


def _extract_scores(sentiment_data: dict) -> dict[str, dict]:
    """
    Extract normalized 0-100 scores from all available sentiment sources.

    Args:
        sentiment_data: Raw sentiment data dict

    Returns:
        Dict of source_name -> {'score': float, 'label': str}
    """
    scores = {}

    # 1. Overall news headline sentiment
    overall = sentiment_data.get("overall_sentiment", {})
    if overall and overall.get("score") is not None:
        scores["news_headlines"] = {
            "score": float(overall["score"]),
            "label": overall.get("label", "neutral"),
        }

    # 2. Fear & Greed Index
    fg = sentiment_data.get("fear_greed", {})
    if fg and fg.get("value") is not None:
        fg_score = float(fg["value"])
        # Fear & Greed is already 0-100, no conversion needed
        scores["fear_greed"] = {
            "score": fg_score,
            "label": fg.get("classification", "Neutral").lower(),
        }

    # 3. Sentiment breakdown ratio (bullish vs bearish count)
    breakdown = sentiment_data.get("breakdown", {})
    if breakdown and breakdown.get("total", 0) > 0:
        total = breakdown["total"]
        bullish = breakdown.get("bullish", 0)
        bearish = breakdown.get("bearish", 0)
        # Convert ratio to 0-100 score: 0 = all bearish, 100 = all bullish
        ratio_score = (bullish / total) * 100 if total > 0 else 50
        scores["source_ratio"] = {
            "score": round(ratio_score, 1),
            "label": "bullish" if ratio_score > 60 else ("bearish" if ratio_score < 40 else "neutral"),
        }

    # 4. Consensus from AlternativeSentimentEngine (if available)
    consensus = sentiment_data.get("consensus_score")
    if consensus is not None:
        scores["alt_consensus"] = {
            "score": float(consensus),
            "label": sentiment_data.get("consensus_label", "neutral"),
        }

    # 5. Per-source breakdown from alt_sentiment
    breakdown_by_source = sentiment_data.get("breakdown", {}).get("scores", {})
    for source_key, source_data in breakdown_by_source.items():
        if isinstance(source_data, dict) and "score" in source_data:
            scores[f"alt_{source_key}"] = {
                "score": float(source_data["score"]),
                "label": source_data.get("label", "neutral"),
            }

    return scores


def _contrarian_convert(score: float) -> float:
    """
    Convert a 0-100 sentiment score to a -1.0 to 1.0 contrarian signal.

    The conversion creates a dead zone in the middle (35-65) where no
    signal is generated. Outside that range, the signal becomes contrarian
    (extreme bullish → negative signal = short, extreme bearish → positive signal = long).

    At score extremes (0 or 100), the signal is strongest.

    Args:
        score: Sentiment score from 0-100

    Returns:
        Contrarian signal from -1.0 to 1.0
    """
    # Dead zone: 35-65 → no signal
    if 35 <= score <= 65:
        return 0.0

    if score < 35:
        # Bearish sentiment → contrarian buy (positive signal)
        # 0 → 1.0, 35 → 0.0
        return (35 - score) / 35

    # Bullish sentiment → contrarian sell (negative signal)
    # 65 → 0.0, 100 → -1.0
    return (65 - score) / 35


def _get_regime_multiplier(regime_label: str) -> float:
    """
    Get the sentiment signal multiplier for the current market regime.

    In high volatility, sentiment is unreliable and should be ignored.
    In trending markets, sentiment can be a useful contrarian indicator.

    Args:
        regime_label: Current regime label

    Returns:
        Multiplier factor (0.0 = ignore, 1.0 = full signal)
    """
    multipliers = {
        "ranging": 0.6,    # Ranging — moderate signal
        "bullish": 0.8,    # Bullish — useful contrarian signal
        "bearish": 0.8,    # Bearish — useful contrarian signal
        "volatile": 0.2,   # Volatile — mostly ignore, sentiment unreliable
    }
    return multipliers.get(regime_label, 0.5)


def create_sentiment_signal(
    symbol: str,
    sentiment_data: dict,
    regime_label: str = "ranging",
) -> TradeSignal | None:
    """
    Create a TradeSignal record from sentiment data.

    This is the integration point between sentiment analysis and the
    quant signal system. Only creates signals when sentiment is extreme
    enough to warrant a trade.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        sentiment_data: Sentiment data dict
        regime_label: Current market regime

    Returns:
        TradeSignal instance if signal generated, None otherwise
    """
    result = sentiment_to_signal(sentiment_data, regime_label)

    # Only create signal if we have meaningful direction and confidence
    if not result["direction"] or result["confidence"] < 0.3:
        return None

    now = datetime.now(timezone.utc)

    signal = TradeSignal.objects.create(
        symbol=symbol,
        signal_type=result["direction"],
        direction=result["direction"],
        strength=result["strength"],
        confidence=result["confidence"],
        source_model="sentiment",
        generated_at=now,
        expiry=now + timedelta(hours=6),  # Sentiment signals expire in 6 hours
        status="active",
        metadata={
            "regime": regime_label,
            "avg_sentiment": result["avg_sentiment"],
            "sources_used": result["sources_used"],
            "regime_multiplier": result["regime_multiplier"],
        },
    )

    logger.info(
        f"Sentiment signal: {symbol} {result['direction'].upper()} "
        f"(conf={result['confidence']:.0%}, sentiment={result['avg_sentiment']:.0f}/100)"
    )

    return signal
