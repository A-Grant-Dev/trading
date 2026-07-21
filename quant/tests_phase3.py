"""
Phase 3 — Alternative Data & Sentiment Signals Tests

Tests cover:
  - Google Trends signal (fallback logic, score classification)
  - GitHub activity signal (repo mapping, edge cases)
  - On-chain metrics (supported/unsupported assets)
  - AlternativeSentimentEngine consensus computation
  - Sentiment → Signal conversion (contrarian logic, regime adjustment)
  - Edge case handling (missing data, API failures, extreme values)
"""

from django.test import TestCase

from quant.services.alt_sentiment import (
    AlternativeSentimentEngine,
    _classify_trend,
    _compute_fallback_trends,
    _default_trends,
    _trends_to_signal,
    get_github_activity_signal,
    get_google_trends_signal,
    get_onchain_metrics,
    get_project_info,
)
from quant.services.sentiment_signals import (
    _contrarian_convert,
    _extract_scores,
    _get_regime_multiplier,
    sentiment_to_signal,
)


# ══════════════════════════════════════════════════════════════════
#  Alternative Sentiment Engine Tests
# ══════════════════════════════════════════════════════════════════


class ProjectInfoTests(TestCase):
    """Test coin-to-project mapping."""

    def test_known_coin(self):
        """Known coin should return full project info."""
        info = get_project_info("BTC")
        self.assertEqual(info["github"], "bitcoin/bitcoin")
        self.assertEqual(info["coingecko_id"], "bitcoin")

    def test_unknown_coin(self):
        """Unknown coin should return sensible defaults."""
        info = get_project_info("SHIBX")
        self.assertIsNone(info["github"])
        self.assertEqual(info["coingecko_id"], "shibx")

    def test_case_insensitive(self):
        """Mapping should be case-insensitive."""
        info_lower = get_project_info("btc")
        info_upper = get_project_info("BTC")
        self.assertEqual(info_lower["github"], info_upper["github"])


class GoogleTrendsTests(TestCase):
    """Test Google Trends signal generation."""

    def test_default_trends(self):
        """Default trends should be neutral."""
        result = _default_trends()
        self.assertEqual(result["interest_score"], 50.0)
        self.assertEqual(result["signal"], "neutral")

    def test_fallback_trends_major_coin(self):
        """Major coins should get a slight boost in fallback."""
        result = _compute_fallback_trends("Bitcoin")
        self.assertGreaterEqual(result["interest_score"], 50.0)

    def test_fallback_trends_unknown(self):
        """Unknown coins should use default fallback."""
        result = _compute_fallback_trends("UNKNOWN_COIN_XYZ")
        self.assertEqual(result["interest_score"], 50.0)

    def test_classify_trend(self):
        """Trend classification should cover all ranges."""
        self.assertEqual(_classify_trend(85), "spiking")
        self.assertEqual(_classify_trend(70), "rising")
        self.assertEqual(_classify_trend(50), "stable")
        self.assertEqual(_classify_trend(30), "declining")
        self.assertEqual(_classify_trend(10), "low")

    def test_trends_to_signal(self):
        """Signal should be contrarian (extreme interest = bearish)."""
        self.assertEqual(_trends_to_signal(10), "bullish")   # Low interest = accumulation
        self.assertEqual(_trends_to_signal(50), "neutral")   # Normal = no signal
        self.assertEqual(_trends_to_signal(90), "bearish")   # Extreme = potential top


class GitHubActivityTests(TestCase):
    """Test GitHub activity signal generation."""

    def test_no_repo(self):
        """None repo should return neutral with note."""
        result = get_github_activity_signal(None)
        self.assertEqual(result["signal"], "neutral")
        self.assertEqual(result["commit_count"], None)

    def test_unknown_repo_format(self):
        """Invalid repo format should still attempt API and handle failure."""
        result = get_github_activity_signal("nonexistent/repo-12345")
        self.assertIn("signal", result)
        # API call should return a note or status message without crashing
        note = result.get("note", "") or result.get("signal", "")
        self.assertIsNotNone(note)

    def test_known_repo_returns_data(self):
        """Known repo should return commit data (or timeout gracefully)."""
        result = get_github_activity_signal("bitcoin/bitcoin")
        self.assertIn("signal", result)
        # May get rate limited, but shouldn't crash
        if result.get("commit_count") is not None:
            self.assertGreaterEqual(result["commit_count"], 0)
            self.assertIn(result["activity_trend"], ["increasing", "decreasing", "stable", "loading", "unknown"])


class OnChainMetricsTests(TestCase):
    """Test on-chain metrics fetching."""

    def test_unsupported_asset(self):
        """Unsupported assets should return not-supported response."""
        result = get_onchain_metrics("SHIBX")
        self.assertFalse(result["supported"])

    def test_supported_asset_returns_data(self):
        """Supported assets should return metrics (or timeout gracefully)."""
        result = get_onchain_metrics("BTC")
        self.assertTrue(result["supported"])
        self.assertEqual(result["symbol"], "BTC")

    def test_case_insensitive(self):
        """Asset lookup should be case-insensitive."""
        result = get_onchain_metrics("btc")
        self.assertTrue(result["supported"])


class AlternativeSentimentEngineTests(TestCase):
    """Test AlternativeSentimentEngine consensus computation."""

    def setUp(self):
        self.engine = AlternativeSentimentEngine()

    def test_consensus_with_news_data(self):
        """Engine should compute consensus from news sentiment."""
        news_data = {
            "overall_sentiment": {"score": 75.0, "label": "bullish"},
            "fear_greed": {"value": 80, "classification": "Extreme Greed"},
        }
        result = self.engine.compute_consensus_score(
            "BTC",
            news_sentiment=news_data,
            fear_greed={"value": 80, "classification": "Extreme Greed"},
        )

        self.assertIn("consensus_score", result)
        self.assertIn("consensus_label", result)
        self.assertIn("breakdown", result)
        # With bullish news and fear-greed, consensus should be bullish
        self.assertGreaterEqual(result["consensus_score"], 50)

    def test_consensus_without_data(self):
        """Engine should compute consensus even without pre-fetched data."""
        result = self.engine.compute_consensus_score("BTC")
        self.assertIn("consensus_score", result)
        self.assertIn("consensus_label", result)
        # Should not crash — all sources will attempt API calls or use fallbacks

    def test_signal_to_score_mapping(self):
        """Signal strings should map to correct scores."""
        self.assertEqual(self.engine._signal_to_score("bullish"), 80.0)
        self.assertEqual(self.engine._signal_to_score("neutral"), 50.0)
        self.assertEqual(self.engine._signal_to_score("bearish"), 20.0)
        self.assertEqual(self.engine._signal_to_score("unknown"), 50.0)

    def test_consensus_unknown_coin(self):
        """Engine should handle unknown coins gracefully."""
        result = self.engine.compute_consensus_score("UNKNOWN_COIN_XYZ")
        self.assertIn("consensus_score", result)
        # Should get a default neutral-ish score
        self.assertGreaterEqual(result["consensus_score"], 0)
        self.assertLessEqual(result["consensus_score"], 100)


# ══════════════════════════════════════════════════════════════════
#  Sentiment → Signal Converter Tests
# ══════════════════════════════════════════════════════════════════


class SentimentSignalTests(TestCase):
    """Test sentiment_to_signal conversion with contrarian logic."""

    def test_extreme_bearish_bullish_signal(self):
        """Extreme bearish sentiment should produce a contrarian buy (long) signal."""
        data = {
            "overall_sentiment": {"score": 5.0, "label": "bearish"},
            "fear_greed": {"value": 8, "classification": "Extreme Fear"},
            "breakdown": {"bullish": 1, "bearish": 9, "neutral": 0, "total": 10},
        }
        result = sentiment_to_signal(data, "ranging")
        self.assertEqual(result["direction"], "long")  # Contrarian: bearish → buy
        self.assertGreater(result["signal"], 0)
        self.assertGreater(result["strength"], 0)

    def test_extreme_bullish_bearish_signal(self):
        """Extreme bullish sentiment should produce a contrarian sell (short) signal."""
        data = {
            "overall_sentiment": {"score": 95.0, "label": "bullish"},
            "fear_greed": {"value": 92, "classification": "Extreme Greed"},
            "breakdown": {"bullish": 9, "bearish": 1, "neutral": 0, "total": 10},
        }
        result = sentiment_to_signal(data, "ranging")
        self.assertEqual(result["direction"], "short")  # Contrarian: bullish → sell
        self.assertLess(result["signal"], 0)
        self.assertGreater(result["strength"], 0)

    def test_neutral_sentiment_no_signal(self):
        """Neutral sentiment should produce no signal."""
        data = {
            "overall_sentiment": {"score": 50.0, "label": "neutral"},
            "fear_greed": {"value": 50, "classification": "Neutral"},
        }
        result = sentiment_to_signal(data, "ranging")
        self.assertIsNone(result["direction"])
        self.assertEqual(result["strength"], 0.0)
        self.assertEqual(result["signal"], 0.0)

    def test_volatile_regime_ignores_sentiment(self):
        """In volatile regime, sentiment signals should be heavily suppressed."""
        data = {
            "overall_sentiment": {"score": 90.0, "label": "bullish"},
            "fear_greed": {"value": 85, "classification": "Extreme Greed"},
        }
        result = sentiment_to_signal(data, "volatile")
        # Signal should be very weak due to regime multiplier
        self.assertIsNone(result["direction"])
        self.assertLess(abs(result["signal"]), 0.3)

    def test_no_data(self):
        """Empty sentiment data should produce zero signal."""
        result = sentiment_to_signal({}, "ranging")
        self.assertEqual(result["signal"], 0.0)
        self.assertEqual(result["sources_used"], 0)

    def test_contrarian_convert_dead_zone(self):
        """Middle sentiment values should produce no signal."""
        self.assertEqual(_contrarian_convert(50), 0.0)
        self.assertEqual(_contrarian_convert(40), 0.0)
        self.assertEqual(_contrarian_convert(60), 0.0)

    def test_contrarian_convert_extremes(self):
        """Extreme sentiment values should produce strong contrarian signals."""
        # Score 0 → bearish extreme → contrarian buy → +1.0
        self.assertAlmostEqual(_contrarian_convert(0), 1.0)
        # Score 100 → bullish extreme → contrarian sell → -1.0
        self.assertAlmostEqual(_contrarian_convert(100), -1.0)
        # Score 35 → boundary bearish → contrarian buy → 0.0
        self.assertAlmostEqual(_contrarian_convert(35), 0.0)
        # Score 65 → boundary bullish → contrarian sell → 0.0
        self.assertAlmostEqual(_contrarian_convert(65), 0.0)

    def test_contrarian_convert_linear(self):
        """Conversion should be linear between boundaries."""
        # Bearish side: score 10 → (35-10)/35 = 0.714
        self.assertAlmostEqual(_contrarian_convert(10), 25/35)
        # Bullish side: score 80 → (65-80)/35 = -0.429
        self.assertAlmostEqual(_contrarian_convert(80), -15/35)

    def test_regime_multiplier_values(self):
        """Regime multipliers should match expectations."""
        self.assertEqual(_get_regime_multiplier("ranging"), 0.6)
        self.assertEqual(_get_regime_multiplier("bullish"), 0.8)
        self.assertEqual(_get_regime_multiplier("bearish"), 0.8)
        self.assertEqual(_get_regime_multiplier("volatile"), 0.2)

    def test_extract_scores(self):
        """Score extraction should pull from all available sources."""
        data = {
            "overall_sentiment": {"score": 75.0, "label": "bullish"},
            "fear_greed": {"value": 80, "classification": "Extreme Greed"},
            "breakdown": {"bullish": 8, "bearish": 2, "neutral": 0, "total": 10},
            "consensus_score": 70.0,
            "consensus_label": "bullish",
        }
        scores = _extract_scores(data)
        self.assertIn("news_headlines", scores)
        self.assertIn("fear_greed", scores)
        self.assertIn("source_ratio", scores)
        self.assertGreater(len(scores), 0)

    def test_complete_signal_with_empty_data(self):
        """Empty data should still produce a valid result structure."""
        result = sentiment_to_signal({})
        self.assertIn("signal", result)
        self.assertIn("strength", result)
        self.assertIn("direction", result)
        self.assertIn("sources_used", result)
        self.assertIn("confidence", result)
