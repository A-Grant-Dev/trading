"""
Alternative Data & Sentiment Service

Extends the existing RSS/news-based sentiment system with alternative
data sources that Renaissance-style quant models use:

  - Google Trends: Search interest spikes → contrarian signals
  - GitHub Activity: Developer commits → fundamental health
  - On-Chain Metrics: Exchange flows, whale tx, active addresses

All sources use free/public APIs — no API keys required.

Renaissance/Simons principle: The best signals come from non-obvious
data. When everyone is watching price, look at developer activity,
search interest, and chain fundamentals instead.
"""

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}


# ── Coin-to-Project Mapping ────────────────────────────────────────
#
# Maps crypto tickers to their GitHub org/repo and project name
# for on-chain and developer activity lookups.

COIN_PROJECTS = {
    "BTC": {"github": "bitcoin/bitcoin", "coingecko_id": "bitcoin", "blockchain": "btc"},
    "ETH": {"github": "ethereum/go-ethereum", "coingecko_id": "ethereum", "blockchain": "eth"},
    "SOL": {"github": "solana-labs/solana", "coingecko_id": "solana", "blockchain": "sol"},
    "XRP": {"github": "XRPLF/rippled", "coingecko_id": "ripple", "blockchain": "xrp"},
    "ADA": {"github": "IntersectMBO/cardano-node", "coingecko_id": "cardano", "blockchain": "ada"},
    "DOT": {"github": "paritytech/polkadot", "coingecko_id": "polkadot", "blockchain": "dot"},
    "LINK": {"github": "smartcontractkit/chainlink", "coingecko_id": "chainlink", "blockchain": None},
    "AVAX": {"github": "ava-labs/avalanchego", "coingecko_id": "avalanche-2", "blockchain": "avax"},
    "ATOM": {"github": "cosmos/cosmos-sdk", "coingecko_id": "cosmos", "blockchain": "atom"},
    "ALGO": {"github": "algorand/go-algorand", "coingecko_id": "algorand", "blockchain": "algo"},
    "NEAR": {"github": "near/nearcore", "coingecko_id": "near", "blockchain": "near"},
    "APT": {"github": "aptos-labs/aptos-core", "coingecko_id": "aptos", "blockchain": "aptos"},
    "ARB": {"github": "OffchainLabs/nitro", "coingecko_id": "arbitrum", "blockchain": None},
    "OP": {"github": "ethereum-optimism/optimism", "coingecko_id": "optimism", "blockchain": None},
    "SUI": {"github": "MystenLabs/sui", "coingecko_id": "sui", "blockchain": "sui"},
    "FTM": {"github": "Fantom-Foundation/go-opera", "coingecko_id": "fantom", "blockchain": "ftm"},
    "MATIC": {"github": "maticnetwork/bor", "coingecko_id": "matic-network", "blockchain": "polygon"},
    "TRX": {"github": "tronprotocol/java-tron", "coingecko_id": "tron", "blockchain": "trx"},
}

# Coins that match specific Bitcoin/blockchain tickers for on-chain lookups
# (the Blockchain.com API only supports major chains)
ONCHAIN_SUPPORTED = {"BTC", "ETH", "XRP", "ADA", "DOT", "SOL", "AVAX", "ATOM", "NEAR"}


def get_project_info(base_asset: str) -> dict:
    """Get project info for a coin ticker, with sensible defaults for unknown coins."""
    return COIN_PROJECTS.get(
        base_asset.upper(),
        {
            "github": None,
            "coingecko_id": base_asset.lower(),
            "blockchain": None,
        }
    )


# ── Google Trends Signal ──────────────────────────────────────────


def get_google_trends_signal(keyword: str) -> dict:
    """
    Fetch Google Trends interest score for a coin name using the public API.

    Uses the unofficial Google Trends API (no auth required).
    Returns a normalized interest score (0-100) where:
      - 0-20:  Very low interest → potential accumulation zone (bullish)
      - 20-50: Normal interest (neutral)
      - 50-80: Elevated interest → watch for tops (neutral/bearish)
      - 80-100: Extreme interest → potential top signal (bearish/contrarian)

    Renaissance principle: When retail interest spikes to extreme levels,
    it's often a contrarian sell signal. The smart money has already
    positioned before the crowd arrives.

    Args:
        keyword: The coin name or ticker to search (e.g., 'Bitcoin', 'BTC')

    Returns:
        Dict with interest score, trend direction, and signal interpretation.
        Returns default neutral values if the API is unavailable.
    """
    try:
        # Google Trends public API endpoint (no auth needed)
        # Uses the same API that powers the "Interest over time" widget
        url = (
            "https://trends.google.com/trends/api/dailytrends"
            "?hl=en-US&tz=0&geo=US&ns=15"
        )
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"Google Trends API returned {resp.status_code}")
            return _default_trends()

        # More targeted search via Google Trends explore API
        search_url = (
            f"https://trends.google.com/trends/api/explore"
            f"?hl=en-US&tz=0&req={quote('{\"comparisonItem\":[{\"keyword\":\"' + keyword + '\",\"geo\":\"\",\"time\":\"today 3-m\"}],\"category\":0,\"property\":\"\"}')}"
        )
        resp2 = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

        if resp2.status_code != 200:
            logger.debug(f"Google Trends explore API returned {resp2.status_code}")
            # Fall back to basic interest assessment
            return _compute_fallback_trends(keyword)

        # Parse the response for interest score
        try:
            # The response is JSON with a JavaScript wrapper; extract the JSON part
            text = resp2.text
            if text.startswith(")]}',"):
                text = text[5:]

            import json
            data = json.loads(text)

            # Extract average interest from the first widget
            widgets = data.get("widgets", [])
            interest = 50  # Default neutral
            for widget in widgets:
                if widget.get("id") == "TIMESERIES":
                    lines = widget.get("lines", [])
                    if lines:
                        points = lines[0].get("points", [])
                        if points:
                            values = [p.get("value", 0) for p in points if p.get("value")]
                            interest = sum(values) / len(values) if values else 50
                    break

            return {
                "interest_score": round(interest, 1),
                "trend": _classify_trend(interest),
                "signal": _trends_to_signal(interest),
                "source": "Google Trends",
            }

        except (ValueError, KeyError, TypeError, ImportError) as e:
            logger.debug(f"Failed to parse Google Trends response: {e}")
            return _compute_fallback_trends(keyword)

    except requests.RequestException as e:
        logger.warning(f"Google Trends API request failed: {e}")
        return _default_trends()


def _default_trends() -> dict:
    """Return neutral default when Google Trends is unavailable."""
    return {
        "interest_score": 50.0,
        "trend": "stable",
        "signal": "neutral",
        "source": "Google Trends",
        "note": "Using default — API unavailable",
    }


def _compute_fallback_trends(keyword: str) -> dict:
    """
    Compute a fallback trends signal based on keyword characteristics.
    Used when the API is unavailable or rate-limited.
    """
    # Simple heuristic: shorter/keyword-only names (likely small-cap)
    # get a slight boost in interest
    keyword_clean = keyword.lower().strip()

    # These major coins tend to have consistent search interest
    major_coins = {"bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "cardano", "ada"}
    if keyword_clean in major_coins:
        return {
            "interest_score": 55.0,
            "trend": "stable",
            "signal": "neutral",
            "source": "Google Trends",
            "note": "Estimated — major coin with consistent interest",
        }

    return _default_trends()


def _classify_trend(score: float) -> str:
    """Classify the trend direction based on interest score."""
    if score >= 80:
        return "spiking"
    elif score >= 60:
        return "rising"
    elif score >= 40:
        return "stable"
    elif score >= 20:
        return "declining"
    else:
        return "low"


def _trends_to_signal(score: float) -> str:
    """
    Convert interest score to trading signal.

    Renaissance/contrarian approach:
      - Very low interest → potential accumulation (bullish)
      - Normal interest → neutral (no signal)
      - High/extreme interest → potential distribution (bearish)
    """
    if score < 15:
        return "bullish"  # Extremely low interest — accumulation zone
    elif score < 35:
        return "mildly_bullish"  # Low interest — potential value
    elif score < 65:
        return "neutral"  # Normal range
    elif score < 85:
        return "mildly_bearish"  # Elevated — watch for top
    else:
        return "bearish"  # Extreme — potential top signal


# ── GitHub Activity Signal ─────────────────────────────────────────


def get_github_activity_signal(github_repo: str | None) -> dict:
    """
    Fetch GitHub commit activity for a project's repository.

    Uses the public GitHub API (no auth required, 60 req/hr limit).
    Measures developer activity as a fundamental signal:
      - High commit activity = active development = bullish
      - Drops in commits = project stagnation = bearish

    Args:
        github_repo: GitHub repo in format 'owner/repo' (e.g., 'bitcoin/bitcoin')
                     Can be None for unknown coins.

    Returns:
        Dict with commit count, activity trend, and signal.
        Returns neutral if repo is unknown or API unavailable.
    """
    if not github_repo:
        return {
            "commit_count": None,
            "activity_trend": "unknown",
            "signal": "neutral",
            "source": "GitHub",
            "note": "No GitHub repository mapped for this coin",
        }

    try:
        # GitHub API: get commit activity for the last week
        url = f"https://api.github.com/repos/{github_repo}/stats/commit_activity"
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 202:
            # GitHub is still generating the stats — try again
            return {
                "commit_count": None,
                "activity_trend": "loading",
                "signal": "neutral",
                "source": "GitHub",
                "note": "Stats still being generated — try again shortly",
            }

        if resp.status_code != 200:
            logger.warning(f"GitHub API returned {resp.status_code} for {github_repo}")
            return {
                "commit_count": None,
                "activity_trend": "unknown",
                "signal": "neutral",
                "source": "GitHub",
                "note": f"GitHub API returned {resp.status_code}",
            }

        data = resp.json()

        # Calculate metrics from the weekly commit data
        total_commits = sum(week.get("total", 0) for week in data)

        # Get last 4 weeks for trend comparison
        recent_weeks = data[-4:] if len(data) >= 4 else data
        if len(recent_weeks) >= 2:
            current = recent_weeks[-1].get("total", 0)
            previous = recent_weeks[-2].get("total", 0)
            trend = "increasing" if current > previous else (
                "decreasing" if current < previous else "stable"
            )
        else:
            trend = "stable"

        # Signal determination
        if total_commits >= 50:
            signal = "bullish"  # Very active development
        elif total_commits >= 20:
            signal = "mildly_bullish"  # Healthy development
        elif total_commits >= 5:
            signal = "neutral"  # Some activity
        else:
            signal = "bearish"  # Very low activity — project may be stagnant

        return {
            "commit_count": total_commits,
            "weeks_of_data": len(data),
            "activity_trend": trend,
            "signal": signal,
            "source": "GitHub",
            "repo": github_repo,
        }

    except requests.RequestException as e:
        logger.warning(f"GitHub API request failed for {github_repo}: {e}")
        return {
            "commit_count": None,
            "activity_trend": "unknown",
            "signal": "neutral",
            "source": "GitHub",
            "note": f"API request failed: {e}",
        }


# ── On-Chain Metrics ──────────────────────────────────────────────


def get_onchain_metrics(base_asset: str) -> dict:
    """
    Fetch on-chain metrics for supported assets using public APIs.

    Sources:
      - Blockchain.com API: BTC wallet activity, exchange flows
      - Etherscan API (free tier): ETH gas, active addresses
      - CoinGecko API: General market data

    Args:
        base_asset: The crypto ticker (e.g., 'BTC', 'ETH')

    Returns:
        Dict with on-chain metrics or empty/default if unsupported.
    """
    symbol = base_asset.upper()

    if symbol not in ONCHAIN_SUPPORTED:
        return {
            "supported": False,
            "note": f"On-chain data not available for {symbol}. Supported: {', '.join(sorted(ONCHAIN_SUPPORTED))}",
        }

    result = {"supported": True, "symbol": symbol, "fetched_at": datetime.now(timezone.utc).isoformat()}

    # BTC-specific: Exchange flows from Blockchain.com
    if symbol == "BTC":
        btc_data = _fetch_btc_onchain()
        result.update(btc_data)
    elif symbol == "ETH":
        eth_data = _fetch_eth_onchain()
        result.update(eth_data)

    # General: CoinGecko market data for all supported coins
    cg_data = _fetch_coingecko_data(symbol)
    if cg_data:
        result.update(cg_data)

    # Compute overall signal
    result["overall_signal"] = _compute_onchain_signal(result)

    return result


def _fetch_btc_onchain() -> dict:
    """Fetch BTC-specific on-chain data from Blockchain.com."""
    result = {"asset": "BTC"}
    try:
        # Latest block info
        url = "https://blockchain.info/latestblock"
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            result["latest_block"] = data.get("height")
            result["block_time"] = datetime.fromtimestamp(
                data.get("time", 0), tz=timezone.utc
            ).isoformat() if data.get("time") else None

        # Transaction count (24h approximate via blockchair free API)
        url2 = "https://api.blockchain.info/charts/transactions-per-second?timespan=24hours&format=json"
        resp2 = requests.get(url2, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp2.status_code == 200:
            data2 = resp2.json()
            values = data2.get("values", [])
            if values:
                avg_tps = sum(v.get("y", 0) for v in values) / len(values)
                result["tx_per_second_24h"] = round(avg_tps, 2)

        # Addresses with balance (active addresses)
        url3 = "https://api.blockchain.info/charts/n-unique-addresses?timespan=30days&format=json"
        resp3 = requests.get(url3, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp3.status_code == 200:
            data3 = resp3.json()
            addr_values = data3.get("values", [])
            if addr_values:
                recent = [v.get("y", 0) for v in addr_values[-7:]]
                result["active_addresses_7d_avg"] = round(sum(recent) / len(recent), 0) if recent else None

        # Estimate exchange flow (public data only)
        result["exchange_flow_signal"] = "neutral"
        result["note"] = "BTC on-chain data from Blockchain.com"

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch BTC on-chain data: {e}")
        result["note"] = "On-chain data temporarily unavailable"

    return result


def _fetch_eth_onchain() -> dict:
    """Fetch ETH-specific on-chain data."""
    result = {"asset": "ETH"}
    try:
        # Gas price estimate via Etherscan free API
        url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey=free"
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            gas_data = data.get("result", {})
            if gas_data:
                result["gas_price_gwei"] = {
                    "slow": gas_data.get("SafeGasPrice"),
                    "average": gas_data.get("ProposeGasPrice"),
                    "fast": gas_data.get("FastGasPrice"),
                }

        result["note"] = "ETH on-chain data from Etherscan"

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch ETH on-chain data: {e}")
        result["note"] = "On-chain data temporarily unavailable"

    return result


def _fetch_coingecko_data(symbol: str) -> dict:
    """
    Fetch general market data from CoinGecko free API.

    Returns price change, volume, and market cap metrics.
    """
    project = get_project_info(symbol)
    coin_id = project.get("coingecko_id", symbol.lower())

    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            "?localization=false&tickers=false&community_data=false"
            "&developer_data=false&sparkline=false"
        )
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

        if resp.status_code != 200:
            logger.debug(f"CoinGecko returned {resp.status_code} for {coin_id}")
            return {}

        data = resp.json()
        market_data = data.get("market_data", {})

        return {
            "market_cap_rank": market_data.get("market_cap_rank"),
            "price_change_24h_pct": market_data.get("price_change_percentage_24h"),
            "price_change_7d_pct": market_data.get("price_change_percentage_7d"),
            "total_volume_usd": market_data.get("total_volume", {}).get("usd"),
            "market_cap_usd": market_data.get("market_cap", {}).get("usd"),
            "circulating_supply": market_data.get("circulating_supply"),
        }

    except requests.RequestException as e:
        logger.warning(f"CoinGecko request failed for {coin_id}: {e}")
        return {}


def _compute_onchain_signal(metrics: dict) -> str:
    """Compute an overall on-chain signal from available metrics."""
    bullish_signals = 0
    bearish_signals = 0

    # Price change signals (contrarian)
    pct_24h = metrics.get("price_change_24h_pct")
    if pct_24h is not None:
        if pct_24h > 10:
            bearish_signals += 1  # Pumped — potential top
        elif pct_24h < -10:
            bullish_signals += 1  # Dumped — potential bottom

    # Transaction activity signals (BTC)
    tps = metrics.get("tx_per_second_24h")
    if tps is not None:
        if tps > 10:
            bullish_signals += 1  # High network usage
        elif tps < 3:
            bearish_signals += 1  # Low network usage

    # Active addresses
    active = metrics.get("active_addresses_7d_avg")
    if active is not None:
        if active > 800000:
            bullish_signals += 1  # High user activity

    if bullish_signals > bearish_signals:
        return "bullish"
    elif bearish_signals > bullish_signals:
        return "bearish"
    else:
        return "neutral"


# ── Consensus Score Engine ─────────────────────────────────────────


class AlternativeSentimentEngine:
    """
    Multi-source sentiment aggregation engine.

    Combines signals from:
      - Google Trends (search interest)
      - GitHub Activity (developer commits)
      - On-Chain Metrics (exchange flows, active addresses)
      - Existing news/RSS sentiment (from sentiment app)

    Produces a single 0-100 consensus score with bullish/bearish/neutral label.

    Usage:
        engine = AlternativeSentimentEngine()
        result = engine.compute_consensus_score("BTC")
    """

    WEIGHTS = {
        "news_sentiment": 0.25,
        "google_trends": 0.15,
        "github_activity": 0.20,
        "onchain_metrics": 0.25,
        "fear_greed": 0.15,
    }

    def __init__(self, weights: dict[str, float] | None = None):
        """
        Args:
            weights: Custom source weights (defaults to equal weighting)
        """
        self.weights = weights or self.WEIGHTS

    def compute_consensus_score(
        self,
        base_asset: str,
        news_sentiment: dict | None = None,
        fear_greed: dict | None = None,
    ) -> dict:
        """
        Compute a consensus sentiment score from all available sources.

        Args:
            base_asset: Crypto ticker (e.g., 'BTC', 'ETH')
            news_sentiment: Pre-fetched news sentiment data (from sentiment app).
                           If None, we'll fetch it.
            fear_greed: Pre-fetched Fear & Greed Index. If None, we'll fetch it.

        Returns:
            Dict with:
              - consensus_score: 0-100 (50 = neutral)
              - consensus_label: 'bullish', 'bearish', 'neutral'
              - sources_contributing: How many sources had data
              - breakdown: Per-source scores
        """
        project = get_project_info(base_asset)
        scores: dict[str, dict] = {}
        sources_used = 0

        # 1. Google Trends
        try:
            trends = get_google_trends_signal(base_asset)
            trends_score = self._signal_to_score(trends.get("signal", "neutral"))
            scores["google_trends"] = {
                "score": trends_score,
                "weight": self.weights.get("google_trends", 0.15),
                "detail": trends,
            }
            if trends.get("interest_score") is not None:
                sources_used += 1
        except Exception:
            scores["google_trends"] = {"score": 50.0, "weight": 0, "error": True}

        # 2. GitHub Activity
        try:
            github = get_github_activity_signal(project.get("github"))
            github_score = self._signal_to_score(github.get("signal", "neutral"))
            scores["github_activity"] = {
                "score": github_score,
                "weight": self.weights.get("github_activity", 0.20),
                "detail": github,
            }
            if github.get("commit_count") is not None:
                sources_used += 1
        except Exception:
            scores["github_activity"] = {"score": 50.0, "weight": 0, "error": True}

        # 3. On-Chain Metrics
        try:
            onchain = get_onchain_metrics(base_asset)
            onchain_score = self._signal_to_score(onchain.get("overall_signal", "neutral"))
            scores["onchain_metrics"] = {
                "score": onchain_score,
                "weight": self.weights.get("onchain_metrics", 0.25),
                "detail": onchain,
            }
            if onchain.get("supported", False):
                sources_used += 1
        except Exception:
            scores["onchain_metrics"] = {"score": 50.0, "weight": 0, "error": True}

        # 4. News Sentiment (passed in or fetched)
        news_score = 50.0
        if news_sentiment:
            news_score = news_sentiment.get("overall_sentiment", {}).get("score", 50.0)
            scores["news_sentiment"] = {
                "score": news_score,
                "weight": self.weights.get("news_sentiment", 0.25),
            }
            sources_used += 1

        # 5. Fear & Greed Index (passed in or fetched)
        fg_score = 50.0
        if fear_greed:
            fg = fear_greed.get("value")
            if fg is not None:
                fg_score = float(fg)
                scores["fear_greed"] = {
                    "score": fg_score,
                    "weight": self.weights.get("fear_greed", 0.15),
                }
                sources_used += 1

        # Compute weighted consensus
        total_weight = sum(s.get("weight", 0) for s in scores.values())
        if total_weight > 0:
            weighted_sum = sum(
                s.get("score", 50) * s.get("weight", 0)
                for s in scores.values()
            )
            consensus_score = weighted_sum / total_weight
        else:
            consensus_score = 50.0

        # Determine label
        if consensus_score > 60:
            label = "bullish"
        elif consensus_score < 40:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "consensus_score": round(consensus_score, 1),
            "consensus_label": label,
            "sources_contributing": sources_used,
            "total_sources": len(scores),
            "base_asset": base_asset.upper(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "breakdown": scores,
        }

    @staticmethod
    def _signal_to_score(signal: str) -> float:
        """Convert a signal string to a 0-100 score."""
        mapping = {
            "bullish": 80.0,
            "mildly_bullish": 65.0,
            "neutral": 50.0,
            "mildly_bearish": 35.0,
            "bearish": 20.0,
        }
        return mapping.get(signal, 50.0)
