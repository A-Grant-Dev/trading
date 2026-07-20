"""
Crypto News Scraper & Sentiment Analyzer

Scrapes free, open-source data sources (no API keys required) and performs
sentiment analysis using TextBlob. Sources include:
  - CoinDesk RSS
  - CoinTelegraph RSS
  - Decrypt RSS
  - Google News RSS (search-based)
  - Reddit .json endpoint (r/cryptocurrency, r/bitcoin, r/ethfinance)
"""

import logging
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import requests
from textblob import TextBlob
from urllib.parse import quote

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────

REQUEST_TIMEOUT = 15
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}
MAX_ARTICLES_PER_SOURCE = 8
MAX_REDDIT_POSTS = 10
CACHE_TTL_SECONDS = 300  # 5 minutes
MAX_AGE_DAYS = 31  # Max age for articles/posts in days

# ── Coin name mapping ──────────────────────────────────────────────

COIN_NAMES = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "XRP": "XRP",
    "ADA": "Cardano",
    "DOGE": "Dogecoin",
    "DOT": "Polkadot",
    "AVAX": "Avalanche",
    "MATIC": "Polygon",
    "LINK": "Chainlink",
    "UNI": "Uniswap",
    "ATOM": "Cosmos",
    "LTC": "Litecoin",
    "BCH": "Bitcoin Cash",
    "XLM": "Stellar",
    "TRX": "TRON",
    "FIL": "Filecoin",
    "APT": "Aptos",
    "ARB": "Arbitrum",
    "OP": "Optimism",
    "SUI": "Sui",
    "PEPE": "Pepe",
    "SHIB": "Shiba Inu",
    "NEAR": "NEAR Protocol",
    "ALGO": "Algorand",
    "FTM": "Fantom",
    "EGLD": "MultiversX",
    "THETA": "Theta Network",
    "VET": "VeChain",
    "ICP": "Internet Computer",
    "SAND": "The Sandbox",
    "MANA": "Decentraland",
    "AXS": "Axie Infinity",
    "AAVE": "Aave",
    "MKR": "Maker",
    "CRV": "Curve DAO",
    "SNX": "Synthetix",
    "COMP": "Compound",
    "YFI": "yearn.finance",
}


def get_coin_name(base_asset):
    """Convert a ticker like 'BTC' to a readable name like 'Bitcoin'."""
    return COIN_NAMES.get(base_asset.upper(), base_asset.upper())


# ── RSS Feed Sources ───────────────────────────────────────────────

RSS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt", "https://decrypt.co/feed"),
]


def fetch_rss_feed(url, max_entries=10):
    """Fetch and parse an RSS feed, returning up to max_entries entries."""
    try:
        feed = feedparser.parse(url)
        entries = []
        for entry in feed.entries[:max_entries]:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    published = None

            entries.append({
                "title": entry.title,
                "link": entry.link,
                "published": published.isoformat() if published else None,
                "source": feed.feed.title if hasattr(feed.feed, "title") else "Unknown",
                "summary": entry.get("summary", "")[:500] if hasattr(entry, "summary") else "",
            })
        return entries
    except Exception as e:
        logger.warning(f"Failed to fetch RSS feed {url}: {e}")
        return []


# ── Google News RSS Search ─────────────────────────────────────────

def fetch_google_news(query, max_entries=8):
    """Search Google News via RSS for a specific crypto query."""
    try:
        search_query = quote(f"{query} cryptocurrency")
        url = (
            f"https://news.google.com/rss/search?q={search_query}"
            f"&hl=en-US&gl=US&ceid=US:en"
        )
        feed = feedparser.parse(url)
        entries = []
        for entry in feed.entries[:max_entries]:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    published = None

            # Google News links are weird - try to extract the real URL
            link = entry.link
            if hasattr(entry, "feedburner_origlink"):
                link = entry.feedburner_origlink

            entries.append({
                "title": entry.title,
                "link": link,
                "published": published.isoformat() if published else None,
                "source": "Google News",
                "summary": "",
            })
        return entries
    except Exception as e:
        logger.warning(f"Failed to fetch Google News for '{query}': {e}")
        return []


# ── Reddit Scraper (RSS-based) ───────────────────────────────────────
#
# Reddit's JSON API is locked down (returns 403).
# We use Reddit's public RSS search instead, which still works.
# This searches r/cryptocurrency for the specific coin, returning
# results even for less popular coins.


def fetch_reddit_posts(query, max_posts=6):
    """
    Fetch Reddit posts mentioning the crypto using Reddit's RSS search.
    Searches r/cryptocurrency for posts matching the coin ticker or name.
    Falls back to the main hot feed if search returns nothing.
    """
    posts = []
    try:
        search_q = quote(query)
        # Reddit search RSS - works for any coin
        search_url = f"https://www.reddit.com/r/cryptocurrency/search.rss?q={search_q}&restrict_sr=on&sort=new&t=month"
        feed = feedparser.parse(search_url)

        entries = feed.entries or []
        if not entries:
            # Fallback: try the general hot feed and filter
            feed = feedparser.parse("https://www.reddit.com/r/cryptocurrency/.rss")
            entries = feed.entries or []

        query_lower = query.lower()
        for entry in entries[:max_posts * 2]:
            title = entry.title
            title_lower = title.lower()

            # Relaxed matching: the search RSS already returns relevant results;
            # just skip clearly off-topic posts (no coin name, no ticker, no crypto mention)
            if query_lower not in title_lower and len(query_lower) <= 4:
                # For short tickers (BTC, ETH, etc.), be more lenient if crypto-related
                if 'crypto' not in title_lower and 'cryptocurrenc' not in title_lower:
                    continue
            elif query_lower not in title_lower and len(query_lower) > 4:
                # For longer names/tickers (HBAR, AVAX, etc.), try first 4 chars
                if query_lower[:4] not in title_lower and 'crypto' not in title_lower:
                    continue

            link = entry.link
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    published = None

            # Skip posts older than MAX_AGE_DAYS
            if published and is_older_than(published, MAX_AGE_DAYS):
                continue

            posts.append({
                "title": title,
                "link": link,
                "published": published.isoformat() if published else None,
                "source": "r/cryptocurrency",
                "summary": entry.get("summary", "")[:300] if hasattr(entry, "summary") else "",
                "score": 0,
                "upvote_ratio": 0.5,
                "num_comments": 0,
            })

            if len(posts) >= max_posts:
                break

    except Exception as e:
        logger.warning(f"Failed to fetch Reddit RSS feed: {e}")

    return posts


# ── Hacker News Scraper (free API, no auth needed) ─────────────────


def fetch_hacker_news(query, max_posts=6):
    """
    Fetch Hacker News posts about the crypto using Algolia's free API.
    No API key required, no rate limiting for moderate usage.
    If no coin-specific results found, does a broader crypto search.
    """
    posts = []
    try:
        search_query = quote(f"{query} crypto")
        url = (
            f"https://hn.algolia.com/api/v1/search?"
            f"query={search_query}&tags=story&hitsPerPage={max_posts * 2}"
        )
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"Hacker News API returned {resp.status_code}")
            return posts

        data = resp.json()
        hits = data.get("hits", [])

        # If no results, try broader search
        if not hits:
            broader_query = quote("cryptocurrency crypto blockchain")
            broader_url = (
                f"https://hn.algolia.com/api/v1/search?"
                f"query={broader_query}&tags=story&hitsPerPage={max_posts}"
            )
            resp2 = requests.get(broader_url, timeout=REQUEST_TIMEOUT)
            if resp2.status_code == 200:
                hits = resp2.json().get("hits", [])

        for hit in hits:
            title = hit.get("title", "")
            title_lower = title.lower()

            # Include if it mentions the coin or is about crypto in general
            if query.lower() not in title_lower and "crypto" not in title_lower and "cryptocurrenc" not in title_lower:
                continue

            created_at = hit.get("created_at")
            published = None
            if created_at:
                try:
                    published = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except Exception:
                    published = None

            # Skip posts older than MAX_AGE_DAYS
            if published and is_older_than(published, MAX_AGE_DAYS):
                continue

            posts.append({
                "title": title,
                "link": hit.get("url") or hit.get("story_url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                "published": published.isoformat() if published else None,
                "source": "Hacker News",
                "summary": hit.get("story_text", "")[:300] if hit.get("story_text") else "",
                "score": hit.get("points", 0),
                "upvote_ratio": 0.5,
                "num_comments": hit.get("num_comments", 0),
            })

            if len(posts) >= max_posts:
                break

    except Exception as e:
        logger.warning(f"Failed to fetch Hacker News for '{query}': {e}")

    return posts


# ── Date Helpers ────────────────────────────────────────────────────

def is_older_than(dt, days):
    """Check if a datetime is older than the specified number of days."""
    if not dt:
        return False
    now = datetime.now(timezone.utc)
    age = (now - dt).total_seconds()
    return age > days * 86400  # days to seconds


def filter_recent(items, date_key="published", max_days=MAX_AGE_DAYS):
    """Filter a list of items to only include those newer than max_days.
    Items without a published date are excluded (we can't verify they're recent).
    """
    filtered = []
    for item in items:
        published_str = item.get(date_key)
        if not published_str:
            continue  # Skip items without dates — can't verify recency
        try:
            published = datetime.fromisoformat(published_str)
            if is_older_than(published, max_days):
                continue
        except Exception:
            continue  # Can't parse date, skip to be safe
        filtered.append(item)
    return filtered


# ── Sentiment Analysis ─────────────────────────────────────────────

# Common crypto bullish/bearish keywords for boosting accuracy
BULLISH_KEYWORDS = {
    "bullish", "buoyant", "surge", "surges", "surged", "soar", "soars", "soared",
    "rally", "rallies", "rallied", "breakout", "break", "breaks", "pump", "moon",
    "adoption", "partnership", "partners", "upgrade", "launch", "launches",
    "approval", "approved", "positive", "growth", "growing", "gains", "gain",
    "upside", "outperform", "outperformed", "accumulate", "accumulation",
    "institutional", "mainstream", "bull run", "bullrun", "ATH", "all-time high",
    "new high", "all time high", "whale accumulation", "hodl", "buy the dip",
}

BEARISH_KEYWORDS = {
    "bearish", "decline", "declines", "declined", "dump", "dumped", "dumping",
    "crash", "crashed", "crashing", "plunge", "plunges", "plunged", "drop", "drops",
    "dropped", "correction", "corrections", "sell-off", "selloff", "liquidation",
    "liquidations", "ban", "banned", "banning", "crackdown", "regulatory",
    "regulation", "sec", "lawsuit", "investigation", "fraud", "hack", "hacked",
    "exploit", "breach", "fud", "fear", "panic", "sell", "selling", "bear market",
    "bearmarket", "downtrend", "resistance", "overbought", "death cross",
    "whale selling", "rug pull", "rugpull", "scam", "pump and dump",
}


def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob + keyword boosting.
    Returns a dict with scores.
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "label": "neutral",
            "score": 0.0,
        }

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1.0 to 1.0
    subjectivity = blob.sentiment.subjectivity  # 0.0 to 1.0

    # Keyword boosting
    text_lower = text.lower()
    
    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)

    keyword_bias = 0.0
    if bullish_count > bearish_count:
        keyword_bias = min(0.3, (bullish_count - bearish_count) * 0.05)
    elif bearish_count > bullish_count:
        keyword_bias = -min(0.3, (bearish_count - bullish_count) * 0.05)

    final_polarity = max(-1.0, min(1.0, polarity + keyword_bias))

    # Determine label
    if final_polarity > 0.15:
        label = "bullish"
    elif final_polarity < -0.15:
        label = "bearish"
    else:
        label = "neutral"

    # Normalize to 0-100 score (50 = neutral)
    score = 50 + (final_polarity * 50)

    return {
        "polarity": round(final_polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "label": label,
        "score": round(score, 1),
    }


def analyze_headline_sentiment(headlines):
    """Analyze sentiment across a list of headline texts."""
    scores = []
    for h in headlines:
        result = analyze_sentiment(h)
        scores.append(result["polarity"])

    if not scores:
        return {"label": "neutral", "score": 50.0, "polarity": 0.0}

    avg_polarity = sum(scores) / len(scores)
    if avg_polarity > 0.15:
        label = "bullish"
    elif avg_polarity < -0.15:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": round(50 + (avg_polarity * 50), 1),
        "polarity": round(avg_polarity, 3),
    }


# ── Fear & Greed Index ───────────────────────────────────────────


def get_fear_greed_index():
    """
    Fetch the Crypto Fear & Greed Index from alternative.me (free, no API key).
    Returns current value, classification, and previous day for comparison.
    """
    cache_key = "fear_greed"
    cached = get_cached(cache_key, ttl=CACHE_TTL_SECONDS)
    if cached:
        return cached

    try:
        url = "https://api.alternative.me/fng/?limit=2"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"Fear & Greed API returned {resp.status_code}")
            return {"value": None, "classification": "Unknown", "previous_value": None}

        data = resp.json()
        items = data.get("data", [])
        if not items:
            return {"value": None, "classification": "Unknown", "previous_value": None}

        current = items[0]
        previous = items[1] if len(items) > 1 else None

        result = {
            "value": int(current.get("value", 50)),
            "classification": current.get("value_classification", "Neutral"),
            "previous_value": int(previous.get("value", 50)) if previous else None,
            "timestamp": current.get("timestamp"),
            "time_until_update": current.get("time_until_update"),
        }

        set_cache(cache_key, result)
        return result

    except Exception as e:
        logger.exception(f"Failed to fetch Fear & Greed Index: {e}")
        return {"value": None, "classification": "Error", "previous_value": None}


# ── Main Orchestrator ──────────────────────────────────────────────

# Simple in-memory cache
_cache = {}
_cache_timestamps = {}


def get_cached(key, ttl=CACHE_TTL_SECONDS):
    """Get value from cache if not expired."""
    if key in _cache and key in _cache_timestamps:
        age = (datetime.now(timezone.utc) - _cache_timestamps[key]).total_seconds()
        if age < ttl:
            return _cache[key]
    return None


def set_cache(key, value):
    """Set value in cache."""
    _cache[key] = value
    _cache_timestamps[key] = datetime.now(timezone.utc)


def get_crypto_sentiment(base_asset):
    """
    Main entry point.
    Given a base asset (e.g. 'BTC'), scrape all sources and return
    aggregated sentiment data with articles and summaries.
    """
    coin_name = get_coin_name(base_asset)
    cache_key = f"sentiment_{base_asset.upper()}"

    cached = get_cached(cache_key)
    if cached:
        return cached

    # Gather data
    all_articles = []
    all_headlines = []
    reddit_posts = []

    # 1. RSS Feeds
    for source_name, feed_url in RSS_FEEDS:
        entries = fetch_rss_feed(feed_url, MAX_ARTICLES_PER_SOURCE)
        # Filter entries that mention the coin
        for entry in entries:
            title_lower = entry["title"].lower()
            summary_lower = entry["summary"].lower()
            if (coin_name.lower() in title_lower or coin_name.lower() in summary_lower
                    or base_asset.lower() in title_lower or base_asset.lower() in summary_lower):
                all_articles.append(entry)
                all_headlines.append(entry["title"])

    # 2. Google News
    google_entries = fetch_google_news(coin_name, MAX_ARTICLES_PER_SOURCE)
    for entry in google_entries:
        title_lower = entry["title"].lower()
        if coin_name.lower() in title_lower or base_asset.lower() in title_lower:
            all_articles.append(entry)
            all_headlines.append(entry["title"])

    # 3. Reddit (RSS-based - JSON API is locked down)
    reddit_posts = fetch_reddit_posts(coin_name, MAX_REDDIT_POSTS)
    for post in reddit_posts:
        all_headlines.append(post["title"])

    # 4. Hacker News (free Algolia API, no auth needed)
    hn_posts = fetch_hacker_news(coin_name, MAX_REDDIT_POSTS)
    for post in hn_posts:
        all_headlines.append(post["title"])

    # Filter to only include recent items (max 1 month old)
    all_articles = filter_recent(all_articles)
    reddit_posts = filter_recent(reddit_posts)
    hn_posts = filter_recent(hn_posts)
    all_headlines = [a["title"] for a in all_articles]
    all_headlines.extend(p["title"] for p in reddit_posts)
    all_headlines.extend(p["title"] for p in hn_posts)

    # Analyze sentiment
    headline_sentiment = analyze_headline_sentiment(all_headlines)

    # Individual article sentiment (lightweight - just titles for speed)
    article_sentiments = []
    for article in all_articles[:20]:
        text_to_analyze = article["title"]
        if article.get("summary"):
            text_to_analyze += " " + article["summary"][:300]
        sent = analyze_sentiment(text_to_analyze)
        article_sentiments.append({
            "title": article["title"],
            "link": article["link"],
            "source": article["source"],
            "published": article["published"],
            "sentiment": sent,
        })

    # Reddit sentiment analysis
    reddit_sentiments = []
    for post in reddit_posts:
        text_to_analyze = post["title"]
        if post.get("summary"):
            text_to_analyze += " " + post["summary"][:200]
        sent = analyze_sentiment(text_to_analyze)
        reddit_sentiments.append({
            "title": post["title"],
            "link": post["link"],
            "source": post["source"],
            "published": post["published"],
            "score": post.get("score"),
            "upvote_ratio": post.get("upvote_ratio"),
            "num_comments": post.get("num_comments"),
            "sentiment": sent,
        })

    # HN sentiment analysis
    hn_sentiments = []
    for post in hn_posts:
        text_to_analyze = post["title"]
        if post.get("summary"):
            text_to_analyze += " " + post["summary"][:200]
        sent = analyze_sentiment(text_to_analyze)
        hn_sentiments.append({
            "title": post["title"],
            "link": post["link"],
            "source": post["source"],
            "published": post["published"],
            "score": post.get("score"),
            "upvote_ratio": post.get("upvote_ratio"),
            "num_comments": post.get("num_comments"),
            "sentiment": sent,
        })

    # Count bullish vs bearish across ALL sources (articles + Reddit + HN)
    all_sentiments = article_sentiments + reddit_sentiments + hn_sentiments
    bullish_count = sum(1 for s in all_sentiments if s["sentiment"]["label"] == "bullish")
    bearish_count = sum(1 for s in all_sentiments if s["sentiment"]["label"] == "bearish")
    neutral_count = sum(1 for s in all_sentiments if s["sentiment"]["label"] == "neutral")

    result = {
        "coin": coin_name,
        "base_asset": base_asset.upper(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "overall_sentiment": headline_sentiment,
        "breakdown": {
            "bullish": bullish_count,
            "bearish": bearish_count,
            "neutral": neutral_count,
            "total": len(all_sentiments),
        },
        "articles": article_sentiments[:15],
        "reddit_posts": reddit_sentiments[:8],
        "hn_posts": hn_sentiments[:8],
        "fear_greed": get_fear_greed_index(),
        "sources_used": [
            "CoinDesk RSS",
            "CoinTelegraph RSS",
            "Decrypt RSS",
            "Google News RSS",
            "Reddit (r/cryptocurrency RSS)",
            "Hacker News (Algolia API)",
            "Fear & Greed Index (alternative.me)",
        ],
    }

    # Cache result
    set_cache(cache_key, result)
    return result
