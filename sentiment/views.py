import logging

from django.http import JsonResponse

from .services import get_crypto_sentiment, get_fear_greed_index

logger = logging.getLogger(__name__)


def sentiment_data(request):
    """
    AJAX endpoint that returns news articles and sentiment analysis
    for a given cryptocurrency base asset.
    
    Query params:
        symbol: The trading pair symbol (e.g. BTCUSDT)
    
    Returns JSON with articles, sentiment scores, and breakdown.
    """
    symbol = request.GET.get("symbol", "").strip().upper()

    if not symbol:
        return JsonResponse({
            "error": "No symbol provided",
            "articles": [],
            "reddit_posts": [],
            "overall_sentiment": {"label": "neutral", "score": 50.0},
        })

    # Extract base asset from trading pair (e.g., BTCUSDT -> BTC)
    # Common quote assets: USDT, USDC, BUSD, BTC, ETH, BNB, etc.
    base_asset = symbol
    for quote in ["USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "PAX"]:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base_asset = symbol[:-len(quote)]
            break

    try:
        data = get_crypto_sentiment(base_asset)
        return JsonResponse(data)
    except Exception as e:
        logger.exception(f"Sentiment analysis failed for {symbol}")
        return JsonResponse({
            "error": f"Failed to fetch sentiment data: {str(e)}",
            "articles": [],
            "reddit_posts": [],
            "overall_sentiment": {"label": "neutral", "score": 50.0},
        })


def fear_greed(request):
    """
    AJAX endpoint returning the Crypto Fear & Greed Index.
    Market-wide sentiment indicator from alternative.me (free, no auth).
    """
    try:
        data = get_fear_greed_index()
        return JsonResponse(data)
    except Exception as e:
        logger.exception("Failed to fetch Fear & Greed Index")
        return JsonResponse({
            "value": None,
            "classification": "Error",
            "previous_value": None,
            "error": str(e),
        })
