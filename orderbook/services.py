import requests

BINANCE_API = 'https://api.binance.com'


def get_depth(symbol: str, limit: int = 20) -> dict:
    """Fetch order book depth snapshot from Binance REST API.

    Returns dict with 'bids' and 'asks' arrays, each containing
    [price, quantity] tuples.
    """
    url = f'{BINANCE_API}/api/v3/depth'
    params = {
        'symbol': symbol.upper(),
        'limit': min(limit, 100),  # Binance max is 100 for this endpoint
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Convert to a structure our frontend already expects
    return {
        'bids': [[float(p), float(q)] for p, q in data.get('bids', [])],
        'asks': [[float(p), float(q)] for p, q in data.get('asks', [])],
    }


def get_formatted_depth(symbol: str, limit: int = 20) -> dict:
    """Fetch depth and compute summary stats (spread, total volume)."""
    raw = get_depth(symbol, limit)

    asks = [[float(p), float(q)] for p, q in raw['asks']]
    bids = [[float(p), float(q)] for p, q in raw['bids']]

    best_ask = asks[0][0] if asks else 0
    best_bid = bids[0][0] if bids else 0
    spread = best_ask - best_bid if best_ask and best_bid else 0
    spread_pct = (spread / best_ask) * 100 if best_ask else 0

    total_ask_vol = sum(qty for _, qty in asks)
    total_bid_vol = sum(qty for _, qty in bids)

    return {
        'asks': asks,
        'bids': bids,
        'best_ask': best_ask,
        'best_bid': best_bid,
        'spread': spread,
        'spread_pct': spread_pct,
        'total_ask_vol': total_ask_vol,
        'total_bid_vol': total_bid_vol,
    }
