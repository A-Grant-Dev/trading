import hashlib
import hmac
import time
import urllib.parse
from datetime import datetime, timezone

import requests

from charts.models import BinanceConfig


BINANCE_API = 'https://api.binance.com'
BINANCE_TESTNET = 'https://testnet.binance.vision'


def get_active_config():
    """Get the active Binance API configuration from the database."""
    configs = BinanceConfig.objects.filter(is_active=True)
    return configs.first()


def _get_base_url(config):
    return BINANCE_TESTNET if config.use_testnet else BINANCE_API


def _sign_request(config, method, endpoint, params=None):
    """Make a signed request to Binance API."""
    if params is None:
        params = {}

    base_url = _get_base_url(config)
    params['timestamp'] = int(time.time() * 1000)
    params['recvWindow'] = 10000

    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(
        config.api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256,
    ).hexdigest()

    url = f'{base_url}{endpoint}?{query_string}&signature={signature}'
    headers = {'X-MBX-APIKEY': config.api_key}

    if method == 'GET':
        resp = requests.get(url, headers=headers, timeout=15)
    else:
        resp = requests.post(url, headers=headers, timeout=15)

    if not resp.ok:
        try:
            binance_error = resp.json()
            msg = binance_error.get('msg', resp.text)
        except Exception:
            msg = resp.text or f'HTTP {resp.status_code}'
        raise requests.HTTPError(
            f'Binance API {endpoint}: {msg} (status {resp.status_code})',
            response=resp,
        )
    return resp.json()


def _public_request(endpoint, params=None):
    """Make a public (unsigned) request to Binance API."""
    if params is None:
        params = {}
    url = f'{BINANCE_API}{endpoint}'
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_account_info(config):
    """Fetch account info with all non-zero balances."""
    return _sign_request(config, 'GET', '/api/v3/account')


# Module-level cache for USD/ZAR rate
_usd_zar_rate = None
_usd_zar_fetched_at = None


def get_usd_zar_rate():
    """Fetch USD to ZAR exchange rate with caching (refreshes every 5 min)."""
    global _usd_zar_rate, _usd_zar_fetched_at
    now = time.time()
    if _usd_zar_rate is not None and _usd_zar_fetched_at and (now - _usd_zar_fetched_at) < 300:
        return _usd_zar_rate
    try:
        resp = requests.get('https://open.er-api.com/v6/latest/USD', timeout=10)
        data = resp.json()
        _usd_zar_rate = data['rates']['ZAR']
    except Exception:
        if _usd_zar_rate is None:
            _usd_zar_rate = 18.50  # fallback
    _usd_zar_fetched_at = now  # update timestamp on success or failure to prevent hammering
    return _usd_zar_rate


def get_all_prices():
    """Fetch all current ticker prices."""
    data = _public_request('/api/v3/ticker/price')
    return {item['symbol']: float(item['price']) for item in data}


def get_my_trades(config, symbol, limit=500):
    """Fetch trade history for a specific symbol."""
    try:
        return _sign_request(config, 'GET', '/api/v3/myTrades', {
            'symbol': symbol,
            'limit': limit,
        })
    except requests.HTTPError:
        return []


def get_asset_balance(config):
    usd_zar = get_usd_zar_rate()
    """Get all non-zero balances with USD values and P&L."""
    account = get_account_info(config)
    all_prices = get_all_prices()

    # Non-zero, non-free balances only
    balances = [
        b for b in account['balances']
        if float(b['free']) > 0 or float(b['locked']) > 0
    ]

    # Stablecoin set
    stablecoins = {'USDT', 'USDC', 'BUSD', 'DAI', 'FDUSD', 'TUSD', 'USDP', 'GUSD', 'PAX', 'BKRW', 'EUR', 'GBP', 'AUD'}

    results = []
    for bal in balances:
        asset = bal['asset']
        free = float(bal['free'])
        locked = float(bal['locked'])
        total_qty = free + locked

        price_usdt = None
        entry_price = None
        pnl = None
        pnl_pct = None
        first_trade_time = None

        # Try direct USDT pair
        symbol_usdt = f'{asset}USDT'
        if symbol_usdt in all_prices:
            price_usdt = all_prices[symbol_usdt]
        else:
            # Try BTC pair then convert
            symbol_btc = f'{asset}BTC'
            if symbol_btc in all_prices and 'BTCUSDT' in all_prices:
                price_usdt = all_prices[symbol_btc] * all_prices['BTCUSDT']
            # Try ETH pair
            elif f'{asset}ETH' in all_prices and 'ETHUSDT' in all_prices:
                price_usdt = all_prices[f'{asset}ETH'] * all_prices['ETHUSDT']

        current_value = total_qty * price_usdt if price_usdt else 0

        # Fetch trade history for P&L (skip stablecoins and small amounts)
        if price_usdt and asset not in stablecoins and current_value > 1:
            trades = get_my_trades(config, symbol_usdt, limit=500)
            if trades:
                buy_trades = [t for t in trades if t.get('isBuyer')]
                if buy_trades:
                    total_qty_bought = sum(float(t['qty']) for t in buy_trades)
                    total_cost = sum(float(t['quoteQty']) for t in buy_trades)
                    if total_qty_bought > 0:
                        entry_price = total_cost / total_qty_bought
                        # Calculate P&L based on current holdings
                        pnl = (price_usdt - entry_price) * total_qty
                        pnl_pct = ((price_usdt / entry_price) - 1) * 100
                        first_trade_time = datetime.fromtimestamp(
                            min(t['time'] for t in trades) / 1000,
                            tz=timezone.utc,
                        )

        results.append({
            'asset': asset,
            'free': free,
            'locked': locked,
            'total_qty': total_qty,
            'price_usdt': price_usdt,
            'current_value': current_value,
            'current_value_zar': current_value * usd_zar,
            'entry_price': entry_price,
            'pnl': pnl,
            'pnl_zar': pnl * usd_zar if pnl is not None else None,
            'pnl_pct': pnl_pct,
            'first_trade_time': first_trade_time,
            'is_stablecoin': asset in stablecoins,
        })

    # Sort by current value descending
    results.sort(key=lambda r: r['current_value'], reverse=True)

    total_value = sum(r['current_value'] for r in results)
    total_pnl = sum(r['pnl'] for r in results if r['pnl'] is not None)

    return {
        'balances': results,
        'total_value': total_value,
        'total_value_zar': total_value * usd_zar,
        'total_pnl': total_pnl,
        'total_pnl_zar': total_pnl * usd_zar,
        'usd_zar_rate': usd_zar,
        'can_trade': account.get('canTrade', False),
        'account_type': account.get('accountType', 'SPOT'),
    }
