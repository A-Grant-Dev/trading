from django.shortcuts import render

from .services import get_asset_balance, get_active_config


def portfolio(request):
    """Display portfolio overview with balances, P&L, and investment details."""

    config = get_active_config()
    context = {
        'has_config': False,
        'error': None,
        'balances': [],
        'total_value': 0,
        'total_pnl': 0,
        'account_type': '—',
    }

    if not config:
        context['error'] = 'No active Binance API configuration found. Please add one in the admin panel.'
        return render(request, 'portfolio/index.html', context)

    try:
        data = get_asset_balance(config)
        context.update({
            'has_config': True,
            'balances': data['balances'],
            'total_value': data['total_value'],
            'total_value_zar': data['total_value_zar'],
            'total_pnl': data['total_pnl'],
            'total_pnl_zar': data['total_pnl_zar'],
            'usd_zar_rate': data['usd_zar_rate'],
            'account_type': data['account_type'],
        })
    except Exception as e:
        context['error'] = f'Failed to fetch portfolio data: {str(e)}'

    return render(request, 'portfolio/index.html', context)
