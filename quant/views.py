"""
Quant Dashboard Views

Provides a live dashboard view and JSON API endpoints for
all quant system data. These are consumed by the frontend
for real-time monitoring of the trading system.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from quant.models import ExecutedTrade, Pair, QuantConfig, TradeSignal
from quant.services.data_feeds import get_cache

logger = logging.getLogger(__name__)


@require_GET
def dashboard(request):
    """Main quant dashboard page."""
    config = QuantConfig.get_config()
    context = {
        "config": config,
    }
    return render(request, "quant/dashboard.html", context)


@require_GET
def api_regime(request):
    """
    API: Get current market regime for all tracked symbols.

    Returns the HMM-detected regime, confidence, and description.
    Refreshed every 15 minutes by the update_regime management command.
    """
    symbol = request.GET.get("symbol", "").strip().upper()

    if symbol:
        data = get_cache(f"regime:{symbol}")
        if not data:
            return JsonResponse({
                "symbol": symbol,
                "regime_label": "unknown",
                "error": "No regime data available. Run 'python manage.py update_regime' first.",
            })
        return JsonResponse(data)

    # Return all
    all_data = get_cache("regime:all", {})
    return JsonResponse(all_data, safe=False)


@require_GET
def api_active_signals(request):
    """
    API: Get all active (non-expired, non-executed) trading signals.
    """
    now = datetime.now(timezone.utc)
    signals = TradeSignal.objects.filter(
        status="active",
        expiry__gt=now,
    ).select_related("pair").order_by("-generated_at")[:50]

    data = []
    for s in signals:
        data.append({
            "id": s.id,
            "symbol": s.symbol,
            "pair": str(s.pair) if s.pair else None,
            "direction": s.direction,
            "signal_type": s.signal_type,
            "strength": s.strength,
            "confidence": s.confidence,
            "source_model": s.source_model,
            "generated_at": s.generated_at.isoformat(),
            "expiry": s.expiry.isoformat(),
        })

    return JsonResponse({"signals": data, "count": len(data)})


@require_GET
def api_portfolio(request):
    """
    API: Get portfolio status — balance, open positions, daily P&L, drawdown.
    """
    config = QuantConfig.get_config()
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Open positions
    open_positions = ExecutedTrade.objects.filter(status="open").count()

    # Today's trades
    today_trades = ExecutedTrade.objects.filter(entry_time__gte=today_start)
    today_pnl = sum(
        float(t.pnl or 0) for t in today_trades if t.pnl is not None
    )

    # Recent closed trades for win rate
    last_30d = now - timedelta(days=30)
    closed_trades = ExecutedTrade.objects.filter(
        status="closed",
        exit_time__gte=last_30d,
    )
    wins = closed_trades.filter(pnl__gt=0).count()
    total = closed_trades.count()
    win_rate = wins / total if total > 0 else 0

    # Latest regime info
    regime_info = get_cache("regime:all", {})

    return JsonResponse({
        "mode": config.mode,
        "is_enabled": config.is_enabled,
        "virtual_balance": float(config.virtual_balance),
        "open_positions": open_positions,
        "max_open_positions": config.max_open_positions,
        "daily_pnl": round(today_pnl, 2),
        "win_rate_30d": round(win_rate, 4),
        "total_trades_30d": total,
        "max_drawdown_pct": config.max_drawdown_pct,
        "kelly_fraction": config.kelly_fraction,
        "updated_at": now.isoformat(),
        "regimes": {
            s: d.get("regime_label", "unknown")
            for s, d in regime_info.items()
        } if regime_info else {},
    })


@require_GET
def api_recent_trades(request):
    """
    API: Get the most recent executed trades.
    """
    limit = int(request.GET.get("limit", 20))
    trades = ExecutedTrade.objects.order_by("-entry_time")[:limit]

    data = []
    for t in trades:
        data.append({
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price) if t.exit_price else None,
            "qty": float(t.qty),
            "pnl": float(t.pnl) if t.pnl else None,
            "pnl_pct": t.pnl_pct,
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "status": t.status,
            "strategy": t.strategy,
        })

    return JsonResponse({"trades": data, "count": len(data)})


@require_GET
def api_pairs(request):
    """
    API: Get all tracked pairs with current z-scores and stats.
    """
    pairs = Pair.objects.filter(is_active=True).order_by("coint_p_value")[:50]

    data = []
    for p in pairs:
        data.append({
            "id": p.id,
            "symbol_a": p.symbol_a,
            "symbol_b": p.symbol_b,
            "coint_p_value": p.coint_p_value,
            "half_life": p.half_life,
            "hedge_ratio": p.hedge_ratio,
            "current_zscore": p.current_zscore,
            "total_signals": p.total_signals,
            "total_trades": p.total_trades,
            "win_rate": p.win_rate,
            "last_tested": p.last_tested.isoformat() if p.last_tested else None,
        })

    return JsonResponse({"pairs": data, "count": len(data)})


@require_GET
def api_performance(request):
    """
    API: Get performance metrics — Sharpe, win rate, source accuracy.
    """
    now = datetime.now(timezone.utc)
    last_30d = now - timedelta(days=30)

    # Per-strategy performance
    strategies = ExecutedTrade.objects.filter(
        status="closed",
        exit_time__gte=last_30d,
    ).values("strategy").distinct()

    strategy_data = []
    for strat in strategies:
        name = strat["strategy"] or "unknown"
        trades = ExecutedTrade.objects.filter(
            strategy=name, status="closed", exit_time__gte=last_30d
        )
        total = trades.count()
        wins = trades.filter(pnl__gt=0).count()
        losses = trades.filter(pnl__lt=0).count()

        # Simplified Sharpe (if we have P&L data)
        pnl_values = [float(t.pnl or 0) for t in trades if t.pnl is not None]
        sharpe = 0.0
        if len(pnl_values) > 1 and np.std(pnl_values) > 0:
            sharpe = round(np.mean(pnl_values) / np.std(pnl_values) * np.sqrt(365), 2)

        strategy_data.append({
            "strategy": name,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total, 4) if total > 0 else 0,
            "sharpe_ratio": sharpe,
        })

    # Per-source performance (from TradeSignals → ExecutedTrades)
    source_data = []
    for source_code, source_label in TradeSignal.SOURCE_MODELS:
        signals = TradeSignal.objects.filter(
            source_model=source_code,
            generated_at__gte=last_30d,
        )
        total_signals = signals.count()
        executed = signals.filter(status="executed").count()
        source_data.append({
            "source": source_label,
            "total_signals": total_signals,
            "executed": executed,
            "execution_rate": round(executed / total_signals, 4) if total_signals > 0 else 0,
        })

    return JsonResponse({
        "strategies": strategy_data,
        "sources": source_data,
        "period_days": 30,
    })
