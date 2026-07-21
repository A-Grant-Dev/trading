"""
Autonomous Trading Bot — Views & API

Provides dashboard views and REST API endpoints for:
- Viewing training history, ParamSets, backtest results
- Triggering commands (download_history, rebuild_features, run_optimization)
- Viewing live signals and trades
- Optimization status and leaderboard
"""

import logging
from datetime import datetime, timezone

from django.db.models import Count, Min, Q, Sum
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from trading_bot.models import (
    AuditLog,
    BacktestRun,
    FeatureSnapshot,
    OHLCV,
    ParamSet,
    Signal,
    Strategy,
    Trade,
)

logger = logging.getLogger(__name__)


# ── Dashboard ──────────────────────────────────────────────────────


@require_GET
def dashboard(request):
    """Main trading bot dashboard."""
    active_strategies = Strategy.objects.filter(is_active=True).count()
    total_signals = Signal.objects.count()
    active_signals = Signal.objects.filter(status="active").count()
    open_trades = Trade.objects.filter(status="open").count()
    total_trades = Trade.objects.count()
    total_backtests = BacktestRun.objects.count()
    total_features = FeatureSnapshot.objects.count()
    ohlcv_count = OHLCV.objects.count()

    # Latest backtests
    latest_backtests = BacktestRun.objects.select_related(
        "param_set__strategy"
    ).order_by("-created_at")[:10]

    # Active signals
    recent_signals = Signal.objects.select_related("strategy", "param_set").filter(
        status__in=["pending", "active"]
    ).order_by("-timestamp")[:20]

    # Latest trades
    recent_trades = Trade.objects.select_related(
        "signal", "strategy"
    ).order_by("-entry_time")[:20]

    # ParamSet leaderboard (candidates ranked by Sharpe)
    live_params = ParamSet.objects.filter(is_live=True).select_related("strategy")
    candidate_params = ParamSet.objects.filter(
        is_candidate=True, is_live=False
    ).select_related("strategy").order_by("-metrics__sharpe_ratio")[:10]

    # Optimization status
    total_paramsets = ParamSet.objects.count()
    optimization_runs = AuditLog.objects.filter(action="optimization_run").count()
    last_optimization = AuditLog.objects.filter(
        action="optimization_run"
    ).order_by("-timestamp").first()

    # Audit log
    recent_audit = AuditLog.objects.order_by("-timestamp")[:20]

    # OHLCV coverage breakdown
    ohlcv_coverage = list(
        OHLCV.objects.values("exchange", "symbol", "interval")
        .annotate(total=Count("id"))
        .order_by("symbol", "interval")
    )

    context = {
        "active_strategies": active_strategies,
        "total_signals": total_signals,
        "active_signals": active_signals,
        "open_trades": open_trades,
        "total_trades": total_trades,
        "total_backtests": total_backtests,
        "total_features": total_features,
        "ohlcv_count": ohlcv_count,
        "ohlcv_coverage": ohlcv_coverage,
        "latest_backtests": latest_backtests,
        "recent_signals": recent_signals,
        "recent_trades": recent_trades,
        "live_params": live_params,
        "candidate_params": candidate_params,
        "recent_audit": recent_audit,
        # Optimization status
        "total_paramsets": total_paramsets,
        "optimization_runs": optimization_runs,
        "last_optimization": last_optimization,
    }

    return render(request, "trading_bot/dashboard.html", context)


# ── API Endpoints ──────────────────────────────────────────────────


@require_GET
def api_stats(request):
    """Return aggregate statistics as JSON."""
    data = {
        "active_strategies": Strategy.objects.filter(is_active=True).count(),
        "total_signals": Signal.objects.count(),
        "active_signals": Signal.objects.filter(status="active").count(),
        "open_trades": Trade.objects.filter(status="open").count(),
        "total_trades": Trade.objects.count(),
        "win_rate": _calculate_win_rate(),
        "total_backtests": BacktestRun.objects.count(),
        "ohlcv_candles": OHLCV.objects.count(),
        "live_params": ParamSet.objects.filter(is_live=True).count(),
        "candidate_params": ParamSet.objects.filter(is_candidate=True, is_live=False).count(),
        "optimization_runs": AuditLog.objects.filter(action="optimization_run").count(),
    }
    return JsonResponse(data)


@require_GET
def api_signals(request):
    """Return recent signals as JSON."""
    limit = int(request.GET.get("limit", 20))
    signals = Signal.objects.select_related("strategy", "param_set").order_by(
        "-timestamp"
    )[:limit]
    data = [
        {
            "id": s.id,
            "timestamp": s.timestamp.isoformat(),
            "symbol": s.symbol,
            "strategy": s.strategy.name,
            "direction": s.direction,
            "confidence": s.confidence,
            "status": s.status,
        }
        for s in signals
    ]
    return JsonResponse({"signals": data})


@require_GET
def api_backtests(request):
    """Return latest backtest runs as JSON."""
    limit = int(request.GET.get("limit", 10))
    backtests = BacktestRun.objects.select_related(
        "param_set__strategy"
    ).order_by("-created_at")[:limit]
    data = [
        {
            "id": b.id,
            "strategy": b.param_set.strategy.name if b.param_set else "?",
            "symbol": b.symbol,
            "status": b.status,
            "sharpe": b.metrics.get("sharpe_ratio"),
            "win_rate": b.metrics.get("win_rate"),
            "total_return_pct": b.total_return_pct,
            "start_date": b.start_date.isoformat(),
            "end_date": b.end_date.isoformat(),
            "duration_seconds": b.duration_seconds,
        }
        for b in backtests
    ]
    return JsonResponse({"backtests": data})


@require_GET
def api_ohlcv_overview(request):
    """Return OHLCV coverage summary by symbol/interval."""
    data = list(
        OHLCV.objects.values("exchange", "symbol", "interval").annotate(
            total=Count("id"),
            earliest=Min("timestamp")
        ).order_by("symbol", "interval")[:50]
    )
    return JsonResponse({"coverage": data})


# ── Optimization API ────────────────────────────────────────────────


@require_GET
def api_optimization_status(request):
    """Return optimization status overview."""
    live_params = ParamSet.objects.filter(is_live=True).select_related("strategy")
    candidate_params = ParamSet.objects.filter(
        is_candidate=True, is_live=False
    ).select_related("strategy").order_by("-metrics__sharpe_ratio")[:20]

    last_opt = AuditLog.objects.filter(
        action="optimization_run"
    ).order_by("-timestamp").first()

    return JsonResponse({
        "total_paramsets": ParamSet.objects.count(),
        "live_count": live_params.count(),
        "candidate_count": candidate_params.count(),
        "optimization_runs": AuditLog.objects.filter(action="optimization_run").count(),
        "last_optimization": {
            "timestamp": last_opt.timestamp.isoformat() if last_opt else None,
            "message": last_opt.message if last_opt else None,
        } if last_opt else None,
        "live_paramsets": [
            {
                "strategy": p.strategy.name,
                "sharpe": p.metrics.get("sharpe_ratio"),
                "win_rate": p.metrics.get("win_rate"),
                "created": p.created_at.isoformat(),
            }
            for p in live_params
        ],
        "candidate_paramsets": [
            {
                "strategy": p.strategy.name,
                "sharpe": p.metrics.get("sharpe_ratio"),
                "win_rate": p.metrics.get("win_rate"),
                "created": p.created_at.isoformat(),
            }
            for p in candidate_params
        ],
    })


# ── Paper Trading API ──────────────────────────────────────────────


@require_GET
def api_paper_trading_status(request):
    """Return paper trading portfolio status as JSON."""
    try:
        from trading_bot.services.executor.paper import get_paper_portfolio_summary
        from trading_bot.models import BotConfig

        config = BotConfig.get_config()
        summary = get_paper_portfolio_summary()
        summary["mode"] = config.mode
        summary["is_enabled"] = config.is_enabled
        summary["max_open_positions"] = config.max_open_positions
        summary["max_position_size_pct"] = config.max_position_size_pct
        return JsonResponse(summary)
    except Exception as e:
        logger.exception("Paper trading status failed")
        return JsonResponse({"error": str(e)}, status=500)


# ── Live Trading API ───────────────────────────────────────────────


@require_GET
def api_live_trading_status(request):
    """Return live trading portfolio status as JSON."""
    try:
        from trading_bot.services.executor.live import get_live_portfolio_summary

        summary = get_live_portfolio_summary()
        return JsonResponse(summary)
    except Exception as e:
        logger.exception("Live trading status failed")
        return JsonResponse({"error": str(e)}, status=500)


@require_POST
def api_live_trading_flatten(request):
    """Emergency kill switch — flatten all open live positions."""
    dry_run = request.GET.get("dry_run", "true").lower() == "true"

    try:
        from trading_bot.services.executor.live import flatten_all_live
        from trading_bot.models import BotConfig

        config = BotConfig.get_config()
        if config.mode != "live":
            return JsonResponse({"error": "Mode is not 'live'"}, status=400)

        n_closed = flatten_all_live(dry_run=dry_run)
        return JsonResponse({
            "n_closed": n_closed,
            "dry_run": dry_run,
            "message": f"Flattened {n_closed} positions" + (" (dry run)" if dry_run else ""),
        })
    except Exception as e:
        logger.exception("Kill switch failed")
        return JsonResponse({"error": str(e)}, status=500)


# ── HTMX Partial Refresh APIs ─────────────────────────────────────


@require_GET
def api_trades(request):
    """Return recent trades as HTML snippet for HTMX refresh."""
    limit = int(request.GET.get("limit", 5))
    trades = Trade.objects.select_related("signal", "strategy").order_by("-entry_time")[:limit]

    if not trades:
        return JsonResponse({"trades": []})

    data = []
    for t in trades:
        data.append({
            "id": t.id,
            "entry_time": t.entry_time.isoformat(),
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": float(t.entry_price),
            "pnl": float(t.pnl) if t.pnl else None,
            "status": t.status,
            "mode": t.mode,
        })
    return JsonResponse({"trades": data})


@require_GET
def api_audit(request):
    """Return recent audit log entries as HTML snippet for HTMX refresh."""
    limit = int(request.GET.get("limit", 10))
    entries = AuditLog.objects.order_by("-timestamp")[:limit]
    data = [
        {
            "timestamp": e.timestamp.isoformat(),
            "action": e.get_action_display(),
            "action_key": e.action,
            "message": e.message[:80],
            "severity": e.severity,
        }
        for e in entries
    ]
    return JsonResponse({"audit": data})


@require_GET
def api_metrics(request):
    """Return Prometheus-style metrics for observability."""
    from django.db.models import Sum, Count
    from datetime import date, timedelta

    today = date.today()
    week_ago = today - timedelta(days=7)

    total_closed = Trade.objects.filter(status="closed").count()
    wins = Trade.objects.filter(status="closed", pnl__gt=0).count()
    total_pnl = Trade.objects.filter(status="closed").aggregate(Sum("pnl"))["pnl__sum"] or 0

    lines = [
        "# HELP trading_bot_strategies_active Number of active strategies",
        "# TYPE trading_bot_strategies_active gauge",
        f"trading_bot_strategies_active {Strategy.objects.filter(is_active=True).count()}",
        "",
        "# HELP trading_bot_signals_total Total signals generated",
        "# TYPE trading_bot_signals_total counter",
        f"trading_bot_signals_total {Signal.objects.count()}",
        "",
        "# HELP trading_bot_trades_total Total trades executed",
        "# TYPE trading_bot_trades_total counter",
        f"trading_bot_trades_total {Trade.objects.count()}",
        "",
        "# HELP trading_bot_trades_open Currently open positions",
        "# TYPE trading_bot_trades_open gauge",
        f"trading_bot_trades_open {Trade.objects.filter(status='open').count()}",
        "",
        "# HELP trading_bot_pnl_total Realized PnL in quote currency",
        "# TYPE trading_bot_pnl_total gauge",
        f"trading_bot_pnl_total {float(total_pnl)}",
        "",
        "# HELP trading_bot_win_rate Win rate from closed trades (0-100)",
        "# TYPE trading_bot_win_rate gauge",
        f"trading_bot_win_rate {(wins / total_closed * 100) if total_closed > 0 else 0}",
        "",
        "# HELP trading_bot_backtests_total Total backtest runs",
        "# TYPE trading_bot_backtests_total counter",
        f"trading_bot_backtests_total {BacktestRun.objects.count()}",
        "",
        "# HELP trading_bot_ohlcv_candles OHLCV candles stored",
        "# TYPE trading_bot_ohlcv_candles gauge",
        f"trading_bot_ohlcv_candles {OHLCV.objects.count()}",
        "",
        "# HELP trading_bot_features_total Feature snapshots stored",
        "# TYPE trading_bot_features_total gauge",
        f"trading_bot_features_total {FeatureSnapshot.objects.count()}",
        "",
        "# HELP trading_bot_paramsets_total Total ParamSets",
        "# TYPE trading_bot_paramsets_total gauge",
        f"trading_bot_paramsets_total {ParamSet.objects.count()}",
        "",
        "# HELP trading_bot_optimization_runs Total optimization cycles",
        "# TYPE trading_bot_optimization_runs counter",
        f"trading_bot_optimization_runs {AuditLog.objects.filter(action='optimization_run').count()}",
    ]
    return HttpResponse("\n".join(lines), content_type="text/plain; charset=utf-8")


def _calculate_win_rate() -> float:
    """Calculate overall win rate from closed trades."""
    closed = Trade.objects.filter(status="closed")
    total = closed.count()
    if total == 0:
        return 0.0
    wins = closed.filter(pnl__gt=0).count()
    return round(wins / total * 100, 1)
