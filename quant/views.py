"""
Quant Dashboard Views

Provides a live dashboard view and JSON API endpoints for
all quant system data. These are consumed by the frontend
for real-time monitoring of the trading system.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_http_methods

from quant.models import ExecutedTrade, Pair, QuantConfig, TradeSignal
from quant.services.alt_sentiment import AlternativeSentimentEngine
from quant.services.data_feeds import get_cache
from quant.services.sentiment_signals import sentiment_to_signal

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
def api_training_logs(request):
    """
    API: Get training history logs.

    Query params:
        model_type: Filter by model type (hmm_regime, random_forest, etc.)
        symbol: Filter by symbol
        limit: Max results (default: 20)

    Returns training history with metrics, timing, and feature importance.
    """
    from quant.models import TrainingLog

    model_type = request.GET.get("model_type", "")
    symbol = request.GET.get("symbol", "").strip().upper()
    limit = int(request.GET.get("limit", 20))

    qs = TrainingLog.objects.all()

    if model_type:
        qs = qs.filter(model_type=model_type)
    if symbol:
        qs = qs.filter(symbol=symbol)

    logs = qs.order_by("-started_at")[:limit]

    data = []
    for log in logs:
        data.append({
            "id": log.id,
            "model_type": log.model_type,
            "model_type_display": log.get_model_type_display(),
            "symbol": log.symbol,
            "interval": log.interval,
            "status": log.status,
            "config": log.config,
            "metrics": log.metrics,
            "feature_importance": log.feature_importance,
            "data_points": log.data_points,
            "feature_count": log.feature_count,
            "started_at": log.started_at.isoformat(),
            "completed_at": log.completed_at.isoformat() if log.completed_at else None,
            "duration_seconds": log.duration_seconds,
            "error_message": log.error_message,
        })

    return JsonResponse({"logs": data, "count": len(data)})


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


@require_GET
def api_alt_sentiment(request):
    """
    API: Get alternative sentiment data (Google Trends, GitHub, on-chain).

    Query params:
        symbol: Trading pair (e.g., BTCUSDT)

    Returns consensus score from AlternativeSentimentEngine.
    """
    symbol = request.GET.get("symbol", "").strip().upper()

    if not symbol:
        return JsonResponse({"error": "No symbol provided"})

    base_asset = symbol
    for quote in ["USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "PAX"]:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base_asset = symbol[: -len(quote)]
            break

    try:
        engine = AlternativeSentimentEngine()
        result = engine.compute_consensus_score(base_asset)
        return JsonResponse(result)
    except Exception as e:
        logger.exception(f"Alt sentiment failed for {symbol}")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_sentiment_signal(request):
    """
    API: Convert sentiment data into a trading signal.

    Query params:
        symbol: Trading pair (e.g., BTCUSDT)
        regime: Current regime label (optional, default: ranging)

    Returns the sentiment-to-signal conversion with contrarian logic.
    """
    symbol = request.GET.get("symbol", "").strip().upper()
    regime = request.GET.get("regime", "ranging").lower()

    if not symbol:
        return JsonResponse({"error": "No symbol provided"})

    base_asset = symbol
    for quote in ["USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "PAX"]:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base_asset = symbol[: -len(quote)]
            break

    try:
        # Get alt sentiment consensus
        engine = AlternativeSentimentEngine()
        alt_data = engine.compute_consensus_score(base_asset)

        # Convert to signal
        signal = sentiment_to_signal(alt_data, regime)
        return JsonResponse(signal)
    except Exception as e:
        logger.exception(f"Sentiment signal failed for {symbol}")
        return JsonResponse({"error": str(e)}, status=500)


# ══════════════════════════════════════════════════════════════════
#  Phase 5 — Execution Layer API Endpoints
# ══════════════════════════════════════════════════════════════════


@require_GET
def api_combine_signals(request):
    """
    API: Combine all active signals and return a trading decision.

    Query params:
        symbol: Trading pair (optional, defaults to all)
        regime: Override regime label (optional)
    """
    symbol = request.GET.get("symbol", "").strip().upper() or None
    regime = request.GET.get("regime", "").strip().lower() or None

    try:
        from quant.services.signal_combiner import SignalCombiner
        combiner = SignalCombiner()
        result = combiner.combine(symbol=symbol, regime=regime)
        if result is None:
            return JsonResponse({
                "action": "hold",
                "reason": "No active signals found",
                "symbol": symbol,
            })
        return JsonResponse(result)
    except Exception as e:
        logger.exception("Signal combination failed")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_execute(request):
    """
    API: Execute a trading decision based on the signal combiner.

    Query params:
        symbol: Trading pair (required)
        force: If 'true', execute even with low confidence (for testing)
    """
    symbol = request.GET.get("symbol", "").strip().upper()
    force = request.GET.get("force", "").lower() == "true"

    if not symbol:
        return JsonResponse({"error": "No symbol provided"})

    try:
        from quant.services.signal_combiner import SignalCombiner
        from quant.services.order_manager import OrderManager

        combiner = SignalCombiner()
        signal = combiner.combine(symbol=symbol)

        if signal is None:
            return JsonResponse({
                "status": "skipped",
                "reason": "No signals available",
            })

        if signal.get("action") != "trade" and not force:
            return JsonResponse({
                "status": "skipped",
                "reason": signal.get("reason", "Confidence below threshold"),
                "confidence": signal.get("confidence"),
            })

        # Override to trade if force=True
        if force and signal.get("action") != "trade":
            signal["action"] = "trade"
            if not signal.get("side"):
                signal["side"] = "buy" if (signal.get("confidence", 0) or 0) >= 0 else "sell"
            signal["confidence"] = max(abs(signal.get("confidence", 0) or 0), 0.55)

        # Select execution strategy
        from quant.services.execution_strategies import select_execution_strategy
        strategy = select_execution_strategy(signal)
        signal["execution_strategy"] = strategy

        # Execute via OrderManager
        config = QuantConfig.get_config()
        manager = OrderManager(mode=config.mode)
        result = manager.execute_signal(signal)

        return JsonResponse({
            "signal": signal,
            "execution": result,
        })

    except Exception as e:
        logger.exception(f"Execution failed for {symbol}")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_execution_strategy(request):
    """
    API: Preview the execution strategy for a proposed order without executing.

    Query params:
        symbol: Trading pair
        notional: Order value in USDT
        side: 'buy' or 'sell'
        confidence: Signal confidence (0-1)
    """
    symbol = request.GET.get("symbol", "").strip().upper()
    notional = request.GET.get("notional", "0")
    side = request.GET.get("side", "buy").lower()
    confidence = request.GET.get("confidence", "0.5")

    if not symbol:
        return JsonResponse({"error": "No symbol provided"})

    try:
        from quant.services.execution_strategies import select_execution_strategy

        order_info = {
            "symbol": symbol,
            "side": side,
            "notional": float(notional),
            "confidence": float(confidence),
        }

        strategy = select_execution_strategy(order_info)
        return JsonResponse(strategy)

    except Exception as e:
        logger.exception(f"Strategy selection failed for {symbol}")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_open_trades(request):
    """
    API: Get currently open trades/positions.
    """
    trades = ExecutedTrade.objects.filter(
        status="open"
    ).order_by("-entry_time")[:50]

    data = []
    for t in trades:
        data.append({
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": float(t.entry_price),
            "qty": float(t.qty),
            "entry_time": t.entry_time.isoformat(),
            "strategy": t.strategy,
            "order_id": t.order_id,
            "notes": t.notes,
        })

    return JsonResponse({"trades": data, "count": len(data)})


@require_http_methods(["GET", "POST"])
def api_config(request):
    """
    API: Get or update quant configuration.

    To update, use POST with params like:
        mode=paper&is_enabled=true&max_open_positions=3
    """
    if request.method == "POST":
        config = QuantConfig.get_config()

        for field in ["mode", "max_open_positions", "max_position_size_pct",
                       "max_daily_loss_pct", "max_drawdown_pct", "kelly_fraction"]:
            val = request.POST.get(field)
            if val is not None:
                try:
                    setattr(config, field, val)
                except (ValueError, TypeError):
                    pass

        is_enabled = request.POST.get("is_enabled")
        if is_enabled is not None:
            config.is_enabled = is_enabled.lower() in ("true", "1", "yes", "on")

        config.save()
        return JsonResponse({"status": "updated", "config": {
            "mode": config.mode,
            "is_enabled": config.is_enabled,
            "max_open_positions": config.max_open_positions,
            "max_position_size_pct": config.max_position_size_pct,
            "max_daily_loss_pct": config.max_daily_loss_pct,
            "max_drawdown_pct": config.max_drawdown_pct,
            "kelly_fraction": config.kelly_fraction,
            "virtual_balance": float(config.virtual_balance),
        }})

    # GET — return current config
    config = QuantConfig.get_config()
    return JsonResponse({
        "mode": config.mode,
        "is_enabled": config.is_enabled,
        "virtual_balance": float(config.virtual_balance),
        "real_balance_limit": float(config.real_balance_limit),
        "max_open_positions": config.max_open_positions,
        "max_position_size_pct": config.max_position_size_pct,
        "max_daily_loss_pct": config.max_daily_loss_pct,
        "max_drawdown_pct": config.max_drawdown_pct,
        "kelly_fraction": config.kelly_fraction,
    })


@require_GET
def api_cancel_order(request):
    """
    API: Cancel an open order.

    Query params:
        trade_id: ExecutedTrade ID to cancel
    """
    trade_id = request.GET.get("trade_id")
    if not trade_id:
        return JsonResponse({"error": "No trade_id provided"})

    try:
        trade = ExecutedTrade.objects.get(id=trade_id, status="open")
        trade.status = "cancelled"
        trade.exit_time = datetime.now(timezone.utc)
        trade.save(update_fields=["status", "exit_time"])
        return JsonResponse({"status": "cancelled", "trade_id": trade_id})
    except ExecutedTrade.DoesNotExist:
        return JsonResponse({"error": "Open trade not found"}, status=404)
    except Exception as e:
        logger.exception(f"Cancel failed for trade {trade_id}")
        return JsonResponse({"error": str(e)}, status=500)


# ══════════════════════════════════════════════════════════════════
#  Phase 6 — Portfolio Management & Risk API Endpoints
# ══════════════════════════════════════════════════════════════════


@require_GET
def api_kelly_size(request):
    """
    API: Calculate optimal position size using Kelly Criterion.

    Query params:
        capital: Available capital
        win_probability: P(profit) (0-1)
        avg_win: Average win % (e.g., 0.02 for 2%)
        avg_loss: Average loss % (e.g., 0.01 for 1%)
        fraction: 'full', 'half', or 'quarter' (default: quarter)
    """
    try:
        capital = float(request.GET.get("capital", 0))
        win_probability = float(request.GET.get("win_probability", 0))
        avg_win = float(request.GET.get("avg_win", 0))
        avg_loss = float(request.GET.get("avg_loss", 0))
        fraction = request.GET.get("fraction", "quarter")

        from quant.services.kelly_sizing import KellyPositionSizer
        sizer = KellyPositionSizer(fraction=fraction)
        config = QuantConfig.get_config()
        result = sizer.calculate_position_size(
            capital=capital,
            win_probability=win_probability,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_position_pct=config.max_position_size_pct,
        )
        return JsonResponse(result)
    except Exception as e:
        logger.exception("Kelly sizing failed")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_risk_check(request):
    """
    API: Check all risk rules for a proposed trade.

    Query params:
        symbol: Trading pair
        side: 'buy' or 'sell'
        notional: Order value in USDT
        confidence: Signal confidence (0-1)
    """
    symbol = request.GET.get("symbol", "").strip().upper()
    side = request.GET.get("side", "buy").lower()
    notional = request.GET.get("notional", "0")
    confidence = request.GET.get("confidence", "0.5")

    if not symbol:
        return JsonResponse({"error": "No symbol provided"})

    try:
        from quant.services.risk_manager import RiskManager

        risk = RiskManager()
        proposed = {
            "symbol": symbol,
            "side": side,
            "notional": float(notional),
            "confidence": float(confidence),
        }
        allowed, reason = risk.can_trade(proposed)

        return JsonResponse({
            "allowed": allowed,
            "reason": reason,
            "proposed_order": proposed,
        })
    except Exception as e:
        logger.exception(f"Risk check failed for {symbol}")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_stops(request):
    """
    API: Calculate stop loss and take profit levels.

    Query params:
        symbol: Trading pair
        side: 'buy' or 'sell'
        entry_price: Entry price
        regime: Current regime label (optional)
        strategy: 'fixed' or 'atr' (optional)
    """
    symbol = request.GET.get("symbol", "").strip().upper()
    side = request.GET.get("side", "buy").lower()
    entry_price = request.GET.get("entry_price", "0")
    regime = request.GET.get("regime", "ranging").lower()
    strategy = request.GET.get("strategy", "atr")

    if not symbol or not entry_price:
        return JsonResponse({"error": "symbol and entry_price required"})

    try:
        from quant.services.stop_loss import StopLossManager

        sl = StopLossManager(strategy=strategy)
        result = sl.calculate_stops(
            entry_price=float(entry_price),
            side=side,
            symbol=symbol,
            regime=regime,
        )
        return JsonResponse(result)
    except Exception as e:
        logger.exception(f"Stop calculation failed for {symbol}")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_check_exit(request):
    """
    API: Check if any exit condition is triggered for a position.

    Query params:
        entry_price, side, stop_loss, take_profit,
        current_price, highest_price (optional), entry_time
    """
    try:
        from quant.services.stop_loss import StopLossManager

        position = {
            "entry_price": float(request.GET.get("entry_price", 0)),
            "side": request.GET.get("side", "buy").lower(),
            "stop_loss": float(request.GET.get("stop_loss", 0)),
            "take_profit": float(request.GET.get("take_profit", 0)),
            "trailing_activation": float(request.GET.get("trailing_activation", 0)),
            "trailing_distance": float(request.GET.get("trailing_distance", 0)),
            "entry_time": request.GET.get("entry_time", ""),
        }
        current_price = float(request.GET.get("current_price", 0))
        highest_price = request.GET.get("highest_price")
        if highest_price:
            highest_price = float(highest_price)

        should_exit, reason = StopLossManager().should_exit(
            position, current_price, highest_price=highest_price
        )

        return JsonResponse({
            "should_exit": should_exit,
            "reason": reason,
        })
    except Exception as e:
        logger.exception("Exit check failed")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
@require_GET
def api_update_pair_signals(request):
    """
    API: Update z-scores and generate signals for all active pairs.
    """
    try:
        from quant.services.cointegration import PairsFinder, fetch_daily_close_prices
        from quant.services.pairs_signals import PairsSignalGenerator
        from quant.services.data_feeds import get_cache

        pairs = Pair.objects.filter(is_active=True)
        updated = 0
        signals_generated = 0

        for pair in pairs:
            try:
                # Fetch latest prices
                price_data = fetch_daily_close_prices(
                    [pair.symbol_a, pair.symbol_b], days=30
                )
                if len(price_data) >= 2:
                    # Compute z-score
                    series_a = price_data.get(pair.symbol_a)
                    series_b = price_data.get(pair.symbol_b)
                    if series_a is not None and series_b is not None:
                        combined = pd.concat(
                            [series_a.rename("a"), series_b.rename("b")],
                            axis=1,
                        ).dropna()
                        if len(combined) >= 20:
                            spread = combined["a"].values.astype(float) - pair.hedge_ratio * combined["b"].values.astype(float)
                            z = PairsFinder.compute_zscore(spread)
                            pair.current_zscore = z
                            pair.save(update_fields=["current_zscore"])

                            # Generate signal if z-score exceeds threshold
                            gen = PairsSignalGenerator()
                            signal = gen.evaluate_pair(pair, current_zscore=z)
                            if signal:
                                signals_generated += 1

                            updated += 1
            except Exception as e:
                logger.debug(f"Pair update failed for {pair}: {e}")

        return JsonResponse({
            "status": "ok",
            "pairs_updated": updated,
            "signals_generated": signals_generated,
        })
    except Exception as e:
        logger.exception("Pair signal update failed")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_portfolio_risk(request):
    """
    API: Get comprehensive portfolio risk snapshot.

    Returns drawdown, daily P&L, open positions, and all risk limits.
    """
    try:
        from quant.services.risk_manager import RiskManager

        config = QuantConfig.get_config()
        drawdown = RiskManager.get_current_drawdown(float(config.virtual_balance))
        daily_pnl = RiskManager.get_daily_pnl()

        return JsonResponse({
            "mode": config.mode,
            "is_enabled": config.is_enabled,
            "virtual_balance": float(config.virtual_balance),
            "current_drawdown": round(drawdown, 6),
            "daily_pnl": round(daily_pnl, 2),
            "max_drawdown_pct": config.max_drawdown_pct,
            "max_daily_loss_pct": config.max_daily_loss_pct,
            "max_open_positions": config.max_open_positions,
            "max_position_size_pct": config.max_position_size_pct,
            "kelly_fraction": config.kelly_fraction,
        })
    except Exception as e:
        logger.exception("Portfolio risk snapshot failed")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_run_command(request):
    """
    API: Run a quant management command from the dashboard.

    Query params:
        cmd: Command name (update_regime, discover_pairs, train_models,
             toggle_mode, reset_balance, status)
        arg: Optional argument for the command
    """
    cmd = request.GET.get("cmd", "")
    arg = request.GET.get("arg", "")

    if not cmd:
        return JsonResponse({"status": "error", "error": "No command specified"})

    from io import StringIO
    from django.core.management import call_command

    output = StringIO()
    try:
        if cmd == "update_regime":
            call_command("update_regime", stdout=output)
        elif cmd == "discover_pairs":
            call_command("discover_pairs", "--limit", "10", "--days", "30", stdout=output)
        elif cmd == "train_models":
            symbol = arg or "BTCUSDT"
            # Train with 1000 candles — models are saved by default now
            call_command("train_models", symbol, "--limit", "1000", stdout=output)
        elif cmd == "toggle_mode":
            action = arg or "status"
            call_command("toggle_mode", action, stdout=output)
        elif cmd == "reset_balance":
            call_command("toggle_mode", "reset", "--balance", "10000", stdout=output)
        elif cmd == "status":
            call_command("toggle_mode", "status", stdout=output)
        else:
            return JsonResponse({"status": "error", "error": f"Unknown command: {cmd}"})

        output_lines = output.getvalue().strip().split("\n") if output.getvalue() else []
        return JsonResponse({"status": "ok", "output": output_lines})

    except Exception as e:
        logger.exception(f"Command {cmd} failed")
        return JsonResponse({"status": "error", "error": str(e)})
