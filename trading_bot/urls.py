"""
Autonomous Trading Bot — URL Configuration
"""

from django.urls import path

from trading_bot import views

app_name = "trading_bot"

urlpatterns = [
    # Dashboard
    path("", views.dashboard, name="dashboard"),
    # API
    path("api/stats/", views.api_stats, name="api-stats"),
    path("api/signals/", views.api_signals, name="api-signals"),
    path("api/backtests/", views.api_backtests, name="api-backtests"),
    path("api/ohlcv-overview/", views.api_ohlcv_overview, name="api-ohlcv-overview"),
    # Optimization API
    path("api/optimization/", views.api_optimization_status, name="api-optimization"),
    # Paper Trading API
    path("api/paper-trading/", views.api_paper_trading_status, name="api-paper-trading"),
    # Live Trading API
    path("api/live-trading/", views.api_live_trading_status, name="api-live-trading"),
    path("api/live-trading/flatten/", views.api_live_trading_flatten, name="api-live-flatten"),
    # HTMX Partial Refresh APIs
    path("api/trades/", views.api_trades, name="api-trades"),
    path("api/audit/", views.api_audit, name="api-audit"),
    # Prometheus-style Metrics
    path("api/metrics/", views.api_metrics, name="api-metrics"),
]
