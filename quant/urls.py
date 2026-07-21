from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="quant_dashboard"),
    path("api/regime/", views.api_regime, name="quant_api_regime"),
    path("api/signals/", views.api_active_signals, name="quant_api_signals"),
    path("api/portfolio/", views.api_portfolio, name="quant_api_portfolio"),
    path("api/trades/recent/", views.api_recent_trades, name="quant_api_recent_trades"),
    path("api/pairs/", views.api_pairs, name="quant_api_pairs"),
    path("api/performance/", views.api_performance, name="quant_api_performance"),
    path("api/training-logs/", views.api_training_logs, name="quant_api_training_logs"),
    path("api/update-pair-signals/", views.api_update_pair_signals, name="quant_api_update_pair_signals"),
    path("api/alt-sentiment/", views.api_alt_sentiment, name="quant_api_alt_sentiment"),
    path("api/sentiment-signal/", views.api_sentiment_signal, name="quant_api_sentiment_signal"),
    # Phase 5 — Execution Layer
    path("api/combine-signals/", views.api_combine_signals, name="quant_api_combine_signals"),
    path("api/execute/", views.api_execute, name="quant_api_execute"),
    path("api/execution-strategy/", views.api_execution_strategy, name="quant_api_execution_strategy"),
    path("api/trades/open/", views.api_open_trades, name="quant_api_open_trades"),
    path("api/config/", views.api_config, name="quant_api_config"),
    path("api/cancel-order/", views.api_cancel_order, name="quant_api_cancel_order"),
    # Phase 6 — Portfolio Management & Risk
    path("api/kelly-size/", views.api_kelly_size, name="quant_api_kelly_size"),
    path("api/risk-check/", views.api_risk_check, name="quant_api_risk_check"),
    path("api/stops/", views.api_stops, name="quant_api_stops"),
    path("api/check-exit/", views.api_check_exit, name="quant_api_check_exit"),
    path("api/portfolio-risk/", views.api_portfolio_risk, name="quant_api_portfolio_risk"),
    # Command Runner
    path("api/run-command/", views.api_run_command, name="quant_api_run_command"),
]
