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
]
