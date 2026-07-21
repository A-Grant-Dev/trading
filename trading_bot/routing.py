"""
Autonomous Trading Bot — WebSocket Routing

Maps WebSocket URL patterns to Channels consumers for
real-time data streaming from Binance.
"""

from django.urls import re_path

from trading_bot import consumers

websocket_urlpatterns = [
    re_path(r'ws/trading-bot/live/(?P<symbol>\w+)/$', consumers.BinanceStreamConsumer.as_asgi()),
]
