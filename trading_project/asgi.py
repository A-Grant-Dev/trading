"""
ASGI config for trading_project project.

Supports both HTTP (via Django WSGI handler) and WebSocket
(via Django Channels) for real-time data streaming.
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_project.settings')

django_asgi_app = get_asgi_application()

# Import routing after Django is fully initialized
import trading_bot.routing  # noqa: E402

application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': AuthMiddlewareStack(
        URLRouter(
            trading_bot.routing.websocket_urlpatterns
        )
    ),
})
