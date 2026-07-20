from django.urls import path

from . import views

urlpatterns = [
    path("chat/", views.chat, name="ai_chat"),
    path("browser-chat/", views.browser_chat, name="ai_browser_chat"),
]
