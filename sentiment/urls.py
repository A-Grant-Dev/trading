from django.urls import path

from . import views

urlpatterns = [
    path("data/", views.sentiment_data, name="sentiment_data"),
]
