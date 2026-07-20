from django.urls import path

from . import views

urlpatterns = [
    path('depth/', views.depth_json, name='orderbook_depth'),
]
