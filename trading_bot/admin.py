"""
Autonomous Trading Bot — Admin Registration
"""

from django.contrib import admin

from trading_bot.models import (
    AuditLog,
    BacktestRun,
    BotConfig,
    FeatureSnapshot,
    OHLCV,
    OrderBookSnapshot,
    ParamSet,
    Signal,
    Strategy,
    Trade,
)


@admin.register(Strategy)
class StrategyAdmin(admin.ModelAdmin):
    list_display = ["name", "is_active", "weight", "created_at"]
    list_filter = ["is_active"]
    search_fields = ["name", "description"]


@admin.register(ParamSet)
class ParamSetAdmin(admin.ModelAdmin):
    list_display = ["strategy", "is_live", "is_candidate", "created_at"]
    list_filter = ["is_live", "is_candidate"]
    search_fields = ["strategy__name"]


@admin.register(BacktestRun)
class BacktestRunAdmin(admin.ModelAdmin):
    list_display = ["param_set", "symbol", "status", "start_date", "end_date", "duration_seconds"]
    list_filter = ["status", "symbol"]
    date_hierarchy = "created_at"


@admin.register(OHLCV)
class OHLCVAdmin(admin.ModelAdmin):
    list_display = ["exchange", "symbol", "interval", "timestamp", "open", "close", "volume"]
    list_filter = ["exchange", "symbol", "interval"]
    date_hierarchy = "timestamp"
    search_fields = ["symbol"]


@admin.register(FeatureSnapshot)
class FeatureSnapshotAdmin(admin.ModelAdmin):
    list_display = ["symbol", "interval", "feature_set_version", "timestamp"]
    list_filter = ["symbol", "feature_set_version", "interval"]
    date_hierarchy = "timestamp"


@admin.register(Signal)
class SignalAdmin(admin.ModelAdmin):
    list_display = ["strategy", "symbol", "direction", "confidence", "status", "timestamp"]
    list_filter = ["status", "direction", "strategy"]
    date_hierarchy = "timestamp"


@admin.register(Trade)
class TradeAdmin(admin.ModelAdmin):
    list_display = [
        "mode", "symbol", "side", "status", "entry_price", "quantity",
        "pnl", "pnl_pct", "entry_time", "exit_time",
    ]
    list_filter = ["mode", "status", "side", "symbol"]
    date_hierarchy = "entry_time"
    search_fields = ["symbol", "exchange_order_id"]


@admin.register(OrderBookSnapshot)
class OrderBookSnapshotAdmin(admin.ModelAdmin):
    list_display = ["symbol", "timestamp", "bid_volume", "ask_volume", "spread", "imbalance_pct"]
    list_filter = ["symbol"]
    date_hierarchy = "timestamp"


@admin.register(BotConfig)
class BotConfigAdmin(admin.ModelAdmin):
    list_display = [
        "mode", "is_enabled", "virtual_balance", "max_open_positions",
        "nightly_optimization_enabled", "updated_at",
    ]
    # Singleton — only 1 row, so disable add/delete
    actions = None

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ["timestamp", "action", "severity", "message"]
    list_filter = ["action", "severity"]
    date_hierarchy = "timestamp"
    search_fields = ["message"]
    readonly_fields = ["timestamp", "action", "message", "details", "severity"]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
