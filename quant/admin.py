from django.contrib import admin

from .models import (
    ExecutedTrade,
    FuturesData,
    HistoricalDataBatch,
    MarketData,
    OrderBookSnapshot,
    Pair,
    QuantConfig,
    TradeRecord,
    TradeSignal,
    TrainingLog,
)


@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = ("symbol", "interval", "open_time", "open", "high", "low", "close", "volume")
    list_filter = ("symbol", "interval")
    search_fields = ("symbol",)
    date_hierarchy = "open_time"
    ordering = ("-open_time",)


@admin.register(HistoricalDataBatch)
class HistoricalDataBatchAdmin(admin.ModelAdmin):
    list_display = ("symbol", "interval", "date_range_start", "date_range_end", "status", "rows_imported")
    list_filter = ("symbol", "interval", "status")
    ordering = ("-created_at",)


@admin.register(OrderBookSnapshot)
class OrderBookSnapshotAdmin(admin.ModelAdmin):
    list_display = ("symbol", "timestamp", "spread", "imbalance_pct", "bid_vol", "ask_vol")
    list_filter = ("symbol",)
    date_hierarchy = "timestamp"


@admin.register(TradeRecord)
class TradeRecordAdmin(admin.ModelAdmin):
    list_display = ("symbol", "trade_id", "price", "qty", "is_buyer_maker", "time")
    list_filter = ("symbol", "is_buyer_maker")
    date_hierarchy = "time"
    ordering = ("-time",)


@admin.register(FuturesData)
class FuturesDataAdmin(admin.ModelAdmin):
    list_display = (
        "symbol", "timestamp", "funding_rate", "open_interest",
        "long_short_ratio", "mark_price",
    )
    list_filter = ("symbol",)
    date_hierarchy = "timestamp"


@admin.register(Pair)
class PairAdmin(admin.ModelAdmin):
    list_display = (
        "symbol_a", "symbol_b", "is_active", "coint_p_value",
        "half_life", "current_zscore", "win_rate",
    )
    list_filter = ("is_active",)
    search_fields = ("symbol_a", "symbol_b")


@admin.register(TradeSignal)
class TradeSignalAdmin(admin.ModelAdmin):
    list_display = (
        "symbol", "pair", "signal_type", "direction", "strength",
        "confidence", "source_model", "generated_at", "expiry", "status",
    )
    list_filter = ("source_model", "signal_type", "status")
    date_hierarchy = "generated_at"
    ordering = ("-generated_at",)


@admin.register(ExecutedTrade)
class ExecutedTradeAdmin(admin.ModelAdmin):
    list_display = (
        "symbol", "side", "entry_price", "exit_price", "qty",
        "pnl", "pnl_pct", "entry_time", "exit_time", "status", "strategy",
    )
    list_filter = ("side", "status", "strategy")
    date_hierarchy = "entry_time"
    ordering = ("-entry_time",)


@admin.register(QuantConfig)
class QuantConfigAdmin(admin.ModelAdmin):
    list_display = (
        "mode", "is_enabled", "virtual_balance", "max_open_positions",
        "kelly_fraction", "updated_at",
    )


@admin.register(TrainingLog)
class TrainingLogAdmin(admin.ModelAdmin):
    list_display = (
        "model_type", "symbol", "status", "data_points",
        "duration_seconds", "started_at",
    )
    list_filter = ("model_type", "status", "symbol")
    date_hierarchy = "started_at"
    ordering = ("-started_at",)
