"""
Autonomous Trading Bot — Data Models

Implements the full data schema for a self-improving spot trading system.
All models are TimescaleDB-compatible for future migration from SQLite.

Requirements from plan:
- Every data source produces at least one model
- Full audit trail for all decisions
- Zero hard-coded secrets
- All timestamps timezone-aware (UTC)
"""

import logging
from decimal import Decimal

from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  Core Strategy Framework
# ═══════════════════════════════════════════════════════════════════


class Strategy(models.Model):
    """
    Registered trading strategy.

    Each strategy is a pure function that accepts a feature matrix
    and returns a signal series (+1 / 0 / -1) + confidence.

    The strategy_class field stores the Python import path so strategies
    can be loaded dynamically at runtime.
    """

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(
        default=False,
        help_text="Whether this strategy is currently generating signals",
    )
    strategy_class = models.CharField(
        max_length=200,
        help_text="Python import path (e.g. trading_bot.services.strategies.technical.MomentumStrategy)",
    )
    weight = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Dynamic ensemble weight (updated by promoter)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Strategies"
        ordering = ["-weight", "name"]

    def __str__(self) -> str:
        return f"{self.name} ({'active' if self.is_active else 'inactive'}, weight={self.weight:.2f})"


class ParamSet(models.Model):
    """
    A specific parameter configuration for a strategy.

    Each training/optimization cycle produces candidate ParamSets.
    The promoter promotes the best one to 'live' status only if it
    beats the current live set on out-of-sample metrics.

    Metrics stored here so the dashboard can rank and compare
    candidate ParamSets.
    """

    strategy = models.ForeignKey(
        Strategy, on_delete=models.CASCADE, related_name="param_sets"
    )
    params = models.JSONField(
        help_text="Strategy parameters as JSON (e.g. {'rsi_period': 14, 'ema_fast': 9})"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    is_live = models.BooleanField(
        default=False,
        help_text="Currently deployed in paper/live trading",
    )
    is_candidate = models.BooleanField(
        default=False,
        help_text="Candidate for promotion (beating current live on OOS metrics)",
    )
    metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text="Performance metrics: sharpe, sortino, win_rate, profit_factor, max_dd, expectancy",
    )

    class Meta:
        indexes = [
            models.Index(fields=["strategy", "-created_at"]),
            models.Index(fields=["is_live", "is_candidate"]),
        ]
        ordering = ["-is_live", "-created_at"]

    def __str__(self) -> str:
        status = "LIVE" if self.is_live else "CANDIDATE" if self.is_candidate else "archived"
        sharpe = self.metrics.get("sharpe", "?")
        return f"{self.strategy.name} [{status}] sharpe={sharpe}"


class BacktestRun(models.Model):
    """
    Result of a single backtest execution.

    Stores the full equity curve and all performance metrics so the
    dashboard can display historical comparisons and the optimizer
    can rank ParamSets.

    The equity_curve is stored as JSON array of [timestamp, equity]
    pairs. For large backtests, this can be gigabytes — in production
    with PostgreSQL, consider storing as a separate timeseries table
    or Parquet file.
    """

    STATUS_CHOICES = [
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    param_set = models.ForeignKey(
        ParamSet, on_delete=models.CASCADE, related_name="backtest_runs"
    )
    symbol = models.CharField(max_length=20, db_index=True)
    interval = models.CharField(max_length=5, default="1h")
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="running")
    metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text="Full metrics: sharpe, sortino, calmar, win_rate, profit_factor, max_dd, "
        "expectancy, total_trades, avg_hold_time, etc.",
    )
    equity_curve = models.JSONField(
        null=True,
        blank=True,
        help_text="JSON array of [timestamp_epoch, equity] pairs",
    )
    error_message = models.TextField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["param_set", "symbol", "-created_at"]),
            models.Index(fields=["status", "-created_at"]),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        sharpe = self.metrics.get("sharpe", "?")
        return f"Backtest {self.param_set.strategy.name} {self.symbol} sharpe={sharpe}"

    @property
    def total_return_pct(self) -> float:
        """Calculate total return % from equity curve.

        Supports two formats:
        1. List of [timestamp_epoch, equity] pairs (Phase 7+)
        2. Flat list of equity values (Phase 5 backtester)
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        # Check format: if first element is a list/tuple, it's [ts, equity] pairs
        first = self.equity_curve[0]
        if isinstance(first, (list, tuple)):
            start_equity = first[1]
            end_equity = self.equity_curve[-1][1]
        else:
            # Flat equity values list
            start_equity = self.equity_curve[0]
            end_equity = self.equity_curve[-1]
        if start_equity == 0:
            return 0.0
        return (end_equity - start_equity) / start_equity * 100


# ═══════════════════════════════════════════════════════════════════
#  Market Data (TimescaleDB-compatible)
# ═══════════════════════════════════════════════════════════════════


class OHLCV(models.Model):
    """
    Raw OHLCV market data — the foundation of all analysis.

    Designed as a TimescaleDB hypertable candidate:
        SELECT create_hypertable('trading_bot_ohlcv', 'timestamp');

    Covers multi-exchange, multi-timeframe storage.
    Supports 1s, 5s, 15s, 1m, 5m, 15m, 1h, 4h, 1d intervals.

    Fields match the Binance kline format for direct ingestion.
    """

    INTERVAL_CHOICES = [
        ("1s", "1 Second"),
        ("5s", "5 Seconds"),
        ("15s", "15 Seconds"),
        ("1m", "1 Minute"),
        ("3m", "3 Minutes"),
        ("5m", "5 Minutes"),
        ("15m", "15 Minutes"),
        ("30m", "30 Minutes"),
        ("1h", "1 Hour"),
        ("2h", "2 Hours"),
        ("4h", "4 Hours"),
        ("6h", "6 Hours"),
        ("8h", "8 Hours"),
        ("12h", "12 Hours"),
        ("1d", "1 Day"),
        ("3d", "3 Days"),
        ("1w", "1 Week"),
    ]

    exchange = models.CharField(
        max_length=20, default="binance", db_index=True,
        help_text="Exchange identifier (binance, coinbase, kraken, etc.)",
    )
    symbol = models.CharField(max_length=20, db_index=True)
    interval = models.CharField(max_length=5, choices=INTERVAL_CHOICES, db_index=True)
    timestamp = models.DateTimeField(db_index=True)  # Hypertable time column candidate
    open = models.DecimalField(max_digits=20, decimal_places=8)
    high = models.DecimalField(max_digits=20, decimal_places=8)
    low = models.DecimalField(max_digits=20, decimal_places=8)
    close = models.DecimalField(max_digits=20, decimal_places=8)
    volume = models.DecimalField(max_digits=24, decimal_places=8)
    quote_volume = models.DecimalField(
        max_digits=24, decimal_places=8, null=True, blank=True
    )
    trades = models.IntegerField(null=True, blank=True)
    taker_buy_volume = models.DecimalField(
        max_digits=24, decimal_places=8, null=True, blank=True
    )
    taker_buy_quote_volume = models.DecimalField(
        max_digits=24, decimal_places=8, null=True, blank=True
    )
    closed = models.BooleanField(default=True)
    fetched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "OHLCV"
        verbose_name_plural = "OHLCV Data"
        indexes = [
            models.Index(fields=["exchange", "symbol", "interval", "timestamp"]),
            models.Index(fields=["symbol", "interval", "-timestamp"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["exchange", "symbol", "interval", "timestamp"],
                name="unique_ohlcv_candle",
            )
        ]
        # TimescaleDB: add this via migration after PostgreSQL migration
        # https://docs.timescale.com/api/latest/distributed-hypertables/create_distributed_hypertable/

    def __str__(self) -> str:
        return f"{self.exchange}:{self.symbol} {self.interval} @ {self.timestamp}"


# ═══════════════════════════════════════════════════════════════════
#  Features & Signals
# ═══════════════════════════════════════════════════════════════════


class FeatureSnapshot(models.Model):
    """
    Point-in-time feature values for ML models.

    Each row is a full feature vector at a given timestamp.
    In production with TimescaleDB, this becomes a hypertable with
    native compression for efficient storage of billions of rows.

    Feature versions are tracked for reproducibility — changing the
    feature engine increments the version so old snapshots remain valid.
    """

    timestamp = models.DateTimeField(db_index=True)
    symbol = models.CharField(max_length=20, db_index=True)
    interval = models.CharField(max_length=5, default="1h")
    feature_set_version = models.CharField(
        max_length=20, default="1.0",
        help_text="Version of the feature pipeline that generated this snapshot",
    )
    features = models.JSONField(
        help_text="Feature name → value dictionary. Keys are feature names, values are floats.",
    )
    source_hash = models.CharField(
        max_length=64, null=True, blank=True,
        help_text="SHA-256 of input data used to generate features (reproducibility)",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "feature_set_version", "-timestamp"]),
            models.Index(fields=["symbol", "interval", "-timestamp"]),
        ]
        ordering = ["-timestamp"]

    def __str__(self) -> str:
        n_features = len(self.features) if self.features else 0
        return f"Features {self.symbol} v{self.feature_set_version} ({n_features} features) @ {self.timestamp}"


class Signal(models.Model):
    """
    A trading signal generated by a strategy with a specific ParamSet.

    Every signal is logged with full provenance: which strategy,
    which parameter set, which features were used, and the confidence.
    This enables full audit trail and post-hoc analysis.

    Direction: +1 = long, 0 = neutral, -1 = short
    Confidence: 0.0 to 1.0
    """

    DIRECTION_CHOICES = [
        (1, "Long"),
        (0, "Neutral"),
        (-1, "Short"),
    ]

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("active", "Active"),
        ("filled", "Filled"),
        ("expired", "Expired"),
        ("cancelled", "Cancelled"),
    ]

    timestamp = models.DateTimeField(db_index=True)
    symbol = models.CharField(max_length=20, db_index=True)
    strategy = models.ForeignKey(
        Strategy, on_delete=models.CASCADE, related_name="signals"
    )
    param_set = models.ForeignKey(
        ParamSet, on_delete=models.CASCADE, related_name="signals",
        null=True, blank=True,
    )
    direction = models.SmallIntegerField(choices=DIRECTION_CHOICES)  # 1, 0, -1
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Model confidence / signal strength",
    )
    features_used = models.JSONField(
        null=True, blank=True,
        help_text="List of feature names that contributed to this signal",
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="pending"
    )
    metadata = models.JSONField(
        null=True, blank=True,
        help_text="Additional context (regime state, market conditions, etc.)",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "status", "-timestamp"]),
            models.Index(fields=["strategy", "status"]),
            models.Index(fields=["-timestamp"]),
        ]
        ordering = ["-timestamp"]

    def __str__(self) -> str:
        direction_label = "LONG" if self.direction == 1 else "SHORT" if self.direction == -1 else "NEUTRAL"
        return f"{self.strategy.name} {direction_label} {self.symbol} ({self.confidence:.0%})"


# ═══════════════════════════════════════════════════════════════════
#  Trade Execution
# ═══════════════════════════════════════════════════════════════════


class Trade(models.Model):
    """
    Base trade model covering both paper and live trades.

    Paper trades use realistic fill simulation (order book slippage + fees).
    Live trades use actual Binance order responses.

    The same model works for both — the 'mode' field distinguishes them.
    This avoids duplicating fields across PaperTrade / LiveTrade tables.
    """

    MODE_CHOICES = [
        ("paper", "Paper Trade"),
        ("live", "Live Trade"),
    ]

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("open", "Open"),
        ("closed", "Closed"),
        ("cancelled", "Cancelled"),
        ("rejected", "Rejected"),
    ]

    SIDE_CHOICES = [
        ("buy", "Buy"),
        ("sell", "Sell"),
    ]

    mode = models.CharField(max_length=10, choices=MODE_CHOICES, db_index=True)
    signal = models.ForeignKey(
        Signal, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="trades",
    )
    symbol = models.CharField(max_length=20, db_index=True)
    side = models.CharField(max_length=10, choices=SIDE_CHOICES)
    entry_price = models.DecimalField(max_digits=20, decimal_places=8)
    exit_price = models.DecimalField(
        max_digits=20, decimal_places=8, null=True, blank=True
    )
    quantity = models.DecimalField(max_digits=20, decimal_places=8)
    entry_time = models.DateTimeField(db_index=True)
    exit_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="pending"
    )
    pnl = models.DecimalField(
        max_digits=20, decimal_places=8, null=True, blank=True,
        help_text="Realized PnL in quote currency",
    )
    pnl_pct = models.FloatField(
        null=True, blank=True,
        help_text="PnL as percentage of entry value",
    )
    entry_fee = models.DecimalField(
        max_digits=20, decimal_places=8, null=True, blank=True
    )
    exit_fee = models.DecimalField(
        max_digits=20, decimal_places=8, null=True, blank=True
    )
    slippage = models.DecimalField(
        max_digits=20, decimal_places=8, null=True, blank=True,
        help_text="Estimated slippage for paper trades",
    )
    strategy = models.ForeignKey(
        Strategy, on_delete=models.SET_NULL, null=True, blank=True,
        help_text="Strategy that generated this trade",
    )
    param_set = models.ForeignKey(
        ParamSet, on_delete=models.SET_NULL, null=True, blank=True,
        help_text="ParamSet used when this trade was opened",
    )
    notes = models.TextField(blank=True)
    exchange_order_id = models.CharField(
        max_length=100, null=True, blank=True,
        help_text="Exchange order ID (live trades only)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "status", "-entry_time"]),
            models.Index(fields=["mode", "status"]),
            models.Index(fields=["strategy", "-entry_time"]),
        ]
        ordering = ["-entry_time"]

    def __str__(self) -> str:
        return f"[{self.mode.upper()}] {self.side.upper()} {self.symbol} {self.quantity} @ {self.entry_price}"


# ═══════════════════════════════════════════════════════════════════
#  Order Book & Market Microstructure
# ═══════════════════════════════════════════════════════════════════


class OrderBookSnapshot(models.Model):
    """
    Full L2/L3 order book depth snapshot.

    Captured at regular intervals for microstructure analysis.
    Stores top N levels (configurable) for both bids and asks.

    Used for:
    - Order book imbalance signals
    - Depth pressure analysis
    - Spread / slippage estimation
    - Absorption / exhaustion detection
    """

    exchange = models.CharField(max_length=20, default="binance")
    symbol = models.CharField(max_length=20, db_index=True)
    timestamp = models.DateTimeField(db_index=True)
    bids = models.JSONField(
        help_text="JSON array of [price, quantity] arrays, sorted best bid first",
    )
    asks = models.JSONField(
        help_text="JSON array of [price, quantity] arrays, sorted best ask first",
    )
    bid_volume = models.DecimalField(
        max_digits=24, decimal_places=8, null=True, blank=True
    )
    ask_volume = models.DecimalField(
        max_digits=24, decimal_places=8, null=True, blank=True
    )
    spread = models.DecimalField(
        max_digits=20, decimal_places=8, null=True, blank=True
    )
    spread_pct = models.FloatField(null=True, blank=True)
    imbalance_pct = models.FloatField(
        null=True, blank=True,
        help_text="bid_volume / (bid_volume + ask_volume) * 100",
    )
    depth_pressure = models.FloatField(
        null=True, blank=True,
        help_text="(bid_top5 - ask_top5) / (bid_top5 + ask_top5)",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "-timestamp"]),
        ]
        ordering = ["-timestamp"]

    def __str__(self) -> str:
        return f"OrderBook {self.symbol} @ {self.timestamp}"


# ═══════════════════════════════════════════════════════════════════
#  System Configuration & Audit
# ═══════════════════════════════════════════════════════════════════


class BotConfig(models.Model):
    """
    Singleton configuration for the autonomous trading bot.

    Controls operational mode, safety limits, and optimization settings.
    Only one row should exist (enforced by save() override).

    All parameters are adjustable via the dashboard and are
    logged in AuditLog whenever changed.
    """

    MODE_CHOICES = [
        ("backtest", "Backtest Only"),
        ("paper", "Paper Trading"),
        ("live", "Live Trading"),
    ]

    mode = models.CharField(
        max_length=20, choices=MODE_CHOICES, default="backtest"
    )
    is_enabled = models.BooleanField(
        default=False,
        help_text="Master on/off switch for automated trading",
    )
    virtual_balance = models.DecimalField(
        max_digits=20, decimal_places=2, default=10000.00
    )
    real_balance_limit = models.DecimalField(
        max_digits=20, decimal_places=2, default=500.00
    )

    # Risk limits
    max_open_positions = models.IntegerField(default=5)
    max_position_size_pct = models.FloatField(
        default=10.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(100.0)],
        help_text="Max single position as % of balance",
    )
    max_daily_loss_pct = models.FloatField(
        default=5.0,
        help_text="Halt trading if daily loss exceeds this %",
    )
    max_drawdown_pct = models.FloatField(
        default=15.0,
        help_text="Halt trading if drawdown exceeds this %",
    )
    circuit_breaker_count = models.IntegerField(
        default=3,
        help_text="Consecutive losses before circuit breaker trip",
    )
    circuit_breaker_hours = models.IntegerField(
        default=24,
        help_text="Hours to wait after circuit breaker trip",
    )

    # Optimization settings
    kelly_fraction = models.FloatField(
        default=0.25,
        validators=[MinValueValidator(0.01), MaxValueValidator(1.0)],
        help_text="Kelly fraction for position sizing (0.25 = Quarter Kelly)",
    )
    nightly_optimization_enabled = models.BooleanField(
        default=False,
        help_text="Run full optimization cycle nightly",
    )
    min_improvement_threshold = models.FloatField(
        default=0.05,
        help_text="Minimum OOS Sharpe improvement to promote new ParamSet (5%)",
    )
    max_trials_per_study = models.IntegerField(
        default=500,
        help_text="Max Optuna trials per optimization run",
    )

    # Data settings
    default_interval = models.CharField(max_length=5, default="1h")
    max_history_days = models.IntegerField(
        default=365 * 2,  # 2 years
        help_text="Default historical data window in days",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Bot Configuration"

    def save(self, *args, **kwargs):  # noqa: PLW0221
        """Enforce singleton pattern — only one config row exists."""
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def get_config(cls):
        """Get or create the singleton config."""
        config, created = cls.objects.get_or_create(pk=1)
        if created:
            logger.info("Created default BotConfig singleton")
        return config

    def __str__(self) -> str:
        return f"BotConfig: {self.get_mode_display()} ({'ON' if self.is_enabled else 'OFF'})"


class AuditLog(models.Model):
    """
    Immutable audit trail for all bot actions.

    Every config change, trade, signal, error, and promotion is
    logged here for full traceability.

    The 'action' field categorizes the log entry:
    - config_change: BotConfig parameter changed
    - signal_generated: New Signal created
    - trade_opened / trade_closed: Trade lifecycle
    - param_promoted: ParamSet promoted to live
    - optimization_run: Full optimization cycle completed
    - error: System error or circuit breaker trip
    - info: Informational message
    """

    ACTION_CHOICES = [
        ("config_change", "Configuration Change"),
        ("signal_generated", "Signal Generated"),
        ("trade_opened", "Trade Opened"),
        ("trade_closed", "Trade Closed"),
        ("param_promoted", "Parameter Set Promoted"),
        ("optimization_run", "Optimization Run"),
        ("error", "Error / Circuit Breaker"),
        ("info", "Information"),
    ]

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    action = models.CharField(max_length=30, choices=ACTION_CHOICES, db_index=True)
    message = models.TextField(help_text="Human-readable description of what happened")
    details = models.JSONField(
        null=True, blank=True,
        help_text="Structured data: trade IDs, signal IDs, metric values, etc.",
    )
    severity = models.CharField(
        max_length=10,
        choices=[
            ("debug", "Debug"),
            ("info", "Info"),
            ("warning", "Warning"),
            ("error", "Error"),
            ("critical", "Critical"),
        ],
        default="info",
    )

    class Meta:
        indexes = [
            models.Index(fields=["action", "-timestamp"]),
            models.Index(fields=["severity", "-timestamp"]),
        ]
        ordering = ["-timestamp"]

    def __str__(self) -> str:
        return f"[{self.get_action_display()}] {self.message[:80]}"
