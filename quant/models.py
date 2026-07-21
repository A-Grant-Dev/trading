"""
Renaissance Quant — Data Models

Core data infrastructure for the quantitative trading platform.
Each model stores a specific type of market data or trading signal.

Inspired by Jim Simons' Renaissance Technologies:
- Statistical arbitrage requires high-resolution, clean historical data
- The Law of Large Numbers demands many data points across time
- Model training needs structured, indexed, queryable data
"""

from django.db import models


class TrainingLog(models.Model):
    """
    Training log for all quant model training runs.

    Persists every training run so the dashboard can display
    historical results, compare settings, and show what changed.

    Every time a command runs or models train, a record is created here.
    """

    MODEL_TYPES = [
        ("hmm_regime", "HMM Regime Detection"),
        ("random_forest", "Random Forest"),
        ("xgboost", "XGBoost"),
        ("ensemble", "Ensemble"),
        ("cointegration", "Cointegration / Pairs"),
    ]

    STATUS_CHOICES = [
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    model_type = models.CharField(max_length=30, choices=MODEL_TYPES, db_index=True)
    symbol = models.CharField(max_length=20, db_index=True)
    interval = models.CharField(max_length=5, default="1h")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="running")

    # Training configuration (what settings were used)
    config = models.JSONField(default=dict, blank=True, help_text="Training parameters used")

    # Results/metrics
    metrics = models.JSONField(default=dict, blank=True, help_text="Performance metrics from training")
    feature_importance = models.JSONField(null=True, blank=True, help_text="Top features by importance")

    # Data info
    data_points = models.IntegerField(null=True, blank=True)
    feature_count = models.IntegerField(null=True, blank=True)

    # Timing
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)

    # Error handling
    error_message = models.TextField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["model_type", "symbol", "-started_at"]),
            models.Index(fields=["-started_at"]),
        ]
        ordering = ["-started_at"]
        verbose_name_plural = "Training logs"

    def __str__(self):
        return f"{self.get_model_type_display()} {self.symbol} @ {self.started_at}"


class MarketData(models.Model):
    """
    OHLCV kline storage — the primary time-series table.

    Every candle (open, high, low, close, volume) for every tracked
    symbol and interval is stored here. This is the data that feeds:
      - HMM regime detection (Phase 1)
      - Cointegration / pairs trading (Phase 2)
      - ML model training (Phase 4)
      - Backtesting (Phase 8)

    Data sources:
      - Historical: Bulk CSV downloads from data.binance.vision
      - Live: WebSocket <symbol>@kline_<interval> stream
    """

    INTERVAL_CHOICES = [
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
        ("1M", "1 Month"),
    ]

    symbol = models.CharField(max_length=20, db_index=True)
    interval = models.CharField(max_length=5, choices=INTERVAL_CHOICES, db_index=True)
    open_time = models.DateTimeField(db_index=True)
    open = models.DecimalField(max_digits=20, decimal_places=8)
    high = models.DecimalField(max_digits=20, decimal_places=8)
    low = models.DecimalField(max_digits=20, decimal_places=8)
    close = models.DecimalField(max_digits=20, decimal_places=8)
    volume = models.DecimalField(max_digits=20, decimal_places=8)
    quote_asset_volume = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    taker_buy_base_vol = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    taker_buy_quote_vol = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    trades = models.IntegerField(null=True, blank=True)
    closed = models.BooleanField(default=True)  # True if candle is confirmed

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "interval", "open_time"]),
            models.Index(fields=["symbol", "open_time"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["symbol", "interval", "open_time"],
                name="unique_kline",
            )
        ]

    def __str__(self):
        return f"{self.symbol} {self.interval} @ {self.open_time}"


class HistoricalDataBatch(models.Model):
    """
    Tracks bulk CSV downloads from data.binance.vision.

    Keeps a record of what historical data has been imported so we
    don't re-download the same ranges. Supports incremental backfilling.
    """

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("downloading", "Downloading"),
        ("importing", "Importing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    symbol = models.CharField(max_length=20, db_index=True)
    interval = models.CharField(max_length=5)
    date_range_start = models.DateField()
    date_range_end = models.DateField()
    file_path = models.CharField(max_length=500, blank=True, null=True)
    file_size_bytes = models.BigIntegerField(null=True, blank=True)
    rows_imported = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "interval", "status"]),
        ]

    def __str__(self):
        return f"{self.symbol} {self.interval} {self.date_range_start}–{self.date_range_end}"


class OrderBookSnapshot(models.Model):
    """
    Periodic order book depth snapshots for ML training and analysis.

    Captures the state of the order book at regular intervals so we
    can compute features like:
      - Bid/ask imbalance ratio
      - Depth pressure (concentrated vs distributed)
      - Spread width changes
      - Order book slope / shape

    These features feed into the ML prediction engine (Phase 4) and
    short-term execution signals.
    """

    symbol = models.CharField(max_length=20, db_index=True)
    timestamp = models.DateTimeField(db_index=True)
    bids_json = models.JSONField(help_text="JSON array of [price, qty] pairs")
    asks_json = models.JSONField(help_text="JSON array of [price, qty] pairs")
    bid_vol = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    ask_vol = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    spread = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    imbalance_pct = models.FloatField(null=True, blank=True, help_text="bid_vol / (bid_vol + ask_vol) * 100")
    first_bid_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    first_ask_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "timestamp"]),
        ]

    def __str__(self):
        return f"{self.symbol} depth @ {self.timestamp}"


class TradeRecord(models.Model):
    """
    Individual trade storage for market microstructure analysis.

    Every trade (aggTrade from WebSocket) is recorded here. This enables:
      - Trade intensity / tick frequency analysis
      - Trade size distribution (whale vs retail)
      - Buyer/seller aggression ratio
      - Microstructure features for ML models

    With enough trades, the Law of Large Numbers reveals patterns
    invisible to the naked eye.
    """

    symbol = models.CharField(max_length=20, db_index=True)
    trade_id = models.BigIntegerField(unique=True)
    price = models.DecimalField(max_digits=20, decimal_places=8)
    qty = models.DecimalField(max_digits=20, decimal_places=8)
    quote_qty = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    is_buyer_maker = models.BooleanField(help_text="True if buyer is the market maker (sell pressure)")
    is_best_match = models.BooleanField(null=True, blank=True)
    time = models.DateTimeField(db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "time"]),
            models.Index(fields=["symbol", "is_buyer_maker"]),
        ]

    def __str__(self):
        side = "SELL" if self.is_buyer_maker else "BUY"
        return f"{self.symbol} {side} {self.qty} @ {self.price}"


class FuturesData(models.Model):
    """
    Futures-specific market sentiment data.

    Tracks funding rates, open interest, and long/short ratios.
    These are powerful sentiment indicators that don't exist in spot markets.

    Signals derived from futures data:
      - High positive funding rate → market over-leveraged long → potential short squeeze
      - Rapid OI increase + price flat → accumulation/distribution
      - L/S ratio extremes → contrarian signals
    """

    symbol = models.CharField(max_length=20, db_index=True)
    timestamp = models.DateTimeField(db_index=True)
    funding_rate = models.FloatField(null=True, blank=True)
    open_interest = models.DecimalField(max_digits=24, decimal_places=8, null=True, blank=True)
    open_interest_quote = models.DecimalField(max_digits=24, decimal_places=8, null=True, blank=True)
    long_short_ratio = models.FloatField(null=True, blank=True, help_text="Long/Short account ratio")
    taker_buy_vol = models.DecimalField(max_digits=24, decimal_places=8, null=True, blank=True)
    taker_sell_vol = models.DecimalField(max_digits=24, decimal_places=8, null=True, blank=True)
    mark_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    index_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    last_funding_time = models.DateTimeField(null=True, blank=True)
    next_funding_time = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "timestamp"]),
        ]
        verbose_name_plural = "Futures data"

    def __str__(self):
        return f"{self.symbol} futures @ {self.timestamp}"


class Pair(models.Model):
    """
    Tradable pairs registry for cointegration analysis.

    Stores discovered cointegrated pairs along with their statistical
    properties. The cointegration engine (Phase 2) populates this table.

    Renaissance connection: Finding hidden statistical relationships
    between assets was the core of Medallion's early success.
    """

    symbol_a = models.CharField(max_length=20)
    symbol_b = models.CharField(max_length=20)
    is_active = models.BooleanField(default=True)
    coint_p_value = models.FloatField(null=True, blank=True)
    coint_statistic = models.FloatField(null=True, blank=True)
    half_life = models.FloatField(null=True, blank=True, help_text="Mean reversion half-life in hours")
    hedge_ratio = models.FloatField(null=True, blank=True)
    current_zscore = models.FloatField(null=True, blank=True)
    last_tested = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    total_signals = models.IntegerField(default=0)
    total_trades = models.IntegerField(default=0)
    win_rate = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["symbol_a", "symbol_b"],
                name="unique_pair",
            )
        ]
        indexes = [
            models.Index(fields=["is_active", "coint_p_value"]),
        ]

    def __str__(self):
        return f"{self.symbol_a}/{self.symbol_b} (p={self.coint_p_value:.4f})"


class TradeSignal(models.Model):
    """
    Generated trading signals from all quant models.

    Every signal from every model (HMM, cointegration, ML, sentiment, etc.)
    is recorded here with a confidence score and expiry time.

    The SignalCombiner (Phase 5) reads from this table to make
    final trading decisions.
    """

    SIGNAL_TYPES = [
        ("long", "Long"),
        ("short", "Short"),
        ("neutral", "Neutral"),
    ]

    SOURCE_MODELS = [
        ("hmm_regime", "HMM Regime"),
        ("cointegration", "Cointegration / Pairs"),
        ("ml_ensemble", "ML Ensemble"),
        ("sentiment", "Sentiment"),
        ("orderbook", "Order Book Imbalance"),
        ("futures", "Futures Data"),
        ("alt_data", "Alternative Data"),
    ]

    STATUS_CHOICES = [
        ("active", "Active"),
        ("expired", "Expired"),
        ("executed", "Executed"),
        ("cancelled", "Cancelled"),
    ]

    symbol = models.CharField(max_length=20, db_index=True, null=True, blank=True)
    pair = models.ForeignKey(Pair, on_delete=models.SET_NULL, null=True, blank=True)
    signal_type = models.CharField(max_length=20, choices=SIGNAL_TYPES)
    direction = models.CharField(max_length=10, choices=[("long", "Long"), ("short", "Short")], null=True, blank=True)
    strength = models.FloatField(help_text="Signal strength 0.0–1.0")
    confidence = models.FloatField(help_text="Model confidence 0.0–1.0")
    source_model = models.CharField(max_length=30, choices=SOURCE_MODELS)
    generated_at = models.DateTimeField(auto_now_add=True, db_index=True)
    expiry = models.DateTimeField(db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")
    metadata = models.JSONField(null=True, blank=True, help_text="Additional signal data (e.g., z-score, regime)")

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "status", "generated_at"]),
            models.Index(fields=["source_model", "status"]),
            models.Index(fields=["expiry", "status"]),
        ]

    def __str__(self):
        return f"{self.get_source_model_display()} {self.direction or self.signal_type} {self.symbol or self.pair} ({self.confidence:.0%})"


class ExecutedTrade(models.Model):
    """
    Actual trades executed by the system.

    Every trade that goes through the Order Manager (Phase 5) is
    recorded here. This is the P&L ledger.

    Used for:
      - Performance tracking
      - Win rate calculation
      - Strategy performance comparison
      - Backtesting validation
      - Tax reporting
    """

    SIDE_CHOICES = [
        ("buy", "Buy"),
        ("sell", "Sell"),
    ]

    STATUS_CHOICES = [
        ("open", "Open"),
        ("closed", "Closed"),
        ("cancelled", "Cancelled"),
        ("paper", "Paper Trade"),
    ]

    signal = models.ForeignKey(TradeSignal, on_delete=models.SET_NULL, null=True, blank=True)
    symbol = models.CharField(max_length=20, db_index=True)
    side = models.CharField(max_length=10, choices=SIDE_CHOICES)
    entry_price = models.DecimalField(max_digits=20, decimal_places=8)
    exit_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    qty = models.DecimalField(max_digits=20, decimal_places=8)
    entry_time = models.DateTimeField(db_index=True)
    exit_time = models.DateTimeField(null=True, blank=True)
    pnl = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    pnl_pct = models.FloatField(null=True, blank=True)
    fee = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    fee_asset = models.CharField(max_length=10, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="open")
    order_id = models.CharField(max_length=100, null=True, blank=True)
    strategy = models.CharField(max_length=50, null=True, blank=True, help_text="Which strategy generated this trade")
    notes = models.TextField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "entry_time"]),
            models.Index(fields=["status", "entry_time"]),
            models.Index(fields=["strategy", "entry_time"]),
        ]

    def __str__(self):
        return f"{self.side.upper()} {self.symbol} {self.qty} @ {self.entry_price}"


class QuantConfig(models.Model):
    """
    Global configuration for the quant trading system.

    Controls the operational mode (backtest / paper / live) and
    safety limits. Only one instance should exist (singleton pattern).
    """

    MODE_CHOICES = [
        ("backtest", "Backtest Only"),
        ("paper", "Paper Trading"),
        ("live", "Live Trading"),
    ]

    mode = models.CharField(max_length=20, choices=MODE_CHOICES, default="backtest")
    is_enabled = models.BooleanField(default=False, help_text="Master on/off switch for automated trading")
    virtual_balance = models.DecimalField(max_digits=20, decimal_places=2, default=10000.00)
    real_balance_limit = models.DecimalField(max_digits=20, decimal_places=2, default=500.00)
    max_open_positions = models.IntegerField(default=5)
    max_position_size_pct = models.FloatField(default=10.0, help_text="Max single position as % of balance")
    max_daily_loss_pct = models.FloatField(default=5.0, help_text="Halt trading if daily loss exceeds this %")
    max_drawdown_pct = models.FloatField(default=15.0, help_text="Halt trading if drawdown exceeds this %")
    kelly_fraction = models.CharField(max_length=20, choices=[
        ("full", "Full Kelly"),
        ("half", "Half Kelly"),
        ("quarter", "Quarter Kelly"),
    ], default="quarter")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Quant Configuration"

    def save(self, *args, **kwargs):
        """Enforce singleton pattern — only one config row exists."""
        self.pk = 1  # Always use the same primary key
        super().save(*args, **kwargs)

    @classmethod
    def get_config(cls):
        """Get or create the singleton config."""
        config, _ = cls.objects.get_or_create(pk=1)
        return config

    def __str__(self):
        return f"Quant Config: {self.get_mode_display()} ({'ON' if self.is_enabled else 'OFF'})"
