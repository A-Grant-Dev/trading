# Renaissance Trading Platform — Implementation Roadmap

> **Inspired by Jim Simons & Renaissance Technologies' Medallion Fund**
> *66% avg annual return (gross) over 30 years | ~51% win rate | 150K–300K trades/day*
> *"Build a statistics machine, not a prediction machine."*

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Phase 0 — Foundation & Data Infrastructure](#phase-0--foundation--data-infrastructure)
3. [Phase 1 — Market Regime Detection (HMM)](#phase-1--market-regime-detection-hmm)
4. [Phase 2 — Statistical Arbitrage & Pairs Trading](#phase-2--statistical-arbitrage--pairs-trading)
5. [Phase 3 — Alternative Data & Sentiment Signals](#phase-3--alternative-data--sentiment-signals)
6. [Phase 4 — Machine Learning Prediction Engine](#phase-4--machine-learning-prediction-engine)
7. [Phase 5 — Execution Layer & Algo Trading](#phase-5--execution-layer--algo-trading)
8. [Phase 6 — Portfolio Management & Risk (Kelly Criterion)](#phase-6--portfolio-management--risk-kelly-criterion)
9. [Phase 7 — Live Dashboard & Monitoring](#phase-7--live-dashboard--monitoring)
10. [Phase 8 — Backtesting Framework, Paper Trading & Go Live](#phase-8--backtesting-framework-paper-trading--go-live)
11. [Dependencies & Installation Order](#dependencies--installation-order)
12. [Appendix: Key Mathematical Formulas](#appendix-key-mathematical-formulas)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RENAISSANCE TRADING PLATFORM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐    │
│  │   DATA LAYER    │    │  SIGNAL LAYER     │    │   EXECUTION LAYER   │    │
│  │                 │    │                   │    │                      │    │
│  │ • Binance REST  │───▶│ • HMM Regime     │───▶│ • Signal Combiner   │    │
│  │ • WebSocket WS  │    │ • Cointegration  │    │ • Order Generator   │    │
│  │ • Sentiment RSS │    │ • ML Predictions │    │ • TWAP/VWAP/ICE     │    │
│  │ • Alt. Data     │    │ • Sentiment      │    │ • Binance Algo API  │    │
│  │ • Historical    │    │ • Alt Data       │    │ • Kelly Sizing      │    │
│  └────────┬────────┘    └────────┬─────────┘    └──────────┬───────────┘    │
│           │                      │                         │               │
│           ▼                      ▼                         ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     DASHBOARD & MONITORING                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ Live     │  │ Pairs    │  │ Regime   │  │ Portfolio &      │   │   │
│  │  │ Signals  │  │ Tracker  │  │ Visual   │  │ Risk Dashboard   │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     BACKTESTING ENGINE                                │   │
│  │  • Historical data replay  • In/Out-of-sample split                  │   │
│  │  • Monte Carlo simulation  • Sharpe/Sortino/Calmar ratios            │   │
│  │  • Walk-forward validation • Parameter sensitivity analysis          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Foundation & Data Infrastructure

**Goal:** Create a robust, scalable data layer that feeds all downstream quant models. This is the absolute foundation — get this wrong and nothing else works.

### 0.1 Create New Django App: `quant`

```bash
python manage.py startapp quant
```

Add `'quant'` to `INSTALLED_APPS` in `trading_project/settings.py`.

### 0.2 Data Models (`quant/models.py`)

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `MarketData` | OHLCV kline storage | `symbol`, `interval`, `open_time`, `open`, `high`, `low`, `close`, `volume`, `taker_buy_vol`, `taker_buy_quote_vol`, `trades` |
| `HistoricalDataBatch` | Tracks bulk CSV downloads from data.binance.vision | `symbol`, `interval`, `date_range_start`, `date_range_end`, `file_path`, `rows_imported`, `status` |
| `OrderBookSnapshot` | Periodic depth snapshots for training | `symbol`, `timestamp`, `bids_json`, `asks_json`, `bid_vol`, `ask_vol`, `spread`, `imbalance_pct` |
| `TradeRecord` | Individual trade storage for ML | `symbol`, `trade_id`, `price`, `qty`, `quote_qty`, `is_buyer_maker`, `time` |
| `FuturesData` | Futures-specific market sentiment (funding rate, OI, L/S) | `symbol`, `timestamp`, `funding_rate`, `open_interest`, `long_short_ratio`, `taker_buy_sell_vol` |
| `Pair` | Tradable pairs registry for cointegration | `symbol_a`, `symbol_b`, `is_active`, `coint_p_value`, `half_life`, `last_tested` |
| `TradeSignal` | Generated trading signals from all models | `symbol`/`pair_id`, `signal_type`, `direction`, `strength`, `confidence`, `source_model`, `generated_at`, `expiry` |
| `ExecutedTrade` | Actual trades executed by the system | `signal_id`, `symbol`, `side`, `entry_price`, `exit_price`, `qty`, `pnl`, `entry_time`, `exit_time`, `status` |

### 0.3 Historical Data Ingestion Service (`quant/services/data_ingestion.py`)

**What it does:**
- Downloads bulk historical CSV data from [data.binance.vision](https://data.binance.vision/) (the official Binance historical data repository)
- Imports into `MarketData` model for backtesting/ML training
- Supports spot + futures klines

**Key functions:**
```python
def download_historical_klines(symbol: str, interval: str, start_date: str, end_date: str) -> int:
    """Download CSV from data.binance.vision, parse, import to MarketData."""

def import_csv_to_marketdata(filepath: str, symbol: str, interval: str) -> int:
    """Parse a Binance CSV file and bulk-insert into MarketData table."""

def get_market_data(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Query MarketData table and return as pandas DataFrame for analysis."""
```

### 0.4 Live Data Feed Services (`quant/services/data_feeds.py`)

**Purpose:** Real-time data ingestion from Binance WebSocket streams — the live counterpart to the historical importer.

**WebSocket Streams to connect (reuse patterns from `orderbook/services.py` and `charts` template):**

| Stream | Purpose |
|--------|---------|
| `<symbol>@kline_<interval>` | Live candle updates → store in MarketData |
| `<symbol>@depth20@100ms` | Order book depth → OrderBookSnapshot + compute imbalance |
| `<symbol>@aggTrade` | Every trade as it happens → TradeRecord |
| `!miniTicker@arr` | All tickers at once → price cache updates |
| (Futures) `<symbol>@markPrice` | Funding rate & mark price → FuturesData |

**Architecture for data_feeds.py:**
```python
class DataFeedManager:
    """Manages multiple WebSocket connections for live data."""

    def __init__(self):
        self.connections: dict[str, WebSocketThread] = {}
        self.callbacks: dict[str, list[Callable]] = {}

    def subscribe_kline(self, symbol: str, interval: str, callback=None):
        """Subscribe to live kline data for a symbol."""

    def subscribe_depth(self, symbol: str, callback=None):
        """Subscribe to order book depth for a symbol."""

    def subscribe_trades(self, symbol: str, callback=None):
        """Subscribe to aggregate trade data."""

    def start_all(self, symbols: list[str]):
        """Start all WebSocket connections for tracked symbols."""

    def stop_all(self):
        """Gracefully close all connections."""
```

**Threading approach:** Each WebSocket runs in its own daemon thread (same pattern as `ai_chat/services.py`'s `_discovery_thread`). Use a shared thread-safe queue or in-memory cache for passing data between threads and the Django process.

### 0.5 Data Export & Transformation Utilities (`quant/services/data_utils.py`)

```python
def ohlcv_to_dataframe(symbol, interval, limit=1000) -> pd.DataFrame:
    """Fetch from MarketData or Binance REST, return pandas DataFrame."""

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, ATR, Stochastic, OBV, VWAP using pandas-ta."""

def resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """Resample 1m data to higher intervals (5m, 15m, 1h, etc.)."""

def compute_order_book_features(symbol: str) -> dict:
    """Compute imbalance ratio, depth pressure, bid/ask concentration."""
```

---

## Phase 1 — Market Regime Detection (HMM)

**Goal:** Build a Hidden Markov Model that detects the current "market regime" (quiet, trending up, trending down, highly volatile). This regime label feeds into all downstream models — different strategies activate in different regimes.

**Why this matters (Renaissance principle):** Simons' models didn't just predict price direction — they first identified which *environment* the market was in, then applied the appropriate math. An HMM is the ideal tool for this.

### 1.1 HMM Regime Classifier (`quant/services/hmm_regime.py`)

```python
from hmmlearn import hmm

class MarketRegimeDetector:
    """
    Hidden Markov Model that identifies latent market regimes.

    States to detect (customizable):
    0: Low volatility / ranging
    1: Bullish trend / low vol
    2: Bearish trend / low vol
    3: High volatility / chaotic
    """

    N_STATES = 4  # Default, tune as needed

    def __init__(self, n_states: int = 4):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=1000,
            random_state=42,
        )
        self.regime_labels: dict[int, str] = {}

    def train(self, df: pd.DataFrame, feature_columns: list[str]):
        """Train HMM on historical data using selected features.

        Recommended features for training:
        - Returns (log returns)
        - Volume change %
        - ATR (volatility)
        - RSI (momentum)
        - Spread / bid-ask width
        """
        X = df[feature_columns].values
        self.model.fit(X)
        self._label_states(df)

    def predict_regime(self, feature_vector: np.ndarray) -> tuple[int, str]:
        """Predict current regime. Returns (state_number, label)."""

    def get_regime_confidence(self, feature_vector: np.ndarray) -> dict:
        """Return probability distribution across all regimes."""

    def _label_states(self, df: pd.DataFrame):
        """Auto-label states based on mean return and volatility."""
```

**Feature engineering for HMM training:**
```python
def build_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform OHLCV data into HMM-ready features."""
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(20).std()
    df['volume_change'] = df['volume'].pct_change()
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr_pct'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']
    return df.dropna()
```

### 1.2 Regime-Aware Signal Weighting

**Architecture for integrating regime into other signals:**
```python
def adjust_signal_for_regime(base_signal: float, regime: int) -> float:
    """
    Adjust the confidence/strength of a trading signal based on market regime.

    Example rules:
    - Regime 0 (ranging, low vol): Reduce signal confidence, widen thresholds
    - Regime 1 (bullish trend): Increase long signal confidence, reduce short
    - Regime 2 (bearish trend): Increase short signal confidence, reduce long
    - Regime 3 (high volatility): Reduce position size, widen stops, reduce all signals
    """
    weights = {
        0: 0.5,   # Ranging — low conviction
        1: 1.2,   # Bullish — amplify longs
        2: 1.2,   # Bearish — amplify shorts
        3: 0.3,   # Volatile — reduce everything
    }
    return base_signal * weights.get(regime, 1.0)
```

### 1.3 Django Periodic Task for Regime Updates

```python
# In quant/services/periodic_tasks.py or use Django management command

def update_regime_detection():
    """Periodic task (every 15 minutes): fetch latest data, recompute regime."""
    for symbol in TRACKED_SYMBOLS:
        df = get_market_data_as_df(symbol, interval='5m', limit=500)
        features = build_hmm_features(df)
        current_features = features.iloc[-1:][FEATURE_COLUMNS]
        regime = detector.predict_regime(current_features.values)
        cache.set(f'regime:{symbol}', regime, timeout=900)
```

**How to schedule:** Use `django-cron` or `python-crontab` (or a management command called by systemd timer) — no need for Celery at this stage. Keep it simple.

---

## Phase 2 — Statistical Arbitrage & Pairs Trading

**Goal:** Implement the core Renaissance strategy — find cointegrated asset pairs, trade the spread when it deviates. This is the highest-signal, lowest-risk strategy in the quant toolkit.

**Renaissance connection:** This is the strategy that launched Medallion. Ax and Baum built models to find hidden correlations between assets, then traded when those correlations temporarily broke.

### 2.1 Cointegration Service (`quant/services/cointegration.py`)

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

class PairsFinder:
    """
    Scans a universe of symbols to discover cointegrated trading pairs.
    """

    def __init__(self, symbols: list[str]):
        self.universe = symbols
        self.pairs: list[dict] = []

    def find_cointegrated_pairs(self, price_data: dict[str, pd.Series],
                                 p_threshold: float = 0.05) -> list[dict]:
        """
        Run Engle-Granger cointegration test on all symbol pairs.

        Returns pairs where p-value < threshold, sorted by half-life.
        Each result: {
            'symbol_a': str, 'symbol_b': str,
            'p_value': float, 'coint_stat': float,
            'half_life': float, 'hedge_ratio': float,
            'spread': pd.Series
        }
        """

    def compute_half_life(self, spread: pd.Series) -> float:
        """Compute mean reversion half-life using OLS on lagged spread."""

    def compute_zscore(self, spread: pd.Series) -> float:
        """Current z-score of the spread — how far from mean."""
```

**Key formula — the spread:**
```
spread = price_a - hedge_ratio * price_b
z_score = (current_spread - mean_spread) / std_spread

Entry threshold: |z_score| > 2.0  (2 standard deviations from mean)
Exit threshold:  |z_score| < 0.5  (reverted back near mean)
Stop threshold:  |z_score| > 3.0  (something broke — get out)
```

### 2.2 Pairs Trading Signal Generator (`quant/services/pairs_signals.py`)

```python
class PairsSignalGenerator:
    """
    Generates long/short signals for cointegrated pairs.

    When spread widens (z > threshold):
        → Short the outperformer, long the underperformer
    When spread narrows (z < exit_threshold):
        → Close both positions
    """

    ENTRY_Z = 2.0
    EXIT_Z = 0.5
    STOP_Z = 3.0

    def evaluate_pair(self, pair: dict) -> TradeSignal | None:
        """Check current z-score and generate signal if threshold breached."""

    def backtest_pair(self, pair: dict, historical_data: pd.DataFrame) -> dict:
        """Run a full backtest on historical data for this pair.

        Returns:
            total_trades, win_rate, sharpe_ratio, max_drawdown,
            cumulative_pnl, trades_list
        """
```

### 2.3 Automated Pair Discovery (`quant/management_commands/discover_pairs.py`)

```python
class Command(BaseCommand):
    """Management command: python manage.py discover_pairs"""

    def handle(self, *args, **options):
        symbols = get_tradable_symbols()  # From Binance exchangeInfo
        # Filter to only USDT pairs
        usdt_pairs = [s for s in symbols if s.endswith('USDT')]
        # Get price data for all
        price_data = fetch_daily_close_prices(usdt_pairs, days=90)
        # Scan for cointegrated pairs
        finder = PairsFinder(usdt_pairs)
        pairs = finder.find_cointegrated_pairs(price_data, p_threshold=0.05)
        # Store in Pair model
        for p in pairs:
            Pair.objects.update_or_create(...)
```

### 2.4 Strategy Performance Table (Django Admin + Dashboard)

| Pair | P-Value | Half-Life | Hedge Ratio | Z-Score | # Trades | Win Rate | Sharpe |
|------|---------|-----------|-------------|---------|----------|----------|--------|
| BTCUSDT/ETHUSDT | 0.003 | 18h | 14.2 | -1.2 | 47 | 53% | 1.8 |
| SOLUSDT/AVAXUSDT | 0.01 | 12h | 3.8 | **2.4** | 23 | 48% | 1.2 |
| LINKUSDT/UNIUSDT | 0.04 | 24h | 1.1 | 0.3 | 12 | 58% | 2.1 |

---

## Phase 3 — Alternative Data & Sentiment Signals

**Goal:** Augment price-based signals with non-traditional data sources — exactly what Renaissance did with weather patterns, linguistics, and other alternative datasets.

### 3.1 Enhanced Sentiment Pipeline (`quant/services/alt_sentiment.py`)

**Building on the existing `sentiment/services.py`:**

```python
class AlternativeSentimentEngine:
    """
    Multi-source sentiment aggregation — extends current RSS-based system.

    Sources (some already exist in sentiment app):
    ✓ CoinDesk RSS (exists)
    ✓ CoinTelegraph RSS (exists)
    ✓ Decrypt RSS (exists)
    ✓ Reddit (r/cryptocurrency) (exists)
    ✓ Hacker News Algolia (exists)
    ✓ Fear & Greed Index (exists)
    ✗ Twitter/X API (need auth) — optional
    ✗ On-chain metrics (Glassnode) — optional
    ✗ Google Trends — NEW
    ✗ GitHub commit activity — NEW
    """

    def get_google_trends_signal(keyword: str) -> float:
        """Fetch Google Trends interest score for a coin name.
        Extreme interest spikes → contrarian signal (top signal)."""

    def get_github_activity_signal(project_repo: str) -> dict:
        """
        Measure developer activity as a signal.
        High dev activity = strong fundamentals (bullish).
        Drops in commits = project abandonment (bearish).
        """

    def compute_consensus_score(self, symbol: str) -> dict:
        """
        Combine all sentiment sources into a single 0-100 score.
        Weighted average with recency bias.
        Returns {score, label, sources_contributing}
        """
```

### 3.2 On-Chain Data Service (`quant/services/onchain.py`)

If the user has/wants on-chain access (optional — can use free sources):

```python
class OnChainAnalyzer:
    """
    On-chain metrics for BTC/ETH using public APIs.

    Sources:
    - Blockchain.com API (free, limited)
    - Glassnode Studio (paid, premium)
    - CoinMetrics (free tier)
    - Messari (free tier)
    """

    def get_exchange_flows(asset: str) -> dict:
        """Net exchange inflows/outflows.
        Large exchange inflow → potential sell pressure (bearish)
        Large exchange outflow → accumulation (bullish)"""

    def get_whale_transactions(asset: str, min_usd: float = 1_000_000) -> list:
        """Large transactions — whale accumulation/distribution signals."""

    def get_active_addresses(asset: str) -> pd.Series:
        """Network activity metric — rising = healthy, falling = dying."""
```

### 3.3 Signal Integration — Sentiment → Quant Signal

```python
def sentiment_to_signal(sentiment_data: dict, regime: int) -> float:
    """
    Convert multi-source sentiment into a -1.0 to 1.0 signal.

    -1.0 = extreme bearish → potential contrarian BUY signal
     1.0 = extreme bullish → potential contrarian SELL signal
     0.0 = neutral → no signal

    Renaissance principle: when sentiment is extreme, it's often a
    contrarian indicator — the market has already priced it in.
    """
    fear_greed = sentiment_data.get('fear_greed', {}).get('value', 50)
    news_sentiment = sentiment_data.get('overall_sentiment', {}).get('score', 50)
    reddit_sentiment = sentiment_data.get('reddit_sentiment', {}).get('score', 50)

    # Average all sources
    avg = (fear_greed + news_sentiment + reddit_sentiment) / 3

    # Regime adjustment
    if regime == 3:  # High volatility — sentiment is unreliable
        return 0.0

    # Convert 0-100 to -1.0 to 1.0 with dead zone in middle
    if 35 < avg < 65:
        return 0.0  # Neutral — no signal
    elif avg <= 35:
        return (35 - avg) / 35  # 0.0 to 1.0 bearish
    else:
        return (65 - avg) / 35  # -1.0 to 0.0 bullish (contrarian)
```

---

## Phase 4 — Machine Learning Prediction Engine

**Goal:** Build ML models that predict short-term price movements (1-5 candles ahead) using the features engineered in Phase 0 and the regime labels from Phase 1.

**Renaissance connection:** Their models don't predict exact prices — they predict statistical probabilities of direction. "The market will likely go up with 53% probability" — that's enough for the law of large numbers to work.

### 4.1 Feature Engineering Pipeline (`quant/services/ml_features.py`)

```python
class FeaturePipeline:
    """
    Builds feature matrices for ML models from raw market data.
    """

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline.

        Feature groups:
        1. Price-based: Returns (1, 5, 15, 60 min), log returns
        2. Technical: RSI, MACD, BB, ATR, Stochastic, OBV, VWAP
        3. Volume: Volume delta, taker buy/sell ratio, volume profile
        4. Order book: Imbalance ratio, depth pressure, spread width
        5. Microstructure: Trade intensity, tick frequency, avg trade size
        6. Cross-asset: BTC correlation, sector correlation
        7. Regime: HMM regime label (one-hot encoded)
        8. Sentiment: Aggregated sentiment score
        9. Time-based: Hour of day, day of week, month (seasonality)
        """

    def get_feature_importance(self, model, feature_names: list) -> pd.DataFrame:
        """Return sorted feature importance for interpretability."""
```

### 4.2 ML Models (`quant/services/ml_models.py`)

```python
# Multiple models for ensemble approach (Renaissance used many models in parallel)

class DirectionPredictor:
    """
    Base class for direction prediction models.
    Predicts P(up) or P(down) over next N candles.
    """

    def predict_proba(self, features: np.ndarray) -> float:
        """Return probability of upward movement (0.0 - 1.0)."""
        ...

class XGBoostDirectionModel(DirectionPredictor):
    """XGBoost classifier — best for tabular feature data."""

class LSTMModel(DirectionPredictor):
    """LSTM neural network — captures sequential dependencies."""

class RandomForestModel(DirectionPredictor):
    """Random Forest — interpretable baseline."""

class EnsemblePredictor:
    """
    Combines multiple models using weighted voting.

    Weights are determined by each model's recent performance
    (walk-forward validation accuracy).

    Renaissance principle: many weak models > one strong model.
    Use the ensemble's average probability as the final signal.
    """
    def __init__(self):
        self.models: list[DirectionPredictor] = [
            XGBoostDirectionModel(),
            LSTMModel(),
            RandomForestModel(),
        ]
        self.weights: list[float] = [1/3, 1/3, 1/3]  # Initial equal weight

    def update_weights(self, recent_performance: list[float]):
        """Recalculate weights based on last N predictions' accuracy."""

    def predict(self, features: np.ndarray) -> float:
        """Weighted average of all model probabilities."""
```

### 4.3 Model Training Pipeline (`quant/services/ml_training.py`)

```python
class ModelTrainer:
    """
    End-to-end training pipeline with Renaissance-grade rigor.

    Key steps:
    1. Split data: 70% train, 15% validation, 15% test (chronological!)
    2. Train multiple models with hyperparameter tuning
    3. Walk-forward validation (train on expanding window)
    4. Purge 99%+ of signals that don't pass statistical significance
    5. Save best models to disk for inference
    """

    def train_and_evaluate(self, symbol: str, interval: str) -> dict:
        """
        Full training cycle for a single symbol.

        Returns:
            accuracy, precision, recall, f1_score, sharpe_of_signals,
            feature_importance, confusion_matrix
        """

class SignalPurger:
    """
    Renaissance famously discarded 99%+ of discovered signals.
    This implements that filtering.

    A signal passes if:
    - Backtest Sharpe ratio > 1.5 (in-sample)
    - Out-of-sample performance > 70% of in-sample
    - P-value of strategy returns < 0.05 (statistically significant)
    - Survives Monte Carlo simulation (95% confidence)
    """
    def should_keep_signal(self, signal_metrics: dict) -> bool:
        """Apply Renaissance-style signal filtering."""
```

---

## Phase 5 — Execution Layer & Algo Trading

**Goal:** The bridge between signal generation and actual trade execution. This is where signals become orders, orders become trades, and trades become P&L.

**Renaissance connection:** They execute 150K-300K trades per day. Speed and automation are everything. The computer decides, the computer executes — no human in the loop.

### 5.1 Signal Combiner (`quant/services/signal_combiner.py`)

```python
class SignalCombiner:
    """
    Aggregates signals from all sources into a single actionable decision.

    Sources:
    - Phase 2: Pairs trading signals
    - Phase 3: Sentiment-based signals
    - Phase 4: ML model predictions
    - Phase 1: Regime override

    Logic:
    1. Collect all active signals for each symbol/pair
    2. Weight by historical accuracy of source
    3. Apply regime adjustment
    4. If combined confidence > threshold → generate trade order
    """

    CONFIDENCE_THRESHOLD = 0.55  # > 55% = trade (law of large numbers)

    WEIGHTS = {
        'pairs_cointegration': 0.40,   # Highest conviction
        'ml_ensemble': 0.30,           # Medium conviction
        'sentiment': 0.15,            # Lower conviction
        'orderbook_imbalance': 0.15,  # Short-term micro-signal
    }

    def combine(self, signals: list[TradeSignal], regime: int) -> dict | None:
        """
        Combine all signals. Returns order info if threshold met.

        Returns: {
            'symbol': str, 'side': 'BUY'|'SELL',
            'confidence': float, 'quantity': float,
            'order_type': 'MARKET'|'LIMIT',
            'reason': str  # Which signals triggered
        } or None
        """
```

### 5.2 Order Manager (`quant/services/order_manager.py`)

```python
class OrderManager:
    """
    Manages the lifecycle of orders from creation to execution.

    Responsible for:
    - Creating orders via Binance REST API (or testnet)
    - Monitoring fill status
    - Handling cancellations
    - Recording trades in ExecutedTrade model
    - Error handling (partial fills, rejections, network issues)
    """

    def execute_signal(self, signal: dict) -> ExecutedTrade:
        """Execute a combined trading signal.

        For large orders (based on Kelly position size):
        - Use Binance Algo TWAP to minimize slippage
        - For small orders: direct MARKET or LIMIT order
        """

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""

    def get_open_orders(self, symbol: str = None) -> list:
        """Get all currently open orders."""

    def get_order_status(self, order_id: str, symbol: str) -> dict:
        """Check fill status of an order."""
```

### 5.3 Binance Algo Trading Integration (`quant/services/algo_execution.py`)

**Leverage the already-researched Binance Algo Trading endpoints:**

```python
class AlgoExecutionService:
    """
    Uses Binance's built-in algo trading products for sophisticated execution.

    Docs reference: binance-api-research.md (Section 8)

    Three algorithm types:
    1. TWAP — Time-weighted average price (accumulate/distribute evenly)
    2. VWAP — Volume-weighted average price (follow market volume)
    3. Iceberg — Hide true order size from the book

    API endpoints (from research doc):
    POST /sapi/v1/algo/spot/newOrderTwap
    POST /sapi/v1/algo/spot/newOrderVwap
    POST /sapi/v1/algo/spot/newOrderIceberg
    """

    def execute_twap(self, symbol: str, side: str, quantity: float,
                      duration_minutes: int, limit_price: float = None) -> dict:
        """Place a TWAP order via Binance Algo API."""

    def execute_vwap(self, symbol: str, side: str, notional: float) -> dict:
        """Place a VWAP order — follows natural volume."""

    def execute_iceberg(self, symbol: str, side: str, total_quantity: float,
                         display_quantity: float) -> dict:
        """Place an Iceberg order — hides true size."""

    def cancel_algo_order(self, algo_id: str) -> bool:
        """Cancel an active algo order."""
```

### 5.4 Execution Strategy Selector

```python
def should_use_algo(order: dict, market_data: dict) -> bool:
    """
    Decide whether to use regular or algo execution.

    Use regular market/limit order if:
    - Order size is small (< $1,000 notional)
    - High liquidity pair (BTC, ETH)

    Use Algo TWAP if:
    - Order size > $10,000
    - Low/mid liquidity pair (ALGO, HBAR, etc.)
    - Current spread > 0.1% (too wide for market order)

    Use Algo VWAP if:
    - Need best average price over a longer period
    - Accumulating/distributing over hours

    Use Iceberg if:
    - Order size > 0.5% of 24h volume
    - Don't want to reveal position to market
    """
```

---

## Phase 6 — Portfolio Management & Risk (Kelly Criterion)

**Goal:** Don't blow up. Position sizing is more important than signal accuracy. This is the risk layer that sits above everything else.

**Renaissance connection:** "It's not about how often you win — it's about how much you win when you're right vs. how much you lose when you're wrong." The Kelly Criterion optimizes for maximum long-term geometric growth.

### 6.1 Kelly Position Sizer (`quant/services/kelly_sizing.py`)

```python
class KellyPositionSizer:
    """
    Implements the Kelly Criterion for optimal position sizing.

    Full Kelly formula:
    f* = (p * win_avg_return - q * loss_avg_return) / (win_avg_return * loss_avg_return)

    Simplified (for even-money bets):
    f* = p - q
    where:
    p = probability of winning (from ML model or historical win rate)
    q = 1 - p = probability of losing

    Since full Kelly is aggressive (high drawdown risk), we use:
    - Quarter Kelly (f* / 4) — conservative, low drawdown
    - Half Kelly (f* / 2) — moderate
    - Full Kelly — aggressive (not recommended for retail)
    """

    def __init__(self, fraction: str = 'quarter'):
        self.kelly_fraction = {'full': 1.0, 'half': 0.5, 'quarter': 0.25}[fraction]

    def calculate_position_size(self, capital: float, win_probability: float,
                                 avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size in quote currency.

        Args:
            capital: Available capital for this trade
            win_probability: P(profit) from model/signal history
            avg_win: Average fractional win (e.g., 0.02 = 2%)
            avg_loss: Average fractional loss (e.g., 0.01 = 1%)

        Returns:
            Position size in quote currency
        """
```

### 6.2 Risk Manager (`quant/services/risk_manager.py`)

```python
class RiskManager:
    """
    Portfolio-level risk controls. The circuit breaker.

    Rules enforced on every trade attempt:
    1. Max position size: No single position > 10% of portfolio
    2. Max daily loss: Halt all trading if daily P&L < -5%
    3. Max drawdown: Halt all trading if drawdown > 15%
    4. Max correlation: Don't over-concentrate in correlated assets
    5. Min time between trades: Avoid overtrading
    6. Max open positions: Global cap on concurrent positions (e.g., 5)
    7. Leverage cap: Max leverage based on regime (lower in regime 3)
    8. Weekend/session check: Only trade during active hours
    """

    def can_trade(self, proposed_order: dict, portfolio: dict) -> tuple[bool, str]:
        """
        Check all risk rules. Returns (allowed: bool, reason: str).

        If any rule is violated, the trade is BLOCKED.
        The dashboard logs the rejection reason.
        """

    def get_daily_pnl(self) -> float:
        """Calculate today's P&L from ExecutedTrade records."""

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak portfolio value."""

    def check_correlation_risk(self, symbol: str, open_positions: list) -> bool:
        """Check if new position would over-concentrate in correlated assets."""
```

### 6.3 Stop Loss & Take Profit (`quant/services/stop_loss.py`)

```python
class StopLossManager:
    """
    Dynamic stop-loss and take-profit levels.

    Stop loss strategies (configurable per strategy):
    1. Fixed percentage: -2% stop, +4% take profit
    2. ATR-based: 2x ATR stop, 4x ATR take profit
    3. Volatility-adjusted: Wider stops in high vol, tighter in low
    4. Trailing stop: Lock in profits as trade moves favorably
    5. Time-based: Exit if trade hasn't hit target within N hours
    """

    def calculate_stops(self, entry_price: float, side: str,
                         atr: float, regime: int) -> dict:
        """Calculate stop loss and take profit levels.

        Returns: {
            'stop_loss': float,
            'take_profit': float,
            'trailing_activation': float,
            'time_exit_hours': int
        }
        """

    def should_exit(self, position: ExecutedTrade, market_data: pd.DataFrame) -> tuple[bool, str]:
        """Check if any exit condition is triggered."""
```

---

## Phase 7 — Live Dashboard & Monitoring

**Goal:** A real-time dashboard inside the existing Django project that shows every layer of the quant system. This is where you see the machine at work.

### 7.1 Quant Dashboard View (`quant/views.py`)

**Pages to create:**

| URL | View | Purpose |
|-----|------|---------|
| `/quant/` | `dashboard` | Main quant dashboard overview |
| `/quant/pairs/` | `pairs_list` | Cointegrated pairs monitor |
| `/quant/signals/` | `active_signals` | All current trading signals |
| `/quant/regime/` | `regime_view` | HMM regime visualization |
| `/quant/trades/` | `trade_history` | Executed trade log |
| `/quant/performance/` | `performance` | P&L, Sharpe, drawdown charts |
| `/quant/backtest/` | `backtest_ui` | Run backtest from UI |

### 7.2 Dashboard Sections (Template: `quant/templates/quant/dashboard.html`)

**Section 1 — Market Regime Banner**
```
┌──────────────────────────────────────────────────────────────────┐
│  MARKET REGIME: 🟢 BULLISH TREND (Confidence: 87%)              │
│  Low volatility, upward momentum.                                 │
│  Strategy: Long bias, normal position sizing.                     │
└──────────────────────────────────────────────────────────────────┘
```

**Section 2 — Active Signals Feed**
```
┌─────────────┬──────────┬──────────┬────────┬────────┬────────────┐
│ Symbol/Pair │ Direction│Strength  │Source  │Expiry  │Confidence  │
├─────────────┼──────────┼──────────┼────────┼────────┼────────────┤
│ BTCUSDT     │  LONG    │ 0.62     │ ML     │14:35   │ 62%        │
│ SOL/AVAX    │  SHORT   │ 0.71     │ Pairs  │15:00   │ 71%        │
│ ETHUSDT     │  LONG    │ 0.55     │ Sent.  │14:45   │ 55%        │
└─────────────┴──────────┴──────────┴────────┴────────┴────────────┘
```

**Section 3 — Portfolio & Risk Status**
```
┌──────────────────────────────────────────────────────────────────┐
│  BALANCE: $12,450  │  OPEN POSITIONS: 2/5  │  DAILY P&L: +1.2% │
│  DRAWDOWN: 3.2%    │  SHARPE (30d): 1.8    │  WIN RATE: 54%    │
│  KELLY ACTIVE: 1/4 │  STOP MODE: ATR-2x    │  STATUS: TRADING  │
└──────────────────────────────────────────────────────────────────┘
```

**Section 4 — Signal Source Performance**
```
┌──────────────┬───────┬────────┬────────┬─────────┬───────┐
│ Source       │#Signals│Won     │Lost    │Accuracy │Sharpe│
├──────────────┼───────┼────────┼────────┼─────────┼───────┤
│ Pairs (Coin) │  47   │ 25     │ 22     │ 53.2%   │ 1.4   │
│ ML Ensemble  │  89   │ 48     │ 41     │ 53.9%   │ 1.6   │
│ Sentiment    │  23   │ 11     │ 12     │ 47.8%   │ 0.8   │
│ Order Book   │  156  │ 82     │ 74     │ 52.6%   │ 1.1   │
└──────────────┴───────┴────────┴────────┴─────────┴───────┘
```

**Section 5 — Recent Trades Feed**
```
┌──────┬────────┬───────┬────────┬──────────┬──────────┬──────┬──────────┐
│ Time │ Symbol │Side   │Entry   │Exit      │P&L       │R:R   │Source    │
├──────┼────────┼───────┼────────┼──────────┼──────────┼──────┼──────────┤
│09:32 │BTCUSDT │LONG   │$64,200 │$64,580   │+$380     │1.9   │ML + OB   │
│09:15 │SOLAVAX │SHORT  │$32.50  │$32.10    │+$40      │2.1   │Pairs     │
│08:45 │ETHUSDT │LONG   │$3,480  │$3,465    │-$15      │0.5   │Sentiment │
└──────┴────────┴───────┴────────┴──────────┴──────────┴──────┴──────────┘
```

### 7.3 API Endpoints for Dashboard AJAX

| Endpoint | Returns | Refresh |
|----------|---------|---------|
| `/quant/api/regime/` | Current regime & confidence | Every 15 min |
| `/quant/api/signals/` | All active/fresh signals | Every 30 sec |
| `/quant/api/portfolio/` | Balance, P&L, drawdown | Every 60 sec |
| `/quant/api/trades/recent/` | Last 20 trades | Every 30 sec |
| `/quant/api/pairs/` | All tracked pairs + z-scores | Every 5 min |
| `/quant/api/performance/` | Rolling Sharpe, win rate | Every 5 min |

### 7.4 Notification System

```python
class QuantNotifier:
    """
    Sends notifications for important events.

    Events that trigger notification:
    - New trade executed (with details)
    - Stop loss hit
    - Take profit hit
    - Regime change detected
    - New cointegrated pair discovered
    - Drawdown exceeds threshold
    - Risk manager blocks a trade (with reason)
    - Daily P&L summary
    """
    def notify_trade(self, trade: ExecutedTrade):
        """Send trade alert — NOT for every trade, only significant ones."""

    def notify_risk_event(self, event_type: str, details: str):
        """Alert on risk rule violations or drawdown breaches."""
```

---

## Phase 8 — Backtesting Framework, Paper Trading & Go Live

**Goal:** Before any real money touches the system, every strategy must be rigorously backtested. Then paper trade. Then — and only then — go live with small capital.

**Renaissance connection:** They discarded 99%+ of discovered signals. They tested relentlessly. They used out-of-sample data that the model had never seen. This is non-negotiable.

### 8.1 Backtesting Engine (`quant/services/backtesting.py`)

```python
class BacktestEngine:
    """
    Event-driven backtesting engine.

    Features:
    - Uses MarketData table for historical data
    - Walks through time candle by candle (not vectorized — more realistic)
    - Supports multiple strategy types simultaneously
    - Tracks every trade: entry, exit, P&L, fees, slippage
    - Reports comprehensive metrics
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.capital = initial_capital
        self.positions: list[dict] = []
        self.trades: list[ExecutedTrade] = []
        self.equity_curve: list[float] = []

    def add_strategy(self, strategy_fn: Callable):
        """Add a signal-generating function to the backtest."""

    def run(self, df: pd.DataFrame) -> 'BacktestResult':
        """Run the backtest over historical data.

        Steps:
        1. Split data into in-sample (70%) and out-of-sample (30%)
        2. Warm up indicators on in-sample
        3. Test on out-of-sample — NEVER touch in-sample during test
        4. Record all metrics
        """

class BacktestResult:
    """Comprehensive backtest report."""

    @property
    def metrics(self) -> dict:
        return {
            'total_return': ...,
            'annualized_return': ...,
            'sharpe_ratio': ...,
            'sortino_ratio': ...,
            'calmar_ratio': ...,
            'max_drawdown': ...,
            'win_rate': ...,
            'profit_factor': ...,
            'avg_win': ...,
            'avg_loss': ...,
            'avg_hold_time': ...,
            'total_trades': ...,
            'number_of_signals_discarded': ...,
        }

    def plot_equity_curve(self) -> str:
        """Generate an equity curve plot (base64 PNG for HTML)."""

    def plot_drawdown(self) -> str:
        """Generate a drawdown chart."""

    def summary_table(self) -> str:
        """Render metrics as an HTML table."""
```

### 8.2 Monte Carlo Simulation (`quant/services/monte_carlo.py`)

```python
class MonteCarloSimulator:
    """
    Stress-test strategies by randomizing trade outcomes.

    Takes the actual trade list and reshuffles results to create
    thousands of alternate histories. Shows the range of possible
    outcomes — not just the single backtest result.

    If less than 95% of simulated outcomes are profitable →
    the strategy is not statistically significant → DISCARD IT.
    """

    def run(self, trades: list[ExecutedTrade], n_simulations: int = 10000) -> dict:
        """
        Returns:
            median_return, percentile_5, percentile_95,
            probability_of_profit, probability_of_ruin
        """
```

### 8.3 Walk-Forward Analysis (`quant/services/walk_forward.py`)

```python
class WalkForwardAnalyzer:
    """
    Advanced validation technique.

    Instead of a single train/test split, walk forward:
    - Train on months 1-6, test on month 7
    - Train on months 1-7, test on month 8
    - Train on months 1-8, test on month 9
    - ... etc.

    Strategy passes only if it performs consistently across ALL windows.
    """

    def analyze(self, strategy_fn: Callable, df: pd.DataFrame,
                 window_size: int = 180, step_size: int = 30) -> dict:
        """
        Returns per-window metrics and overall consistency score.
        """
```

### 8.4 Paper Trading Mode

**A mode where signals generate orders but they're NOT sent to Binance:**
- Order manager creates `ExecutedTrade` records with `status='paper'`
- Prices are taken from the live feed as if executed
- P&L is tracked in a "virtual balance"
- This runs alongside live signals for N days/weeks before going real

**Toggle in Django Admin:**
```python
# In quant/models.py
class QuantConfig(models.Model):
    mode = models.CharField(
        max_length=20,
        choices=[
            ('backtest', 'Backtest Only'),
            ('paper', 'Paper Trading'),
            ('live', 'Live Trading'),
        ],
        default='backtest',
    )
    virtual_balance = models.DecimalField(max_digits=20, decimal_places=2, default=10000.00)
    real_balance_limit = models.DecimalField(max_digits=20, decimal_places=2, default=500.00)
```

### 8.5 Go-Live Checklist

Before switching to `live` mode:
- [ ] All models pass backtesting with Sharpe > 1.0 out-of-sample
- [ ] Monte Carlo shows > 95% probability of profitability
- [ ] Walk-forward analysis shows consistency across all windows
- [ ] Paper trading for minimum 14 days with positive P&L
- [ ] Risk manager tested with all edge cases
- [ ] Stop-losses verified working
- [ ] Binance testnet configuration active and tested
- [ ] Daily/weekly reporting configured
- [ ] Maximum position size cap set
- [ ] Emergency kill-switch tested (can halt all trading instantly)
- [ ] Drawdown limit configured

---

## Dependencies & Installation Order

### Python Libraries

```bash
# Phase 0 — Data
pip install pandas numpy pandas-ta  # Already partially used

# Phase 1 — HMM
pip install hmmlearn

# Phase 2 — Cointegration & Statistics
pip install statsmodels scipy

# Phase 3 — Sentiment (most already installed)
pip install textblob feedparser requests  # Already in sentiment app

# Phase 4 — Machine Learning
pip install scikit-learn xgboost torch  # PyTorch for LSTM

# Phase 5 — Execution
pip install ccxt  # Alternative/additional exchange connectivity

# Phase 6 — Additional utilities
pip install matplotlib plotly  # For charts and reports
pip install tabulate  # For formatted tables
```

### Installation Order

1. `pandas`, `numpy`, `pandas-ta` — Foundation data manipulation
2. `statsmodels`, `scipy` — Cointegration, statistical tests
3. `hmmlearn` — Hidden Markov Models for regime detection
4. `scikit-learn`, `xgboost` — ML models
5. `matplotlib`, `plotly` — Visualization for backtesting
6. `ccxt` — Exchange connectivity (supplemental)
7. `torch` — LSTM/neural network models (install last, largest)

---

## Appendix: Key Mathematical Formulas

### Cointegration Test (Engle-Granger)
```
1. Regress: price_a = α + β * price_b + ε
2. Test ε for stationarity using Augmented Dickey-Fuller test
3. If ADF p-value < 0.05 → cointegrated!
```

### Half-Life of Mean Reversion
```
Compute using OLS on: Δ(spread_t) = θ * (spread_{t-1} - μ) + ε
Half-life = ln(2) / θ
```

### Kelly Criterion (General)
```
f* = (b*p - q) / b
where:
b = net odds received on the bet (profit/loss ratio)
p = probability of winning
q = probability of losing = 1 - p
```

### Sharpe Ratio
```
Sharpe = (R_p - R_f) / σ_p
where:
R_p = portfolio return
R_f = risk-free rate
σ_p = standard deviation of portfolio returns
```

### Order Book Imbalance
```
Imbalance = bid_volume / (bid_volume + ask_volume) * 100
> 60% = strong buy pressure
< 40% = strong sell pressure
```

### Maximum Drawdown
```
MDD = (Peak_value - Trough_value) / Peak_value
```

---

> **Final Renaissance Wisdom:** *"Trading is about exploiting edges so small they seem invisible. The edge is not in predicting — it's in the mathematics of large numbers, disciplined risk management, and the relentless pursuit of signals that others overlook. Build the machine, trust the machine, and let statistics do the work."*
> — Inspired by Jim Simons (1925–2024)
